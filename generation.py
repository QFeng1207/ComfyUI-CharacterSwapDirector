from __future__ import annotations

import json
import re
import time
import subprocess
import uuid
from pathlib import Path

import cv2
import av
import numpy as np
import torch
from PIL import Image, ImageOps

import nodes as comfy_nodes
import comfy.model_management as model_management
from comfy_extras.nodes_custom_sampler import BasicGuider, SamplerCustomAdvanced
try:
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide, MiniMaxH3FunControlNetApply, MiniMaxH3ReferenceToVideo, align_frame_count
except ImportError:
    MiniMaxH3AddGuide = MiniMaxH3FunControlNetApply = MiniMaxH3ReferenceToVideo = align_frame_count = None

from .media import VideoWriter, extract_audio, ffmpeg_path, iter_video, nearest_frame_indices, probe_video, read_video
from .paths import cache_dir, input_path, output_path
from .plan import active_segments, normalize_audio_mode
from .tracking import read_masks
from .resolution import output_size


def canvas_size(width, height, long_edge):
    scale = min(1.0, float(long_edge) / max(width, height))
    return max(32, round(width * scale / 32) * 32), max(32, round(height * scale / 32) * 32)


def load_reference(path, size):
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert("RGBA")
        background = Image.new("RGBA", image.size, "white")
        background.alpha_composite(image)
        image = background.convert("RGB")
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        return torch.from_numpy(np.array(image)).float().div_(255).unsqueeze(0)


def swap_prompt(role, reference_count, segment, alpha, has_audio=False):
    centers = torch.nonzero(alpha[0] > .5)
    location = "the selected person"
    if len(centers):
        y, x = centers.float().mean(dim=0).tolist()
        location = f"the person centered at x={x / alpha.shape[2]:.2f}, y={y / alpha.shape[1]:.2f} in the source video"
    auxiliary = " ".join(f"<Picture {i}> provides additional identity and body-view details of the same <Subject 1>." for i in range(2, reference_count + 1))
    description = role.get("description", "").strip()
    duration = alpha.shape[0] / 24
    audio_definition = "\n<Audio 1> is the synchronized source soundtrack; its speech rhythm guides the selected person's lip movement." if has_audio else ""
    audio_retention = "\n<Audio 1>: fully_copy - retain source speech timing and delivery throughout this clip." if has_audio else ""
    soundscape = "Follow the synchronized <Audio 1> for dialogue timing and ambience, without adding words or sound effects." if has_audio else "Silence; do not introduce dialogue or sound effects."
    return (
        "subject_definitions:\n"
        f"<Subject 1> is the replacement person whose complete appearance comes from <Picture 1>. {auxiliary} {description}\n"
        f"<Subject 2> is the precise motion, expression and lip movement of {location} in <Video 1>.\n"
        "<Subject 3> is the original environment, props and all other people in <Video 1>.\n"
        f"<Video 1> is the source video to edit, including its camera movement and action timeline.{audio_definition}\n\n"
        "summary:\n"
        "[video editing + reference generation" + (" + audio reuse" if has_audio else "") + "] The target video is an edited version of <Video 1>. "
        "Replace the selected person's full appearance with <Subject 1>, while retaining <Subject 2>, <Subject 3> and source timing.\n\n"
        "retention_analysis:\n"
        "<Subject 1> (appears throughout [Shot 1]): fully_preserved - retain face, hair, body proportions, clothing, shoes and accessories from the primary picture.\n"
        "<Subject 2> (throughout [Shot 1]): attribute_transfer - apply the exact source poses, gestures, path, expression and lip motion to <Subject 1>.\n"
        "<Subject 3> (throughout [Shot 1]): fully_preserved - retain the source environment, objects and other people.\n"
        f"<Video 1> (camera and temporal structure): fully_preserved - retain framing, camera motion and action timing.{audio_retention}\n\n"
        "detailed_description:\n"
        f"[Shot 1] (0.00s–{duration:.2f}s): Match the composition and lighting of <Video 1> frame by frame. "
        "<Subject 1> occupies the selected source person's position and follows <Subject 2> at the exact same times. "
        "Their face, hair, body and entire outfit match the primary picture across every frame, with natural scene lighting and occlusion. "
        "Retain <Subject 3> around them. Reconstruct only newly exposed background within the editable region. "
        "Keep the original camera path and playback speed. Do not add cuts, slow motion, objects or new gestures. "
        f"Additional direction: {segment.get('prompt', '').strip()}\n\n"
        f"overall_soundscape:\n{soundscape}\n\n"
        "non_diegetic_music:\nRetain only source music if present; do not add music."
    )


def pose_control(images, mode, resolution):
    if mode == "none":
        return None
    if mode == "canny":
        frames = images.mul(255).byte().numpy()
        edges = [cv2.cvtColor(cv2.Canny(frame, 80, 160), cv2.COLOR_GRAY2RGB) for frame in frames]
        return torch.from_numpy(np.stack(edges)).float().div_(255)
    if "DWPreprocessor" not in comfy_nodes.NODE_CLASS_MAPPINGS:
        raise RuntimeError("动作控制需要 comfyui_controlnet_aux 的 DWPreprocessor。请启用该插件并安装 DWPose 模型，或将动作控制选择为边缘/关闭。")
    processor = comfy_nodes.NODE_CLASS_MAPPINGS["DWPreprocessor"]()
    result = processor.estimate_pose(images, detect_hand="enable", detect_body="enable", detect_face="enable", resolution=resolution,
                                     bbox_detector="yolox_l.onnx", pose_estimator="dw-ll_ucoco_384.onnx")
    # The installed DWPose node documents a legacy result dictionary (IMAGE, POSE_KEYPOINT).
    return result["result"][0]


def _padded(tensor, count):
    if tensor.shape[0] >= count:
        return tensor[:count]
    return torch.cat((tensor, tensor[-1:].expand(count - tensor.shape[0], *tensor.shape[1:])), dim=0)


def composite_frame(original, generated, alpha):
    generated = cv2.resize(generated, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_CUBIC)
    if generated.dtype == np.uint8:
        generated = generated.astype(np.float32)
    else:
        generated = np.asarray(generated, np.float32) * 255
    alpha = np.clip(np.asarray(alpha, np.float32), 0, 1)[..., None]
    output = np.rint(original.astype(np.float32) * (1 - alpha) + generated * alpha).clip(0, 255).astype(np.uint8)
    output[alpha[..., 0] == 0] = original[alpha[..., 0] == 0]
    return output


def _cached_file(value):
    candidate = Path(value).resolve()
    root = cache_dir("renders").resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise ValueError("生成缓存不在插件缓存目录中，或文件已被清理。")
    return candidate


def _sample_frame(frames, position, interpolation):
    position = min(max(0.0, position), len(frames) - 1)
    left = int(position)
    if interpolation == "nearest":
        return frames[round(position)].numpy()
    right = min(left + 1, len(frames) - 1)
    weight = position - left
    return (frames[left] * (1 - weight) + frames[right] * weight).numpy()


def render_part(video, segment, role, state, track, model, clip, video_vae, audio_vae, model_patch, noise, sampler, sigmas,
                long_edge, control_mode, control_strength, interpolation, directory, progress, previous_role=None, previous_segment=None):
    original = input_path(video["file"])
    start, end = float(segment["start"]), float(segment["end"])
    info = video["info"]
    width, height = canvas_size(info["width"], info["height"], long_edge)
    prefix = None
    generation_start = start
    # Carry a five-frame clip only across adjacent pieces of the same shot.
    if previous_segment is not None and abs(previous_segment["end"] - start) < 1e-4 and segment.get("scene_id") is not None and segment.get("scene_id") == previous_segment.get("scene_id"):
        generation_start = max(previous_segment["start"], start - 5 / 24)
        prefix, _ = read_video(_cached_file(previous_segment["path"]), generation_start - previous_segment["start"], start - previous_segment["start"], fps=24, size=(width, height))
        prefix = _padded(prefix, 5)
        generation_start = start - 5 / 24
    if previous_role:
        source_offset = previous_role.get("source_offset", 0.)
        source, times = read_video(_cached_file(previous_role["path"]), 0, end - start - source_offset, fps=24, size=(width, height))
        times = [time + start + source_offset for time in times]
    else:
        source, times = read_video(original, start, end, fps=24, size=(width, height))
    if prefix is not None:
        source = torch.cat((prefix, source))
        times = [generation_start + i / 24 for i in range(5)] + times
    length = align_frame_count(max(5, source.shape[0]))
    if length > 362:
        raise ValueError("单段加连续性上下文后超过 H3 的 15 秒训练范围；请缩短分段（建议 2–10 秒）。")
    source = _padded(source, length)
    indices = nearest_frame_indices(track["timestamps"], times)
    alpha = _padded(read_masks(track, role["id"], indices, size=(width, height)), length)
    if not bool((alpha > 0).any()):
        raise ValueError(f"{role.get('name', role['id'])} 在此分段的蒙版为空，请检查绑定或停用该段中的角色。")
    progress(.08, f"提取动作控制：{role.get('name', role['id'])}")
    control = pose_control(source, control_mode, max(width, height))
    photos = {item["id"]: item for item in state["images"]}
    image_ids = role["image_ids"]
    if not image_ids:
        raise ValueError(f"角色 {role.get('name', role['id'])} 尚未绑定参考图。")
    primary = role.get("primary_image_id") or image_ids[0]
    ordered = [primary] + [item for item in image_ids if item != primary]
    refs = {f"ref_image_{i}": load_reference(input_path(photos[item]["file"]), long_edge) for i, item in enumerate(ordered)}
    source_audio = extract_audio(original, max(0, generation_start), end)
    progress(.15, "编码主参考图、辅助图与原动作视频")
    conditioning, latent = MiniMaxH3ReferenceToVideo.execute(clip, swap_prompt(role, len(refs), segment, alpha, source_audio is not None), width, height, length,
        ref_image_size="match", vae=video_vae, audio_vae=audio_vae, ref_images=refs,
        ref_videos={"ref_video_0": source}, ref_video_audios={"ref_video_audio_0": source_audio} if source_audio else None).result
    if prefix is not None or source_audio is not None:
        conditioning = MiniMaxH3AddGuide.execute(conditioning, latent, 0, vae=video_vae, audio_vae=audio_vae,
                                                  image=prefix, audio=source_audio).result[0]
    controlled = MiniMaxH3FunControlNetApply.execute(model, model_patch, video_vae, control_strength, 0., 1.,
                                                     control_video=control, mask=alpha, source_video=source).result[0]
    guider = BasicGuider.execute(controlled, conditioning).result[0]
    progress(.25, "开始人物生成；采样进度显示在 ComfyUI 进度条")
    _, denoised = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent).result
    video_latent = denoised["samples"].tensors[0]
    progress(.78, "解码人物视频")
    generated = video_vae.decode(video_latent).detach().float().cpu()
    # Native VAEDecode flattens a temporal VAE's [B,T,H,W,C] to IMAGE [N,H,W,C].
    if generated.ndim == 5:
        generated = generated.reshape(-1, *generated.shape[-3:])
    if generated.ndim != 4 or generated.shape[-1] < 3:
        raise ValueError("H3 视频 VAE 输出格式无效，请接入 MiniMax H3 video VAE。")
    generated = generated[..., :3].clamp_(0, 1)
    del conditioning, latent, guider, controlled, denoised, video_latent, source, refs, control, alpha
    model_management.soft_empty_cache()
    progress(.85, "合成原尺寸画面，保留蒙版外原像素")
    return write_composite_part(video, segment, role, track, generated, generation_start, directory, interpolation,
                                previous_role=previous_role)


def write_composite_part(video, segment, role, track, generated, generation_start, directory, interpolation,
                         previous_role=None, generation_fps=24):
    original = input_path(video["file"])
    info = video["info"]
    start, end = float(segment["start"]), float(segment["end"])
    target = directory / (uuid.uuid4().hex + ".mp4")
    base = iter_video(_cached_file(previous_role["path"])) if previous_role else iter_video(original, start, end)
    frame_count = 0
    first_source_time = None
    with VideoWriter(target, info["width"], info["height"], info["fps_num"], info["fps_den"], lossless=True, duration=end - start) as writer:
        for original_frame, source_time in base:
            absolute_time = source_time + start + previous_role.get("source_offset", 0.) if previous_role else source_time
            if first_source_time is None:
                first_source_time = absolute_time
            mask_index = nearest_frame_indices(track["timestamps"], [absolute_time])[0]
            editing = read_masks(track, role["id"], [mask_index])[0].numpy()
            sample = _sample_frame(generated, (absolute_time - generation_start) * generation_fps, interpolation)
            writer.write(composite_frame(original_frame, sample, editing), time=max(0, absolute_time - start))
            frame_count += 1
    return {"path": str(target), "start": start, "end": end, "segment_id": segment["id"], "scene_id": segment.get("scene_id"),
            "frames": frame_count, "source_offset": first_source_time - start, "roles": (previous_role["roles"] if previous_role else []) + [role["id"]]}


def render_batch(plan, tracks, model, clip, video_vae, audio_vae, model_patch, noise, sampler, sigmas,
                 long_edge=512, control_mode="pose", control_strength=1., interpolation="linear", progress=None):
    if MiniMaxH3ReferenceToVideo is None:
        raise RuntimeError("H3 实验分支需要宿主提供原生 MiniMax H3 节点；Wan 工作流请连接 Wan Animate 生成节点。")
    def operation(video, segment, role, state, track, directory, part_progress, previous_role, previous_segment):
        return render_part(video, segment, role, state, track, model, clip, video_vae, audio_vae, model_patch, noise, sampler, sigmas,
                           long_edge, control_mode, control_strength, interpolation, directory, part_progress,
                           previous_role=previous_role, previous_segment=previous_segment)
    return render_segments(plan, tracks, operation, long_edge, progress)


def render_segments(plan, tracks, operation, long_edge, progress=None, segments=None):
    state = plan["state"]
    roles = {item["id"]: item for item in state["roles"]}
    work = cache_dir("renders") / uuid.uuid4().hex
    work.mkdir(parents=True)
    segments = active_segments(plan) if segments is None else segments
    count = sum(len(role_ids) for _, _, role_ids in segments)
    if not count:
        raise ValueError("请在导演台绑定至少一个人物，然后选择替换分段。")
    results, last_segments = {}, {}
    done = 0
    for video, segment, role_ids in segments:
        if video["id"] not in tracks["videos"]:
            raise ValueError("人物跟踪未覆盖当前视频；请重新运行跟踪节点。")
        result = None
        for role_id in role_ids:
            if role_id not in tracks["videos"][video["id"]]["roles"]:
                raise ValueError("分段选择的角色没有绑定到这个视频。")
            def part_progress(fraction, message):
                if progress:
                    progress((done + fraction) / count, message, video["id"])
                model_management.throw_exception_if_processing_interrupted()
            result = operation(video, segment, roles[role_id], state, tracks["videos"][video["id"]], work, part_progress,
                               result, last_segments.get((video["id"], role_id)))
            last_segments[(video["id"], role_id)] = result
            done += 1
        results.setdefault(video["id"], []).append(result)
    if progress:
        progress(1, "人物生成与原画面合成完成", "")
    return {"plan": plan, "videos": results, "work_dir": str(work), "processing_long_edge": long_edge}


def assemble_video(video, parts, target, trial=False, base_path=None):
    info = video["info"]
    width, height = output_size(video)
    ordered = sorted(parts, key=lambda part: part["start"])
    start = ordered[0]["start"] if trial else 0
    end = ordered[-1]["end"] if trial else info["duration"]
    readers = {part["path"]: iter(iter_video(_cached_file(part["path"]))) for part in ordered}
    frame_count = 0
    base_reader = iter(iter_video(base_path, start, end)) if base_path else None
    with VideoWriter(target, width, height, info["fps_num"], info["fps_den"], lossless=True, duration=end - start) as writer:
        for frame, time in iter_video(input_path(video["file"]), start, end):
            if base_reader is not None:
                frame, base_time = next(base_reader)
                if abs(base_time - time) > 1 / 90000 + 1e-6:
                    raise ValueError("上一版帧时间与原素材不一致，不能复用；请先完整生成")
            part = next((item for item in ordered if item["start"] - 1e-9 <= time < item["end"] - 1e-9), None)
            if part is not None:
                rendered, _ = next(readers[part["path"]])
                frame = rendered
            elif frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
            writer.write(frame, time=max(0., time - start))
            frame_count += 1
    return {"start": start, "end": end, "duration": end - start, "frames": frame_count}


def mux_and_save(video_only, source_audio, target, duration, *, audio_start=0, compatible=False, lossless_audio=True):
    arguments = [ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(video_only)]
    if source_audio:
        # Input seeking drops AAC priming; trim decoded samples for a trial.
        arguments += ["-i", str(source_audio)]
    arguments += ["-map", "0:v:0"]
    audio_encoding = "none"
    if source_audio:
        with av.open(str(source_audio)) as source:
            codec = source.streams.audio[0].codec_context.name if source.streams.audio else None
        movable = codec in ("aac", "mp3", "mp2", "alac", "ac3", "eac3", "pcm_s16le", "pcm_s24le", "pcm_s32le", "pcm_f32le", "pcm_f64le")
        if codec is not None:
            audio_encoding = "aac" if compatible else ("copy" if movable and lossless_audio and audio_start == 0 else "pcm_f32le")
            arguments += ["-map", "1:a:0", "-c:a", audio_encoding]
            if audio_start > 0:
                arguments += ["-af", f"atrim=start={audio_start}:duration={duration},asetpts=PTS-STARTPTS"]
    arguments += (["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv444p"] if compatible else ["-c:v", "copy"])
    if compatible:
        arguments += ["-movflags", "+faststart"]
    arguments += ["-t", str(duration), str(target)]
    completed = subprocess.run(arguments, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode:
        raise RuntimeError("视频保存失败：" + completed.stderr.strip()[-1000:])
    return {"path": str(target), "audio_encoding": audio_encoding}


def save_results(results, audio, filename_prefix, compatible=False, progress=None):
    state = results["plan"]["state"]
    trial = bool(state["settings"].get("trial", {}).get("enabled"))
    written = []
    for video in state["videos"]:
        parts = results["videos"].get(video["id"])
        if not parts:
            continue
        temporary = cache_dir("renders") / (uuid.uuid4().hex + ".mp4")
        candidate = state["settings"].get("candidate", {}).get("enabled", False)
        timing = assemble_video(video, parts, temporary, trial or candidate, results.get("base_videos", {}).get(video["id"]))
        mode = normalize_audio_mode(state["settings"]["audio_mode"])
        audio_source = input_path(video["file"]) if mode == "original" else None
        audio_report = {"mode": mode}
        prefix = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', str(filename_prefix)).strip(' .')[:60] or "CharacterSwapH3"
        name = f"{prefix}_{video['id']}_{uuid.uuid4().hex[:8]}" + ("_trial" if trial else "")
        target = output_path(name + (".mp4" if compatible else ".mov"))
        result_id = "h3_" + uuid.uuid4().hex if results.get("backend") == "minimax_h3_reference" else target.stem
        encoded = mux_and_save(temporary, audio_source, target, timing["duration"], audio_start=timing["start"], compatible=compatible)
        preview = target if compatible else cache_dir("previews") / (uuid.uuid4().hex + ".mp4")
        if not compatible:
            mux_and_save(temporary, audio_source, preview, timing["duration"], audio_start=timing["start"], compatible=True)
        temporary.unlink(missing_ok=True)
        report = {**timing, "source": video["file"], "source_signature": video.get("source_signature"), "video_id": video["id"], "planner_node": results["plan"].get("planner_node"),
                  "result_id": result_id, "output_file": target.name, "created": time.time(), "parent_result_id": state["settings"].get("revision", {}).get("base_result_id") if results.get("base_videos", {}).get(video["id"]) else None,
                  "resolution": list(output_size(video)), "source_resolution": [video["info"]["width"], video["info"]["height"]], "original_timestamps": True,
                  "outside_mask_rgb_lossless": not compatible if results.get("backend") != "minimax_h3_reference" else False,
                  "background_target": "visual_consistency" if results.get("backend") == "minimax_h3_reference" else "outside_mask_pixel_exact",
                  "trial": trial, "candidate": candidate, "audio_mode": mode, "audio": audio_report, "audio_encoding": encoded["audio_encoding"],
                  "processing_long_edge": results["processing_long_edge"], "backend": results.get("backend", "minimax_h3"),
                  "generation_fps": results.get("generation_fps", 24), "segments": parts,
                  "plugin_version": results.get("plugin_version"), "sampling": results.get("sampling", {}),
                  "replacement_inference_slices": len({part.get("generation_group_id", part["segment_id"]) for part in parts if part.get("sampling_steps", 0)}),
                  "source_shots": video.get("shots", []), "warnings": list(dict.fromkeys(message for part in parts for message in part.get("warnings", [])))}
        report["expanded_center_rgb_lossless"] = not compatible and any(part.get("expansion", {}).get("center_rgb_preserved") for part in parts)
        # Persist a browser-playable preview alongside the version, independent of temp cleanup.
        if results.get("backend") == "minimax_h3_reference" and not compatible:
            persistent_preview = target.with_name(target.stem + "_preview.mp4")
            preview.replace(persistent_preview)
            preview = persistent_preview
        report["preview_file"] = preview.name
        if candidate:
            absolute_time = state["settings"]["candidate"]["time"]
            frame = next(iter_video(target, absolute_time - timing["start"], timing["duration"]), None)
            if frame is None:
                raise ValueError("候选时间未找到有效画面")
            candidate_path = target.with_suffix(".png")
            Image.fromarray(frame[0]).save(candidate_path)
            report["candidate_file"] = candidate_path.name
            report["candidate_time"] = absolute_time
        output_path(result_id + ".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append({"name": target.name, "result_id": result_id, "video_id": video["id"], "planner_node": results["plan"].get("planner_node"), "path": str(target), "preview_path": str(preview), "report": report})
        if progress:
            progress(len(written) / len(results["videos"]), "已保存 " + target.name, video["id"])
    return written
