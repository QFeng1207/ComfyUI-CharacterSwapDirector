"""H3 Ref2VA replacement; background expansion with an exact RGB center."""
from __future__ import annotations

import uuid
import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
import comfy.model_management as model_management
from comfy_extras.nodes_custom_sampler import BasicGuider, SamplerCustomAdvanced
from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide, MiniMaxH3ReferenceToVideo, align_frame_count

from .generation import canvas_size, load_reference, _sample_frame, _cached_file
from .media import VideoWriter, iter_video, read_video, nearest_frame_indices
from .paths import cache_dir, input_path
from .plan import active_segments
from .reference_plan import frame_references, revision_base
from .resolution import center_rect, output_size
from .shot_plan import shot_plan


def decoded_video(vae, sample):
    frames = vae.decode(sample).detach().float().cpu()
    if frames.ndim == 5:
        frames = frames.reshape(-1, *frames.shape[-3:])
    if frames.ndim != 4 or frames.shape[-1] < 3:
        raise ValueError("H3 视频 VAE 输出格式无效")
    return frames[..., :3].clamp_(0, 1)


def fit_source_clip(frames, times, length):
    """Distribute the source motion over the full H3 clip, including its last frame."""
    first, last = times[0], times[-1]
    scale = (length - 1) / (24 * (last - first)) if last > first else 1.
    clock = {"source_first": first, "source_last": last, "scale": scale, "frames": length,
             "method": "uniform_source_time"}
    targets = np.linspace(first, last, length)
    indices = nearest_frame_indices(times, targets)
    return frames[indices].contiguous(), clock


def frame_position(time, clock):
    return (time - clock["source_first"]) * clock["scale"] * 24


def expansion_prompt(length, rectangle, reference_count=1):
    x, y, width, height = rectangle
    references = ", ".join(f"<Video {i + 1}>" for i in range(reference_count))
    return (f"subject_definitions:\n<Subject 1> is ONLY the original background visible in {references}: its walls, floor, ceiling, lighting and existing structures. "
            "These videos are original environment references. No replacement-character picture or video is provided.\n\n"
            "summary:\n[empty background video generation] Generate ONLY the original environment, with absolutely NO people anywhere in the image. This background plate will fill the new canvas.\n\n"
            "retention_analysis:\nAll source people: removed - reconstruct only the environment. "
            "<Subject 1>: fully_preserved - original environment, materials, geometry, colors and illumination; extend only visible background surfaces.\n\n"
            f"detailed_description:\n[Shot 1] (0s–{length / 24:.3f}s): Extend the existing background around the original view at x={x}, y={y}, width={width}, height={height}, using the SAME architecture, surfaces and perspective. "
            "Maintain the original camera motion and timing. Every pixel must depict empty background, never a person. "
            "Do not extend a cropped head, hand, arm, leg, hair or garment into the image. No faces, skin, silhouettes, clothes or human reflections. "
            "Do not create a new scene, extra people, new furniture, props, text, borders or duplicate subjects. Do not stretch or alter the central picture. "
            "There must be no blank bars in the generated surrounding area.\n\n"
            "overall_soundscape:\nSilent generation; original sound is separately copied.\n\nnon_diegetic_music:\nNo generated music.")


def background_crops(original, video, roles):
    """Prefer original areas outside the union of the user's person boxes."""
    height, width = original.shape[1:3]
    bounds = []
    selected = {role["id"] for role in roles}
    for binding in video.get("bindings", []):
        if binding["role_id"] not in selected:
            continue
        for anchor in binding.get("anchors", []):
            if anchor.get("box"):
                bounds.append(anchor["box"])
            else:
                for point in anchor.get("points", []):
                    if point.get("label", 1):
                        bounds.append([max(0, point["x"] - .3), 0, min(1, point["x"] + .3), 1])
    if not bounds:
        return [original], []
    left = max(0, int((min(b[0] for b in bounds) - .06) * width))
    right = min(width, int(np.ceil((max(b[2] for b in bounds) + .06) * width)))
    top = max(0, int((min(b[1] for b in bounds) - .06) * height))
    bottom = min(height, int(np.ceil((max(b[3] for b in bounds) + .06) * height)))
    rectangles = [(0,0,left,height),(right,0,width-right,height),(0,0,width,top),(0,bottom,width,height-bottom)]
    rectangles = sorted((r for r in rectangles if min(r[2:]) >= 32), key=lambda r:r[2]*r[3], reverse=True)[:2]
    return [original[:,y:y+h,x:x+w].contiguous() for x,y,w,h in rectangles] or [original], rectangles


def fill_background_borders(frames, rectangle):
    """Remove generated black padding within exterior regions only."""
    x,y,cw,ch = rectangle
    height,width = frames.shape[1:3]
    changes = []
    for left,top,w,h in [(0,0,width,y),(0,y+ch,width,height-y-ch),(0,y,x,ch),(x+cw,y,width-x-cw,ch)]:
        if not w or not h:
            continue
        region = frames[:,top:top+h,left:left+w]
        black = (region <= 1 / 255).all(dim=-1)
        # Use the largest padding seen in any frame, keeping the crop fixed in time.
        rows,cols = black.all(dim=-1).any(dim=0),black.all(dim=-2).any(dim=0)
        a,b,l,r = 0,h,0,w
        while a<b and bool(rows[a]): a+=1
        while b>a and bool(rows[b-1]): b-=1
        while l<r and bool(cols[l]): l+=1
        while r>l and bool(cols[r-1]): r-=1
        if (a,b,l,r) == (0,h,0,w) or min(b-a,r-l)<4:
            continue
        cropped = region[:,a:b,l:r].permute(0,3,1,2)
        frames[:,top:top+h,left:left+w] = torch.nn.functional.interpolate(cropped,size=(h,w),mode="bicubic",align_corners=False).permute(0,2,3,1).clamp(0,1)
        changes.append({"region":[left,top,w,h],"used_crop":[l,a,r-l,b-a]})
    return frames,changes


def expand_frames(core, original, model, clip, vae, noise, sampler, sigmas, size, progress=None, background_refs=None):
    """Generate an H3 background plate; use only its border and copy center RGB."""
    width, height = size
    core = core.detach().float().cpu().clamp(0, 1)
    core = torch.round(core * 255) / 255  # define the preserved RGB before VAE round-trip
    count, core_height, core_width, _ = core.shape
    x, y = (width - core_width) // 2, (height - core_height) // 2
    if core_width == width and core_height == height:
        return core, {}
    if core_width > width or core_height > height or any(v % 16 for v in (x, y, core_width, core_height)):
        raise ValueError("扩画幅的中心矩形必须在目标画布内并按 H3 像素网格对齐")
    rectangle = [x, y, core_width, core_height]
    background_refs = background_refs or [original]
    prompt = expansion_prompt(count, rectangle, len(background_refs))
    if progress:
        progress(.05, "使用原片背景条件编码新增画幅")
    conditioning, latent = MiniMaxH3ReferenceToVideo.execute(clip, prompt, width, height, count, vae=vae,
        ref_videos={f"ref_video_{i}":frames for i,frames in enumerate(background_refs)}).result
    guider = BasicGuider.execute(model, conditioning).result[0]
    if progress:
        progress(.25, "H3 生成背景底图；仅合成外围，中心原样保留")
    _, denoised = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent).result
    expanded = decoded_video(vae, denoised["samples"].tensors[0])
    if expanded.shape != (count, height, width, 3):
        raise ValueError("H3 扩画幅输出尺寸或帧数与目标不同")
    expanded, border_fit = fill_background_borders(expanded, rectangle)
    # Only the background plate's exterior is used; the center never replaces core RGB.
    expanded[:, y:y + core_height, x:x + core_width] = core
    del conditioning, latent, guider, denoised
    model_management.soft_empty_cache()
    return expanded, {"center_rect": rectangle, "center_rgb_preserved": True, "background_reference": "original_only",
                      "native_border_noise_mask": False, "method": "h3_background_plate_border_composite", "background_border_fit": border_fit,
                      "sampling_steps": len(sigmas) - 1, "prompt": prompt}


def character_views(path, sheet_mode):
    image = load_reference(path, 2048)
    if sheet_mode != "auto" or image.shape[2] < image.shape[1] * 1.75:
        return [image]
    # Only crop a clearly separated white multi-view sheet; ordinary photos stay intact.
    rgb = image[0].numpy()
    occupied = np.mean(np.min(rgb, axis=-1) < .92, axis=0) > .02
    runs, start = [], None
    for i, value in enumerate(np.r_[occupied, False]):
        if value and start is None:
            start = i
        if not value and start is not None:
            if i - start > rgb.shape[1] * .06:
                runs.append((start, i))
            start = None
    if len(runs) < 2 or len(runs) > 6:
        return [image]
    margin = max(2, round(rgb.shape[1] * .01))
    return [image[:, :, max(0, left - margin):min(rgb.shape[1], right + margin)].contiguous() for left, right in runs]


def character_image(path, sheet_mode):
    return character_views(path, sheet_mode)[0]


def location_description(video, role, start, end, clock_start=None, time_map=None):
    clock_start = start if clock_start is None else clock_start
    def local_time(time):
        return max(0., frame_position(time, time_map) / 24) if time_map else time - clock_start
    windows = shot_plan(video, {"start": start, "end": end, "role_mode": "manual", "role_ids": [role["id"]]})
    lines = []
    for shot in windows:
        for track in shot["tracks"]:
            for position in track["positions"]:
                if position["interpolated"]:
                    continue
                left, top, right, bottom = position["box"]
                method = "user anchor"
                lines.append(f"at {local_time(position['time']):.6f}s: {method}, normalized box left={left:.6f}, top={top:.6f}, right={right:.6f}, bottom={bottom:.6f}, "
                             f"center x={(left + right) / 2:.2f}, y={(top + bottom) / 2:.2f}")
            for anchor in track["anchors"]:
                if not start <= anchor["time"] < end:
                    continue
                for point in anchor.get("points", []):
                    label = "include" if point.get("label", 1) else "exclude"
                    lines.append(f"at {local_time(anchor['time']):.6f}s: {label} point x={point['x']:.6f}, y={point['y']:.6f}")
                for stroke in anchor.get("strokes", []):
                    label = "include" if stroke.get("label", 1) else "exclude"
                    points = ", ".join(f"({p['x']:.6f},{p['y']:.6f})" for p in stroke.get("points", []))
                    lines.append(f"at {local_time(anchor['time']):.6f}s: {label} brush radius={stroke.get('radius', .015):.6f}, path={points}")
    return "; ".join(lines) or "no user location inside this interval; follow the matched performer in <Video 1>"


def references_and_prompt(video, segment, roles, state, start, end, length, sheet_mode, target_core=None, time_map=None):
    time_map = time_map or {"source_first": start, "scale": length / (24 * (end - start)), "frames": length}
    duration = length / 24
    def local_time(time):
        return min(duration, max(0., frame_position(time, time_map) / 24))
    images = {item["id"]: item for item in state["images"]}
    refs, definitions, retention, guide_images = {}, [], [], []
    shots = shot_plan(video, segment, start, end)
    mixed_partial_cast = len(roles) > 1 and any(track.get("body_part") for shot in shots for track in shot["tracks"])
    subject_ids = {role["id"]: f"<Subject {number}>" for number, role in enumerate(roles, 1)}
    frame_refs = frame_references(state, video["id"], segment["start"], segment["end"])
    explicit_count = sum(len(role["image_ids"]) for role in roles) + len(frame_refs)
    if explicit_count > 9:
        raise ValueError("H3 每段最多接受 9 张参考图（角色图与画面参考合计）；请减少本段参考图，或分段处理")
    extra_budget = 9 - explicit_count
    role_labels, extra_views = {}, []
    for role in roles:
        primary = role.get("primary_image_id") or (role.get("image_ids") or [None])[0]
        if not primary:
            raise ValueError(f"角色 {role.get('name', role['id'])} 尚未绑定参考图片")
        ordered = [primary] + [item for item in role["image_ids"] if item != primary]
        labels = []
        for index, image_id in enumerate(ordered):
            labels.append(f"<Picture {len(refs) + 1}>")
            views = character_views(input_path(images[image_id]["file"]), sheet_mode if index == 0 else "whole_image")
            refs[f"ref_image_{len(refs)}"] = views[0]
            if index == 0 and not mixed_partial_cast:
                extra_views.append((role["id"], views[1:]))
        role_labels[role["id"]] = labels
    # Explicit references take priority; distribute automatic extra views between roles.
    while extra_budget and any(views for _, views in extra_views):
        for role_id, views in extra_views:
            if not views or not extra_budget:
                continue
            role_labels[role_id].append(f"<Picture {len(refs) + 1}>")
            refs[f"ref_image_{len(refs)}"] = views.pop(0)
            extra_budget -= 1
    for number, role in enumerate(roles, 1):
        labels = role_labels[role["id"]]
        subject = f"<Subject {number}>"
        definitions.append(f"{subject} is the replacement person: identity, face, hair, body proportions, clothing, footwear and accessories from {labels[0]}; "
                           f"additional views of the SAME person: {', '.join(labels[1:]) or 'none'}. {role.get('description', '')}")
        binding = next(b for b in video["bindings"] if b["role_id"] == role["id"])
        if binding.get("description", "").strip():
            # In mixed partial views, keep old full-person traits out of target subject definitions.
            locator_section = retention if mixed_partial_cast else definitions
            locator_section.append(f"The INPUT performer assigned to {subject} is identified in <Video 1> by: {binding['description'].strip()} "
                               "These are INPUT-only locator traits, never OUTPUT appearance instructions. Discard this performer's original appearance and costume. Replace this source identity and every visible garment from the FIRST frame, including partial limbs; the OUTPUT appearance comes exclusively from the replacement pictures.")
        appearances = ", ".join(f"[Shot {n}]" for n, shot in enumerate(shots, 1) if role["id"] in shot["role_ids"])
        retention.append(f"{subject} (assigned only to {appearances}): fully_preserved - appearance and entire outfit from {labels[0]}; "
                         "attribute_transfer - action, expression, position and occlusion of its matched source performer. Never transfer this identity to a different source performer.")
    environment = f"<Subject {len(roles) + 1}>"
    definitions += [f"{environment} is the original scene, architecture, floor, props, lighting, textures and all unselected people in <Video 1>.",
                    "<Video 1> is the original video being edited, providing the action timeline, original background, framing, camera movement and shot order."]
    retention.append(f"{environment}: fully_preserved - scene layout, objects, colors and lighting. "
                     "<Video 1>: partially_preserved - transfer ONLY the camera trajectory, framing, actions, expressions and environment. "
                     "The selected source performer's identity, face, hair, body and costume: removed - completely replaced by the assigned picture identity.")
    for reference in frame_refs:
        label = f"<Picture {len(refs) + 1}>"
        image = load_reference(input_path(images[reference["image_id"]]["file"]), 2048)
        if target_core and reference["kind"] == "guide":
            # Corrected expanded screenshots include the border; replacement uses their center.
            out_width, out_height, x, y, core_width, core_height = target_core
            if image.shape[2] / image.shape[1] == out_width / out_height:
                left, top = round(x / out_width * image.shape[2]), round(y / out_height * image.shape[1])
                right = round((x + core_width) / out_width * image.shape[2])
                bottom = round((y + core_height) / out_height * image.shape[1])
                image = image[:, top:bottom, left:right].contiguous()
        refs[f"ref_image_{len(refs)}"] = image
        reference_time = local_time(reference["time"])
        if reference["kind"] == "guide":
            shot_number = next(n for n, shot in enumerate(shots, 1) if shot["start"] <= reference["time"] < shot["end"])
            definitions.append(f"{label} is the provided complete target frame at {reference_time:.3f}s of [Shot {shot_number}], defining the desired composition, background and replacement appearance. {reference.get('purpose', '')}")
            retention.append(f"{label}: fully_preserved as the visual anchor at {reference_time:.3f}s; maintain coherent scene and character on either side.")
            guide_images.append((min(length - 1, round(reference_time * 24)), image, reference["id"]))
        else:
            definitions.append(f"{label} is an unedited source scene/action reference around {reference_time:.3f}s; use its environment and composition, with the original performer replaced by the assigned subject. {reference.get('purpose', '')}")
    if len(refs) > 9:
        raise ValueError("H3 每段最多接受 9 张参考图（角色图与画面参考合计）；请减少本段参考图，或分段处理")
    prompt = "subject_definitions:\n" + "\n".join(definitions)
    prompt += "\n\nsummary:\n[video editing + reference generation] Recreate <Video 1> with the selected performer already replaced by the assigned picture person from the first frame onward. "
    prompt += "The replacement occupies the original performer's space and performs the same actions in the same scene. Keep the reference video's composition and camera movement."
    prompt += "\n\nretention_analysis:\n" + "\n".join(retention)
    directions = []
    for number, shot in enumerate(shots, 1):
        text = (f"[Shot {number}] ({local_time(shot['start']):.6f}s–{local_time(shot['end']):.6f}s): Match the corresponding frames of <Video 1> throughout this entire interval. "
                "Keep its shot size, viewpoint, camera movement, background, props, lighting, visible actions and occlusions in their reference order. "
                "Retain the source crop: a face close-up stays a face close-up, and a cropped torso stays cropped. Reference portraits define appearance, never a new full-body composition. "
                "Follow the source action in chronological order from its FIRST to LAST frame: never use the ending action as the opening, replay a stage, or invent a new pose. ")
        if number > 1:
            text += "Make the original hard cut at this boundary; do not dissolve, delay the cut or carry a person across it. "
        for track in shot["tracks"]:
            role = next(r for r in roles if r["id"] == track["role_id"])
            subject = subject_ids[role["id"]]
            partial = {"hand_arm": "hand and forearm", "hands": "hands", "feet": "feet", "torso": "cropped torso", "head": "head"}.get(track.get("body_part"))
            text += (f"{subject} replaces only its matched source performer, and only while that performer is visible. "
                     "Every visible body part and garment already uses this replacement's appearance at the start; there is no transformation or costume change. "
                     "The original selected person is absent, with exactly one replacement in its place. "
                     "Follow the complete reference performance, including entrances, exits, occlusions and interactions. "
                     "Preserve partial views: when the reference shows only a hand, arm or foot, show only the corresponding replacement body part, with the face and torso remaining outside the crop. "
                     "User location markers on the reference video's time axis (normalized coordinates, origin at top left): "
                     + location_description(video, role, shot["start"], shot["end"], clock_start=start, time_map=time_map) + ". ")
            if partial:
                text += (f"This performer's assigned view is only its visible {partial} in <Video 1>. "
                         "Replace these visible parts and their clothing, without adding an off-screen face, torso or second person. "
                         "Do not transfer this partial performer's identity to the other visible performer. ")
            if track.get("description"):
                text += f"Source location and visibility for {subject}: {track['description']}. "
            text += "These markers identify the source performer only at the marked instants. Follow <Video 1> for positions and visibility between and after markers; never force a person to remain in the last box. "
            text += ("Adapt the performance naturally to the replacement's anatomy and proportions, while keeping the original movement direction, timing, framing and contact points. "
                     "Preserve hand-to-prop grips, foreground/background occlusion and the original prop shapes; do not turn a prop into clothing or add a new prop. "
                     "When the source performer exits, keep the vacated scene and do not bring the replacement back into frame. ")
        for role in roles:
            if role["id"] not in shot["role_ids"]:
                text += f"{subject_ids[role['id']]} is not assigned to this shot. Do not insert it or use it to replace another person. "
        if not shot["role_ids"]:
            text += "No performer is selected in this shot. Preserve the source picture and all its people. "
        if shot.get("prompt"):
            text += f"Specific source shot action/camera sequence ({shot['source_start']:.6f}s–{shot['source_end']:.6f}s on the original video): {shot['prompt']} "
        text += segment.get("prompt", "")
        directions.append(text)
    prompt += "\n\ndetailed_description:\n" + "\n".join(directions)
    prompt += "\n\noverall_soundscape:\nSilent generation. The original soundtrack is separately copied at export when selected.\n\nnon_diegetic_music:\nNo generated music."
    return refs, prompt, guide_images


def write_fullframe_part(video, segment, generated, generation_start, directory, interpolation, roles, preserve_center=None,
                         unchanged_windows=(), base_path=None, time_map=None):
    target = directory / (uuid.uuid4().hex + ".mp4")
    info = video["info"]
    width, height = output_size(video)
    start, end = segment["start"], segment["end"]
    count = 0
    offset = None
    center_start = preserve_center.get("start", start) if preserve_center else start
    center_reader = iter(iter_video(_cached_file(preserve_center["path"]), max(0, start - center_start), max(0, end - center_start))) if preserve_center else None
    base_reader = iter(iter_video(_cached_file(base_path), start, end)) if base_path and unchanged_windows else None
    with VideoWriter(target, width, height, info["fps_num"], info["fps_den"], lossless=True, duration=end - start) as writer:
        for original, time in iter_video(input_path(video["file"]), start, end):
            if offset is None:
                offset = time - start
            unchanged = any(left - 1e-9 <= time < right - 1e-9 for left, right in unchanged_windows)
            base_frame, base_time = next(base_reader) if base_reader is not None else (original, time)
            if base_reader is not None and abs(base_time - time) > 1 / 90000 + 1e-6:
                raise ValueError("底稿画面时间与原片不一致")
            if unchanged and preserve_center is None:
                if base_path and base_frame.shape[1::-1] != (width, height):
                    bx, by, bw, bh = center_rect(info["width"], info["height"], base_frame.shape[1], base_frame.shape[0])
                    base_frame = base_frame[by:by + bh, bx:bx + bw]
                frame = base_frame if base_frame.shape[1::-1] == (width, height) else cv2.resize(base_frame, (width, height), interpolation=cv2.INTER_CUBIC)
            else:
                index = frame_position(time, time_map) if time_map else (time - generation_start) * 24
                sample = _sample_frame(generated, index, interpolation)
                frame = cv2.resize(sample, (width, height), interpolation=cv2.INTER_CUBIC)
                frame = np.rint(frame * 255).clip(0, 255).astype(np.uint8)
            if center_reader is not None:
                center_frame, center_time = next(center_reader)
                if abs(center_time - max(0, time - center_start)) > 1 / 90000 + 1e-6:
                    raise ValueError("扩画幅中心帧时间与原片不一致")
                left, top, cw, ch = preserve_center["rect"]
                frame[top:top + ch, left:left + cw] = center_frame
            writer.write(frame, time=max(0, time - start))
            count += 1
    return {"path": str(target), "start": start, "end": end, "segment_id": segment["id"], "scene_id": segment.get("scene_id"), "frames": count, "source_offset": offset or 0, "roles": [role["id"] for role in roles]}


def generation_segments(plan):
    """Keep user ranges intact while isolating every source cut at inference."""
    result = []
    for video, segment, _ in active_segments(plan):
        windows = shot_plan(video, segment)
        for index, window in enumerate(windows, 1):
            part = {**segment, "start": window["start"], "end": window["end"],
                    "id": segment["id"] if len(windows) == 1 else f"{segment['id']}_shot_{index}",
                    "parent_segment_id": segment["id"], "source_shot_id": window["id"]}
            result.append((video, part, window["role_ids"]))
    return result


def generation_groups(plan):
    """Sample adjoining identical settings within one source shot together."""
    groups = []
    state = plan["state"]
    role_images = {role["id"]: len(role["image_ids"]) for role in state["roles"]}
    for video, part, role_ids in generation_segments(plan):
        if groups:
            previous_video, previous, previous_roles = groups[-1]
            if (role_ids and video["id"] == previous_video["id"] and set(role_ids) == set(previous_roles)
                    and part["source_shot_id"] == previous["source_shot_id"]
                    and abs(previous["end"] - part["start"]) < 1e-6
                    and part.get("prompt", "").strip() == previous.get("prompt", "").strip()
                    and part.get("role_mode") == previous.get("role_mode")
                    and set(part.get("role_ids", [])) == set(previous.get("role_ids", []))
                    and align_frame_count(max(124, int(np.ceil((part["end"] - previous["start"]) * 24)))) <= 362
                    and sum(role_images[r] for r in role_ids) + len(frame_references(state, video["id"], previous["start"], part["end"])) <= 9):
                previous["end"] = part["end"]
                previous["members"].append(part)
                continue
        groups.append((video, {**part, "members": [part]}, role_ids))
    return groups


def render_batch(plan, model, clip, video_vae, audio_vae, noise, sampler, sigmas, long_edge=1344,
                 interpolation="nearest", ref_image_size="max", sheet_mode="auto", progress=None):
    state = plan["state"]
    segments = generation_groups(plan)
    custom = state["settings"].get("resolution", {}).get("enabled", False)
    for video in state["videos"]:
        video["output_size"] = list(output_size(video, state["settings"]))
    role_map = {role["id"]: role for role in state["roles"]}
    directory = cache_dir("renders") / uuid.uuid4().hex
    directory.mkdir()
    output, bases = {}, {}
    for video, _, _ in segments:
        if video["id"] not in bases:
            base = revision_base(state, video)
            bases[video["id"]] = str(base[0]) if base else None
            width, height = output_size(video)
            x, y, cw, ch = center_rect(video["info"]["width"], video["info"]["height"], width, height) if custom else (0, 0, width, height)
            if (cw, ch) != (width, height) and not base and not any(state["settings"].get(k, {}).get("enabled") for k in ("trial", "candidate")):
                ordered = sorted((part for v, part, _ in segments if v["id"] == video["id"]), key=lambda part: part["start"])
                covered = 0.
                for part in ordered:
                    if abs(part["start"] - covered) > 1 / 90000:
                        raise ValueError("首次扩画幅必须启用并覆盖整片分段，避免未生成区域留下空边；可先单段试跑")
                    covered = part["end"]
                if abs(covered - video["info"]["duration"]) > 1 / 90000:
                    raise ValueError("首次扩画幅必须覆盖整片；已有同尺寸完整底稿后可仅修正所选段")
    for number, (video, segment, role_ids) in enumerate(segments):
        roles = [role_map[item] for item in role_ids]
        windows = shot_plan(video, segment)
        unchanged = [(s["start"], s["end"]) for s in windows if not s["role_ids"]]
        out_width, out_height = output_size(video)
        x, y, width, height = center_rect(video["info"]["width"], video["info"]["height"], out_width, out_height) if custom else (0, 0, *canvas_size(video["info"]["width"], video["info"]["height"], long_edge))
        expanding = custom and (width, height) != (out_width, out_height)
        start, end = segment["start"], segment["end"]
        generation_start = start
        def update(fraction, text):
            model_management.throw_exception_if_processing_interrupted()
            if progress:
                progress((number + fraction) / len(segments), text, video["id"])
        update(.03, "读取原片镜头与完整人物时间定位")
        warnings = []
        for window in windows:
            if not window["role_ids"]:
                warnings.append(f"{window['start']:.3f}–{window['end']:.3f}s 未选择或未标记人物，沿用底稿" if bases[video["id"]] else f"{window['start']:.3f}–{window['end']:.3f}s 未选择或未标记人物，沿用原片")
            if window["missing_role_ids"]:
                names = "、".join(role_map[r].get("name", r) for r in window["missing_role_ids"])
                warnings.append(f"{window['start']:.3f}–{window['end']:.3f}s 的 {names} 缺少本镜头定位，不跨镜头借框")
        if warnings:
            update(.05, "；".join(warnings))
        source, length, time_map = None, 0, None
        if roles or expanding:
            source, source_times = read_video(input_path(video["file"]), generation_start, end, size=(width, height))
            length = align_frame_count(max(124, int(np.ceil((end - generation_start) * 24))))
            if length > 362:
                raise ValueError("本段超过 H3 的 15 秒范围，请缩短分段")
            source, time_map = fit_source_clip(source, source_times, length)
        prompt = ""
        if roles:
            refs, prompt, guides = references_and_prompt(video, segment, roles, state, generation_start, end, length, sheet_mode,
                (out_width, out_height, x, y, width, height) if expanding else None, time_map=time_map)
            update(.12, "编码镜头人物定位与参考图")
            conditioning, latent = MiniMaxH3ReferenceToVideo.execute(clip, prompt, width, height, length, ref_image_size=ref_image_size,
                vae=video_vae, audio_vae=audio_vae, ref_images=refs, ref_videos={"ref_video_0": source}).result
            seen = set()
            for index, image, reference_id in guides:
                if index in seen:
                    raise ValueError("同一段同一 H3 帧只能放一张修正图片，请调整时间或关闭重复参考")
                seen.add(index)
                conditioning = MiniMaxH3AddGuide.execute(conditioning, latent, index, vae=video_vae, image=image).result[0]
            guider = BasicGuider.execute(model, conditioning).result[0]
            update(.25, "H3 全画面生成；采样进度显示在 ComfyUI 进度条")
            _, denoised = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent).result
            update(.78, "解码全画面视频")
            generated = decoded_video(video_vae, denoised["samples"].tensors[0])
            reference_count = len(refs)
            del conditioning, latent, guider, denoised, refs, guides
            model_management.soft_empty_cache()
        else:
            generated = source
        expansion = {}
        preserve_center = None
        if expanding:
            center_video = {**video, "output_size": [width, height]}
            center_part = write_fullframe_part(center_video, segment, generated, generation_start, directory, interpolation, roles,
                unchanged_windows=unchanged, base_path=bases[video["id"]], time_map=time_map)
            background_refs, background_rects = background_crops(source, video, roles)
            generated, expansion = expand_frames(generated, source, model, clip, video_vae, noise, sampler, sigmas,
                (out_width, out_height), lambda fraction, text: update(.80 + fraction * .12, text), background_refs)
            expansion["background_crop_rects"] = background_rects
            expansion["center_path"] = center_part["path"]
            preserve_center = {"path": center_part["path"], "rect": expansion["center_rect"], "start": start}
        del source
        update(.95, "写入所选尺寸并保持原帧时间" if custom else "恢复原尺寸和原帧时间")
        for member in segment["members"]:
            part = write_fullframe_part(video, member, generated, generation_start, directory, interpolation, roles, preserve_center,
                unchanged_windows=unchanged, base_path=bases[video["id"]], time_map=time_map)
            part.update(reference_ids=[item["id"] for item in frame_references(state, video["id"], member["start"], member["end"])], prompt=prompt,
                        parent_segment_id=member["parent_segment_id"], source_shot_id=member["source_shot_id"],
                        generation_start=generation_start, generation_end=end, generation_group_id=segment["id"], generation_frames=length,
                        time_map=time_map, automatic_prefix=False, processing_size=[width, height],
                        sampling_steps=len(sigmas) - 1 if roles else 0, reference_image_size=ref_image_size, reference_sheet_mode=sheet_mode,
                        output_size=[out_width, out_height], expansion=expansion, shots=shot_plan(video, member), warnings=warnings)
            if roles:
                part["reference_count"] = reference_count
            output.setdefault(video["id"], []).append(part)
    if progress:
        progress(1, "H3 全画面生成完成，等待保存和效果检查", "")
    return {"plan": plan, "videos": output, "base_videos": bases, "work_dir": str(directory), "processing_long_edge": long_edge,
            "backend": "minimax_h3_reference", "generation_fps": 24, "plugin_version": "0.3.3"}
