from __future__ import annotations

import copy
import math

import cv2
import numpy as np
import torch

import nodes as comfy_nodes
import comfy.model_management as model_management
from comfy_extras.nodes_custom_sampler import CFGGuider, SamplerCustomAdvanced
from comfy_extras.nodes_wan import WanAnimateToVideo

from .generation import _cached_file, _padded, canvas_size, load_reference, render_segments, write_composite_part
from .media import nearest_frame_indices, read_video
from .paths import input_path
from .plan import active_segments
from .tracking import read_masks

FPS = 16
WINDOW_FRAMES = 77
PREFIX_FRAMES = 5


def primary_view(image, sheet_mode="auto"):
    """Crop the first full-body view from a clearly separated white sheet."""
    height, width = image.shape[1:3]
    full = [0., 0., 1., 1.]
    if sheet_mode != "auto" or width < height * 1.75:
        return image, full
    pixels = image[0].mul(255).round().byte().numpy()
    foreground = (pixels.min(axis=-1) < 235).astype(np.uint8)
    if float(foreground.mean()) > .7:
        return image, full
    _, _, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    candidates = [item for item in stats[1:] if item[3] >= height * .65 and item[4] >= width * height * .02
                  and width * .04 <= item[2] <= width * .4]
    if len(candidates) < 2:
        return image, full
    x, y, w, h, _ = min(candidates, key=lambda item: item[0])
    pad = max(4, round(height * .025))
    left, top = max(0, x - pad), max(0, y - pad)
    right, bottom = min(width, x + w + pad), min(height, y + h + pad)
    return image[:, top:bottom, left:right], [left / width, top / height, right / width, bottom / height]


def fit_reference(image, width, height):
    # Wan's native conditioning center-crops its reference. Fit the entire
    # primary view first so a portrait does not become a waist-only crop.
    h, w = image.shape[1:3]
    scale = min(width / w, height / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    pixels = cv2.resize(image[0].numpy(), (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    canvas = torch.ones((1, height, width, 3), dtype=image.dtype)
    left, top = (width - new_w) // 2, (height - new_h) // 2
    canvas[0, top:top + new_h, left:left + new_w] = torch.from_numpy(pixels)
    return canvas


def prepare_references(role, state, clip_vision, width, height, sheet_mode):
    photos = {item["id"]: item for item in state["images"]}
    primary = role.get("primary_image_id") or role["image_ids"][0]
    ordered = [primary] + [item for item in role["image_ids"] if item != primary]
    images = [load_reference(input_path(photos[item]["file"]), 1024) for item in ordered]
    images[0], crop = primary_view(images[0], sheet_mode)
    # A centre crop removes a portrait's hat, coat and boots. Wan's official
    # Animate workflow encodes the entire reference with cropping disabled.
    encoded = [clip_vision.encode_image(image, crop=False) for image in images]
    vision = copy.copy(encoded[0])
    if len(encoded) > 1:
        # Appearance/body layout comes from the primary latent. Additional
        # photos contribute identity features without creating extra people.
        auxiliary = torch.stack([item.penultimate_hidden_states.float() for item in encoded[1:]]).mean(dim=0)
        vision.penultimate_hidden_states = (encoded[0].penultimate_hidden_states.float() * 2 / 3 + auxiliary / 3).to(encoded[0].penultimate_hidden_states.dtype)
    return fit_reference(images[0], width, height), vision, crop


def conditioning_mask(alpha, block_size=32):
    """Give the generator complete latent blocks; composite with the original mask."""
    binary = (alpha > .1).to(alpha.dtype).unsqueeze(1)
    blocks = torch.nn.functional.max_pool2d(binary, block_size, block_size, ceil_mode=True)
    expanded = blocks.repeat_interleave(block_size, -2).repeat_interleave(block_size, -1)
    return expanded[:, 0, :alpha.shape[-2], :alpha.shape[-1]]


def keypoint_pixels(values, keyframe, width, height):
    points = np.asarray(values or [], np.float32).reshape(-1, 3)
    points = points[(points[:, 2] > .3) & np.isfinite(points).all(axis=1)]
    if not len(points):
        return points
    if bool(((points[:, :2] >= 0) & (points[:, :2] <= 1)).all()):
        points[:, 0] *= width
        points[:, 1] *= height
    else:
        points[:, 0] *= width / keyframe["canvas_width"]
        points[:, 1] *= height / keyframe["canvas_height"]
    return points


def selected_person(keyframe, mask):
    height, width = mask.shape
    def inside(points):
        return sum(0 <= x < width and 0 <= y < height and mask[int(y), int(x)] > .1 for x, y, _ in points)
    candidates = []
    for person in keyframe["people"]:
        values = person.get("pose_keypoints_2d") or []
        raw = np.asarray(values, np.float32).reshape(-1, 3)
        body = keypoint_pixels(values, keyframe, width, height)
        core = keypoint_pixels(raw[[index for index in (1, 2, 5, 8, 11) if index < len(raw)]].flatten().tolist(), keyframe, width, height)
        # A neighbour's hand can overlap the selected mask while their torso is
        # outside it. Select the person first, then render only that skeleton.
        core_hits, body_hits = inside(core), inside(body)
        if len(core) and not core_hits:
            continue
        if body_hits:
            candidates.append(((core_hits / max(1, len(core)), body_hits / len(body), body_hits), person))
        elif not len(body):
            face = keypoint_pixels(person.get("face_keypoints_2d"), keyframe, width, height)
            hits = inside(face)
            if hits:
                candidates.append(((0., hits / len(face), hits), person))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def controls_from_keypoints(source, alpha, keypoints, preserve_face_motion):
    if "RenderPeopleKps" not in comfy_nodes.NODE_CLASS_MAPPINGS:
        raise RuntimeError("请启用或更新 comfyui_controlnet_aux，人物替换需要 RenderPeopleKps 节点。")
    renderer = comfy_nodes.NODE_CLASS_MAPPINGS["RenderPeopleKps"]()
    selected_frames = []
    pose_frames = []
    for mask, keyframe in zip(alpha, keypoints):
        person = selected_person(keyframe, mask.numpy())
        selected = {**keyframe, "people": [person] if person is not None else []}
        selected_frames.append(selected)
        pose_frames.append(renderer.render(selected, True, True, preserve_face_motion)[0])
    pose = torch.cat(pose_frames)
    if pose.shape[1:3] != source.shape[1:3]:
        pose = torch.from_numpy(np.stack([cv2.resize(frame.numpy(), (source.shape[2], source.shape[1])) for frame in pose]))
    selected = torch.from_numpy(np.stack([cv2.dilate((mask.numpy() > .1).astype(np.uint8), np.ones((9, 9), np.uint8)) for mask in alpha]))
    pose = pose[..., :3] * selected.unsqueeze(-1)
    if not preserve_face_motion:
        return pose, None
    faces = []
    for frame, keyframe in zip(source, selected_frames):
        points = keypoint_pixels(keyframe["people"][0].get("face_keypoints_2d"), keyframe, source.shape[2], source.shape[1]) if keyframe["people"] else []
        if len(points) < 6:
            faces.append(np.zeros((256, 256, 3), np.float32))
            continue
        left, top = points[:, :2].min(axis=0)
        right, bottom = points[:, :2].max(axis=0)
        pad = max(right - left, bottom - top) * .25
        left, top = max(0, math.floor(left - pad)), max(0, math.floor(top - pad))
        right, bottom = min(frame.shape[1], math.ceil(right + pad)), min(frame.shape[0], math.ceil(bottom + pad))
        if right <= left or bottom <= top:
            faces.append(np.zeros((256, 256, 3), np.float32))
        else:
            faces.append(cv2.resize(frame[top:bottom, left:right].numpy(), (256, 256), interpolation=cv2.INTER_AREA))
    return pose, torch.from_numpy(np.stack(faces))


def motion_controls(source, alpha, resolution, preserve_face_motion):
    if "DWPreprocessor" not in comfy_nodes.NODE_CLASS_MAPPINGS:
        raise RuntimeError("Wan 人物替换需要启用 comfyui_controlnet_aux 的 DWPose 节点。")
    result = comfy_nodes.NODE_CLASS_MAPPINGS["DWPreprocessor"]().estimate_pose(
        source, detect_hand="enable", detect_body="enable", detect_face="enable" if preserve_face_motion else "disable",
        resolution=resolution, bbox_detector="yolox_l.onnx", pose_estimator="dw-ll_ucoco_384.onnx")["result"]
    return controls_from_keypoints(source, alpha, result[1], preserve_face_motion)


def bounded_segments(plan):
    chunks = []
    max_seconds = (WINDOW_FRAMES - PREFIX_FRAMES) / FPS
    for video, segment, role_ids in active_segments(plan):
        pieces = math.ceil((segment["end"] - segment["start"]) / max_seconds)
        for index in range(pieces):
            item = {**segment, "start": segment["start"] + index * max_seconds,
                    "end": min(segment["end"], segment["start"] + (index + 1) * max_seconds)}
            if pieces > 1:
                item["id"] = f"{segment['id']}_wan_{index}"
                item["scene_id"] = segment.get("scene_id") or f"csd_segment_{segment['id']}"
            chunks.append((video, item, role_ids))
    return chunks


def render_part(video, segment, role, state, track, model, clip, vae, clip_vision, noise, sampler, sigmas,
                long_edge, cfg, interpolation, preserve_face_motion, sheet_mode, directory, progress,
                reference_cache, previous_role=None, previous_segment=None):
    start, end = float(segment["start"]), float(segment["end"])
    width, height = canvas_size(video["info"]["width"], video["info"]["height"], long_edge)
    original = input_path(video["file"])
    prefix = None
    generation_start = start
    if previous_segment is not None and abs(previous_segment["end"] - start) < 1e-4 and segment.get("scene_id") is not None and segment.get("scene_id") == previous_segment.get("scene_id"):
        generation_start = max(previous_segment["start"], start - PREFIX_FRAMES / FPS)
        prefix, _ = read_video(_cached_file(previous_segment["path"]), generation_start - previous_segment["start"], start - previous_segment["start"], fps=FPS, size=(width, height))
        prefix = _padded(prefix, PREFIX_FRAMES)
        generation_start = start - PREFIX_FRAMES / FPS
    if previous_role:
        offset = previous_role.get("source_offset", 0.)
        source, times = read_video(_cached_file(previous_role["path"]), 0, end - start - offset, fps=FPS, size=(width, height))
        times = [time + start + offset for time in times]
    else:
        source, times = read_video(original, start, end, fps=FPS, size=(width, height))
    if prefix is not None:
        source = torch.cat((prefix, source))
        times = [generation_start + i / FPS for i in range(PREFIX_FRAMES)] + times
    if len(source) > WINDOW_FRAMES:
        raise ValueError("Wan 内部分段超过 77 帧窗口，请缩短分段。")
    alpha = read_masks(track, role["id"], nearest_frame_indices(track["timestamps"], times), size=(width, height))
    if not bool((alpha > 0).any()):
        raise ValueError("当前角色在此分段的蒙版为空，请检查人物绑定。")
    progress(.08, "提取所选人物的身体、手部与可选表情控制")
    pose, faces = motion_controls(source, alpha, max(width, height), preserve_face_motion)
    cache_key = (role["id"], width, height)
    if cache_key not in reference_cache:
        progress(.12, "读取主视图与辅助人物参考")
        reference_cache[cache_key] = prepare_references(role, state, clip_vision, width, height, sheet_mode)
    reference, vision, crop = reference_cache[cache_key]
    prompt = f"One replacement character matching the reference image, including face, hair, body, clothing and shoes. {role.get('description', '').strip()} Natural movement and source scene lighting. {segment.get('prompt', '').strip()}"
    positive = comfy_nodes.CLIPTextEncode().encode(clip, prompt)[0]
    negative = comfy_nodes.CLIPTextEncode().encode(clip, "extra people, duplicate person, character sheet, static image, deformed body, ghosting, watermark, captions")[0]
    length = max(5, 1 + math.ceil((len(source) - 1) / 4) * 4)
    source = _padded(source, length)
    alpha = _padded(alpha, length)
    pose = _padded(pose, length)
    faces = _padded(faces, length) if faces is not None else None
    background = source * (alpha <= .5).unsqueeze(-1)
    progress(.18, "编码人物替换区域与动作序列")
    positive, negative, latent, trim_latent, _, _ = WanAnimateToVideo.execute(
        positive, negative, vae, width, height, length, 1, PREFIX_FRAMES, 0,
        reference_image=reference, clip_vision_output=vision, face_video=faces, pose_video=pose,
        continue_motion=prefix, background_video=background, character_mask=conditioning_mask(alpha)).result
    guider = CFGGuider.execute(model, positive, negative, cfg).result[0]
    progress(.25, "开始 Wan 人物替换；采样进度显示在 ComfyUI 进度条")
    _, denoised = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent).result
    progress(.78, "解码替换人物，移除参考图的潜帧")
    generated = vae.decode(denoised["samples"][:, :, trim_latent:]).detach().float().cpu()
    if generated.ndim == 5:
        generated = generated.reshape(-1, *generated.shape[-3:])
    if generated.ndim != 4 or generated.shape[-1] < 3:
        raise ValueError("Wan 视频 VAE 输出格式无效。")
    generated = generated[..., :3].clamp_(0, 1)
    del positive, negative, latent, guider, denoised, source, alpha, pose, faces, background
    model_management.soft_empty_cache()
    progress(.85, "按原时间戳合成，保留人物蒙版外的原画面")
    result = write_composite_part(video, segment, role, track, generated, generation_start, directory, interpolation,
                                   previous_role=previous_role, generation_fps=FPS)
    result["reference_crop"] = crop
    return result


def render_batch(plan, tracks, model, clip, vae, clip_vision, noise, sampler, sigmas, long_edge=512, cfg=1.,
                 interpolation="linear", preserve_face_motion=False, sheet_mode="auto", progress=None):
    reference_cache = {}
    def operation(video, segment, role, state, track, directory, part_progress, previous_role, previous_segment):
        return render_part(video, segment, role, state, track, model, clip, vae, clip_vision, noise, sampler, sigmas,
                           long_edge, cfg, interpolation, preserve_face_motion, sheet_mode, directory, part_progress,
                           reference_cache, previous_role, previous_segment)
    result = render_segments(plan, tracks, operation, long_edge, progress, bounded_segments(plan))
    result.update(backend="wan2.2_animate", generation_fps=FPS)
    return result
