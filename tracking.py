from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import shutil
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from . import paths
from .media import VideoWriter, iter_video, nearest_frame_indices, probe_video

SAM2_CHECKPOINT = "sam2.1_hiera_tiny.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
WINDOW_FRAMES = 48


def checkpoint_path() -> Path:
    import folder_paths

    return Path(folder_paths.models_dir) / "sam2" / SAM2_CHECKPOINT


def dependencies() -> dict:
    model = checkpoint_path()
    packages = {name: importlib.util.find_spec(name) is not None for name in ("sam2", "hydra", "iopath")}
    return {"available": model.is_file() and all(packages.values()), "packages": packages,
            "model_path": str(model), "model_present": model.is_file(), "model_url": SAM2_URL,
            "backend": "SAM2.1 Hiera tiny", "window_frames": WINDOW_FRAMES}


def _cache_path(path) -> Path:
    root = paths.cache_dir("tracks").resolve()
    candidate = Path(path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("人物蒙版缓存路径超出插件缓存目录。") from exc
    return candidate


def _point(point):
    x, y = float(point["x"]), float(point["y"])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("人物标记坐标无效。")
    return min(1.0, max(0.0, x)), min(1.0, max(0.0, y))


def _anchor_points(anchor, width, height):
    points, labels = [], []
    for point in anchor.get("points", []):
        x, y = _point(point)
        points.append([x * width, y * height])
        labels.append(1 if int(point.get("label", 1)) else 0)
    for stroke in anchor.get("strokes", []):
        coordinates = stroke.get("points", [])
        step = max(1, len(coordinates) // 64)
        for point in coordinates[::step]:
            x, y = _point(point)
            points.append([x * width, y * height])
            labels.append(1 if int(stroke.get("label", 1)) else 0)
    box = anchor.get("box")
    if box is not None:
        if len(box) != 4:
            raise ValueError("人物框选需要四个归一化坐标。")
        box = np.clip(np.asarray(box, np.float32), 0, 1)
        if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("人物框选区域无效。")
        box = box * np.asarray([width, height, width, height], np.float32)
    return np.asarray(points, np.float32).reshape(-1, 2), np.asarray(labels, np.int32), box


def _positive_anchor(anchor):
    return anchor.get("box") is not None or any(int(p.get("label", 1)) for p in anchor.get("points", [])) or any(
        int(stroke.get("label", 1)) and stroke.get("points") for stroke in anchor.get("strokes", []))


def paint_strokes(mask: np.ndarray, strokes: list) -> np.ndarray:
    result = np.asarray(mask, np.uint8).copy()
    height, width = result.shape
    for stroke in strokes:
        coordinates = [_point(point) for point in stroke.get("points", [])]
        if not coordinates:
            continue
        radius = float(stroke.get("radius", 0.015))
        if not math.isfinite(radius) or radius < 0:
            raise ValueError("笔刷半径无效。")
        radius = max(1, round(min(width, height) * min(radius, 1)))
        value = 1 if int(stroke.get("label", 1)) else 0
        pixels = [(min(width - 1, round(x * width)), min(height - 1, round(y * height))) for x, y in coordinates]
        for first, second in zip(pixels, pixels[1:]):
            cv2.line(result, first, second, value, thickness=radius * 2 + 1)
        for pixel in pixels:
            cv2.circle(result, pixel, radius, value, thickness=-1)
    return result


def editing_alpha(mask: np.ndarray, padding=16, feather=4, strokes=None) -> np.ndarray:
    padding, feather = int(padding), float(feather)
    if padding < 0 or feather < 0 or not math.isfinite(feather):
        raise ValueError("蒙版扩展与羽化不能为负数。")
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if padding:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (padding * 2 + 1, padding * 2 + 1))
        binary = cv2.dilate(binary, kernel)
    alpha = binary.astype(np.float32)
    if feather:
        # Keep a solid inner region and fade only within a finite outer band.
        distance = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 3)
        alpha = np.clip(1.0 - distance / feather, 0, 1)
    result = np.rint(alpha * 255).astype(np.uint8)
    negative_strokes = [{**stroke, "label": 1} for stroke in (strokes or []) if not int(stroke.get("label", 1))]
    if negative_strokes:
        protected = paint_strokes(np.zeros(binary.shape, np.uint8), negative_strokes)
        result[protected > 0] = 0
    return result


def _add_anchor(predictor, state, frame_index, anchor, width, height):
    points, labels, box = _anchor_points(anchor, width, height)
    if not len(points) and box is None:
        return
    _, object_ids, logits = predictor.add_new_points_or_box(
        state, frame_idx=frame_index, obj_id=1,
        points=points if len(points) else None, labels=labels if len(points) else None,
        box=box, normalize_coords=True, clear_old_points=True,
    )
    if anchor.get("strokes"):
        index = list(object_ids).index(1)
        mask = (logits[index, 0].detach().cpu().numpy() > 0).astype(np.uint8)
        predictor.add_new_mask(state, frame_idx=frame_index, obj_id=1, mask=paint_strokes(mask, anchor["strokes"]))


def _window_frames(source_dir: Path, window_dir: Path, start: int, end: int):
    window_dir = _cache_path(window_dir)
    window_dir.mkdir(parents=True, exist_ok=True)
    for frame in window_dir.glob("*.jpg"):
        frame.unlink()
    for local_index, global_index in enumerate(range(start, end)):
        source = source_dir / f"{global_index:08d}.jpg"
        destination = window_dir / f"{local_index:08d}.jpg"
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)


def _track_direction(predictor, frame_dir, work_dir, role_dir, frame_count, anchors, seed_index,
                     width, height, padding, feather, reverse, progress, progress_base, progress_span):
    cursor = seed_index
    carry = None
    done = 0
    total = seed_index + 1 if reverse else frame_count - seed_index
    while 0 <= cursor < frame_count:
        if reverse:
            start, end = max(0, cursor - WINDOW_FRAMES + 1), cursor + 1
            seed_local = end - start - 1
        else:
            start, end = cursor, min(frame_count, cursor + WINDOW_FRAMES)
            seed_local = 0
        _window_frames(frame_dir, work_dir, start, end)
        state = predictor.init_state(str(work_dir), offload_video_to_cpu=True, offload_state_to_cpu=True,
                                     async_loading_frames=False)
        try:
            if carry is not None:
                predictor.add_new_mask(state, frame_idx=seed_local, obj_id=1, mask=carry)
            for index, anchor in anchors.items():
                if start <= index < end:
                    _add_anchor(predictor, state, index - start, anchor, width, height)
            terminal = None
            for local_index, object_ids, logits in predictor.propagate_in_video(state, start_frame_idx=seed_local, reverse=reverse):
                global_index = start + local_index
                obj_index = list(object_ids).index(1)
                raw_mask = (logits[obj_index, 0].detach().cpu().numpy() > 0).astype(np.uint8)
                if raw_mask.shape != (height, width):
                    raw_mask = cv2.resize(raw_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                anchor = anchors.get(global_index)
                if anchor is not None and anchor.get("strokes"):
                    raw_mask = paint_strokes(raw_mask, anchor["strokes"])
                Image.fromarray(editing_alpha(raw_mask, padding, feather, anchor.get("strokes") if anchor else None)).save(
                    role_dir / f"{global_index:08d}.png")
                terminal = raw_mask
                done += 1
                if progress is not None and (done % 5 == 0 or done >= total):
                    progress(min(1, progress_base + progress_span * min(done, total) / total), "正在跟踪人物并保存蒙版")
            carry = terminal
        finally:
            predictor.reset_state(state)
            del state
        if carry is None:
            raise RuntimeError("SAM2 没有产生人物蒙版。请重新选择人物并检查模型。")
        if reverse:
            if start == 0:
                break
            cursor = start
        else:
            if end == frame_count:
                break
            cursor = end - 1


def read_masks(track: dict, role_id: str, indices: list[int], size=None) -> torch.Tensor:
    if role_id not in track.get("roles", {}):
        raise ValueError("人物尚未完成跟踪。")
    directory = _cache_path(track["roles"][role_id]["mask_path"])
    result = []
    for index in indices:
        index = int(index)
        if index < 0 or index >= int(track["frames"]):
            raise ValueError("人物蒙版索引超出原视频帧范围。")
        filename = _cache_path(directory / f"{index:08d}.png")
        if not filename.is_file():
            raise FileNotFoundError("人物蒙版缓存不完整，请重新跟踪。")
        with Image.open(filename) as image:
            if size is not None:
                image = image.resize((int(size[0]), int(size[1])), Image.Resampling.BILINEAR)
            result.append(np.asarray(image, np.float32) / 255.0)
    if not result:
        height, width = (int(size[1]), int(size[0])) if size else (track["height"], track["width"])
        return torch.empty((0, height, width), dtype=torch.float32)
    return torch.from_numpy(np.stack(result))


def create_preview(track: dict, source_path: Path, progress=None) -> dict:
    directory = _cache_path(track["cache_dir"])
    info = probe_video(source_path)
    scale = min(1, 480 / info["width"], 270 / info["height"])
    width, height = max(2, int(info["width"] * scale)), max(2, int(info["height"] * scale))
    preview = directory / "preview.mp4"
    role_ids = list(track["roles"])
    palette = [(255, 77, 159), (45, 210, 247), (255, 185, 54), (122, 239, 96)]
    timestamps = track["timestamps"]
    with VideoWriter(preview, width, height, 6, 1) as writer:
        for frame, time in iter_video(source_path, fps=6, size=(width, height)):
            original_index = nearest_frame_indices(timestamps, [time])[0]
            overlay = frame.astype(np.float32)
            for index, role_id in enumerate(role_ids):
                alpha = read_masks(track, role_id, [original_index], size=(width, height))[0].numpy()[..., None] * 0.5
                color = np.asarray(palette[index % len(palette)], np.float32)
                overlay = overlay * (1 - alpha) + color * alpha
            writer.write(np.rint(overlay).astype(np.uint8))
    mask_frames = []
    for role_id, role in track["roles"].items():
        anchor_indices = nearest_frame_indices(timestamps, [a["time"] for a in role["anchors"]])
        selected = sorted(set(anchor_indices + [round(i * (track["frames"] - 1) / 4) for i in range(5)]))
        for frame_index in selected:
            mask_frames.append({"time": timestamps[frame_index], "role_id": role_id,
                                "path": str(Path(role["mask_path"]) / f"{frame_index:08d}.png")})
    return {"preview_path": str(preview), "mask_frames": mask_frames}


def track_video(video: dict, roles: list, cache_dir: Path, progress=None) -> dict:
    source = paths.input_path(video["file"])
    bindings = video.get("bindings", [])
    if not bindings:
        raise ValueError("请先在原视频上点选或框选人物，并绑定参考角色。")
    role_ids = {str(role["id"]) for role in roles}
    for binding in bindings:
        if str(binding.get("role_id")) not in role_ids:
            raise ValueError("视频绑定的参考角色不存在。")
        if not binding.get("anchors"):
            raise ValueError("每个替换人物至少需要一个点选或框选标记。")
        has_positive = any(_positive_anchor(anchor) for anchor in binding["anchors"])
        if not has_positive:
            raise ValueError("人物至少需要一个正向点选、笔刷或框选标记。")
    model_path = checkpoint_path()
    signature = {"version": 2, "file": str(source), "mtime": source.stat().st_mtime_ns,
                 "size": source.stat().st_size, "bindings": bindings,
                 "model": SAM2_CHECKPOINT, "model_mtime": model_path.stat().st_mtime_ns if model_path.is_file() else None}
    key = hashlib.sha256(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]
    directory = _cache_path(cache_dir) / key
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "tracks.json"
    if manifest.is_file():
        cached = json.loads(manifest.read_text(encoding="utf-8"))
        if all(len(list(_cache_path(role["mask_path"]).glob("*.png"))) == cached["frames"] for role in cached["roles"].values()):
            if progress:
                progress(1, "已复用人物跟踪缓存")
            return cached
    if not model_path.is_file():
        raise FileNotFoundError(f"缺少人物跟踪模型：{model_path}。请运行插件 setup.ps1 安装 SAM2.1 tiny。")
    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as exc:
        raise RuntimeError("SAM2 跟踪依赖不可用，请运行插件 setup.ps1；不要替换 ComfyUI 的 Torch。") from exc
    import comfy.model_management as model_management

    if progress:
        progress(0.01, "正在整理原视频帧与时间索引")
    info = probe_video(source)
    frame_dir = _cache_path(directory / "source_frames")
    frame_dir.mkdir(exist_ok=True)
    timestamps = []
    for index, (frame, time) in enumerate(iter_video(source)):
        Image.fromarray(frame).save(frame_dir / f"{index:08d}.jpg", quality=95)
        timestamps.append(time)
        if progress and index % 30 == 0:
            progress(min(0.12, 0.01 + 0.11 * (index + 1) / max(1, info["frames"])), "正在缓存原视频帧")
    if not timestamps:
        raise ValueError("原视频没有可解码帧。")
    info["frames"] = len(timestamps)
    result = {"video_id": video["id"], **info, "timestamps": timestamps, "roles": {}, "cache_dir": str(directory),
              "backend": "SAM2.1 Hiera tiny"}
    model_management.unload_all_models()
    model_management.soft_empty_cache()
    device = model_management.get_torch_device()
    predictor = None
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    try:
        predictor = build_sam2_video_predictor(SAM2_CONFIG, str(model_path), device=device,
                                               apply_postprocessing=False, vos_optimized=False)
        with autocast:
            for binding_index, binding in enumerate(bindings):
                role_id = str(binding["role_id"])
                role_dir = _cache_path(directory / ("role_" + hashlib.sha256(role_id.encode()).hexdigest()[:16]))
                role_dir.mkdir(exist_ok=True)
                anchors = {}
                for anchor in binding["anchors"]:
                    frame_index = nearest_frame_indices(timestamps, [float(anchor["time"])])[0]
                    if frame_index in anchors:
                        old = anchors[frame_index]
                        old["points"].extend(anchor.get("points", []))
                        old["strokes"].extend(anchor.get("strokes", []))
                        if anchor.get("box") is not None:
                            old["box"] = anchor["box"]
                    else:
                        anchors[frame_index] = {**anchor, "points": list(anchor.get("points", [])), "strokes": list(anchor.get("strokes", []))}
                seed = min(index for index, anchor in anchors.items() if _positive_anchor(anchor))
                base = 0.12 + 0.78 * binding_index / len(bindings)
                span = 0.78 / len(bindings)
                _track_direction(predictor, frame_dir, directory / "window", role_dir, len(timestamps), anchors,
                                 seed, info["width"], info["height"], binding.get("padding", 16), binding.get("feather", 4),
                                 False, progress, base, span * (len(timestamps) - seed) / len(timestamps))
                if seed > 0:
                    _track_direction(predictor, frame_dir, directory / "window", role_dir, len(timestamps), anchors,
                                     seed, info["width"], info["height"], binding.get("padding", 16), binding.get("feather", 4),
                                     True, progress, base + span * (len(timestamps) - seed) / len(timestamps), span * seed / len(timestamps))
                result["roles"][role_id] = {"mask_path": str(role_dir), "anchors": binding["anchors"],
                                           "padding": binding.get("padding", 16), "feather": binding.get("feather", 4)}
    finally:
        if predictor is not None:
            predictor.to("cpu")
            del predictor
        model_management.soft_empty_cache()
        shutil.rmtree(_cache_path(directory / "window"), ignore_errors=True)
        shutil.rmtree(_cache_path(frame_dir), ignore_errors=True)
    if progress:
        progress(0.92, "正在生成蒙版预览")
    result.update(create_preview(result, source, progress))
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    temporary.replace(manifest)
    if progress:
        progress(1, "人物跟踪完成，可检查蒙版并补充修正标记")
    return result
