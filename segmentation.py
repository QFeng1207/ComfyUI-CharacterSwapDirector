from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .media import iter_video, probe_video


def _scene_distance(previous, current):
    difference = np.abs(previous.astype(np.float32) - current.astype(np.float32)).mean() / 255.0
    old_hist = cv2.calcHist([previous], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    new_hist = cv2.calcHist([current], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    cv2.normalize(old_hist, old_hist, alpha=1, norm_type=cv2.NORM_L1)
    cv2.normalize(new_hist, new_hist, alpha=1, norm_type=cv2.NORM_L1)
    histogram = cv2.compareHist(old_hist, new_hist, cv2.HISTCMP_BHATTACHARYYA)
    return float(0.55 * difference + 0.45 * histogram)


def detect_segments(path: Path, max_seconds=10, threshold=0.35) -> list[dict]:
    max_seconds, threshold = float(max_seconds), float(threshold)
    if max_seconds <= 0 or not math.isfinite(max_seconds):
        raise ValueError("最大分段时长必须大于 0。")
    info = probe_video(path)
    duration = info["duration"]
    rate = info["fps_num"] / info["fps_den"]
    boundaries = [0.0]
    previous = None
    for frame, time in iter_video(path, size=(160, 90)):
        if previous is not None and _scene_distance(previous, frame) >= threshold:
            if time > boundaries[-1] + 0.5 / rate:
                boundaries.append(time)
        previous = frame
    if duration > boundaries[-1]:
        boundaries.append(duration)
    segments = []
    max_frames = max(1, int(math.floor(max_seconds * rate)))
    for scene_index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        cursor = start
        while cursor < end - 1e-9:
            next_end = min(end, cursor + max_frames / rate)
            segments.append({"id": f"seg_{len(segments) + 1}", "start": cursor, "end": next_end,
                             "scene_id": f"scene_{scene_index + 1}", "scene_start": start, "scene_end": end,
                             "enabled": True, "role_mode": "auto", "role_ids": [], "prompt": ""})
            cursor = next_end
    return segments
