"""Resolve cast and person locations within actual source shots."""
from __future__ import annotations

from bisect import bisect_right


def anchor_bounds(anchor):
    if anchor.get("box"):
        return list(anchor["box"])
    regions = [[p["x"], p["y"], p["x"], p["y"]] for p in anchor.get("points", []) if p.get("label", 1)]
    for stroke in anchor.get("strokes", []):
        if not stroke.get("label", 1):
            continue
        radius = stroke.get("radius", .015)
        regions.extend([max(0, p["x"] - radius), max(0, p["y"] - radius), min(1, p["x"] + radius), min(1, p["y"] + radius)] for p in stroke.get("points", []))
    if not regions:
        return None
    return [min(r[0] for r in regions), min(r[1] for r in regions), max(r[2] for r in regions), max(r[3] for r in regions)]


def shot_anchors(binding, shot):
    return sorted((a for a in binding.get("anchors", []) if shot["start"] <= a["time"] < shot["end"] and anchor_bounds(a) is not None), key=lambda a: a["time"])


def source_shots(video):
    return video.get("shots") or [{"id": "source_shot_1", "start": 0., "end": video["info"]["duration"]}]


def shot_override(video, shot):
    return next((item for item in video.get("shot_overrides", [])
                 if abs(item["start"] - shot["start"]) < 1e-5 and abs(item["end"] - shot["end"]) < 1e-5), {})


def same_shot(video, before, after):
    return any(s["start"] <= before < after < s["end"] for s in source_shots(video))


def location_at(anchors, time):
    times = [a["time"] for a in anchors]
    right = bisect_right(times, time)
    if not right:
        bounds = anchor_bounds(anchors[0])
    elif right == len(anchors):
        bounds = anchor_bounds(anchors[-1])
    else:
        left, following = anchors[right - 1], anchors[right]
        fraction = (time - left["time"]) / (following["time"] - left["time"])
        bounds = [a + (b - a) * fraction for a, b in zip(anchor_bounds(left), anchor_bounds(following))]
    return {"time": time, "box": bounds, "interpolated": time not in times}


def shot_plan(video, segment, start=None, end=None):
    start = segment["start"] if start is None else start
    end = segment["end"] if end is None else end
    manual = segment.get("role_mode", "manual" if segment.get("role_ids") else "auto") == "manual"
    selected = set(segment.get("role_ids", []))
    rate = video["info"].get("fps_num", 30) / video["info"].get("fps_den", 1)
    result = []
    for source in source_shots(video):
        left, right = max(start, source["start"]), min(end, source["end"])
        if left >= right:
            continue
        override = shot_override(video, source)
        choices = {item["role_id"]: item for item in override.get("roles", [])}
        tracks, missing, unlocated = [], [], []
        for binding in video.get("bindings", []):
            if manual and binding["role_id"] not in selected:
                continue
            choice = choices.get(binding["role_id"], {})
            if choice.get("presence") == "absent":
                continue
            anchors = shot_anchors(binding, source)
            if not anchors and choice.get("presence") != "visible":
                if manual:
                    missing.append(binding["role_id"])
                else:
                    unlocated.append(binding["role_id"])
                continue
            # Positions stop at hard cuts. A manual processing boundary does not split an identity track.
            last = max(left, right - 1 / rate)
            times = sorted({left, last, *(a["time"] for a in anchors if left <= a["time"] < right)})
            positions = [location_at(anchors, t) for t in times] if anchors else []
            tracks.append({"role_id": binding["role_id"], "anchors": anchors, "positions": positions,
                           "body_part": choice.get("body_part", ""), "description": choice.get("description", ""),
                           "presence": choice.get("presence", "auto")})
        result.append({"id": source["id"], "start": left, "end": right, "source_start": source["start"], "source_end": source["end"],
                       "role_ids": [t["role_id"] for t in tracks], "tracks": tracks, "missing_role_ids": missing,
                       "unlocated_role_ids": unlocated, "prompt": override.get("prompt", "")})
    return result
