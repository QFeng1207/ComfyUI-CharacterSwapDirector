from __future__ import annotations

import copy
import json
import math

from .paths import input_path, safe_id, file_key
from .media import probe_video
from .segmentation import detect_segments
from .resolution import normalize_resolution, output_size
from .shot_plan import shot_plan


def empty_state():
    return {"version": 1, "videos": [], "images": [], "roles": [], "frame_references": [], "settings": {"person_mode": "multi", "audio_mode": "original", "segment_mode": "auto", "max_segment_seconds": 10, "scene_threshold": 0.2, "resolution": {"enabled": False, "width": 0, "height": 0}, "trial": {"enabled": False, "video_id": "", "segment_id": ""}, "revision": {"enabled": False, "video_id": "", "base_result_id": "", "scope": "selected", "segment_ids": []}, "candidate": {"enabled": False, "video_id": "", "segment_id": "", "time": 0}}, "selected_video_id": "", "selected_role_id": ""}


def normalize_audio_mode(mode):
    # Old workflows remain usable without voice models or reference recordings.
    if mode in ("replace", "reference-timbre"):
        return "original"
    if mode not in ("original", "mute"):
        raise ValueError("请选择保留原声或静音")
    return mode


def finite_number(value, name, low=None, high=None):
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 的数值无效") from error
    if not math.isfinite(value) or (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{name} 的数值无效")
    return value


def validate_state(value, *, require_media=True):
    state = json.loads(value) if isinstance(value, str) else copy.deepcopy(value)
    if not isinstance(state, dict) or state.get("version", 1) != 1:
        raise ValueError("导演台配置版本无效")
    settings = {**empty_state()["settings"], **state.get("settings", {})}
    settings["audio_mode"] = normalize_audio_mode(settings["audio_mode"])
    settings["resolution"] = normalize_resolution(settings.get("resolution", {}))
    if settings["person_mode"] not in ("single", "multi") or settings["segment_mode"] not in ("auto", "manual"):
        raise ValueError("人物模式或分段模式无效")
    settings["max_segment_seconds"] = finite_number(settings["max_segment_seconds"], "分段长度", .2)
    settings["scene_threshold"] = finite_number(settings["scene_threshold"], "镜头阈值", 0, 1)
    state["settings"] = settings
    trial = settings.get("trial")
    if not isinstance(trial, dict):
        raise ValueError("试跑设置必须包含视频和分段编号")
    if not isinstance(trial.get("enabled", False), bool):
        raise ValueError("试跑开关无效")
    state.pop("audios", None)
    for key in ("videos", "images", "roles"):
        if not isinstance(state.get(key, []), list):
            raise ValueError(f"{key} 配置必须是列表")
        state.setdefault(key, [])
        identifiers = set()
        for item in state[key]:
            if not isinstance(item, dict):
                raise ValueError("素材或角色设置无效")
            safe_id(item["id"])
            if item["id"] in identifiers:
                raise ValueError("素材或角色编号重复")
            identifiers.add(item["id"])
            if key != "roles":
                input_path(item["file"])
    if require_media and not state["videos"]:
        raise ValueError("请先在导演台上传或加载参考视频")
    images = {item["id"] for item in state["images"]}
    roles = {item["id"]: item for item in state["roles"]}
    for role in roles.values():
        role.pop("voice_id", None)
        role.setdefault("image_ids", [])
        if not isinstance(role["image_ids"], list):
            raise ValueError("角色参考图必须是列表")
        if any(image_id not in images for image_id in role["image_ids"]):
            raise ValueError(f"角色 {role.get('name', role['id'])} 绑定的参考图已删除")
        if role.get("primary_image_id") and role["primary_image_id"] not in role["image_ids"]:
            raise ValueError("主参考图必须属于该角色的参考图库")
    for video in state["videos"]:
        video.pop("speakers", None)
        bound_roles = set()
        for name in ("bindings", "segments"):
            if not isinstance(video.get(name, []), list):
                raise ValueError("视频人物或分段设置必须是列表")
        for binding in video.get("bindings", []):
            if binding["role_id"] not in roles:
                raise ValueError("视频人物绑定到了不存在的角色")
            if binding["role_id"] in bound_roles:
                raise ValueError("同一个视频中，同一角色只能有一组人物绑定；请在该绑定中添加多个锚点。")
            bound_roles.add(binding["role_id"])
            for anchor in binding.get("anchors", []):
                anchor["time"] = finite_number(anchor["time"], "人物锚点时间", 0)
                for point in anchor.get("points", []):
                    point["x"] = finite_number(point["x"], "点选横坐标", 0, 1)
                    point["y"] = finite_number(point["y"], "点选纵坐标", 0, 1)
                    if point.get("label", 1) not in (0, 1):
                        raise ValueError("点选标记必须是保留或移除")
                for stroke in anchor.get("strokes", []):
                    stroke["radius"] = finite_number(stroke.get("radius", .015), "笔刷半径", .0001, 1)
                    if stroke.get("label", 1) not in (0, 1):
                        raise ValueError("笔刷标记无效")
                    for point in stroke.get("points", []):
                        point["x"] = finite_number(point["x"], "笔刷横坐标", 0, 1)
                        point["y"] = finite_number(point["y"], "笔刷纵坐标", 0, 1)
                box = anchor.get("box")
                if box is not None:
                    if len(box) != 4:
                        raise ValueError("人物框选坐标无效")
                    box = [finite_number(coordinate, "人物框选坐标", 0, 1) for coordinate in box]
                    anchor["box"] = box
                    if box[2] <= box[0] or box[3] <= box[1]:
                        raise ValueError("人物框选区域为空")
            binding["padding"] = finite_number(binding.get("padding", 16), "蒙版扩展", 0, 4096)
            binding["feather"] = finite_number(binding.get("feather", 4), "蒙版羽化", 0, 4096)
        overrides = video.setdefault("shot_overrides", [])
        if not isinstance(overrides, list):
            raise ValueError("镜头设置必须是列表")
        ranges = set()
        for override in overrides:
            if not isinstance(override, dict) or not isinstance(override.get("roles", []), list):
                raise ValueError("镜头角色设置无效")
            start = finite_number(override.get("start"), "镜头开始", 0)
            end = finite_number(override.get("end"), "镜头结束", 0)
            if end <= start or (start, end) in ranges:
                raise ValueError("镜头设置范围为空或重复")
            ranges.add((start, end))
            override.update(start=start, end=end)
            if not isinstance(override.get("prompt", ""), str):
                raise ValueError("镜头动作说明必须是文字")
            chosen = set()
            for choice in override.get("roles", []):
                if not isinstance(choice, dict) or choice.get("role_id") not in bound_roles or choice["role_id"] in chosen:
                    raise ValueError("镜头中的角色未绑定到视频或重复")
                chosen.add(choice["role_id"])
                if choice.get("presence", "auto") not in ("auto", "visible", "absent"):
                    raise ValueError("镜头人物参与方式无效")
                if not isinstance(choice.get("body_part", ""), str) or not isinstance(choice.get("description", ""), str):
                    raise ValueError("镜头人物局部说明必须是文字")
        segment_ids = set()
        for segment in video.get("segments", []):
            safe_id(segment["id"])
            if segment["id"] in segment_ids:
                raise ValueError("分段编号重复")
            segment_ids.add(segment["id"])
            segment.setdefault("role_mode", "manual" if segment.get("role_ids") else "auto")
            if segment["role_mode"] not in ("auto", "manual") or not isinstance(segment.get("role_ids", []), list):
                raise ValueError("请选择按镜头自动选角或手动选择角色")
            if any(role_id not in bound_roles for role_id in segment.get("role_ids", [])):
                raise ValueError("分段中的角色没有绑定到当前视频")
            start = finite_number(segment["start"], "分段开始", 0)
            end = finite_number(segment["end"], "分段结束", 0)
            segment.update(start=start, end=end)
            if end <= start:
                raise ValueError("分段结束时间必须晚于开始时间")
    video_map = {item["id"]: item for item in state["videos"]}
    refs = state.setdefault("frame_references", [])
    if not isinstance(refs, list):
        raise ValueError("画面参考必须是列表")
    reference_ids = set()
    for reference in refs:
        if not isinstance(reference, dict):
            raise ValueError("画面参考配置无效")
        safe_id(reference.get("id", ""))
        if reference["id"] in reference_ids:
            raise ValueError("画面参考编号重复")
        reference_ids.add(reference["id"])
        if reference.get("image_id") not in images or reference.get("video_id") not in video_map:
            raise ValueError("画面参考所用图片或原视频已删除")
        if reference.get("kind") not in ("scene", "guide"):
            raise ValueError("请选择原片场景参考或已修正整帧")
        reference["time"] = finite_number(reference.get("time"), "画面参考时间", 0)
        if not isinstance(reference.get("enabled", True), bool):
            raise ValueError("画面参考开关无效")
    for name in ("revision", "candidate"):
        option = settings[name]
        if not isinstance(option, dict) or not isinstance(option.get("enabled", False), bool):
            raise ValueError("修正或候选设置无效")
        option = settings[name] = {**empty_state()["settings"][name], **option}
        if not option["enabled"]:
            continue
        video = video_map.get(option["video_id"])
        if video is None:
            raise ValueError("修正或候选视频不存在")
        segments = {item["id"] for item in video.get("segments", [])}
        if name == "revision":
            safe_id(option["base_result_id"])
            if option["scope"] not in ("selected", "all") or not isinstance(option["segment_ids"], list):
                raise ValueError("修正范围无效")
            if any(item not in segments for item in option["segment_ids"]):
                raise ValueError("修正分段已改变，请重新选择")
            if option["scope"] == "selected" and not option["segment_ids"]:
                raise ValueError("请勾选至少一个需要修正的分段")
        else:
            if option["segment_id"] not in segments:
                raise ValueError("候选分段不存在")
            option["time"] = finite_number(option["time"], "候选画面时间", 0)
    if trial.get("enabled"):
        safe_id(trial.get("video_id", ""))
        safe_id(trial.get("segment_id", ""))
        selected = next((item for item in state["videos"] if item["id"] == trial["video_id"]), None)
        if selected is None or not any(item["id"] == trial["segment_id"] for item in selected.get("segments", [])):
            raise ValueError("试跑选择的视频或分段已经删除，请重新选择")
    return state


def prepare_plan(project_json):
    state = validate_state(project_json)
    for video in state["videos"]:
        path = input_path(video["file"])
        video["info"] = probe_video(path)
        video["output_size"] = list(output_size(video, state["settings"]))
        video["source_signature"] = file_key({"file": video["file"], "size": path.stat().st_size, "mtime": path.stat().st_mtime_ns})
        duration = video["info"]["duration"]
        video["shots"] = detect_segments(path, duration + 1, state["settings"]["scene_threshold"])
        for binding in video.get("bindings", []):
            if any(anchor["time"] >= duration for anchor in binding.get("anchors", [])):
                raise ValueError("人物锚点超出了原视频长度")
        if not video.get("segments"):
            if state["settings"]["segment_mode"] == "auto":
                video["segments"] = detect_segments(path, state["settings"]["max_segment_seconds"], state["settings"]["scene_threshold"])
            else:
                video["segments"] = [{"id": "segment_1", "start": 0, "end": video["info"]["duration"], "enabled": True, "role_ids": [], "prompt": ""}]
        segments = sorted(video["segments"], key=lambda item: item["start"])
        end = 0
        for segment in segments:
            if segment["start"] < end - .001:
                raise ValueError("同一视频的分段不能重叠；请合并或调整边界")
            if segment["end"] > video["info"]["duration"] + 1 / 90000:
                raise ValueError("分段超出了原视频长度")
            segment["end"] = min(segment["end"], video["info"]["duration"])
            end = segment["end"]
        if any(item["time"] >= duration for item in state["frame_references"] if item["video_id"] == video["id"]):
            raise ValueError("画面参考时间超出原视频")
        candidate = state["settings"]["candidate"]
        if candidate["enabled"] and candidate["video_id"] == video["id"]:
            segment = next(item for item in video["segments"] if item["id"] == candidate["segment_id"])
            if not segment["start"] <= candidate["time"] < segment["end"]:
                raise ValueError("候选画面时间需在所选分段内")
    return {"state": state}


def active_segments(plan):
    state = plan["state"]
    trial = state["settings"].get("trial", {})
    revision = state["settings"].get("revision", {})
    candidate = state["settings"].get("candidate", {})
    result = []
    for video in state["videos"]:
        if candidate.get("enabled") and video["id"] != candidate["video_id"]:
            continue
        if revision.get("enabled") and video["id"] != revision["video_id"]:
            continue
        if trial.get("enabled") and video["id"] != trial.get("video_id"):
            continue
        for segment in sorted(video["segments"], key=lambda item: item["start"]):
            if candidate.get("enabled") and segment["id"] != candidate["segment_id"]:
                continue
            if revision.get("enabled") and revision["scope"] == "selected" and segment["id"] not in revision["segment_ids"]:
                continue
            if not segment.get("enabled", True):
                continue
            if trial.get("enabled") and segment["id"] != trial.get("segment_id"):
                continue
            shots = shot_plan(video, segment)
            role_ids = [role for shot in shots for role in shot["role_ids"]]
            if state["settings"].get("person_mode") == "single":
                if any(len(shot["role_ids"]) > 1 for shot in shots):
                    raise ValueError("同一镜头有多个参与角色，请在导演台选择多人模式，或仅选择一个角色")
            result.append((video, segment, list(dict.fromkeys(role_ids))))
    if not result:
        raise ValueError("没有启用的生成分段；请选择试跑分段或启用视频分段")
    return result
