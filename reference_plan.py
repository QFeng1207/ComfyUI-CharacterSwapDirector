"""Reference frames and persistent revision bases, independent of segmentation models."""
from __future__ import annotations

import json
from pathlib import Path

from .paths import output_path, safe_id
from .resolution import output_size


def saved_result(result_id):
    """Resolve only plugin-owned outputs with a matching, persistent report."""
    safe_id(result_id)
    report_path = output_path(result_id + ".json")
    if not report_path.is_file():
        raise ValueError("上一版结果记录不存在，请重新选择版本")
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    if report.get("result_id") != result_id or report.get("backend") != "minimax_h3_reference":
        raise ValueError("所选结果不是此导演台的 H3 全画面版本")
    filename = report.get("output_file", "")
    if Path(filename).name != filename or Path(filename).suffix.lower() not in (".mov", ".mp4"):
        raise ValueError("结果文件记录无效")
    path = output_path(filename)
    if not path.is_file():
        raise ValueError("上一版视频已被移动或删除")
    return path, report


def frame_references(state, video_id, start, end):
    return [item for item in state.get("frame_references", [])
            if item.get("enabled", True) and item["video_id"] == video_id
            and start - 1e-6 <= item["time"] < end - 1e-6]


def revision_base(state, video):
    revision = state["settings"].get("revision", {})
    if not revision.get("enabled") or revision.get("video_id") != video["id"]:
        return None
    path, report = saved_result(revision["base_result_id"])
    if report.get("source") != video["file"] or report.get("trial") or report.get("candidate"):
        raise ValueError("修正底稿必须是当前原视频的完整生成版本；试跑或候选不能作为完整底稿")
    if report.get("resolution") != list(output_size(video, state["settings"])) or abs(report["duration"] - video["info"]["duration"]) > 1e-4:
        raise ValueError("底稿尺寸与当前输出设置不同，或时长不一致；改变尺寸后请先完整生成一版")
    signature = report.get("source_signature")
    if signature and signature != video.get("source_signature"):
        raise ValueError("原素材已改变，不能复用旧版本；请先完整生成")
    return path, report
