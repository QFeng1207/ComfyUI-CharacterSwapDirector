"""Original soundtrack and mute helpers; no voice models are loaded."""
from __future__ import annotations

from . import media
from .paths import input_path
from .plan import normalize_audio_mode


def dependency_status():
    return {"available": True, "modes": ["original", "mute"], "requires_models": False}


def prepare_audio_file(video: dict, roles: list, assets: list, mode: str, progress=None, work_dir=None):
    """Keep the source file for muxing; legacy voice modes use original audio."""
    mode = normalize_audio_mode(mode)
    if mode == "mute":
        return None
    source = input_path(video["file"])
    if video.get("info", {}).get("has_audio") is False:
        return None
    if progress:
        progress(1, "保留原声")
    return {"audio_file": str(source), "report": {"mode": "original"}}


def prepare_audio(video: dict, roles: list, assets: list, mode: str, progress=None, work_dir=None):
    if normalize_audio_mode(mode) == "mute":
        return None
    return media.extract_audio(input_path(video["file"]))
