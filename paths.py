from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlencode

import folder_paths

PLUGIN_NAME = "character_swap_director"


def contained_path(root: Path, relative: str, *, must_exist=False) -> Path:
    root = root.resolve()
    value = Path(str(relative))
    if value.is_absolute() or not str(relative) or "\x00" in str(relative):
        raise ValueError("素材路径必须位于 ComfyUI/input 内")
    candidate = (root / value).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("路径超出了指定素材目录")
    if must_exist and not candidate.is_file():
        raise FileNotFoundError(f"找不到素材：{relative}")
    return candidate


def input_path(relative: str) -> Path:
    return contained_path(Path(folder_paths.get_input_directory()), relative, must_exist=True)


def cache_dir(key: str) -> Path:
    root = Path(folder_paths.get_temp_directory()) / PLUGIN_NAME
    path = contained_path(root, key)
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_path(name: str) -> Path:
    root = Path(folder_paths.get_output_directory()) / PLUGIN_NAME
    path = contained_path(root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def file_key(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:24]


def safe_id(value: str) -> str:
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", value):
        raise ValueError("素材或角色编号无效")
    return value


def media_url(path: Path) -> str:
    path = path.resolve()
    for kind, root in (("input", folder_paths.get_input_directory()), ("temp", folder_paths.get_temp_directory()), ("output", folder_paths.get_output_directory())):
        root = Path(root).resolve()
        if path.is_relative_to(root):
            relative = path.relative_to(root)
            return "/view?" + urlencode({"filename": relative.name, "subfolder": relative.parent.as_posix() if relative.parent != Path(".") else "", "type": kind})
    raise ValueError("预览文件必须位于 ComfyUI 媒体目录")
