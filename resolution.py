"""Director canvas settings; MP uses ComfyUI's 1024² convention."""
import math

MULTIPLE = 32
MAX_SIZE = 16384


def aligned(value):
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("分辨率宽高的数值无效") from error
    if not math.isfinite(value) or not MULTIPLE <= value <= MAX_SIZE:
        raise ValueError("分辨率宽高必须在 32–16384 像素之间")
    return min(MAX_SIZE, max(MULTIPLE, round(value / MULTIPLE) * MULTIPLE))


def normalize_resolution(value):
    if not isinstance(value, dict) or not isinstance(value.get("enabled", False), bool):
        raise ValueError("分辨率设置无效")
    result = {"enabled": value.get("enabled", False), "width": value.get("width", 0), "height": value.get("height", 0)}
    if result["enabled"]:
        result.update(width=aligned(result["width"]), height=aligned(result["height"]))
    return result


def output_size(video, settings=None):
    resolution = (settings or {}).get("resolution", {})
    if resolution.get("enabled"):
        return aligned(resolution["width"]), aligned(resolution["height"])
    if settings is not None:
        return video["info"]["width"], video["info"]["height"]
    return tuple(video.get("output_size", (video["info"]["width"], video["info"]["height"])))


def center_rect(source_width, source_height, width, height):
    scale = min(width / source_width, height / source_height)
    core_width = min(width, max(MULTIPLE, round(source_width * scale / MULTIPLE) * MULTIPLE))
    core_height = min(height, max(MULTIPLE, round(source_height * scale / MULTIPLE) * MULTIPLE))
    return (width - core_width) // 2, (height - core_height) // 2, core_width, core_height
