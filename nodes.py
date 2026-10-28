from __future__ import annotations

import json
from pathlib import Path

import comfy.model_management as model_management
from server import PromptServer

from .generation import render_batch, save_results
from .paths import cache_dir, file_key, input_path, media_url
from .plan import active_segments, empty_state, normalize_audio_mode, prepare_plan, validate_state
from .server import GPU_LOCK
from .tracking import track_video
from .fullframe_generation import render_batch as render_reference_batch
from .sampling_setup import sampling_setup

try:
    from .wan_generation import render_batch as render_animate_batch
except ImportError:
    render_animate_batch = None

CATEGORY = "Character Swap Director/人物替换"


def progress_for(plan, stage, unique_id):
    def progress(fraction, message, video_id=""):
        model_management.throw_exception_if_processing_interrupted()
        PromptServer.instance.send_sync("csd.progress", {"node": plan.get("planner_node", str(unique_id)), "execution_node": str(unique_id),
            "stage": stage, "progress": float(fraction), "message": str(message), "video_id": video_id})
    return progress


class GpuTask:
    def __enter__(self):
        while not GPU_LOCK.acquire(timeout=.5):
            model_management.throw_exception_if_processing_interrupted()
        return self

    def __exit__(self, exc_type, exc, traceback):
        GPU_LOCK.release()


class MaterialPlanner:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"project_json": ("STRING", {"default": json.dumps(empty_state(), ensure_ascii=False), "multiline": True})},
                "hidden": {"unique_id": "UNIQUE_ID"}}
    RETURN_TYPES = ("CSD_PLAN",)
    RETURN_NAMES = ("导演计划",)
    FUNCTION = "build"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, project_json, **kwargs):
        state = validate_state(project_json, require_media=False)
        stamps = []
        for key in ("videos", "images"):
            for asset in state.get(key, []):
                source = input_path(asset["file"])
                stamps.append((asset["file"], source.stat().st_size, source.stat().st_mtime_ns))
        return file_key({"state": state, "files": stamps})

    def build(self, project_json, unique_id):
        plan = prepare_plan(project_json)
        plan["planner_node"] = str(unique_id)
        return (plan,)


class TrackCharacters:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"plan": ("CSD_PLAN",)}, "hidden": {"unique_id": "UNIQUE_ID"}}
    RETURN_TYPES = ("CSD_TRACKS",)
    RETURN_NAMES = ("人物蒙版与追踪",)
    FUNCTION = "track"
    CATEGORY = CATEGORY

    def track(self, plan, unique_id):
        raise RuntimeError("人物追踪节点已停用。请加载 examples/character_swap_local.json，使用 H3 全画面参考生成节点。")
        state = plan["state"]
        videos = {video["id"]: video for video, _, _ in active_segments(plan)}
        output = {}
        report = progress_for(plan, "tracking", unique_id)
        with GpuTask():
            for number, video in enumerate(videos.values()):
                def progress(fraction, message):
                    report((number + fraction) / len(videos), message, video["id"])
                output[video["id"]] = track_video(video, state["roles"], cache_dir("tracks") / file_key(video), progress=progress)
        return ({"videos": output},)


class AudioProcess:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"plan": ("CSD_PLAN",)}, "hidden": {"unique_id": "UNIQUE_ID"}}
    RETURN_TYPES = ("CSD_AUDIO",)
    RETURN_NAMES = ("声音结果",)
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, plan, unique_id):
        # Compatibility adapter for workflows that still connect this node.
        mode = normalize_audio_mode(plan["state"]["settings"]["audio_mode"])
        report = progress_for(plan, "audio", unique_id)
        report(1, "保留原声" if mode == "original" else "静音")
        return ({"videos": {}, "mode": mode},)


class RenderBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"plan": ("CSD_PLAN",), "tracks": ("CSD_TRACKS",), "model": ("MODEL",), "clip": ("CLIP",),
                "video_vae": ("VAE",), "audio_vae": ("VAE",), "model_patch": ("MODEL_PATCH",), "noise": ("NOISE",), "sampler": ("SAMPLER",), "sigmas": ("SIGMAS",),
                "processing_long_edge": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 32, "tooltip": "内部生成尺寸；最终仍按原视频尺寸合成。12GB 显存建议先用 512。"}),
                "motion_control": (["pose", "canny", "none"], {"default": "pose"}),
                "control_strength": ("FLOAT", {"default": 1., "min": .05, "max": 3., "step": .05}),
                "frame_interpolation": (["linear", "nearest"], {"default": "linear"})}, "hidden": {"unique_id": "UNIQUE_ID"}}
    RETURN_TYPES = ("CSD_RESULTS",)
    RETURN_NAMES = ("换人分段结果",)
    FUNCTION = "render"
    CATEGORY = CATEGORY

    def render(self, plan, tracks, model, clip, video_vae, audio_vae, model_patch, noise, sampler, sigmas, processing_long_edge, motion_control, control_strength, frame_interpolation, unique_id):
        raise RuntimeError("局部重绘流程已停用，请加载新的 H3 全画面参考工作流。")
        if len(sigmas) < 2:
            raise ValueError("采样步数为空，请检查外部 Sigma 调度节点。")
        with GpuTask():
            result = render_batch(plan, tracks, model, clip, video_vae, audio_vae, model_patch, noise, sampler, sigmas,
                                  long_edge=processing_long_edge, control_mode=motion_control, control_strength=control_strength,
                                  interpolation=frame_interpolation, progress=progress_for(plan, "generation", unique_id))
        return (result,)


class RenderAnimateBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"plan": ("CSD_PLAN",), "tracks": ("CSD_TRACKS",), "model": ("MODEL",), "clip": ("CLIP",),
                "vae": ("VAE",), "clip_vision": ("CLIP_VISION",), "noise": ("NOISE",), "sampler": ("SAMPLER",), "sigmas": ("SIGMAS",),
                "processing_long_edge": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 32, "tooltip": "内部生成尺寸；最终保持原视频尺寸。12GB 显存先用 512。"}),
                "cfg": ("FLOAT", {"default": 1., "min": 0., "max": 20., "step": .1}),
                "frame_interpolation": (["linear", "nearest"], {"default": "linear"}),
                "preserve_face_motion": ("BOOLEAN", {"default": False, "tooltip": "开启后传递原人物表情；可能影响参考人物的脸部外观。"}),
                "reference_sheet_mode": (["auto", "whole_image"], {"default": "auto", "tooltip": "auto 将清楚分开的白底多视图角色表裁为第一个完整人物；whole_image 使用整张主图。"})},
                "hidden": {"unique_id": "UNIQUE_ID"}}
    RETURN_TYPES = ("CSD_RESULTS",)
    RETURN_NAMES = ("换人分段结果",)
    FUNCTION = "render"
    CATEGORY = CATEGORY

    def render(self, plan, tracks, model, clip, vae, clip_vision, noise, sampler, sigmas, processing_long_edge=512, cfg=1.,
               frame_interpolation="linear", preserve_face_motion=False, reference_sheet_mode="auto", unique_id=""):
        raise RuntimeError("Wan Animate 流程已停用，请加载新的 H3 全画面参考工作流。")
        if render_animate_batch is None:
            raise RuntimeError("当前 ComfyUI 缺少 WanAnimateToVideo 原生节点，请更新 ComfyUI 后重启。")
        if len(sigmas) < 2:
            raise ValueError("采样步数为空，请检查外部 Sigma 调度节点。")
        with GpuTask():
            result = render_animate_batch(plan, tracks, model, clip, vae, clip_vision, noise, sampler, sigmas,
                                          long_edge=processing_long_edge, cfg=cfg, interpolation=frame_interpolation,
                                          preserve_face_motion=preserve_face_motion, sheet_mode=reference_sheet_mode,
                                          progress=progress_for(plan, "generation", unique_id))
        return (result,)


class RenderReferenceBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"plan": ("CSD_PLAN",), "model": ("MODEL",), "clip": ("CLIP",),
            "video_vae": ("VAE",), "noise": ("NOISE",), "sampler": ("SAMPLER",), "sigmas": ("SIGMAS",),
            "processing_long_edge": ("INT", {"default": 1344, "min": 256, "max": 2048, "step": 32, "tooltip": "默认内部生成长边。导演台启用自定义分辨率后，改用导演台宽高生成与输出。"}),
            "frame_interpolation": (["nearest", "linear"], {"default": "nearest"}),
            "reference_image_size": (["max", "match"], {"default": "max"}),
            "reference_sheet_mode": (["auto", "whole_image"], {"default": "auto"})}, "optional": {"audio_vae": ("VAE",)}, "hidden": {"unique_id": "UNIQUE_ID", "prompt": "PROMPT"}}
    RETURN_TYPES = ("CSD_RESULTS",)
    RETURN_NAMES = ("H3 全画面分段结果",)
    FUNCTION = "render"
    CATEGORY = CATEGORY

    def render(self, plan, model, clip, video_vae, noise, sampler, sigmas, processing_long_edge=1344,
               frame_interpolation="nearest", reference_image_size="max", reference_sheet_mode="auto", unique_id="", audio_vae=None, prompt=None):
        if len(sigmas) < 2:
            raise ValueError("请设置有效采样步数；Ref2VA 基线建议 20 步")
        sampling, warnings = sampling_setup(prompt, unique_id, len(sigmas) - 1, reference_image_size)
        report_progress = progress_for(plan, "generation", unique_id)
        if warnings:
            report_progress(0, "；".join(warnings))
        with GpuTask():
            result = render_reference_batch(plan, model, clip, video_vae, audio_vae, noise, sampler, sigmas,
                long_edge=processing_long_edge, interpolation=frame_interpolation, ref_image_size=reference_image_size,
                sheet_mode=reference_sheet_mode, progress=report_progress)
        result["sampling"] = sampling
        for parts in result["videos"].values():
            for part in parts:
                part["warnings"].extend(warnings)
        return (result,)


class SaveResults:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"results": ("CSD_RESULTS",), "filename_prefix": ("STRING", {"default": "CharacterSwap"}),
                "format": (["lossless_mov", "compatible_mp4"], {"default": "lossless_mov", "tooltip": "MOV 无损保留画面及扩画幅中心；MP4 有损压缩。默认原尺寸，自定义时按导演台尺寸输出；时长与播放速度保持。"})},
                "optional": {"audio": ("CSD_AUDIO",)}, "hidden": {"unique_id": "UNIQUE_ID"}}
    RETURN_TYPES = ("CSD_RESULTS",)
    RETURN_NAMES = ("已保存视频",)
    FUNCTION = "save"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def save(self, results, filename_prefix, format, unique_id, audio=None):
        written = save_results(results, audio, filename_prefix, compatible=format == "compatible_mp4", progress=progress_for(results["plan"], "save", unique_id))
        for item in written:
            item["url"] = media_url(Path(item["path"]))
            item["preview_url"] = media_url(Path(item["preview_path"]))
            if item["report"].get("candidate_file"):
                item["candidate_url"] = media_url(Path(item["path"]).with_name(item["report"]["candidate_file"]))
        return {"ui": {"csd_results": written}, "result": ({**results, "saved": written},)}


NODE_CLASS_MAPPINGS = {"CSD_RenderReferenceBatch": RenderReferenceBatch, "CSD_MaterialPlanner": MaterialPlanner, "CSD_TrackCharacters": TrackCharacters, "CSD_AudioProcess": AudioProcess,
                       "CSD_RenderBatch": RenderBatch, "CSD_RenderAnimateBatch": RenderAnimateBatch, "CSD_SaveResults": SaveResults}
NODE_DISPLAY_NAME_MAPPINGS = {"CSD_RenderReferenceBatch": "H3 人物替换 · 全画面参考生成", "CSD_MaterialPlanner": "人物替换导演台", "CSD_TrackCharacters": "已停用：旧人物追踪与蒙版", "CSD_AudioProcess": "声音选择（旧工作流兼容）",
                               "CSD_RenderBatch": "人物替换批量生成（H3 实验）", "CSD_RenderAnimateBatch": "人物替换批量生成（Wan Animate）", "CSD_SaveResults": "人物替换保存与预览"}
