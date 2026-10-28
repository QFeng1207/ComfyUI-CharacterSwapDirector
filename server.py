from __future__ import annotations

import asyncio
import concurrent.futures
import importlib.util
import json
import subprocess
import threading
import time
import uuid
from pathlib import Path

from aiohttp import web
import folder_paths
from PIL import Image

from .paths import PLUGIN_NAME, cache_dir, file_key, input_path, media_url, safe_id, output_path
from .plan import validate_state
from .media import ffmpeg_path, probe_video, iter_video
from .reference_plan import saved_result

KINDS = {"video": {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}, "image": {".png", ".jpg", ".jpeg", ".webp", ".bmp"}}
JOBS = {}
JOB_LOCK = threading.Lock()
GPU_LOCK = threading.RLock()
WORKERS = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="csd-media")


class CancelledJob(Exception):
    pass


def asset_for(path, kind, *, probe=False):
    relative = path.resolve().relative_to(Path(folder_paths.get_input_directory()).resolve()).as_posix()
    result = {"id": file_key(relative), "file": relative, "name": path.name, "kind": kind, "url": media_url(path)}
    if kind == "video" and probe:
        result["info"] = probe_video(path)
        codec = result["info"].get("video_codec")
        if path.suffix.lower() not in (".mp4", ".webm") or result["info"].get("rotation", 0) or codec not in (None, "h264", "vp8", "vp9", "av1"):
            signature = file_key({"path": relative, "size": path.stat().st_size, "mtime": path.stat().st_mtime_ns})
            proxy = cache_dir("previews") / (signature + ".mp4")
            if not proxy.is_file():
                temporary = proxy.with_name(signature + ".part.mp4")
                subprocess.run([ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(path), "-vf", "scale=960:960:force_original_aspect_ratio=decrease:force_divisible_by=2", "-c:v", "libx264", "-crf", "24", "-preset", "fast", "-c:a", "aac", "-movflags", "+faststart", str(temporary)], check=True, capture_output=True)
                temporary.replace(proxy)
            result["proxy_url"] = media_url(proxy)
    return result


def start_job(operation):
    job_id = uuid.uuid4().hex
    job = {"id": job_id, "status": "queued", "progress": 0, "message": "等待处理", "created": time.time(), "cancel": threading.Event()}
    with JOB_LOCK:
        completed = [key for key, item in JOBS.items() if item["status"] in ("complete", "error", "cancelled")]
        for key in completed[:-20]:
            JOBS.pop(key, None)
        JOBS[job_id] = job

    def progress(fraction, message):
        if job["cancel"].is_set():
            raise CancelledJob()
        with JOB_LOCK:
            job.update(progress=float(fraction), message=str(message))

    def run():
        try:
            progress(0, "开始处理")
            job["status"] = "running"
            # Current auxiliary jobs only decode media or detect cuts; GPU generation runs in graph nodes.
            result = operation(progress)
            progress(1, "处理完成")
            with JOB_LOCK:
                job.update(status="complete", result=result)
        except CancelledJob:
            job.update(status="cancelled", message="已取消")
        except Exception as error:
            job.update(status="error", message=str(error))
    WORKERS.submit(run)
    return {"job_id": job_id}


def _video(state, video_id):
    video = next((item for item in state["videos"] if item["id"] == video_id), None)
    if video is None:
        raise web.HTTPBadRequest(text="所选视频不存在")
    return video


async def selected_video(request):
    try:
        body = await request.json()
        state = validate_state(body["state"])
        return state, _video(state, body["video_id"])
    except (ValueError, TypeError, KeyError, FileNotFoundError) as error:
        raise web.HTTPBadRequest(text="请求设置无效：" + str(error)) from error


def _available():
    model_root = Path(folder_paths.models_dir)
    h3_files = [("diffusion_models", "minimax_h3_ref2va_pruned_int8_convrot.safetensors"),
                 ("text_encoders", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"),
                 ("vae", "minimax_h3_video_vae_int8_convrot.safetensors")]
    resolve_model = getattr(folder_paths, "get_full_path", lambda category, name: str(model_root / category / name))
    missing = [name for category, name in h3_files if not (path := resolve_model(category, name)) or not Path(path).is_file()]
    return {"dependencies": [{"name": "H3 全画面参考生成模型", "available": not missing, "message": "缺少：" + "、".join(missing) if missing else "Ref2VA、Qwen 与 H3 视频 VAE 已找到；原声直接复制"}], "torch_unchanged": True}


def register_routes(prompt_server):
    routes = prompt_server.routes

    @routes.get("/csd/assets")
    async def assets(request):
        kind = request.query.get("kind", "video")
        if kind not in KINDS:
            raise web.HTTPBadRequest(text="素材类型无效")
        root = Path(folder_paths.get_input_directory()).resolve()
        items = []
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in KINDS[kind] and path.resolve().is_relative_to(root):
                items.append(asset_for(path, kind))
        return web.json_response({"assets": sorted(items, key=lambda item: item["name"].lower())})

    @routes.get("/csd/asset")
    async def asset(request):
        kind = request.query.get("kind", "video")
        if kind not in KINDS:
            raise web.HTTPBadRequest(text="素材类型无效")
        try:
            path = input_path(request.query.get("file", ""))
            if path.suffix.lower() not in KINDS[kind]:
                raise ValueError("素材格式与类型不匹配")
            return web.json_response(await asyncio.to_thread(asset_for, path, kind, probe=kind == "video"))
        except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as error:
            raise web.HTTPBadRequest(text=str(error)) from error

    @routes.post("/csd/upload")
    async def upload(request):
        reader = await request.multipart()
        part = await reader.next()
        if part is None or part.name != "file" or not part.filename:
            raise web.HTTPBadRequest(text="请选择要上传的素材")
        suffix = Path(part.filename).suffix.lower()
        kind = next((key for key, extensions in KINDS.items() if suffix in extensions), None)
        if kind is None:
            raise web.HTTPBadRequest(text="不支持此素材格式")
        directory = Path(folder_paths.get_input_directory()) / PLUGIN_NAME
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / (uuid.uuid4().hex + suffix)
        try:
            with target.open("wb") as output:
                while chunk := await part.read_chunk(1024 * 1024):
                    output.write(chunk)
            if kind == "image":
                with Image.open(target) as image:
                    image.verify()
            result = await asyncio.to_thread(asset_for, target, kind, probe=kind == "video")
            result["name"] = Path(part.filename).name
            return web.json_response(result)
        except Exception as error:
            target.unlink(missing_ok=True)
            raise web.HTTPBadRequest(text=str(error)) from error

    def check_queue():
        if prompt_server.prompt_queue and prompt_server.prompt_queue.get_current_queue()[0]:
            raise web.HTTPConflict(text="正在生成视频，请等待当前生成完成后再进行素材预处理")

    @routes.post("/csd/analyse")
    async def analyse(request):
        from .media import probe_video
        from .segmentation import detect_segments

        state, video = await selected_video(request)
        path = input_path(video["file"])
        def operation(progress):
            progress(0, "读取视频与检测镜头")
            segments = detect_segments(path, state["settings"]["max_segment_seconds"], state["settings"]["scene_threshold"])
            shots = []
            for segment in segments:
                if not shots or shots[-1]["scene_id"] != segment.get("scene_id", segment["id"]):
                    shots.append({"id": segment.get("scene_id", segment["id"]), "scene_id": segment.get("scene_id", segment["id"]),
                                  "start": segment.get("scene_start", segment["start"]), "end": segment.get("scene_end", segment["end"])})
                else:
                    shots[-1]["end"] = max(shots[-1]["end"], segment["end"])
            return {"segments": segments, "shots": shots, "info": probe_video(path)}
        return web.json_response(start_job(operation))

    @routes.post("/csd/track")
    async def track(request):
        return web.json_response({"error": "新方案使用 H3 全画面生成；人物框选用于识别目标，不再进行追踪或生成蒙版。请加载新工作流。"}, status=410)

    @routes.get("/csd/results")
    async def result_history(request):
        source = request.query.get("source", "")
        items = []
        for report_path in output_path("history_placeholder").parent.glob("*.json"):
            try:
                report = json.loads(report_path.read_text(encoding="utf-8-sig"))
                if report.get("backend") != "minimax_h3_reference" or (source and report.get("source") != source):
                    continue
                path, report = saved_result(report["result_id"])
                preview_name = report.get("preview_file", path.name)
                if Path(preview_name).name != preview_name:
                    continue
                preview = output_path(preview_name)
                items.append({"result_id": report["result_id"], "video_id": report["video_id"], "name": path.name, "url": media_url(path),
                    "preview_url": media_url(preview if preview.is_file() else path), "report": report})
                candidate_file = report.get("candidate_file")
                if candidate_file and Path(candidate_file).name == candidate_file:
                    candidate = output_path(candidate_file)
                    if candidate.is_file():
                        items[-1]["candidate_url"] = media_url(candidate)
            except (ValueError, OSError, KeyError, TypeError):
                continue
        return web.json_response({"results": sorted(items, key=lambda item: item["report"].get("created", 0), reverse=True)})

    @routes.post("/csd/frame")
    async def extract_frame(request):
        try:
            body = await request.json()
            state = validate_state(body["state"])
            video = _video(state, body["video_id"])
            time_value = float(body["time"])
            path = input_path(video["file"])
            offset = 0.
            result_id = body.get("result_id")
            if result_id:
                path, report = saved_result(result_id)
                if report["source"] != video["file"]:
                    raise ValueError("结果与当前原视频不匹配")
                offset = report["start"]
            info = await asyncio.to_thread(probe_video, path)
            local_time = time_value - offset
            if not 0 <= local_time < info["duration"]:
                raise ValueError("截图时间超出了当前视频")
            def operation():
                iterator = iter_video(path, local_time, info["duration"])
                try:
                    value = next(iterator, None)
                finally:
                    iterator.close()
                if value is None:
                    raise ValueError("未找到该时间的画面")
                target = Path(folder_paths.get_input_directory()) / PLUGIN_NAME / (uuid.uuid4().hex + ".png")
                target.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(value[0]).save(target)
                return {"asset": asset_for(target, "image"), "video_id": video["id"], "time": value[1] + offset, "source_result_id": result_id}
            return web.json_response(await asyncio.to_thread(operation))
        except (ValueError, KeyError, TypeError, OSError) as error:
            return web.json_response({"error": str(error)}, status=400)

    @routes.post("/csd/speakers")
    async def speakers(request):
        return web.json_response({"error": "声音功能已简化为保留原声或静音，请刷新导演台。"}, status=410)

    @routes.get("/csd/jobs/{job_id}")
    async def job_state(request):
        try:
            safe_id(request.match_info["job_id"])
        except ValueError as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        with JOB_LOCK:
            job = JOBS.get(request.match_info["job_id"])
            if job is None:
                raise web.HTTPNotFound(text="作业不存在或已过期")
            result = {key: value for key, value in job.items() if key not in ("cancel", "created")}
        return web.json_response(result)

    @routes.post("/csd/jobs/{job_id}/cancel")
    async def cancel_job(request):
        with JOB_LOCK:
            job = JOBS.get(request.match_info["job_id"])
            if job is None:
                raise web.HTTPNotFound(text="作业不存在")
            job["cancel"].set()
        return web.json_response({"status": "cancelling"})

    @routes.get("/csd/dependencies")
    async def dependencies(request):
        return web.json_response(_available())
