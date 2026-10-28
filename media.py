from __future__ import annotations

from bisect import bisect_left
from fractions import Fraction
from pathlib import Path
from typing import Iterable

import av
import imageio_ffmpeg
import numpy as np
import torch


def ffmpeg_path() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def _rate(stream) -> Fraction:
    return Fraction(stream.average_rate or stream.base_rate or stream.guessed_rate or 24)


def probe_video(path: Path) -> dict:
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("素材没有视频轨道。")
        stream = container.streams.video[0]
        rate = _rate(stream)
        decoded = container.decode(stream)
        first = next(decoded, None)
        if first is None:
            raise ValueError("素材没有可解码的视频帧。")
        rotation = first.rotation % 360
        width, height = first.width, first.height
        if rotation in (90, 270):
            width, height = height, width
        duration = float(stream.duration * stream.time_base) if stream.duration is not None else 0.0
        frames = int(stream.frames)
        if not frames or not duration:
            origin = float((stream.start_time or 0) * stream.time_base)
            count = 1
            last_time = float(first.time) - origin if first.pts is not None else 0.0
            for frame in decoded:
                last_time = float(frame.time) - origin if frame.pts is not None else count / float(rate)
                count += 1
            frames = count
            duration = max(duration, last_time + 1.0 / float(rate))
        return {
            "width": width, "height": height,
            "coded_width": first.width, "coded_height": first.height,
            "rotation": rotation, "video_codec": stream.codec_context.name,
            "duration": duration, "fps_num": rate.numerator, "fps_den": rate.denominator,
            "frames": frames, "has_audio": bool(container.streams.audio),
        }


def _decoded_frames(path: Path, start=0.0, end=None, size=None):
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("素材没有视频轨道。")
        stream = container.streams.video[0]
        rate = float(_rate(stream))
        origin = float((stream.start_time or 0) * stream.time_base)
        if start > 0:
            container.seek(int((start + origin) / float(stream.time_base)), stream=stream, backward=True)
        for index, frame in enumerate(container.decode(stream)):
            time = float(frame.time) - origin if frame.pts is not None else index / rate
            if end is not None and time >= end - 1e-9:
                break
            rotation = frame.rotation % 360
            if size is not None:
                width, height = size[::-1] if rotation in (90, 270) else size
                frame = frame.reformat(width=int(width), height=int(height), format="rgb24")
            image = frame.to_ndarray(format="rgb24")
            if rotation in (90, 180, 270):
                image = np.ascontiguousarray(np.rot90(image, rotation // 90))
            yield image, max(0.0, time)


def iter_video(path: Path, start=0.0, end=None, fps=None, size=None):
    """Decode incrementally; sampled times always identify the selected original frame."""
    start = max(0.0, float(start))
    if end is None:
        end = probe_video(path)["duration"]
    end = float(end)
    if end <= start:
        return
    decoded = _decoded_frames(path, start, end, size)
    if fps is None:
        for frame, time in decoded:
            if time >= start - 1e-9:
                yield frame, time
        return
    rate = Fraction(str(fps)) if not isinstance(fps, Fraction) else fps
    if rate <= 0:
        raise ValueError("采样帧率必须大于 0。")
    target_index = 0
    target = start
    previous = None
    for current in decoded:
        while target <= current[1] + 1e-9 and target < end - 1e-9:
            selected = current
            if previous is not None and abs(target - previous[1]) <= abs(current[1] - target):
                selected = previous
            yield selected
            target_index += 1
            target = start + float(target_index / rate)
        previous = current
    if previous is not None:
        while target < end - 1e-9:
            yield previous
            target_index += 1
            target = start + float(target_index / rate)


def read_video(path: Path, start=0.0, end=None, fps=None, size=None):
    images, times = [], []
    for frame, time in iter_video(path, start, end, fps, size):
        images.append(frame)
        times.append(time)
    if not images:
        raise ValueError("指定时间范围内没有视频帧。")
    return torch.from_numpy(np.stack(images)).float().div_(255), times


def nearest_frame_indices(timestamps: Iterable[float], times: Iterable[float]) -> list[int]:
    stamps = list(timestamps)
    if not stamps:
        raise ValueError("源视频时间索引为空。")
    result = []
    for time in times:
        right = bisect_left(stamps, float(time))
        if right == 0:
            result.append(0)
        elif right == len(stamps):
            result.append(len(stamps) - 1)
        else:
            result.append(right - 1 if time - stamps[right - 1] <= stamps[right] - time else right)
    return result


def extract_audio(path: Path, start=0.0, end=None) -> dict | None:
    start = max(0.0, float(start))
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return None
        stream = container.streams.audio[0]
        video_origin = float((container.streams.video[0].start_time or 0) * container.streams.video[0].time_base) if container.streams.video else 0.0
        source_rate = stream.codec_context.sample_rate or 48000
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=source_rate)
        # Decode the prefix too: seeking loses AAC priming and decoder noise state,
        # so even a preroll cannot produce the exact samples of a full decode.
        chunks = []
        expected_sample = int(round(start * source_rate))
        end_sample = int(round(end * source_rate)) if end is not None else None
        for packet in container.demux(stream):
            for source in packet.decode():
                for frame in resampler.resample(source):
                    frame_start = int(round((float(frame.time) - video_origin) * source_rate)) if frame.pts is not None else expected_sample
                    data = frame.to_ndarray()
                    left = max(0, expected_sample - frame_start)
                    right = len(data[0]) if end_sample is None else min(len(data[0]), end_sample - frame_start)
                    if right <= left:
                        continue
                    if frame_start + left > expected_sample:
                        chunks.append(np.zeros((2, frame_start + left - expected_sample), np.float32))
                    chunks.append(data[:, left:right].copy())
                    expected_sample = frame_start + right
                if end_sample is not None and expected_sample >= end_sample:
                    break
            if end_sample is not None and expected_sample >= end_sample:
                break
        for frame in resampler.resample(None):
            frame_start = int(round((float(frame.time) - video_origin) * source_rate)) if frame.pts is not None else expected_sample
            data = frame.to_ndarray()
            left = max(0, expected_sample - frame_start)
            right = len(data[0]) if end_sample is None else min(len(data[0]), end_sample - frame_start)
            if right > left:
                chunks.append(data[:, left:right].copy())
                expected_sample = frame_start + right
        if not chunks:
            return None
        return {"waveform": torch.from_numpy(np.concatenate(chunks, axis=1))[None], "sample_rate": source_rate}


def _rgb(frame) -> np.ndarray:
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[-1] < 3:
        raise ValueError("视频帧必须为 H × W × RGB。")
    if frame.dtype != np.uint8:
        frame = np.rint(np.clip(frame, 0, 1) * 255).astype(np.uint8)
    return np.ascontiguousarray(frame[..., :3])


class VideoWriter:
    """Bounded video/audio encoder, suitable for full-resolution frame-by-frame compositing."""
    def __init__(self, path: Path, width: int, height: int, fps_num: int, fps_den: int, audio=None, lossless=False, duration=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rate = Fraction(int(fps_num), int(fps_den))
        self.container = av.open(str(self.path), "w")
        self.stream = self.container.add_stream("libx264rgb" if lossless else "libx264", rate=self.rate)
        self.stream.width, self.stream.height = int(width), int(height)
        self.stream.pix_fmt = "rgb24" if lossless else ("yuv420p" if width % 2 == 0 and height % 2 == 0 else "yuv444p")
        self.stream.options = {"crf": "0" if lossless else "18", "preset": "fast"}
        self.frames = 0
        self.duration = float(duration) if duration is not None else None
        self.pending = None
        self.variable_timestamps = False
        self.timestamp_base = Fraction(1, 90000)
        self.last_interval = float(1 / self.rate)
        self.packet_durations = {}
        self.audio_offset = 0
        self.audio_stream = None
        self.waveform = None
        if audio is not None:
            wave = audio["waveform"]
            if isinstance(wave, torch.Tensor):
                wave = wave.detach().cpu().numpy()
            wave = np.asarray(wave)
            if wave.ndim == 3:
                wave = wave[0]
            if wave.ndim == 1:
                wave = wave[None]
            self.waveform = np.ascontiguousarray(wave[:2], dtype=np.float32)
            self.audio_rate = int(audio["sample_rate"])
            self.audio_layout = "mono" if len(self.waveform) == 1 else "stereo"
            self.audio_stream = self.container.add_stream("aac", rate=self.audio_rate)
            self.audio_stream.layout = self.audio_layout
        self.closed = False

    def _audio_until(self, limit):
        if self.audio_stream is None:
            return
        limit = min(int(limit), self.waveform.shape[-1])
        while self.audio_offset < limit:
            stop = min(limit, self.audio_offset + 4096)
            frame = av.AudioFrame.from_ndarray(self.waveform[:, self.audio_offset:stop], format="fltp", layout=self.audio_layout)
            frame.sample_rate = self.audio_rate
            frame.pts = self.audio_offset
            frame.time_base = Fraction(1, self.audio_rate)
            for packet in self.audio_stream.encode(frame):
                self.container.mux(packet)
            self.audio_offset = stop

    def _mux_video(self, packets):
        for packet in packets:
            if self.variable_timestamps:
                time_base = packet.time_base or self.timestamp_base
                original_pts = round(packet.pts * time_base / self.timestamp_base)
                duration = self.packet_durations.pop(original_pts, None)
                if duration is not None:
                    packet.duration = max(1, round(duration * self.timestamp_base / time_base))
            self.container.mux(packet)

    def _encode_frame(self, image, pts, time_base, duration=None):
        video_frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        if duration is not None:
            video_frame.duration = max(1, round(duration / time_base))
            self.packet_durations[pts] = video_frame.duration
        self._mux_video(self.stream.encode(video_frame))

    def _encode_pending(self, next_time):
        image, time = self.pending
        interval = next_time - time
        if interval <= 0:
            raise ValueError("原视频帧时间必须严格递增。")
        self.last_interval = interval
        self._encode_frame(image, round(time / self.timestamp_base), self.timestamp_base, interval)
        if self.audio_stream is not None:
            self._audio_until(round(next_time * self.audio_rate))

    def write(self, frame, time=None):
        image = _rgb(frame)
        if image.shape[:2] != (self.stream.height, self.stream.width):
            raise ValueError("输出帧尺寸与原视频不一致。")
        if self.frames == 0 and time is not None:
            self.variable_timestamps = True
            self.stream.time_base = self.timestamp_base
            self.stream.codec_context.time_base = self.timestamp_base
            self.stream.codec_context.max_b_frames = 0
        if self.variable_timestamps:
            if time is None:
                raise ValueError("保留原时序模式下，每帧都需要原视频时间。")
            time = float(time)
            if not np.isfinite(time) or time < 0:
                raise ValueError("原视频帧时间无效。")
            if self.pending is not None:
                self._encode_pending(time)
            self.pending = (image.copy(), time)
            self.frames += 1
            return
        if time is not None:
            raise ValueError("请从第一帧开始提供原视频时间。")
        self._encode_frame(image, self.frames, 1 / self.rate)
        self.frames += 1
        if self.audio_stream is not None:
            self._audio_until(round(self.frames / self.rate * self.audio_rate))

    def close(self):
        if self.closed:
            return self.path
        try:
            if self.pending is not None:
                end_time = self.duration if self.duration is not None else self.pending[1] + self.last_interval
                self._encode_pending(end_time)
                self.pending = None
            self._mux_video(self.stream.encode(None))
            if self.audio_stream is not None:
                for packet in self.audio_stream.encode(None):
                    self.container.mux(packet)
        finally:
            self.container.close()
            self.closed = True
        return self.path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def write_video(frames, fps_num, fps_den, path: Path, audio=None, lossless=False, timestamps=None, duration=None) -> Path:
    iterator = iter(frames)
    try:
        first = _rgb(next(iterator))
    except StopIteration as exc:
        raise ValueError("无法保存空视频。") from exc
    height, width = first.shape[:2]
    times = iter(timestamps) if timestamps is not None else None
    with VideoWriter(path, width, height, fps_num, fps_den, audio, lossless, duration) as writer:
        writer.write(first, next(times) if times is not None else None)
        for frame in iterator:
            writer.write(frame, next(times) if times is not None else None)
    return Path(path)
