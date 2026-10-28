import importlib
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import av
import numpy as np
import torch
from PIL import Image

PLUGIN = Path(__file__).resolve().parents[1]
PACKAGE = "csd_generation_tests"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(PLUGIN)]
sys.modules[PACKAGE] = package


class FakeReference:
    dimensions = (5, 32, 32)

    @classmethod
    def execute(cls, clip, prompt, width, height, length, **kwargs):
        cls.dimensions = (length, height, width)
        return types.SimpleNamespace(result=("conditioning", {}))


class FakeNode:
    @classmethod
    def execute(cls, *args, **kwargs):
        return types.SimpleNamespace(result=(args[0],))


class FakeSampler:
    @classmethod
    def execute(cls, *args, **kwargs):
        return types.SimpleNamespace(result=(None, {"samples": types.SimpleNamespace(tensors=[torch.empty(0)])}))


stubs = {}
for name in ("nodes", "comfy", "comfy.model_management", "comfy_extras",
             "comfy_extras.nodes_custom_sampler", "comfy_extras.nodes_minimax_h3"):
    stubs[name] = types.ModuleType(name)
stubs["comfy"].__path__ = []
stubs["comfy_extras"].__path__ = []
stubs["comfy.model_management"].soft_empty_cache = lambda: None
stubs["comfy.model_management"].throw_exception_if_processing_interrupted = lambda: None
stubs["comfy"].model_management = stubs["comfy.model_management"]
stubs["nodes"].NODE_CLASS_MAPPINGS = {}
stubs["comfy_extras.nodes_custom_sampler"].BasicGuider = FakeNode
stubs["comfy_extras.nodes_custom_sampler"].SamplerCustomAdvanced = FakeSampler
stubs["comfy_extras.nodes_minimax_h3"].MiniMaxH3ReferenceToVideo = FakeReference
stubs["comfy_extras.nodes_minimax_h3"].MiniMaxH3AddGuide = FakeNode
stubs["comfy_extras.nodes_minimax_h3"].MiniMaxH3FunControlNetApply = FakeNode
stubs["comfy_extras.nodes_minimax_h3"].align_frame_count = lambda length: length + ((5 - length) % 17)
if "folder_paths" not in sys.modules:
    stubs["folder_paths"] = types.ModuleType("folder_paths")
    stubs["folder_paths"].models_dir = str(PLUGIN / "test_models")
with patch.dict(sys.modules, stubs):
    generation = importlib.import_module(PACKAGE + ".generation")
    media = importlib.import_module(PACKAGE + ".media")
    tracking = importlib.import_module(PACKAGE + ".tracking")


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for name in ("input", "renders", "tracks"):
            (self.root / name).mkdir()
        self.patches = [patch.object(generation, "cache_dir", side_effect=lambda key: self.root / key),
                        patch.object(generation, "input_path", side_effect=lambda value: self.root / "input" / value),
                        patch.object(tracking.paths, "cache_dir", side_effect=lambda key: self.root / key)]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temporary.cleanup()

    def test_lossless_composite_keeps_every_pixel_outside_editable_region(self):
        original = np.random.default_rng(41).integers(0, 256, (18, 32, 3), dtype=np.uint8)
        generated = np.ones((9, 16, 3), np.float32)
        alpha = np.zeros((18, 32), np.float32)
        alpha[4:14, 8:24] = 1
        alpha[3, 8:24] = 0.5
        result = generation.composite_frame(original, generated, alpha)
        self.assertTrue(np.array_equal(result[alpha == 0], original[alpha == 0]))
        self.assertTrue(np.all(result[alpha == 1] == 255))
        target = self.root / "renders" / "lossless.mkv"
        media.write_video([result], 24, 1, target, lossless=True)
        self.assertTrue(np.array_equal(next(media.iter_video(target))[0], result))

    def test_assemble_vfr_segments_preserves_rgb_gaps_and_full_duration(self):
        timestamps = [0, 0.04, 0.095, 0.13, 0.2, 0.24, 0.31, 0.39]
        frames = [np.full((24, 32, 3), index * 25, np.uint8) for index in range(8)]
        source = self.root / "input" / "source.mp4"
        media.write_video(frames, 24, 1, source, timestamps=timestamps, duration=0.45, lossless=True)
        info = media.probe_video(source)
        video = {"file": "source.mp4", "info": info}
        expected = [frame.copy() for frame in frames]
        parts = []
        for number, (start, end) in enumerate([(0.04, 0.2), (0.31, 0.45)]):
            selected = [i for i, time in enumerate(timestamps) if start <= time < end]
            changed = []
            for index in selected:
                frame = frames[index].copy()
                frame[4:10, 7:13] = [230, 15, 90]
                expected[index] = frame
                changed.append(frame)
            part_path = self.root / "renders" / f"part_{number}.mkv"
            media.write_video(changed, info["fps_num"], info["fps_den"], part_path, lossless=True,
                              timestamps=[timestamps[index] - start for index in selected], duration=end - start)
            parts.append({"segment_id": f"part_{number}", "start": start, "end": end, "path": str(part_path)})
        target = self.root / "renders" / "assembled.mp4"
        report = generation.assemble_video(video, parts, target)
        actual = list(media.iter_video(target))
        self.assertEqual(report["frames"], 8)
        self.assertAlmostEqual(report["duration"], 0.45, places=5)
        for index, (frame, time) in enumerate(actual):
            self.assertTrue(np.array_equal(frame, expected[index]))
            self.assertAlmostEqual(time, timestamps[index], delta=1 / 90000)
        self.assertAlmostEqual(media.probe_video(target)["duration"], 0.45, places=5)

    def test_original_audio_mux_does_not_reencode_lossless_rgb(self):
        video_only, source, target = [self.root / "renders" / name for name in ("video.mp4", "source.mp4", "final.mov")]
        frames = [np.full((24, 32, 3), index * 9, np.uint8) for index in range(12)]
        wave = (torch.sin(torch.arange(24000) / 48000 * 2 * torch.pi * 300) * 0.1)[None, None].repeat(1, 2, 1)
        media.write_video(frames, 24, 1, video_only, lossless=True)
        media.write_video(frames, 24, 1, source, audio={"sample_rate": 48000, "waveform": wave})
        generation.mux_and_save(video_only, source, target, 0.5)
        actual = list(media.iter_video(target))
        self.assertEqual(len(actual), len(frames))
        self.assertTrue(all(np.array_equal(image, original) for (image, _), original in zip(actual, frames)))
        audio = media.extract_audio(target, end=0.5)
        self.assertIsNotNone(audio)
        self.assertGreater(float(audio["waveform"].abs().mean()), 0.04)
        self.assertTrue(torch.equal(audio["waveform"], media.extract_audio(source, end=0.5)["waveform"]))
        with av.open(str(target)) as container:
            self.assertEqual(container.streams.video[0].start_time, 0)
            self.assertAlmostEqual(container.duration / 1e6, 0.5, places=5)
        with av.open(str(source)) as first, av.open(str(target)) as second:
            first_packets = [bytes(packet) for packet in first.demux(first.streams.audio[0]) if packet.size]
            second_packets = [bytes(packet) for packet in second.demux(second.streams.audio[0]) if packet.size]
        self.assertTrue(all(packet in first_packets for packet in second_packets))

    def aac_source(self, duration=1.0):
        video = self.root / "renders" / "video_only.mp4"
        source = self.root / "input" / "ffmpeg_audio.mp4"
        frames = [np.full((24, 32, 3), index * 3, np.uint8) for index in range(round(duration * 24))]
        media.write_video(frames, 24, 1, video, lossless=True)
        subprocess.run([media.ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
                        "-f", "lavfi", "-i", f"sine=frequency=220:sample_rate=48000:duration={duration}",
                        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", str(source)], check=True)
        return video, source, frames

    def test_full_original_aac_keeps_priming_packets_and_decoded_samples(self):
        video, source, frames = self.aac_source()
        target = self.root / "renders" / "original.mov"
        report = generation.mux_and_save(video, source, target, 1.0)
        self.assertEqual(report["audio_encoding"], "copy")
        with av.open(str(source)) as first, av.open(str(target)) as second:
            original = [packet for packet in first.demux(first.streams.audio[0]) if packet.size]
            copied = [packet for packet in second.demux(second.streams.audio[0]) if packet.size]
            self.assertLess(original[0].pts, 0)
            self.assertEqual([bytes(packet) for packet in original], [bytes(packet) for packet in copied])
        self.assertTrue(torch.equal(media.extract_audio(source, end=1.0)["waveform"],
                                    media.extract_audio(target, end=1.0)["waveform"]))
        for index, (image, time) in enumerate(media.iter_video(target)):
            self.assertTrue(np.array_equal(image, frames[index]))
            self.assertAlmostEqual(time, index / 24, delta=1 / 90000)

    def test_trial_audio_is_sample_aligned_without_changing_video_pts(self):
        _, source, _ = self.aac_source(duration=2.0)
        video = self.root / "renders" / "trial_video.mp4"
        frames = [np.full((24, 32, 3), index * 8, np.uint8) for index in range(12)]
        media.write_video(frames, 24, 1, video, lossless=True)
        target = self.root / "renders" / "trial.mov"
        report = generation.mux_and_save(video, source, target, 0.5, audio_start=0.37)
        self.assertEqual(report["audio_encoding"], "pcm_f32le")
        expected = media.extract_audio(source, start=0.37, end=0.87)["waveform"]
        actual = media.extract_audio(target, end=0.5)["waveform"]
        self.assertEqual(actual.shape[-1], 24000)
        self.assertTrue(torch.equal(expected, actual))
        for index, (image, time) in enumerate(media.iter_video(target)):
            self.assertTrue(np.array_equal(image, frames[index]))
            self.assertAlmostEqual(time, index / 24, delta=1 / 90000)

    def test_assemble_phone_rotation_outputs_display_dimensions_and_pixels(self):
        unrotated = self.root / "input" / "unrotated.mp4"
        source = self.root / "input" / "phone.mp4"
        frames = list(np.random.default_rng(65).integers(0, 256, (3, 24, 32, 3), dtype=np.uint8))
        media.write_video(frames, 24, 1, unrotated, lossless=True)
        subprocess.run([media.ffmpeg_path(), "-loglevel", "error", "-y", "-display_rotation:v:0", "90",
                        "-i", str(unrotated), "-c", "copy", str(source)], check=True, capture_output=True)
        info = media.probe_video(source)
        target = self.root / "renders" / "display.mp4"
        report = generation.assemble_video({"file": "phone.mp4", "info": info}, [], target)
        self.assertEqual(report["frames"], 3)
        output_info = media.probe_video(target)
        self.assertEqual((output_info["width"], output_info["height"], output_info["rotation"]), (24, 32, 0))
        for index, (image, time) in enumerate(media.iter_video(target)):
            self.assertTrue(np.array_equal(image, np.rot90(frames[index])))
            self.assertAlmostEqual(time, index / 24, places=5)

    def test_save_audio_policy_ignores_old_voice_conversion_cache(self):
        video = {"id": "video1", "file": "source.mp4", "info": {"width": 32, "height": 24, "duration": 1}}
        obsolete_audio = {"videos": {"video1": {"path": "missing-converted.wav", "report": {"mode": "replace"}}}}
        for mode in ("original", "mute", "replace", "reference-timbre"):
            with self.subTest(mode=mode):
                results = {"plan": {"state": {"videos": [video], "settings": {"audio_mode": mode}}}, "videos": {"video1": [{}]}, "processing_long_edge": 512}
                with patch.object(generation, "assemble_video", return_value={"start": 0, "duration": 1, "frames": 24}), \
                     patch.object(generation, "output_path", side_effect=lambda name: self.root / name), \
                     patch.object(generation, "mux_and_save", return_value={"audio_encoding": "none" if mode == "mute" else "copy"}) as mux:
                    written = generation.save_results(results, obsolete_audio, "policy", compatible=True)
                source = None if mode == "mute" else self.root / "input/source.mp4"
                self.assertEqual(mux.call_args.args[1], source)
                self.assertEqual(written[0]["report"]["audio_mode"], "mute" if mode == "mute" else "original")

    def test_muted_export_has_no_audio_stream_and_keeps_video_frames(self):
        source = self.root / "renders/source.mp4"
        frames = [np.full((24, 32, 3), index * 40, np.uint8) for index in range(3)]
        media.write_video(frames, 24, 1, source, lossless=True)
        target = self.root / "renders/mute.mov"
        report = generation.mux_and_save(source, None, target, 3 / 24)
        self.assertEqual(report["audio_encoding"], "none")
        with av.open(str(target)) as container:
            self.assertEqual(len(container.streams.audio), 0)
        for index, (frame, time) in enumerate(media.iter_video(target)):
            self.assertTrue(np.array_equal(frame, frames[index]))
            self.assertAlmostEqual(time, index / 24, delta=1 / 90000)

    def test_second_role_uses_original_timeline_after_midframe_vfr_boundary(self):
        timestamps = [0, 0.07, 0.29, 0.38, 0.53, 0.69, 0.89]
        frames = [np.full((24, 32, 3), 60 + index * 10, np.uint8) for index in range(7)]
        source = self.root / "input" / "source.mp4"
        media.write_video(frames, 24, 1, source, timestamps=timestamps, duration=1.1, lossless=True)
        Image.new("RGB", (32, 32), "white").save(self.root / "input" / "ref.png")
        roles = [{"id": "role1", "name": "role1", "image_ids": ["ref"], "primary_image_id": "ref"},
                 {"id": "role2", "name": "role2", "image_ids": ["ref"], "primary_image_id": "ref"}]
        track = {"timestamps": timestamps, "frames": 7, "width": 32, "height": 24, "roles": {}}
        for role_index, role in enumerate(roles):
            directory = self.root / "tracks" / role["id"]
            directory.mkdir()
            for index in range(7):
                alpha = np.zeros((24, 32), np.uint8)
                if role_index == 0:
                    alpha[:, :2] = 255
                else:
                    alpha[:, index * 4:index * 4 + 2] = 255
                Image.fromarray(alpha).save(directory / f"{index:08d}.png")
            track["roles"][role["id"]] = {"mask_path": str(directory)}
        video = {"id": "video1", "file": "source.mp4", "info": media.probe_video(source)}
        segment = {"id": "segment1", "start": 0.15, "end": 0.70}
        state = {"images": [{"id": "ref", "file": "ref.png"}]}

        class FakeVAE:
            color = (1, 0, 0)

            def decode(self, latent):
                count, height, width = FakeReference.dimensions
                return torch.tensor(self.color, dtype=torch.float32).expand(count, height, width, 3).clone()

        vae = FakeVAE()
        args = (video, segment, roles[0], state, track, "model", "clip", vae, "audio_vae", "patch", "noise", "sampler", "sigmas",
                32, "none", 1, "nearest", self.root / "renders", lambda *args: None)
        first = generation.render_part(*args)
        vae.color = (0, 1, 0)
        args = list(args)
        args[2] = roles[1]
        second = generation.render_part(*args, previous_role=first)
        rendered = next(media.iter_video(Path(second["path"])))[0]
        self.assertTrue(np.all(rendered[:, 8:10] == [0, 255, 0]))
        self.assertTrue(np.all(rendered[:, 4:6] == frames[2][:, 4:6]))


if __name__ == "__main__":
    unittest.main()
