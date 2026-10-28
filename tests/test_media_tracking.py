import importlib
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

PLUGIN = Path(__file__).resolve().parents[1]
PACKAGE = "csd_media_tests"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(PLUGIN)]
sys.modules[PACKAGE] = package
if "folder_paths" not in sys.modules:
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = str(PLUGIN / "test_models")
    sys.modules["folder_paths"] = folder_paths
media = importlib.import_module(PACKAGE + ".media")
segmentation = importlib.import_module(PACKAGE + ".segmentation")
tracking = importlib.import_module(PACKAGE + ".tracking")


class MediaTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_rational_fps_lossless_streaming_and_original_indices(self):
        video = self.root / "rational.mp4"
        frames = [np.full((24, 32, 3), i * 2, np.uint8) for i in range(90)]
        media.write_video(iter(frames), 30000, 1001, video, lossless=True)
        info = media.probe_video(video)
        self.assertEqual((info["fps_num"], info["fps_den"], info["frames"]), (30000, 1001, 90))
        original = list(media.iter_video(video))
        self.assertEqual(len(original), 90)
        self.assertTrue(np.array_equal(original[27][0], frames[27]))
        sampled, times = media.read_video(video, start=0.7, end=1.5, fps=24)
        indices = media.nearest_frame_indices([time for _, time in original], times)
        self.assertEqual(len(times), 20)
        for frame, source_index in zip(sampled, indices):
            self.assertTrue(torch.equal(frame, torch.from_numpy(frames[source_index]).float() / 255))
        self.assertAlmostEqual(info["duration"], 90 * 1001 / 30000, places=5)

    def test_audio_mux_and_trim(self):
        video = self.root / "sound.mp4"
        rate = 48000
        time = torch.arange(rate * 2) / rate
        waveform = (torch.sin(time * 440 * 2 * torch.pi) * 0.2)[None, None].repeat(1, 2, 1)
        media.write_video((np.zeros((24, 32, 3), np.uint8) for _ in range(48)), 24, 1, video,
                          audio={"sample_rate": rate, "waveform": waveform})
        self.assertTrue(media.probe_video(video)["has_audio"])
        full = media.extract_audio(video, end=1.75)
        for start, end in ((0.25, 0.75), (1.25, 1.75)):
            with self.subTest(start=start):
                clipped = media.extract_audio(video, start=start, end=end)
                self.assertEqual(clipped["waveform"].shape, (1, 2, rate // 2))
                self.assertGreater(float(clipped["waveform"].abs().mean()), 0.08)
                self.assertTrue(torch.equal(clipped["waveform"], full["waveform"][..., round(start * rate):round(end * rate)]))

    def test_phone_display_rotation_normalizes_probe_decode_and_resize(self):
        source = self.root / "unrotated.mp4"
        frames = list(np.random.default_rng(83).integers(0, 256, (3, 24, 32, 3), dtype=np.uint8))
        media.write_video(frames, 24, 1, source, lossless=True)
        for rotation in (90, 180, 270):
            with self.subTest(rotation=rotation):
                rotated = self.root / f"phone_{rotation}.mp4"
                subprocess.run([media.ffmpeg_path(), "-loglevel", "error", "-y", "-display_rotation:v:0", str(rotation),
                                "-i", str(source), "-c", "copy", str(rotated)], check=True, capture_output=True)
                info = media.probe_video(rotated)
                dimensions = (24, 32) if rotation in (90, 270) else (32, 24)
                self.assertEqual((info["width"], info["height"]), dimensions)
                self.assertEqual((info["coded_width"], info["coded_height"], info["rotation"]), (32, 24, rotation))
                self.assertEqual(info["video_codec"], "h264")
                decoded = list(media.iter_video(rotated))
                self.assertEqual(len(decoded), len(frames))
                for index, (image, time) in enumerate(decoded):
                    self.assertTrue(np.array_equal(image, np.rot90(frames[index], rotation // 90)))
                    self.assertAlmostEqual(time, index / 24, places=5)
                resized, _ = media.read_video(rotated, start=1 / 24, size=(12, 8))
                self.assertEqual(resized.shape, (2, 8, 12, 3))

    def test_variable_timestamps_and_final_frame_duration(self):
        video = self.root / "vfr.mp4"
        times = [0, 0.04, 0.095, 0.13, 0.2, 0.24]
        duration = 0.3
        frames = [np.full((24, 32, 3), index * 30, np.uint8) for index in range(len(times))]
        audio = {"sample_rate": 48000, "waveform": torch.full((1, 2, 14400), 0.1)}
        media.write_video(frames, 24, 1, video, timestamps=times, duration=duration, lossless=True, audio=audio)
        decoded = list(media.iter_video(video))
        self.assertEqual(len(decoded), len(times))
        for (_, actual), expected in zip(decoded, times):
            self.assertAlmostEqual(actual, expected, delta=1 / 90000)
        self.assertAlmostEqual(media.probe_video(video)["duration"], duration, places=5)
        self.assertEqual(media.extract_audio(video, end=duration)["waveform"].shape[-1], 14400)

    def test_scene_first_duration_splits_cover_every_frame(self):
        video = self.root / "cuts.mp4"
        frames = [np.full((24, 32, 3), color, np.uint8) for color in ([0] * 12 + [255] * 12 + [[255, 0, 0]] * 24)]
        media.write_video(frames, 24, 1, video, lossless=True)
        segments = segmentation.detect_segments(video, max_seconds=0.3, threshold=0.35)
        self.assertEqual(segments[0]["start"], 0)
        self.assertEqual(segments[-1]["end"], 2)
        for first, second in zip(segments, segments[1:]):
            self.assertEqual(first["end"], second["start"])
        self.assertTrue(all(0 < s["end"] - s["start"] <= 0.3 + 1e-9 for s in segments))
        ends = [s["end"] for s in segments]
        self.assertIn(0.5, ends)
        self.assertIn(1, ends)
        self.assertEqual({s["scene_id"] for s in segments if s["start"] < 0.5}, {"scene_1"})
        self.assertEqual({s["scene_id"] for s in segments if 0.5 <= s["start"] < 1}, {"scene_2"})


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.patcher = patch.object(tracking.paths, "cache_dir", return_value=self.root)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.temporary.cleanup()

    def test_padding_feather_has_finite_support(self):
        mask = np.zeros((32, 32), np.uint8)
        mask[12:20, 12:20] = 1
        alpha = tracking.editing_alpha(mask, padding=2, feather=3)
        self.assertEqual(alpha[15, 15], 255)
        self.assertEqual(alpha[0, 0], 0)
        self.assertGreater(alpha[9, 15], 0)
        self.assertLess(alpha[9, 15], 255)

    def test_positive_and_negative_brush_edits(self):
        strokes = [{"points": [{"x": 0.2, "y": 0.2}, {"x": 0.8, "y": 0.8}], "radius": 0.08, "label": 1},
                   {"points": [{"x": 0.5, "y": 0.5}], "radius": 0.08, "label": 0}]
        mask = tracking.paint_strokes(np.zeros((32, 32), np.uint8), strokes)
        self.assertEqual(mask[6, 6], 1)
        self.assertEqual(mask[16, 16], 0)
        self.assertEqual(mask[0, 31], 0)
        alpha = tracking.editing_alpha(mask, padding=16, feather=4, strokes=strokes)
        self.assertEqual(alpha[16, 16], 0)

    def test_read_masks_uses_source_index_and_rejects_escape(self):
        role_dir = self.root / "role"
        role_dir.mkdir()
        for index in range(3):
            Image.fromarray(np.full((8, 8), index * 100, np.uint8)).save(role_dir / f"{index:08d}.png")
        track = {"roles": {"person": {"mask_path": str(role_dir)}}, "frames": 3, "width": 8, "height": 8}
        masks = tracking.read_masks(track, "person", [2, 0, 2])
        self.assertTrue(torch.equal(masks[0], masks[2]))
        self.assertEqual(float(masks[1].sum()), 0)
        with self.assertRaises(ValueError):
            tracking.read_masks(track, "person", [-1])
        track["roles"]["person"]["mask_path"] = str(self.root.parent)
        with self.assertRaises(ValueError):
            tracking.read_masks(track, "person", [0])

    def test_sam2_window_adapter_covers_before_and_after_midvideo_anchor(self):
        frame_dir, role_dir = self.root / "frames", self.root / "role"
        frame_dir.mkdir()
        role_dir.mkdir()
        for index in range(80):
            Image.fromarray(np.zeros((8, 16, 3), np.uint8)).save(frame_dir / f"{index:08d}.jpg")

        class FakePredictor:
            carries = 0

            def init_state(self, path, **kwargs):
                return {"frames": len(list(Path(path).glob("*.jpg")))}

            def add_new_mask(self, state, **kwargs):
                self.carries += 1

            def add_new_points_or_box(self, state, **kwargs):
                return kwargs["frame_idx"], [1], torch.ones((1, 1, 8, 16))

            def propagate_in_video(self, state, start_frame_idx, reverse):
                indices = range(start_frame_idx, -1, -1) if reverse else range(start_frame_idx, state["frames"])
                for index in indices:
                    yield index, [1], torch.ones((1, 1, 8, 16))

            def reset_state(self, state):
                state.clear()

        predictor = FakePredictor()
        anchors = {31: {"time": 31 / 30, "points": [{"x": 0.5, "y": 0.5, "label": 1}]}}
        args = (predictor, frame_dir, self.root / "window", role_dir, 80, anchors, 31, 16, 8, 0, 0)
        tracking._track_direction(*args, False, None, 0, 0.5)
        tracking._track_direction(*args, True, None, 0.5, 0.5)
        self.assertEqual(len(list(role_dir.glob("*.png"))), 80)
        self.assertGreaterEqual(predictor.carries, 1)


if __name__ == "__main__":
    unittest.main()
