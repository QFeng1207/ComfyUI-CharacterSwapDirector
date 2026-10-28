import importlib
import copy
import json
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import test_generation as legacy

with patch.dict(legacy.sys.modules, legacy.stubs):
    full = importlib.import_module(legacy.PACKAGE + ".fullframe_generation")
    reference_plan = importlib.import_module(legacy.PACKAGE + ".reference_plan")

class GuideOnly:
    @classmethod
    def execute(cls, *args, **kwargs):
        return types.SimpleNamespace(result=(args[0],))

full.MiniMaxH3AddGuide = GuideOnly


class ReferenceGenerationTests(unittest.TestCase):
    def setUp(self):
        legacy.GenerationTests.setUp(self)
        self.full_patches = [patch.object(full, "cache_dir", side_effect=lambda key: self.root / key),
                             patch.object(full, "input_path", side_effect=lambda value: self.root / "input" / value),
                             patch.object(full, "_cached_file", side_effect=legacy.generation._cached_file)]
        for item in self.full_patches:
            item.start()
        frames = [np.full((32, 64, 3), 15 + i, np.uint8) for i in range(30)]
        legacy.media.write_video(frames, 30, 1, self.root / "input/source.mp4", lossless=True)
        Image.new("RGB", (64, 64), "red").save(self.root / "input/person.png")
        Image.new("RGB", (64, 32), "blue").save(self.root / "input/frame.png")
        self.video = {"id": "v", "file": "source.mp4", "info": legacy.media.probe_video(self.root / "input/source.mp4"),
            "bindings": [{"role_id": "r", "anchors": [{"time": 0, "box": [.2, .1, .8, .9]}]}],
            "segments": [{"id": "s", "start": 0., "end": 1., "scene_id": "scene_1", "role_ids": [], "enabled": True}]}
        self.state = {"version": 1, "videos": [self.video], "images": [{"id": "i", "file": "person.png"}, {"id": "f", "file": "frame.png"}],
                      "roles": [{"id": "r", "name": "Person", "image_ids": ["i"], "primary_image_id": "i"}],
                      "frame_references": [], "settings": {"person_mode": "multi", "audio_mode": "original", "trial": {}, "revision": {}, "candidate": {}}}

    def tearDown(self):
        for item in reversed(self.full_patches):
            item.stop()
        legacy.GenerationTests.tearDown(self)

    def run_full(self):
        vae = types.SimpleNamespace(decode=lambda sample: torch.ones((124, 32, 64, 3)) * .75)
        return full.render_batch({"state": self.state}, None, None, vae, None, None, None, [1, 0], long_edge=256)

    def add_second_role(self):
        Image.new("RGB", (64, 64), (0, 255, 0)).save(self.root / "input/second.png")
        self.state["images"].append({"id": "i2", "file": "second.png"})
        self.state["roles"][0]["description"] = "replacement with red hair and teal robes"
        self.state["roles"].append({"id": "r2", "name": "Other person", "image_ids": ["i2"],
                                    "primary_image_id": "i2", "description": "replacement with green hair and a black coat"})
        self.video["bindings"][0]["description"] = "source performer wearing a red-and-white uniform"
        self.video["bindings"].append({"role_id": "r2", "description": "source performer wearing a white-and-gold uniform",
                                       "anchors": [{"time": .6, "box": [.3, .2, .7, 1.]}]})

    def two_source_shots(self):
        source = [np.full((32, 64, 3), [120 + index, 20, 10], np.uint8) for index in range(15)]
        source += [np.full((32, 64, 3), [10, 20, 120 + index], np.uint8) for index in range(15)]
        legacy.media.write_video(source, 30, 1, self.root / "input/source.mp4", lossless=True)
        self.video["shots"] = [{"id": "actual_cut_1", "start": 0., "end": .5},
                                {"id": "actual_cut_2", "start": .5, "end": 1.}]
        self.video["bindings"][0]["anchors"] = [{"time": .1, "box": [.2, .1, .8, .9]}]

    def manual_halves(self):
        self.video["segments"] = [{"id": "first_half", "start": 0., "end": .5, "enabled": True, "role_ids": [], "scene_id": "user_scene_a"},
                                    {"id": "second_half", "start": .5, "end": 1., "enabled": True, "role_ids": [], "scene_id": "user_scene_b"}]

    def assemble(self, result, filename="assembled.mp4"):
        path = self.root / "renders" / filename
        legacy.generation.assemble_video(self.video, result["videos"]["v"], path, base_path=result["base_videos"].get("v"))
        return list(legacy.media.iter_video(path))

    def test_initial_generation_needs_no_guide_and_overwrites_entire_frame(self):
        with patch.object(full.MiniMaxH3AddGuide, "execute") as guide:
            result = self.run_full()
        guide.assert_not_called()
        part = result["videos"]["v"][0]
        frames = list(legacy.media.iter_video(part["path"]))
        self.assertEqual(len(frames), 30)
        for i, (frame, time) in enumerate(frames):
            self.assertTrue(np.all(frame == 191))
            self.assertEqual(frame.shape, (32, 64, 3))
            self.assertAlmostEqual(time, i / 30, delta=1 / 90000)
        self.assertNotIn("tracks", result)
        self.assertEqual(result["backend"], "minimax_h3_reference")

    def test_scene_reference_does_not_anchor_original_person_guide_does(self):
        self.state["frame_references"] = [{"id": "scene", "image_id": "f", "video_id": "v", "time": .2, "kind": "scene"},
                                          {"id": "guide", "image_id": "f", "video_id": "v", "time": .5, "kind": "guide"}]
        with patch.object(full.MiniMaxH3AddGuide, "execute", wraps=legacy.FakeNode.execute) as guide:
            result = self.run_full()
        self.assertEqual(guide.call_count, 1)
        self.assertEqual(guide.call_args.args[2], 64)
        self.assertEqual(result["videos"]["v"][0]["reference_ids"], ["scene", "guide"])

    def test_multiple_people_generated_together_with_distinct_reference_labels(self):
        other = {"id": "r2", "image_ids": ["i"], "primary_image_id": "i", "description": "second performer"}
        self.state["roles"].append(other)
        self.video["bindings"].append({"role_id": "r2", "anchors": [{"time": 0, "box": [.8, .1, 1, .9]}]})
        with patch.object(full.SamplerCustomAdvanced, "execute", wraps=legacy.FakeSampler.execute) as sample:
            result = self.run_full()
        self.assertEqual(sample.call_count, 1)
        self.assertEqual(result["videos"]["v"][0]["roles"], ["r", "r2"])
        self.assertIn("<Picture 2>", result["videos"]["v"][0]["prompt"])

    def test_selected_revision_keeps_previous_good_frames_and_source_pts(self):
        source_frames = list(legacy.media.iter_video(self.root / "input/source.mp4"))
        base = self.root / "renders/base.mp4"
        expected = [np.full_like(frame, 87) for frame, _ in source_frames]
        legacy.media.write_video(expected, 30, 1, base, lossless=True)
        part = full.write_fullframe_part(self.video, {"id": "changed", "start": .2, "end": .6}, torch.ones((124, 32, 64, 3)), .2, self.root / "renders", "nearest", [self.state["roles"][0]])
        target = self.root / "renders/revised.mp4"
        legacy.generation.assemble_video(self.video, [part], target, base_path=base)
        output = list(legacy.media.iter_video(target))
        self.assertEqual(len(output), len(source_frames))
        for (frame, time), (_, source_time) in zip(output, source_frames):
            self.assertAlmostEqual(time, source_time, delta=1 / 90000)
            self.assertTrue(np.all(frame == (255 if .2 - 1e-9 <= time < .6 - 1e-9 else 87)))

    def test_over_limit_references_fail_instead_of_silently_dropping_images(self):
        self.state["frame_references"] = [{"id": f"frame{x}", "image_id": "f", "video_id": "v", "time": .2, "kind": "scene"} for x in range(9)]
        with self.assertRaisesRegex(ValueError, "9"):
            full.references_and_prompt(self.video, self.video["segments"][0], self.state["roles"], self.state, 0, 1, 124, "auto")

    def test_candidate_export_keeps_original_time_and_persistent_chinese_version(self):
        self.video["segments"][0].update(start=.4, end=.8)
        self.state["settings"]["candidate"] = {"enabled": True, "video_id": "v", "segment_id": "s", "time": .55}
        result = self.run_full()
        output = self.root / "output"
        output.mkdir()
        (self.root / "previews").mkdir()
        with patch.object(legacy.generation, "output_path", side_effect=lambda name: output / name), \
             patch.object(reference_plan, "output_path", side_effect=lambda name: output / name):
            saved = legacy.generation.save_results(result, None, "舞蹈人物替换")[0]
            report = saved["report"]
            self.assertTrue(saved["name"].startswith("舞蹈人物替换_"))
            self.assertRegex(saved["result_id"], r"^h3_[a-f0-9]{32}$")
            self.assertNotEqual(saved["result_id"], Path(saved["path"]).stem)
            self.assertEqual(reference_plan.saved_result(saved["result_id"])[0], Path(saved["path"]))
            self.assertTrue(Path(saved["preview_path"]).is_file())
            self.assertEqual(Path(saved["preview_path"]).parent, output)
            self.assertTrue(report["candidate"])
            self.assertEqual(report["candidate_time"], .55)
            self.assertAlmostEqual(report["start"], .4)
            expected = next(legacy.media.iter_video(saved["path"], .55 - report["start"]))[0]
            actual = np.array(Image.open(output / report["candidate_file"]))
            self.assertTrue(np.array_equal(actual, expected))

    def test_three_roles_keep_their_own_source_descriptions_coordinates_and_pictures(self):
        self.state["roles"] = []
        self.video["bindings"] = []
        for number, description in enumerate(("left red coat", "middle gray shirt", "right green dress"), 1):
            image_id = f"person{number}"
            self.state["images"].append({"id": image_id, "file": "person.png"})
            self.state["roles"].append({"id": f"r{number}", "image_ids": [image_id], "primary_image_id": image_id})
            self.video["bindings"].append({"role_id": f"r{number}", "description": description,
                "anchors": [{"time": 0, "points": [{"x": number / 4, "y": .5}]}]})
        refs, prompt, _ = full.references_and_prompt(self.video, self.video["segments"][0], self.state["roles"], self.state, 0, 1, 124, "auto")
        self.assertEqual(len(refs), 3)
        for number, binding in enumerate(self.video["bindings"], 1):
            self.assertIn(f"<Subject {number}> is the replacement person", prompt)
            self.assertIn(f"from <Picture {number}>", prompt)
            self.assertIn(binding["description"], prompt)
            self.assertIn(f"x={number / 4:.2f}", prompt)

    def test_one_user_segment_generates_alternating_shots_with_only_their_own_reference(self):
        self.add_second_role()
        self.two_source_shots()
        original_segments = copy.deepcopy(self.video["segments"])
        with patch.object(full.MiniMaxH3ReferenceToVideo, "execute", wraps=legacy.FakeReference.execute) as reference, \
             patch.object(full.SamplerCustomAdvanced, "execute", wraps=legacy.FakeSampler.execute) as sample, \
             patch.object(full.MiniMaxH3AddGuide, "execute", wraps=GuideOnly.execute) as guide:
            result = self.run_full()
        self.assertEqual(reference.call_count, 2)
        self.assertEqual(sample.call_count, 2)
        guide.assert_not_called()
        for call, expected, excluded in zip(reference.call_args_list, self.state["roles"], reversed(self.state["roles"])):
            inputs = call.kwargs["ref_images"]
            self.assertEqual(len(inputs), 1)
            color = torch.tensor([1., 0., 0.] if expected["id"] == "r" else [0., 1., 0.])
            self.assertTrue(torch.allclose(inputs["ref_image_0"].mean(dim=(0, 1, 2)), color))
            self.assertIn(expected["description"], call.args[1])
            self.assertNotIn(excluded["description"], call.args[1])
            self.assertIn("source action in chronological order", call.args[1])
        parts = result["videos"]["v"]
        self.assertEqual([part["roles"] for part in parts], [["r"], ["r2"]])
        self.assertEqual([(part["start"], part["end"]) for part in parts], [(0., .5), (.5, 1.)])
        self.assertEqual([part["parent_segment_id"] for part in parts], ["s", "s"])
        self.assertEqual(len({part["segment_id"] for part in parts}), 2)
        self.assertEqual(self.video["segments"], original_segments)
        output = self.assemble(result)
        self.assertEqual(len(output), 30)
        for index, (frame, time) in enumerate(output):
            self.assertTrue(np.all(frame == 191))
            self.assertAlmostEqual(time, index / 30, delta=1 / 90000)

    def test_single_shot_keeps_original_segment_id_and_records_parent_id(self):
        part = self.run_full()["videos"]["v"][0]
        self.assertEqual(part["segment_id"], "s")
        self.assertEqual(part["parent_segment_id"], "s")

    def test_unlocated_shot_does_not_sample_and_preserves_original_rgb_and_pts(self):
        self.two_source_shots()
        original = list(legacy.media.iter_video(self.root / "input/source.mp4"))
        with patch.object(full.MiniMaxH3ReferenceToVideo, "execute", wraps=legacy.FakeReference.execute) as reference, \
             patch.object(full.SamplerCustomAdvanced, "execute", wraps=legacy.FakeSampler.execute) as sample:
            result = self.run_full()
        self.assertEqual(reference.call_count, 1)
        self.assertEqual(sample.call_count, 1)
        unchanged = result["videos"]["v"][1]
        self.assertEqual(unchanged["roles"], [])
        self.assertEqual(unchanged["sampling_steps"], 0)
        self.assertTrue(any("沿用原片" in warning for warning in unchanged["warnings"]))
        output = self.assemble(result)
        self.assertEqual(len(output), len(original))
        for (frame, time), (source, source_time) in zip(output, original):
            self.assertAlmostEqual(time, source_time, delta=1 / 90000)
            self.assertTrue(np.array_equal(frame, source) if time >= .5 else np.all(frame == 191))

    def test_unlocated_revision_shot_preserves_saved_base_rgb_instead_of_original(self):
        self.two_source_shots()
        base = self.root / "renders/base.mp4"
        base_frames = [np.full((32, 64, 3), 60 + index, np.uint8) for index in range(30)]
        legacy.media.write_video(base_frames, 30, 1, base, lossless=True)
        self.state["settings"]["revision"] = {"enabled": True, "video_id": "v", "base_result_id": "base", "scope": "all"}
        with patch.object(full, "revision_base", return_value=(base, {})), \
             patch.object(full.SamplerCustomAdvanced, "execute", wraps=legacy.FakeSampler.execute) as sample:
            result = self.run_full()
        self.assertEqual(sample.call_count, 1)
        self.assertTrue(any("沿用底稿" in warning for warning in result["videos"]["v"][1]["warnings"]))
        output = self.assemble(result)
        for index, (frame, time) in enumerate(output):
            self.assertTrue(np.array_equal(frame, base_frames[index]) if time >= .5 else np.all(frame == 191))
            self.assertAlmostEqual(time, index / 30, delta=1 / 90000)

    def test_manual_empty_role_selection_never_calls_h3_and_keeps_all_frames(self):
        self.video["segments"][0].update(role_mode="manual", role_ids=[])
        original = list(legacy.media.iter_video(self.root / "input/source.mp4"))
        with patch.object(full.MiniMaxH3ReferenceToVideo, "execute") as reference, \
             patch.object(full.SamplerCustomAdvanced, "execute") as sample, \
             patch.object(full, "read_video") as read:
            result = self.run_full()
        reference.assert_not_called()
        sample.assert_not_called()
        read.assert_not_called()
        self.assertEqual(result["videos"]["v"][0]["generation_frames"], 0)
        for (frame, time), (source, source_time) in zip(self.assemble(result), original):
            self.assertTrue(np.array_equal(frame, source))
            self.assertAlmostEqual(time, source_time, delta=1 / 90000)

    def test_identical_manual_split_samples_once_and_preserves_both_user_ranges(self):
        self.manual_halves()
        self.video["shots"] = [{"id": "one_actual_shot", "start": 0., "end": 1.}]
        self.video["bindings"][0]["anchors"].append({"time": .701778, "box": [.1, .2, .7, .95]})
        with patch.object(full.MiniMaxH3AddGuide, "execute", wraps=GuideOnly.execute) as guide, \
             patch.object(full.SamplerCustomAdvanced, "execute", wraps=legacy.FakeSampler.execute) as sampler:
            result = self.run_full()
        guide.assert_not_called()
        self.assertEqual(sampler.call_count, 1)
        second = result["videos"]["v"][1]
        self.assertEqual(second["generation_start"], 0.)
        self.assertEqual(second["generation_end"], 1.)
        self.assertFalse(second["automatic_prefix"])
        self.assertIn("at 3.720633s:", second["prompt"])
        self.assertNotIn("at 0.201778s:", second["prompt"])
        self.assertEqual([part["parent_segment_id"] for part in result["videos"]["v"]], ["first_half", "second_half"])
        self.assertEqual([(part["start"], part["end"]) for part in result["videos"]["v"]], [(0., .5), (.5, 1.)])
        self.assertEqual(result["videos"]["v"][0]["time_map"], second["time_map"])

    def test_same_shot_changed_manual_cast_does_not_carry_previous_person_as_guide(self):
        self.add_second_role()
        self.manual_halves()
        self.video["shots"] = [{"id": "one_actual_shot", "start": 0., "end": 1.}]
        self.video["segments"][0].update(role_mode="manual", role_ids=["r"])
        self.video["segments"][1].update(role_mode="manual", role_ids=["r2"])
        with patch.object(full.MiniMaxH3AddGuide, "execute", wraps=GuideOnly.execute) as guide:
            result = self.run_full()
        guide.assert_not_called()
        self.assertEqual(result["videos"]["v"][1]["generation_start"], .5)

    def test_hard_cut_never_adds_prefix_even_with_same_role_and_user_scene_id(self):
        self.two_source_shots()
        self.manual_halves()
        self.video["segments"][1]["scene_id"] = self.video["segments"][0]["scene_id"]
        self.video["bindings"][0]["anchors"].append({"time": .6, "box": [.2, .1, .8, .9]})
        with patch.object(full.MiniMaxH3AddGuide, "execute", wraps=GuideOnly.execute) as guide:
            result = self.run_full()
        guide.assert_not_called()
        self.assertEqual([part["generation_start"] for part in result["videos"]["v"]], [0., .5])

    def test_corrected_guide_belongs_only_to_its_source_shot_and_uses_local_h3_frame(self):
        self.add_second_role()
        self.two_source_shots()
        self.state["frame_references"] = [{"id": "later_fix", "image_id": "f", "video_id": "v", "time": .7, "kind": "guide"}]
        with patch.object(full.MiniMaxH3ReferenceToVideo, "execute", wraps=legacy.FakeReference.execute) as reference, \
             patch.object(full.MiniMaxH3AddGuide, "execute", wraps=GuideOnly.execute) as guide:
            result = self.run_full()
        self.assertEqual([len(call.kwargs["ref_images"]) for call in reference.call_args_list], [1, 2])
        self.assertEqual(guide.call_count, 1)
        self.assertEqual(guide.call_args.args[2], 53)
        self.assertEqual([part["reference_ids"] for part in result["videos"]["v"]], [[], ["later_fix"]])

    def test_prompt_maps_six_real_shots_with_complete_locations_and_source_only_descriptions(self):
        self.add_second_role()
        boundaries = [0., .2, .3, .6, .7, .8, 1.]
        self.video["shots"] = [{"id": f"cut_{number}", "start": start, "end": end}
                                for number, (start, end) in enumerate(zip(boundaries, boundaries[1:]), 1)]
        self.video["bindings"][0]["anchors"] = [{"time": time, "box": [.2, .1, .8, .9]} for time in (.01, .1, .35, .5, .65, .75)]
        self.video["bindings"][1]["anchors"] = [{"time": time, "box": [.3, .2, .7, 1.]} for time in (.25, .85, .95)]
        self.video["bindings"][0]["anchors"][1].update(
            points=[{"x": .4, "y": .5, "label": 1}, {"x": .9, "y": .9, "label": 0}],
            strokes=[{"radius": .02, "label": 1, "points": [{"x": .3, "y": .4}]},
                     {"radius": .01, "label": 0, "points": [{"x": .8, "y": .8}]}])
        _, prompt, _ = full.references_and_prompt(self.video, self.video["segments"][0], self.state["roles"], self.state, 0., 1., 124, "auto")
        sections = prompt.split("\n\n")
        self.assertEqual([section.split(":\n", 1)[0] for section in sections],
                         ["subject_definitions", "summary", "retention_analysis", "detailed_description", "overall_soundscape", "non_diegetic_music"])
        details = sections[3].split(":\n", 1)[1].splitlines()
        self.assertEqual(len(details), 6)
        for number, (start, end) in enumerate(zip(boundaries, boundaries[1:]), 1):
            self.assertIn(f"[Shot {number}] ({start * 124 / 24:.6f}s–{end * 124 / 24:.6f}s)", details[number - 1])
        for number, subject in enumerate((1, 2, 1, 1, 1, 2), 1):
            line = details[number - 1]
            self.assertIn(f"<Subject {subject}> replaces only its matched source performer", line)
            self.assertIn(f"<Subject {3 - subject}> is not assigned to this shot", line)
        self.assertIn("left=0.200000, top=0.100000, right=0.800000, bottom=0.900000", details[0])
        self.assertIn("at 0.051667s: user anchor", details[0])
        self.assertIn("at 0.516667s: user anchor", details[0])
        self.assertIn("include point x=0.400000, y=0.500000", details[0])
        self.assertIn("exclude point x=0.900000, y=0.900000", details[0])
        self.assertIn("include brush radius=0.020000", details[0])
        self.assertIn("exclude brush radius=0.010000", details[0])
        for binding in self.video["bindings"]:
            self.assertIn(binding["description"], sections[0])
            self.assertNotIn(binding["description"], sections[2])
            self.assertNotIn(binding["description"], sections[3])
        self.assertIn("INPUT-only locator traits", sections[0])
        self.assertIn("Discard this performer's original appearance and costume", sections[0])
        self.assertIn("Retain the source crop", details[0])
        self.assertIn("never a new full-body composition", details[0])
        self.assertIn("from the first frame onward", prompt)
        self.assertNotIn("Padding only", prompt)
        self.assertNotIn("<Video 1>: fully_preserved", prompt)
        self.assertIn("identity, face, hair, body and costume: removed", prompt)

    def test_uniform_time_fit_uses_whole_source_motion_and_exact_endpoints(self):
        source, times = legacy.media.read_video(self.root / "input/source.mp4")
        fitted, clock = full.fit_source_clip(source, times, 124)
        self.assertTrue(torch.equal(fitted[0], source[0]))
        self.assertTrue(torch.equal(fitted[-1], source[-1]))
        self.assertEqual(len(torch.unique(fitted[:, 0, 0, 0])), len(source))
        self.assertAlmostEqual(full.frame_position(times[0], clock), 0.)
        self.assertAlmostEqual(full.frame_position(times[-1], clock), 123.)
        self.assertLess(int(torch.count_nonzero(fitted[:, 0, 0, 0] == fitted[-1, 0, 0, 0])), 5)
        part = full.write_fullframe_part(self.video, self.video["segments"][0], fitted, 0., self.root / "renders", "nearest", self.state["roles"], time_map=clock)
        for (actual, time), (expected, source_time) in zip(legacy.media.iter_video(part["path"]), legacy.media.iter_video(self.root / "input/source.mp4")):
            self.assertEqual(time, source_time)
            self.assertTrue(np.array_equal(actual, expected))

    def test_single_role_prompt_sends_source_traits_only_as_locator(self):
        self.video["bindings"][0]["description"] = "old face, purple hair and original orange uniform"
        _, prompt, _ = full.references_and_prompt(self.video, self.video["segments"][0], self.state["roles"], self.state, 0., 1., 124, "auto")
        self.assertIn(self.video["bindings"][0]["description"], prompt)
        self.assertIn("INPUT-only locator traits, never OUTPUT appearance instructions", prompt)
        self.assertIn("from the FIRST frame", prompt)
        self.assertIn("exactly one replacement in its place", prompt)
        self.assertIn("left=0.200000, top=0.100000, right=0.800000, bottom=0.900000", prompt)

    def test_explicit_partial_performer_sends_reference_and_visibility_without_invented_box(self):
        self.add_second_role()
        self.two_source_shots()
        self.video["shot_overrides"] = [{"start": .5, "end": 1., "prompt": "Turn, then lift both feet toward the camera and exit.",
            "roles": [{"role_id": "r", "presence": "visible", "body_part": "hand_arm", "description": "left foreground holding the revolver"}]}]
        second = {"start": .5, "end": 1., "role_mode": "auto", "role_ids": []}
        refs, prompt, guides = full.references_and_prompt(self.video, second, self.state["roles"], self.state, .5, 1., 124, "auto")
        self.assertEqual(len(refs), 2)
        self.assertIn("only its visible hand and forearm", prompt)
        self.assertIn("left foreground holding the revolver", prompt)
        self.assertIn("Turn, then lift both feet", prompt)
        self.assertIn("source performer wearing a red-and-white uniform", prompt)
        self.assertNotIn("no performer is selected", prompt.lower())
        self.assertNotIn("at 0.000000s: user anchor", prompt)
        self.assertEqual(guides, [])

    def test_explicit_guides_take_priority_over_automatic_character_views(self):
        sheet = Image.new("RGB", (300, 100), "white")
        for left in (10, 110, 210):
            sheet.paste("red", (left, 10, left + 50, 90))
        sheet.save(self.root / "input/person.png")
        clock = {"source_first": 0., "scale": 123 / 24, "frames": 124}
        refs, prompt, guides = full.references_and_prompt(self.video, self.video["segments"][0], self.state["roles"], self.state,
            0., 1., 124, "auto", time_map=clock)
        self.assertEqual(len(refs), 3)
        self.assertEqual(guides, [])
        self.assertIn("additional views of the SAME person: <Picture 2>, <Picture 3>", prompt)
        self.assertIn("never use the ending action as the opening", prompt)
        self.state["frame_references"] = [{"id": f"fix{index}", "image_id": "f", "video_id": "v", "time": index / 10, "kind": "guide"} for index in range(8)]
        refs, prompt, guides = full.references_and_prompt(self.video, self.video["segments"][0], self.state["roles"], self.state,
            0., 1., 124, "auto", time_map=clock)
        self.assertEqual(len(refs), 9)
        self.assertEqual(len(guides), 8)
        self.assertNotIn("UNEDITED source action/composition checkpoint", prompt)

    def test_mixed_partial_cast_keeps_explicit_pictures_and_source_traits_out_of_target_definitions(self):
        self.add_second_role()
        self.two_source_shots()
        sheet = Image.new("RGB", (300, 100), "white")
        for left in (10, 110, 210):
            sheet.paste("red", (left, 10, left + 50, 90))
        for name in ("person.png", "second.png"):
            sheet.save(self.root / "input" / name)
        self.state["roles"][0]["image_ids"].append("f")
        self.video["shot_overrides"] = [{"start": .5, "end": 1., "roles": [
            {"role_id": "r", "presence": "visible", "body_part": "hand_arm", "description": "left foreground hand only"}]}]
        refs, prompt, _ = full.references_and_prompt(self.video, {"start": .5, "end": 1., "role_mode": "auto", "role_ids": []},
            self.state["roles"], self.state, .5, 1., 124, "auto")
        self.assertEqual(len(refs), 3)  # Two primaries plus the explicitly selected auxiliary.
        targets, remainder = prompt.split("\n\nsummary:", 1)
        self.assertNotIn("source performer wearing", targets)
        self.assertIn("source performer wearing a red-and-white uniform", remainder)
        self.assertIn("source performer wearing a white-and-gold uniform", remainder)
        self.assertIn("additional views of the SAME person: <Picture 2>", targets)
        self.assertNotIn("<Picture 4>", prompt)

    def test_prompt_does_not_invent_an_end_location_from_a_person_that_may_have_exited(self):
        self.video["bindings"][0]["anchors"] = [{"time": .2, "box": [.6, .1, 1., .9]}]
        _, prompt, _ = full.references_and_prompt(self.video, self.video["segments"][0], self.state["roles"], self.state, 0., 1., 124, "auto")
        self.assertIn("at 1.033333s: user anchor", prompt)
        self.assertNotIn("at 0.000000s: user anchor", prompt)
        self.assertNotIn("at 4.994444s: user anchor", prompt)
        self.assertNotIn("interpolated position", prompt)
        self.assertIn("only a hand, arm or foot", prompt)
        self.assertIn("never force a person to remain in the last box", prompt)

    def test_render_restores_all_original_motion_frames_when_vae_returns_the_reference(self):
        encoded = []
        execute = legacy.FakeReference.execute
        def reference(*args, **kwargs):
            encoded.append(kwargs["ref_videos"]["ref_video_0"])
            return execute(*args, **kwargs)
        vae = types.SimpleNamespace(decode=lambda sample: encoded[-1])
        with patch.object(full.MiniMaxH3ReferenceToVideo, "execute", side_effect=reference):
            result = full.render_batch({"state": self.state}, None, None, vae, None, None, None, [1, 0], long_edge=256)
        part = result["videos"]["v"][0]
        self.assertEqual(part["time_map"]["method"], "uniform_source_time")
        for (actual, time), (expected, source_time) in zip(self.assemble(result), legacy.media.iter_video(self.root / "input/source.mp4")):
            self.assertEqual(time, source_time)
            self.assertTrue(np.array_equal(actual, expected))

    def test_manual_split_restores_source_motion_from_one_shared_generation(self):
        self.manual_halves()
        self.video["shots"] = [{"id": "one_actual_shot", "start": 0., "end": 1.}]
        encoded = []
        execute = legacy.FakeReference.execute
        def reference(*args, **kwargs):
            encoded.append(kwargs["ref_videos"]["ref_video_0"])
            return execute(*args, **kwargs)
        vae = types.SimpleNamespace(decode=lambda sample: encoded[-1])
        with patch.object(full.MiniMaxH3ReferenceToVideo, "execute", side_effect=reference), \
             patch.object(full.MiniMaxH3AddGuide, "execute", wraps=GuideOnly.execute) as guide:
            result = full.render_batch({"state": self.state}, None, None, vae, None, None, None, [1, 0], long_edge=256)
        self.assertEqual(len(encoded), 1)
        guide.assert_not_called()
        output = self.assemble(result)
        original = list(legacy.media.iter_video(self.root / "input/source.mp4"))
        self.assertEqual(len(output), len(original))
        for (actual, time), (expected, source_time) in zip(output, original):
            self.assertEqual(time, source_time)
            self.assertTrue(np.array_equal(actual, expected))

    def test_different_prompts_keep_separate_sampling_and_do_not_inherit_previous_pixels(self):
        self.manual_halves()
        self.video["shots"] = [{"id": "one_actual_shot", "start": 0., "end": 1.}]
        self.video["segments"][0]["prompt"] = "Maintain the close-up."
        self.video["segments"][1]["prompt"] = "Preserve the hand movement."
        with patch.object(full.MiniMaxH3AddGuide, "execute") as guide, \
             patch.object(full.SamplerCustomAdvanced, "execute", wraps=legacy.FakeSampler.execute) as sampler:
            result = self.run_full()
        self.assertEqual(sampler.call_count, 2)
        guide.assert_not_called()
        self.assertEqual([part["generation_start"] for part in result["videos"]["v"]], [0., .5])

    def test_revision_selected_range_is_not_widened_and_saved_base_is_not_an_automatic_guide(self):
        self.manual_halves()
        self.video["shots"] = [{"id": "one_actual_shot", "start": 0., "end": 1.}]
        base = self.root / "renders/base.mp4"
        legacy.media.write_video([np.full((32, 64, 3), 50 + index, np.uint8) for index in range(30)], 30, 1, base, lossless=True)
        self.state["settings"]["revision"] = {"enabled": True, "video_id": "v", "base_result_id": "base", "scope": "selected", "segment_ids": ["second_half"]}
        with patch.object(full, "revision_base", return_value=(base, {})), patch.object(full.MiniMaxH3AddGuide, "execute") as guide:
            result = self.run_full()
        guide.assert_not_called()
        self.assertEqual([(part["start"], part["end"]) for part in result["videos"]["v"]], [(.5, 1.)])
        self.assertEqual(result["videos"]["v"][0]["generation_start"], .5)
        for index, (frame, time) in enumerate(self.assemble(result)):
            self.assertTrue(np.all(frame == (50 + index if time < .5 else 191)))

    def test_merge_does_not_exceed_reference_budget(self):
        self.manual_halves()
        self.video["shots"] = [{"id": "one_actual_shot", "start": 0., "end": 1.}]
        self.state["frame_references"] = [{"id": f"frame_{number}", "image_id": "f", "video_id": "v", "time": .04 + number * .09, "kind": "scene"} for number in range(10)]
        groups = full.generation_groups({"state": self.state})
        self.assertEqual([(group["start"], group["end"]) for _, group, _ in groups], [(0., .5), (.5, 1.)])

    def test_saved_report_records_real_shots_and_missing_localization_warnings(self):
        self.two_source_shots()
        result = self.run_full()
        output = self.root / "output"
        output.mkdir()
        (self.root / "previews").mkdir()
        with patch.object(legacy.generation, "output_path", side_effect=lambda name: output / name):
            saved = legacy.generation.save_results(result, None, "shot_plan")[0]
        report = saved["report"]
        self.assertEqual(report["source_shots"], self.video["shots"])
        self.assertTrue(any("沿用原片" in warning for warning in report["warnings"]))
        self.assertEqual([part["parent_segment_id"] for part in report["segments"]], ["s", "s"])
        self.assertEqual(report["frames"], 30)

    def test_background_generation_restores_center_rgb_and_uses_original_references(self):
        class Nested:
            def __init__(self, tensors): self.tensors = list(tensors)
        class Ref:
            @staticmethod
            def execute(*args, **kwargs):
                return types.SimpleNamespace(result=("condition", {"samples": Nested((torch.zeros(1), torch.zeros(1,32,2,8)))}))
        class Vae:
            def encode(self, frames): return torch.zeros((1,24,2,frames.shape[1]//16,frames.shape[2]//16))
            def decode(self, sample): return torch.ones((124,64,64,3)) * .15
        core = torch.rand((124,32,64,3))
        native = types.ModuleType("comfy.nested_tensor"); native.NestedTensor = Nested
        with patch.dict(legacy.sys.modules, {**legacy.stubs,"comfy.nested_tensor":native}), \
             patch.object(full,"MiniMaxH3ReferenceToVideo",Ref), \
             patch.object(full.SamplerCustomAdvanced,"execute",wraps=legacy.FakeSampler.execute) as sample:
            expanded, report = full.expand_frames(core, core, None, None, Vae(), None, None, [1,0], (64,64))
        self.assertTrue(torch.equal(expanded[:,16:48], torch.round(core*255)/255))
        self.assertNotIn("noise_mask",sample.call_args.args[-1])
        self.assertEqual(report["method"],"h3_background_plate_border_composite")
        self.assertEqual(report["background_reference"],"original_only")
        self.assertIn("Do not create a new scene",report["prompt"])

    def test_custom_dimensions_survive_real_video_write_assembly_and_revision_validation(self):
        self.state["settings"]["resolution"] = {"enabled":True,"width":128,"height":64}
        self.video["output_size"] = [128,64]
        generated = torch.rand((124,64,128,3))
        part = full.write_fullframe_part(self.video,self.video["segments"][0],generated,0,self.root/"renders","nearest",self.state["roles"])
        path = self.root/"renders/custom.mp4"
        legacy.generation.assemble_video(self.video,[part],path)
        self.assertEqual(legacy.media.probe_video(path)["width"],128)
        self.assertEqual(legacy.media.probe_video(path)["height"],64)
        frames=list(legacy.media.iter_video(path))
        self.assertEqual(len(frames),30)
        for _,time in frames:
            self.assertAlmostEqual(time*30,round(time*30),delta=.001)
        report={"source":"source.mp4","resolution":[128,64],"duration":1}
        self.state["settings"]["revision"]={"enabled":True,"video_id":"v","base_result_id":"base"}
        with patch.object(reference_plan,"saved_result",return_value=(path,report)):
            self.assertEqual(reference_plan.revision_base(self.state,self.video)[0],path)
            self.state["settings"]["resolution"]["width"]=256
            with self.assertRaisesRegex(ValueError,"改变尺寸"):
                reference_plan.revision_base(self.state,self.video)

    def test_expansion_final_writer_copies_center_rgb_exactly_with_linear_timing(self):
        original = torch.rand((124,32,64,3))
        center = full.write_fullframe_part(self.video,self.video["segments"][0],original,0,self.root/"renders","linear",self.state["roles"])
        expanded_video = {**self.video,"output_size":[64,64]}
        expanded = torch.rand((124,64,64,3))
        with patch.object(full,"_cached_file",side_effect=Path):
            part = full.write_fullframe_part(expanded_video,self.video["segments"][0],expanded,0,self.root/"renders","linear",self.state["roles"],
                {"path":center["path"],"rect":[0,16,64,32]})
        for (wanted,time),(actual,new_time) in zip(legacy.media.iter_video(center["path"]),legacy.media.iter_video(part["path"])):
            self.assertEqual(time,new_time)
            self.assertTrue(np.array_equal(wanted,actual[16:48]))

    def test_second_user_range_reads_its_matching_frames_from_a_shared_expansion_center(self):
        original = torch.rand((124, 32, 64, 3))
        source, times = legacy.media.read_video(self.root / "input/source.mp4")
        _, clock = full.fit_source_clip(source, times, 124)
        center = full.write_fullframe_part(self.video, self.video["segments"][0], original, 0., self.root / "renders", "linear", self.state["roles"], time_map=clock)
        second = {"id": "second_half", "start": .5, "end": 1.}
        expanded_video = {**self.video, "output_size": [64, 64]}
        expanded = torch.rand((124, 64, 64, 3))
        with patch.object(full, "_cached_file", side_effect=Path):
            part = full.write_fullframe_part(expanded_video, second, expanded, 0., self.root / "renders", "linear", self.state["roles"],
                {"path": center["path"], "rect": [0, 16, 64, 32], "start": 0.}, time_map=clock)
        frames = list(legacy.media.iter_video(part["path"]))
        expected = list(legacy.media.iter_video(center["path"], .5, 1.))
        self.assertEqual(len(frames), 15)
        self.assertEqual(len(frames), len(expected))
        for (wanted, source_time), (actual, time) in zip(expected, frames):
            self.assertAlmostEqual(source_time - .5, time, delta=1 / 90000)
            self.assertTrue(np.array_equal(wanted, actual[16:48]))

    def test_first_full_expansion_rejects_timeline_gaps_before_sampling(self):
        self.state["settings"]["resolution"] = {"enabled":True,"width":64,"height":64}
        self.video["segments"][0]["end"] = .5
        with patch.object(full.SamplerCustomAdvanced,"execute") as sample:
            with self.assertRaisesRegex(ValueError,"覆盖整片"):
                self.run_full()
            sample.assert_not_called()

    def test_background_references_use_original_crops_without_replacement_people(self):
        original = torch.rand((5,128,256,3))
        self.video["bindings"][0]["anchors"] = [{"time":0,"box":[.4,0,.6,1]}]
        refs,rects = full.background_crops(original,self.video,self.state["roles"])
        self.assertEqual(len(refs),2)
        for frames,(x,y,w,h) in zip(refs,rects):
            self.assertTrue(torch.equal(frames,original[:,y:y+h,x:x+w]))
            self.assertTrue(x+w<.4*256 or x>.6*256)
        prompt = full.expansion_prompt(124,[0,112,512,288],len(refs))
        self.assertIn("<Video 2>",prompt)
        self.assertIn("Do not extend a cropped head",prompt)

    def test_generated_background_padding_is_removed_only_outside_the_center(self):
        frames = torch.zeros((3,64,64,3))
        center = torch.rand((3,32,64,3));frames[:,16:48]=center
        frames[:,8:16]=.6;frames[:,48:56]=.3
        frames[1,:12]=0;frames[2,52:]=0
        output,changes = full.fill_background_borders(frames,[0,16,64,32])
        self.assertTrue(torch.equal(output[:,16:48],center))
        self.assertTrue(torch.allclose(output[:,:16],torch.full_like(output[:,:16],.6),atol=1e-6))
        self.assertTrue(torch.allclose(output[:,48:],torch.full_like(output[:,48:],.3),atol=1e-6))
        self.assertEqual(len(changes),2)


if __name__ == "__main__":
    unittest.main()
