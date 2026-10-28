import importlib
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from test_generation import PACKAGE, FakeNode, generation, stubs


wan_stubs = dict(stubs)
wan_stubs["comfy_extras.nodes_wan"] = types.ModuleType("comfy_extras.nodes_wan")
wan_stubs["comfy_extras.nodes_wan"].WanAnimateToVideo = FakeNode
wan_stubs["comfy_extras.nodes_custom_sampler"].CFGGuider = FakeNode
with patch.dict("sys.modules", wan_stubs):
    wan = importlib.import_module(PACKAGE + ".wan_generation")


class WanReferenceTests(unittest.TestCase):
    def test_full_portrait_survives_wide_reference_fit(self):
        image = torch.zeros((1, 200, 60, 3))
        image[:, :20, :, 0] = 1  # hat
        image[:, -20:, :, 2] = 1  # boots
        result = wan.fit_reference(image, 320, 180)
        self.assertEqual(tuple(result.shape), (1, 180, 320, 3))
        self.assertGreater(float(result[0, :18, 140:180, 0].mean()), .95)
        self.assertGreater(float(result[0, -18:, 140:180, 2].mean()), .95)
        self.assertTrue(torch.all(result[:, :, :100] == 1))

    def test_separate_character_sheet_uses_one_view_and_keeps_switch(self):
        image = torch.ones((1, 120, 420, 3))
        for index, color in enumerate(([.2, .1, .3], [.1, .2, .3], [.3, .1, .2])):
            image[0, 10:110, 30 + index * 130:85 + index * 130] = torch.tensor(color)
        result, crop = wan.primary_view(image)
        self.assertLess(result.shape[2], 90)
        self.assertTrue(torch.any(result[..., 0] == .2))
        self.assertLess(crop[2], .3)
        whole, full = wan.primary_view(image, "whole_image")
        self.assertIs(whole, image)
        self.assertEqual(full, [0., 0., 1., 1.])

    def test_wide_scene_with_only_one_person_is_not_cropped(self):
        image = torch.ones((1, 120, 420, 3))
        image[:, 10:110, 180:240] = 0
        result, crop = wan.primary_view(image)
        self.assertIs(result, image)
        self.assertEqual(crop, [0., 0., 1., 1.])

    def test_secondary_portraits_do_not_become_extra_reference_latent_frames(self):
        primary = torch.full((1, 40, 20, 3), .25)
        auxiliary = torch.full((1, 40, 20, 3), .75)
        def encode(image, crop):
            self.assertFalse(crop, "Portrait references must keep the hat and boots")
            return types.SimpleNamespace(penultimate_hidden_states=torch.full((1, 2, 3), float(image.mean())),
                                         image_embeds=torch.tensor([float(image.mean())]))
        vision = types.SimpleNamespace(encode_image=encode)
        role = {"id": "r", "primary_image_id": "primary", "image_ids": ["aux", "primary"]}
        state = {"images": [{"id": "primary", "file": "primary.png"}, {"id": "aux", "file": "aux.png"}]}
        with patch.object(wan, "input_path", side_effect=lambda value: value), patch.object(wan, "load_reference", side_effect=[primary, auxiliary]):
            reference, output, _ = wan.prepare_references(role, state, vision, 64, 64, "whole_image")
        self.assertEqual(reference.shape[0], 1)
        self.assertAlmostEqual(float(reference[0, 32, 32, 0]), .25)
        self.assertAlmostEqual(float(output.penultimate_hidden_states.mean()), .25 * 2 / 3 + .75 / 3)
        self.assertAlmostEqual(float(output.image_embeds[0]), .25)


class WanControlsTests(unittest.TestCase):
    def test_generator_mask_covers_partial_blocks_without_changing_compositor_mask(self):
        alpha = torch.zeros((2, 35, 70))
        alpha[0, 15, 31] = 1
        alpha[0, 34, 69] = .5
        before = alpha.clone()
        mask = wan.conditioning_mask(alpha)
        self.assertEqual(mask.shape, alpha.shape)
        self.assertTrue(torch.all(mask[0, :32, :32] == 1))
        self.assertTrue(torch.all(mask[0, 32:, 64:] == 1))
        self.assertEqual(float(mask[1].sum()), 0.)
        self.assertTrue(torch.equal(alpha, before))

    def test_unselected_people_are_removed_from_pose_and_face_is_optional(self):
        source = torch.ones((2, 32, 64, 3))
        alpha = torch.zeros((2, 32, 64))
        alpha[:, 8:24, 4:20] = 1
        class Preprocessor:
            def estimate_pose(self, *args, **kwargs):
                self.assert_face = kwargs["detect_face"]
                person = {"pose_keypoints_2d": [10., 10., 1.] * 18}
                return {"result": (torch.ones_like(source), [{"people": [person], "canvas_width": 64, "canvas_height": 32}] * 2)}
        class Renderer:
            def render(self, *args):
                return (torch.ones((1, 32, 64, 3)),)
        with patch.dict(wan.comfy_nodes.NODE_CLASS_MAPPINGS, {"DWPreprocessor": Preprocessor, "RenderPeopleKps": Renderer}):
            pose, face = wan.motion_controls(source, alpha, 64, False)
        self.assertIsNone(face)
        self.assertTrue(torch.all(pose[:, :, 32:] == 0))
        self.assertTrue(torch.all(pose[:, 10:20, 8:16] == 1))

    def test_empty_face_detection_is_a_valid_control_video(self):
        source = torch.ones((1, 32, 64, 3))
        alpha = torch.ones((1, 32, 64))
        face = [10., 10., 1.] * 8  # degenerate detector output
        keyframe = {"people": [{"face_keypoints_2d": face}], "canvas_width": 64, "canvas_height": 32}
        class Preprocessor:
            def estimate_pose(self, *args, **kwargs):
                return {"result": (torch.ones_like(source), [keyframe])}
        class Renderer:
            def render(self, *args):
                return (torch.zeros((1, 32, 64, 3)),)
        with patch.dict(wan.comfy_nodes.NODE_CLASS_MAPPINGS, {"DWPreprocessor": Preprocessor, "RenderPeopleKps": Renderer}):
            _, faces = wan.motion_controls(source, alpha, 64, True)
        self.assertEqual(tuple(faces.shape), (1, 256, 256, 3))
        self.assertEqual(float(faces.sum()), 0.)

    def test_overlapping_neighbours_hand_cannot_supply_the_selected_persons_motion(self):
        mask = torch.zeros((32, 64))
        mask[6:28, 4:25] = 1
        target = {"id": "target", "pose_keypoints_2d": [12., 16., 1.] * 18}
        neighbour_points = [48., 16., 1.] * 18
        neighbour_points[4 * 3:5 * 3] = [12., 16., 1.]  # wrist crosses into target mask
        neighbour = {"id": "neighbour", "pose_keypoints_2d": neighbour_points}
        keyframe = {"people": [neighbour, target], "canvas_width": 64, "canvas_height": 32}
        selected = wan.selected_person(keyframe, mask.numpy())
        self.assertIs(selected, target)
        keyframe["people"] = [neighbour]
        self.assertIsNone(wan.selected_person(keyframe, mask.numpy()))

    def test_normalized_keypoints_map_to_source_pixels(self):
        keyframe = {"canvas_width": 640, "canvas_height": 320}
        points = wan.keypoint_pixels([.25, .5, 1., .5, .75, 1.], keyframe, 64, 32)
        self.assertTrue(torch.equal(torch.from_numpy(points[:, :2]), torch.tensor([[16., 16.], [32., 24.]])))

    def test_long_manual_segment_splits_without_linking_unrelated_segments(self):
        segments = [("video", {"id": "first", "start": 0., "end": 10., "scene_id": None}, ["r"]),
                    ("video", {"id": "next", "start": 10., "end": 11., "scene_id": None}, ["r"])]
        with patch.object(wan, "active_segments", return_value=segments):
            result = wan.bounded_segments({})
        self.assertEqual(len(result), 4)
        self.assertEqual(len({item[1]["id"] for item in result}), 4)
        self.assertEqual(result[0][1]["start"], 0.)
        self.assertEqual(result[2][1]["end"], 10.)
        self.assertEqual(result[0][1]["scene_id"], result[2][1]["scene_id"])
        self.assertNotEqual(result[2][1]["scene_id"], result[3][1]["scene_id"])
        self.assertTrue(all(item[1]["end"] - item[1]["start"] <= 4.5 for item in result))

    def test_replacement_hides_source_person_and_removes_reference_latent_before_decoding(self):
        source = torch.full((4, 32, 64, 3), .75)
        alpha = torch.zeros((4, 32, 64))
        alpha[:, 6:26, 20:40] = 1
        captured = {}
        class NativeAnimate:
            @classmethod
            def execute(cls, *args, **kwargs):
                captured.update(kwargs)
                captured["length"] = args[5]
                return types.SimpleNamespace(result=("pos", "neg", {"samples": torch.zeros((1, 16, 3, 4, 8))}, 1, 0, 5))
        class Sampler:
            @classmethod
            def execute(cls, *args):
                samples = torch.ones((1, 16, 3, 4, 8))
                samples[:, :, 0] = 99  # reference-image latent must never be decoded as a video frame
                return types.SimpleNamespace(result=(None, {"samples": samples}))
        class Vae:
            def decode(self, samples):
                self.assert_trimmed = bool((samples == 1).all())
                captured["trimmed"] = self.assert_trimmed
                return torch.ones((5, 32, 64, 3))
        video = {"file": "source.mp4", "info": {"width": 64, "height": 32}}
        segment = {"id": "segment", "start": 0., "end": .25}
        role = {"id": "role"}
        cache = {("role", 64, 32): (torch.ones((1, 32, 64, 3)), "vision", [0., 0., 1., 1.])}
        encoder = types.SimpleNamespace(encode=lambda *args: ("text",))
        with patch.object(wan, "input_path", return_value=Path("source.mp4")), \
             patch.object(wan, "read_video", return_value=(source, [0., .0625, .125, .1875])), \
             patch.object(wan, "read_masks", return_value=alpha), \
             patch.object(wan, "motion_controls", return_value=(torch.zeros_like(source), None)), \
             patch.object(wan.comfy_nodes, "CLIPTextEncode", return_value=encoder, create=True), \
             patch.object(wan, "WanAnimateToVideo", NativeAnimate), \
             patch.object(wan, "SamplerCustomAdvanced", Sampler), \
             patch.object(wan, "write_composite_part", return_value={}) as composite:
            wan.render_part(video, segment, role, {}, {"timestamps": [0., .0625, .125, .1875]},
                            "model", "clip", Vae(), "vision", "noise", "sampler", [1, 0],
                            64, 1., "linear", False, "auto", Path("renders"), lambda *args: None, cache)
        self.assertTrue(captured["trimmed"])
        self.assertEqual(captured["length"], 5)
        self.assertIsNone(captured["face_video"])
        self.assertTrue(torch.all(captured["background_video"][:, 6:26, 20:40] == 0))
        self.assertTrue(torch.all(captured["background_video"][:, :, :10] == .75))
        self.assertEqual(composite.call_args.kwargs["generation_fps"], 16)


if __name__ == "__main__":
    unittest.main()
