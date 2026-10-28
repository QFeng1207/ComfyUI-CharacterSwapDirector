import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import test_plan_server as project


SPEC = importlib.util.spec_from_file_location("csd_shot_plan_tests", Path(__file__).resolve().parents[1] / "shot_plan.py")
shots = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shots)


def alternating_video():
    boundaries = [0., 5 / 3, 2.6, 6.5, 7 + 22 / 30, 8.5, 10.]
    dan_times = [.002138, .478, 1.020555, 2.607919, 2.966357, 3.358931,
                 4.007531, 4.235111, 4.553721, 6.664518, 7.085540, 7.404150,
                 7.842240, 8.462394]
    blade_times = [2.255, 2.545335, 8.570494, 9.241852, 9.486500]
    return {"info": {"duration": 10., "fps_num": 30, "fps_den": 1},
            "shots": [{"id": f"shot_{index + 1}", "start": start, "end": end}
                      for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]))],
            "bindings": [{"role_id": "dan", "anchors": [{"time": time, "box": [.2, .1, .8, .9]} for time in dan_times]},
                         {"role_id": "blade", "anchors": [{"time": time, "box": [.3, .2, .7, 1.]} for time in blade_times]}]}


class PartialShotTests(unittest.TestCase):
    def test_visible_partial_role_is_local_to_cut_and_does_not_fabricate_anchors(self):
        video = alternating_video()
        video["shot_overrides"] = [{"start": 5 / 3, "end": 2.6, "prompt": "Keep the foreground gun hand.",
            "roles": [{"role_id": "dan", "presence": "visible", "body_part": "hand_arm", "description": "left foreground"}]}]
        result = shots.shot_plan(video, {"start": 0., "end": 10., "role_mode": "auto", "role_ids": []})
        self.assertEqual([part["role_ids"] for part in result], [["dan"], ["dan", "blade"], ["dan"], ["dan"], ["dan"], ["blade"]])
        partial = result[1]["tracks"][0]
        self.assertEqual(partial["anchors"], [])
        self.assertEqual(partial["positions"], [])
        self.assertEqual(partial["body_part"], "hand_arm")
        self.assertEqual(result[1]["prompt"], "Keep the foreground gun hand.")
        self.assertEqual(result[-1]["unlocated_role_ids"], ["dan"])

    def test_absent_and_manual_empty_override_visibility_without_losing_other_shots(self):
        video = alternating_video()
        video["shot_overrides"] = [{"start": 0., "end": 5 / 3, "roles": [{"role_id": "dan", "presence": "absent"}]}]
        result = shots.shot_plan(video, {"start": 0., "end": 10., "role_mode": "auto", "role_ids": []})
        self.assertEqual(result[0]["role_ids"], [])
        self.assertEqual(result[2]["role_ids"], ["dan"])
        video["shot_overrides"][0]["roles"][0]["presence"] = "visible"
        result = shots.shot_plan(video, {"start": 0., "end": 10., "role_mode": "manual", "role_ids": []})
        self.assertTrue(all(part["role_ids"] == [] for part in result))

    def test_shot_override_shared_by_user_boundary_but_not_stale_cut(self):
        video = alternating_video()
        video["shot_overrides"] = [{"start": 2.6, "end": 6.5, "prompt": "Face leaves frame, only the hand remains.", "roles": []}]
        before = shots.shot_plan(video, {"start": 2.6, "end": 4.033333})[0]
        after = shots.shot_plan(video, {"start": 4.033333, "end": 6.5})[0]
        self.assertEqual(before["prompt"], after["prompt"])
        self.assertTrue(before["prompt"])
        video["shots"][2]["end"] = 6.4
        self.assertEqual(shots.shot_plan(video, {"start": 2.6, "end": 4.033333})[0]["prompt"], "")


class AnchorBoundsTests(unittest.TestCase):
    def assertBoxAlmostEqual(self, actual, expected):
        self.assertEqual(len(actual), 4)
        for coordinate, wanted in zip(actual, expected):
            self.assertAlmostEqual(coordinate, wanted)

    def test_complete_box_is_preserved_without_exclusion_points_expanding_it(self):
        anchor = {"box": [.2, .1, .8, .9], "points": [{"x": 1., "y": 0., "label": 0}]}
        bounds = shots.anchor_bounds(anchor)
        self.assertEqual(bounds, [.2, .1, .8, .9])
        bounds[0] = 0
        self.assertEqual(anchor["box"], [.2, .1, .8, .9])

    def test_positive_points_define_location_and_negative_points_do_not(self):
        anchor = {"points": [{"x": .2, "y": .3, "label": 1},
                             {"x": .6, "y": .7}, {"x": .99, "y": .99, "label": 0}]}
        self.assertBoxAlmostEqual(shots.anchor_bounds(anchor), [.2, .3, .6, .7])

    def test_positive_brush_includes_radius_and_clips_to_the_image(self):
        anchor = {"points": [{"x": .2, "y": .3}],
                  "strokes": [{"radius": .1, "label": 1, "points": [{"x": .99, "y": .02}]},
                              {"radius": .5, "label": 0, "points": [{"x": 0., "y": 1.}]}]}
        self.assertBoxAlmostEqual(shots.anchor_bounds(anchor), [.2, 0., 1., .3])

    def test_exclusion_only_or_empty_anchors_are_not_person_locations(self):
        for anchor in ({}, {"points": [{"x": .5, "y": .5, "label": 0}]},
                       {"strokes": [{"radius": .2, "label": 0, "points": [{"x": .5, "y": .5}]}]}):
            with self.subTest(anchor=anchor):
                self.assertIsNone(shots.anchor_bounds(anchor))

    def test_shot_anchors_are_sorted_and_belong_to_half_open_hard_cut_window(self):
        anchors = [{"time": 2., "box": [.1, .1, .4, .8]},
                   {"time": 1.5, "points": [{"x": .5, "y": .5, "label": 0}]},
                   {"time": 1.2, "box": [.2, .1, .5, .8]},
                   {"time": 1., "points": [{"x": .3, "y": .4}]},
                   {"time": .99, "box": [.4, .1, .7, .8]}]
        result = shots.shot_anchors({"anchors": anchors}, {"start": 1., "end": 2.})
        self.assertEqual([anchor["time"] for anchor in result], [1., 1.2])
        self.assertEqual([anchor["time"] for anchor in anchors], [2., 1.5, 1.2, 1., .99])

    def test_location_interpolates_all_four_corners_and_holds_nearest_endpoint(self):
        anchors = [{"time": 1., "box": [.1, .2, .4, .6]},
                   {"time": 3., "box": [.5, .4, .9, .8]}]
        midpoint = shots.location_at(anchors, 2.)
        self.assertBoxAlmostEqual(midpoint["box"], [.3, .3, .65, .7])
        self.assertTrue(midpoint["interpolated"])
        self.assertEqual(shots.location_at(anchors, 0.)["box"], anchors[0]["box"])
        self.assertEqual(shots.location_at(anchors, 4.)["box"], anchors[1]["box"])
        self.assertFalse(shots.location_at(anchors, 1.)["interpolated"])


class ShotPlanTests(unittest.TestCase):
    def test_original_six_shots_select_the_alternating_cast_without_cross_cut_fallback(self):
        video = alternating_video()
        segment = {"id": "user_whole_video", "start": 0., "end": 10., "scene_id": "scene_1", "role_ids": []}
        original = copy.deepcopy((video, segment))
        plan = shots.shot_plan(video, segment)
        self.assertEqual([part["role_ids"] for part in plan], [["dan"], ["blade"], ["dan"], ["dan"], ["dan"], ["blade"]])
        self.assertEqual([len(part["tracks"][0]["anchors"]) for part in plan], [3, 2, 6, 3, 2, 3])
        self.assertEqual([(part["start"], part["end"]) for part in plan],
                         [(source["start"], source["end"]) for source in video["shots"]])
        self.assertEqual((video, segment), original)

    def test_unmarked_hard_cut_does_not_borrow_a_nearby_actor(self):
        video = alternating_video()
        video["bindings"][0]["anchors"] = [anchor for anchor in video["bindings"][0]["anchors"]
                                            if not 7 + 22 / 30 <= anchor["time"] < 8.5]
        segment = {"start": 7 + 22 / 30, "end": 8.5, "role_ids": []}
        automatic = shots.shot_plan(video, segment)[0]
        self.assertEqual(automatic["role_ids"], [])
        self.assertEqual(automatic["tracks"], [])
        segment.update(role_mode="manual", role_ids=["dan"])
        manual = shots.shot_plan(video, segment)[0]
        self.assertEqual(manual["role_ids"], [])
        self.assertEqual(manual["missing_role_ids"], ["dan"])

    def test_manual_split_in_one_source_shot_carries_its_outside_anchor(self):
        video = alternating_video()
        video["bindings"][0]["anchors"] = [{"time": 3., "box": [.2, .1, .8, .9]}]
        segment = {"start": 4 + 1 / 30, "end": 6.5, "role_ids": [], "scene_id": "user_renamed_scene"}
        plan = shots.shot_plan(video, segment)[0]
        self.assertEqual(plan["role_ids"], ["dan"])
        self.assertEqual((plan["source_start"], plan["source_end"]), (2.6, 6.5))
        self.assertEqual([anchor["time"] for anchor in plan["tracks"][0]["anchors"]], [3.])
        self.assertEqual(plan["tracks"][0]["positions"][0]["time"], segment["start"])
        self.assertEqual(plan["tracks"][0]["positions"][0]["box"], [.2, .1, .8, .9])
        self.assertTrue(plan["tracks"][0]["positions"][0]["interpolated"])

    def test_positions_include_every_mark_and_stop_before_the_next_hard_cut(self):
        video = alternating_video()
        segment = {"start": 2.6, "end": 6.5, "role_ids": []}
        positions = shots.shot_plan(video, segment)[0]["tracks"][0]["positions"]
        self.assertEqual([position["time"] for position in positions],
                         [2.6, 2.607919, 2.966357, 3.358931, 4.007531, 4.235111, 4.553721, 6.5 - 1 / 30])
        self.assertTrue(all(2.6 <= position["time"] < 6.5 for position in positions))
        self.assertTrue(all(len(position["box"]) == 4 for position in positions))

    def test_explicit_window_keeps_absolute_source_times_and_source_shot_bounds(self):
        video = alternating_video()
        segment = {"start": 4 + 1 / 30, "end": 10., "role_ids": []}
        plan = shots.shot_plan(video, segment, start=1.6, end=2.7)
        self.assertEqual([(part["start"], part["end"]) for part in plan], [(1.6, 5 / 3), (5 / 3, 2.6), (2.6, 2.7)])
        self.assertEqual([part["role_ids"] for part in plan], [["dan"], ["blade"], ["dan"]])
        self.assertEqual(plan[0]["source_start"], 0.)
        self.assertEqual(plan[2]["source_end"], 6.5)

    def test_manual_selection_filters_cast_and_an_explicit_empty_selection_stays_empty(self):
        video = alternating_video()
        segment = {"start": 0., "end": 10., "role_mode": "manual", "role_ids": ["blade"]}
        plan = shots.shot_plan(video, segment)
        self.assertEqual([part["role_ids"] for part in plan], [[], ["blade"], [], [], [], ["blade"]])
        self.assertEqual([part["missing_role_ids"] for part in plan], [["blade"], [], ["blade"], ["blade"], ["blade"], []])
        segment["role_ids"] = []
        empty = shots.shot_plan(video, segment)
        self.assertTrue(all(part["role_ids"] == [] and part["tracks"] == [] for part in empty))
        self.assertTrue(all(part["missing_role_ids"] == [] for part in empty))

    def test_old_nonempty_role_selection_remains_manual_but_old_empty_selection_is_auto(self):
        video = alternating_video()
        manual = shots.shot_plan(video, {"start": 0., "end": 10., "role_ids": ["blade"]})
        automatic = shots.shot_plan(video, {"start": 0., "end": 10., "role_ids": []})
        self.assertEqual(manual[0]["role_ids"], [])
        self.assertEqual(automatic[0]["role_ids"], ["dan"])

    def test_simultaneous_people_keep_independent_locations_and_both_roles(self):
        video = alternating_video()
        video["bindings"][1]["anchors"].append({"time": .7, "box": [.82, .2, .99, .8]})
        first = shots.shot_plan(video, {"start": 0., "end": 5 / 3, "role_ids": []})[0]
        self.assertEqual(first["role_ids"], ["dan", "blade"])
        self.assertEqual(first["tracks"][0]["positions"][0]["box"], [.2, .1, .8, .9])
        self.assertEqual(first["tracks"][1]["positions"][0]["box"], [.82, .2, .99, .8])

    def test_source_cut_controls_continuity_not_manual_segment_scene_ids(self):
        video = alternating_video()
        self.assertTrue(shots.same_shot(video, 4., 4 + 1 / 30))
        self.assertFalse(shots.same_shot(video, 2.59, 2.6))
        self.assertFalse(shots.same_shot(video, 6.49, 6.5))
        self.assertTrue(shots.same_shot(video, 2.6, 3.))

    def test_subframe_window_emits_a_single_position_without_crossing_its_end(self):
        video = alternating_video()
        part = shots.shot_plan(video, {"start": 3., "end": 3.01, "role_ids": []})[0]
        self.assertEqual([position["time"] for position in part["tracks"][0]["positions"]], [3.])


class ShotSelectionTests(project.ProjectFixture, unittest.TestCase):
    def setUp(self):
        self.open_fixture()

    def tearDown(self):
        self.close_fixture()

    def make_alternating_project(self):
        state = self.make_project()
        state["roles"].append({"id": "role2", "image_ids": ["image1"]})
        video = state["videos"][0]
        video["bindings"].append({"role_id": "role2", "anchors": [{"time": 2.5, "box": [.5, .1, .9, .9]}]})
        video["segments"] = [{"id": "whole", "start": 0., "end": 4., "role_ids": []}]
        source = [{"id": "cut1", "start": 0., "end": 2.}, {"id": "cut2", "start": 2., "end": 4.}]
        return state, source

    def test_validate_partial_shot_settings_keeps_data_and_rejects_invalid_role_or_mode(self):
        state, _ = self.make_alternating_project()
        state["videos"][0]["shot_overrides"] = [{"start": 0., "end": 2., "prompt": "Face then hands", "roles": [
            {"role_id": "role2", "presence": "visible", "body_part": "hand_arm", "description": "foreground"}]}]
        result = project.plan.validate_state(state)
        self.assertEqual(result["videos"][0]["shot_overrides"], state["videos"][0]["shot_overrides"])
        for change in ({"role_id": "unknown"}, {"presence": "force_everywhere"}, {"description": {"bad": 1}}):
            invalid = copy.deepcopy(state)
            invalid["videos"][0]["shot_overrides"][0]["roles"][0].update(change)
            with self.assertRaises(ValueError):
                project.plan.validate_state(invalid)

    def test_prepare_plan_replaces_stale_source_shots_without_rewriting_user_segments(self):
        state, source = self.make_alternating_project()
        state["videos"][0]["shots"] = [{"id": "wrong", "start": 0., "end": 4.}]
        original = copy.deepcopy(state)
        with patch.object(project.plan, "detect_segments", return_value=source) as detect:
            prepared = project.plan.prepare_plan(state)
        self.assertEqual(prepared["state"]["videos"][0]["shots"], source)
        normalized_segments = prepared["state"]["videos"][0]["segments"]
        self.assertEqual([{key: segment[key] for key in original_segment}
                          for segment, original_segment in zip(normalized_segments, state["videos"][0]["segments"])],
                         state["videos"][0]["segments"])
        self.assertEqual(len(normalized_segments), 1)
        self.assertEqual(state, original)
        self.assertEqual(detect.call_args.args[1:], (5., .35))

    def test_single_mode_accepts_different_people_in_different_source_shots(self):
        state, source = self.make_alternating_project()
        state["settings"]["person_mode"] = "single"
        with patch.object(project.plan, "detect_segments", return_value=source):
            active = project.plan.active_segments(project.plan.prepare_plan(state))
        self.assertEqual([(segment["id"], roles) for _, segment, roles in active], [("whole", ["role1", "role2"])])

    def test_single_mode_rejects_simultaneous_people_and_respects_manual_filter(self):
        state, source = self.make_alternating_project()
        state["settings"]["person_mode"] = "single"
        state["videos"][0]["bindings"][1]["anchors"].append({"time": .8, "box": [.5, .1, .9, .9]})
        with patch.object(project.plan, "detect_segments", return_value=source):
            prepared = project.plan.prepare_plan(state)
        with self.assertRaisesRegex(ValueError, "多人模式"):
            project.plan.active_segments(prepared)
        prepared["state"]["videos"][0]["segments"][0].update(role_mode="manual", role_ids=["role1"])
        self.assertEqual(project.plan.active_segments(prepared)[0][2], ["role1"])

    def test_trial_and_selected_revision_keep_the_original_user_segment_scope(self):
        for mode in ("trial", "revision"):
            with self.subTest(mode=mode):
                state, source = self.make_alternating_project()
                state["videos"][0]["segments"] = [{"id": "first", "start": 0., "end": 1., "role_ids": []},
                                                    {"id": "selected", "start": 1., "end": 4., "role_ids": []}]
                state["settings"][mode] = {"enabled": True, "video_id": "video1", "segment_id": "selected",
                                           "base_result_id": "base", "scope": "selected", "segment_ids": ["selected"]}
                with patch.object(project.plan, "detect_segments", return_value=source):
                    active = project.plan.active_segments(project.plan.prepare_plan(state))
                self.assertEqual([(segment["id"], roles) for _, segment, roles in active], [("selected", ["role1", "role2"])])
                video, segment, _ = active[0]
                self.assertEqual([(part["start"], part["end"]) for part in shots.shot_plan(video, segment)], [(1., 2.), (2., 4.)])


if __name__ == "__main__":
    unittest.main()
