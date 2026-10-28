import copy
import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


PLUGIN = Path(__file__).resolve().parents[1]
PACKAGE = "csd_plan_server_tests"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(PLUGIN)]
sys.modules[PACKAGE] = package

folder_paths = types.ModuleType("folder_paths")
folder_paths.get_input_directory = lambda: str(PLUGIN / "test_input")
folder_paths.get_temp_directory = lambda: str(PLUGIN / "test_temp")
folder_paths.get_output_directory = lambda: str(PLUGIN / "test_output")
folder_paths.models_dir = str(PLUGIN / "test_models")
media = types.ModuleType(PACKAGE + ".media")
media.probe_video = lambda path: {"duration": 4.0, "width": 64, "height": 32}
media.ffmpeg_path = lambda: "ffmpeg"
media.iter_video = lambda *args: iter(())
segmentation = types.ModuleType(PACKAGE + ".segmentation")
segmentation.detect_segments = lambda *args: [{"id": "auto_1", "start": 0.0, "end": 4.0, "enabled": True, "role_ids": []}]
audio = types.ModuleType(PACKAGE + ".audio")
audio.detect_speakers = lambda *args, **kwargs: []
tracking = types.ModuleType(PACKAGE + ".tracking")
tracking.track_video = lambda *args, **kwargs: {}
with patch.dict(sys.modules, {"folder_paths": folder_paths, media.__name__: media,
                            segmentation.__name__: segmentation, audio.__name__: audio, tracking.__name__: tracking}):
    paths = importlib.import_module(PACKAGE + ".paths")
    plan = importlib.import_module(PACKAGE + ".plan")
    server = importlib.import_module(PACKAGE + ".server")
for module in (media, segmentation, audio, tracking, paths, plan, server):
    sys.modules[module.__name__] = module


class ProjectFixture:
    def make_project(self):
        return {
            "version": 1,
            "videos": [{"id": "video1", "file": "source.mp4", "bindings": [{
                "role_id": "role1", "padding": 16, "feather": 4,
                "anchors": [{"time": 0.5, "points": [{"x": 0.4, "y": 0.6, "label": 1}],
                             "box": [0.1, 0.2, 0.8, 0.9],
                             "strokes": [{"radius": 0.015, "label": 0, "points": [{"x": 0.2, "y": 0.3}]}]}]}],
                "segments": [{"id": "segment1", "start": 0, "end": 2, "enabled": True, "role_ids": ["role1"]}],
                "speakers": [{"id": "speaker1", "role_id": "role1", "intervals": [{"start": 0.2, "end": 1.5}]}]}],
            "images": [{"id": "image1", "file": "reference.png"}],
            "audios": [{"id": "voice1", "file": "voice.wav"}],
            "roles": [{"id": "role1", "name": "Person", "image_ids": ["image1"],
                       "primary_image_id": "image1", "voice_id": "voice1"}],
            "settings": {"person_mode": "multi", "audio_mode": "original", "segment_mode": "manual",
                         "max_segment_seconds": 10, "scene_threshold": 0.35,
                         "trial": {"enabled": False, "video_id": "", "segment_id": ""}},
        }

    def open_fixture(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for name in ("input", "temp", "output"):
            (self.root / name).mkdir()
        for name in ("source.mp4", "reference.png", "voice.wav"):
            (self.root / "input" / name).write_bytes(b"test fixture")
        self.patches = [patch.object(folder_paths, "get_input_directory", return_value=str(self.root / "input")),
                        patch.object(folder_paths, "get_temp_directory", return_value=str(self.root / "temp")),
                        patch.object(folder_paths, "get_output_directory", return_value=str(self.root / "output"))]
        for item in self.patches:
            item.start()

    def close_fixture(self):
        for item in reversed(self.patches):
            item.stop()
        self.temporary.cleanup()


class PlanTests(ProjectFixture, unittest.TestCase):
    def setUp(self):
        self.open_fixture()

    def tearDown(self):
        self.close_fixture()

    def test_valid_state_is_copied_and_defaults_are_applied(self):
        original = self.make_project()
        original.pop("settings")
        validated = plan.validate_state(json.dumps(original))
        self.assertEqual(validated["settings"], plan.empty_state()["settings"])
        self.assertNotIn("settings", original)
        validated["roles"][0]["name"] = "Changed"
        self.assertEqual(original["roles"][0]["name"], "Person")

    def test_invalid_role_references_are_rejected(self):
        cases = {
            "missing bound role": lambda state: state["videos"][0]["bindings"][0].update(role_id="deleted"),
            "duplicate binding": lambda state: state["videos"][0]["bindings"].append(copy.deepcopy(state["videos"][0]["bindings"][0])),
            "missing image": lambda state: state["roles"][0].update(image_ids=["deleted"]),
            "foreign primary image": lambda state: state["roles"][0].update(primary_image_id="deleted"),
            "duplicate role": lambda state: state["roles"].append(copy.deepcopy(state["roles"][0])),
        }
        for name, change in cases.items():
            with self.subTest(name=name):
                state = self.make_project()
                change(state)
                with self.assertRaises(ValueError):
                    plan.validate_state(state)

    def test_invalid_metadata_shapes_are_rejected(self):
        cases = {
            "settings list": lambda state: state.update(settings=[]),
            "null trial": lambda state: state["settings"].update(trial=None),
            "scalar video": lambda state: state.update(videos=["source.mp4"]),
            "missing file": lambda state: state["videos"][0].pop("file"),
            "null images": lambda state: state["roles"][0].update(image_ids=None),
            "scalar bindings": lambda state: state["videos"][0].update(bindings="role1"),
            "null anchors": lambda state: state["videos"][0]["bindings"][0].update(anchors=None),
            "scalar segment": lambda state: state["videos"][0].update(segments=[1]),
        }
        for name, change in cases.items():
            with self.subTest(name=name):
                state = self.make_project()
                change(state)
                with self.assertRaises((ValueError, TypeError, KeyError)):
                    plan.validate_state(state)

    def test_legacy_voice_workflows_use_original_audio_without_voice_assets(self):
        for mode in ("replace", "reference-timbre", "original", "mute"):
            with self.subTest(mode=mode):
                state = self.make_project()
                state["settings"]["audio_mode"] = mode
                state["audios"] = [{"id": "missing", "file": "deleted-reference.wav"}]
                state["roles"][0]["voice_id"] = "deleted"
                state["videos"][0]["speakers"] = [1]
                normalized = plan.prepare_plan(state)["state"]
                self.assertEqual(normalized["settings"]["audio_mode"], "mute" if mode == "mute" else "original")
                self.assertNotIn("audios", normalized)
                self.assertNotIn("voice_id", normalized["roles"][0])
                self.assertNotIn("speakers", normalized["videos"][0])
                self.assertEqual(normalized["videos"][0]["bindings"], state["videos"][0]["bindings"])
                self.assertEqual(state["settings"]["audio_mode"], mode)

    def test_unknown_audio_mode_is_rejected(self):
        state = self.make_project()
        state["settings"]["audio_mode"] = "unexpected"
        with self.assertRaisesRegex(ValueError, "保留原声或静音"):
            plan.validate_state(state)

    def test_segments_and_original_timeline_bounds_are_validated(self):
        cases = {
            "unbound role": lambda video: video["segments"][0].update(role_ids=["deleted"]),
            "unsafe segment id": lambda video: video["segments"][0].update(id="../segment"),
            "duplicate segment": lambda video: video["segments"].append(copy.deepcopy(video["segments"][0])),
            "overlap": lambda video: video["segments"].append({"id": "segment2", "start": 1.5, "end": 3}),
            "past video end": lambda video: video["segments"][0].update(end=4.2),
            "anchor at video end": lambda video: video["bindings"][0]["anchors"][0].update(time=4),
        }
        for name, change in cases.items():
            with self.subTest(name=name):
                state = self.make_project()
                change(state["videos"][0])
                with self.assertRaises(ValueError):
                    plan.prepare_plan(state)

    def test_enabled_trial_must_reference_an_existing_segment(self):
        for video_id, segment_id in (("deleted", "segment1"), ("video1", "deleted")):
            with self.subTest(video_id=video_id, segment_id=segment_id):
                state = self.make_project()
                state["settings"]["trial"] = {"enabled": True, "video_id": video_id, "segment_id": segment_id}
                with self.assertRaises(ValueError):
                    plan.prepare_plan(state)

    def test_trial_on_a_disabled_segment_fails_before_execution(self):
        state = self.make_project()
        state["settings"]["trial"] = {"enabled": True, "video_id": "video1", "segment_id": "segment1"}
        state["videos"][0]["segments"][0]["enabled"] = False
        with self.assertRaises(ValueError):
            plan.active_segments(plan.prepare_plan(state))

    def test_numeric_strings_are_normalized_before_timeline_operations(self):
        state = self.make_project()
        state["settings"].update(max_segment_seconds="10", scene_threshold="0.35")
        video = state["videos"][0]
        binding = video["bindings"][0]
        binding.update(padding="16", feather="4")
        anchor = binding["anchors"][0]
        anchor["time"] = "0.5"
        anchor["points"][0].update(x="0.4", y="0.6")
        anchor["box"] = ["0.1", "0.2", "0.8", "0.9"]
        anchor["strokes"][0]["radius"] = "0.015"
        anchor["strokes"][0]["points"][0].update(x="0.2", y="0.3")
        video["segments"][0].update(start="0", end="2")
        result = plan.prepare_plan(state)
        normalized = result["state"]["videos"][0]
        bound = normalized["bindings"][0]
        marked = bound["anchors"][0]
        scalars = [bound["padding"], bound["feather"], marked["time"], *marked["box"],
                   marked["points"][0]["x"], marked["points"][0]["y"], marked["strokes"][0]["radius"],
                   marked["strokes"][0]["points"][0]["x"], marked["strokes"][0]["points"][0]["y"],
                   normalized["segments"][0]["start"], normalized["segments"][0]["end"]]
        self.assertTrue(all(isinstance(value, (int, float)) for value in scalars))
        self.assertEqual(plan.active_segments(result)[0][1]["end"], 2.0)
        self.assertEqual(state["videos"][0]["segments"][0]["end"], "2")

    def test_single_mode_never_silently_omits_bound_roles(self):
        state = self.make_project()
        state["roles"].append({"id": "role2", "image_ids": []})
        state["videos"][0]["bindings"].append({"role_id": "role2", "anchors": [{"time": 1.5, "box": [.7, .1, .9, .9]}]})
        state["videos"][0]["segments"] = [{"id": "segment1", "start": 0, "end": 1, "enabled": False},
                                          {"id": "segment2", "start": 1, "end": 4}]
        state["settings"].update(person_mode="single", trial={"enabled": True, "video_id": "video1", "segment_id": "segment2"})
        with self.assertRaisesRegex(ValueError, "多人模式"):
            plan.active_segments(plan.prepare_plan(state))
        state["videos"][0]["segments"][1]["role_ids"] = ["role1"]
        active = plan.active_segments(plan.prepare_plan(state))
        self.assertEqual([(video["id"], segment["id"], roles) for video, segment, roles in active],
                         [("video1", "segment2", ["role1"])])

    def test_auto_segments_are_created_when_no_trial_is_selected(self):
        state = self.make_project()
        state["videos"][0]["segments"] = []
        state["settings"].update(segment_mode="auto")
        result = plan.prepare_plan(state)
        self.assertEqual(plan.active_segments(result)[0][1]["id"], "auto_1")

    def test_resolution_default_custom_alignment_and_invalid_values(self):
        state = self.make_project()
        self.assertEqual(plan.prepare_plan(state)["state"]["videos"][0]["output_size"], [64, 32])
        state["settings"]["resolution"] = {"enabled": True, "width": 1024, "height": 1080}
        prepared = plan.prepare_plan(state)
        self.assertEqual(prepared["state"]["videos"][0]["output_size"], [1024, 1088])
        for width in (None, float("nan"), -2, 16385):
            state["settings"]["resolution"]["width"] = width
            with self.assertRaises((ValueError, TypeError)):
                plan.prepare_plan(state)

    def test_asset_paths_cannot_escape_input(self):
        outside = self.root / "outside.mp4"
        outside.write_bytes(b"outside")
        for relative in ("../outside.mp4", "..\\outside.mp4", str(outside), "source.mp4\x00", ""):
            with self.subTest(relative=relative):
                state = self.make_project()
                state["videos"][0]["file"] = relative
                with self.assertRaises(ValueError):
                    plan.validate_state(state)

    def test_frame_reference_and_revision_sources_are_validated(self):
        state = self.make_project()
        image_id = state["images"][0]["id"]
        reference = {"id": "frame1", "image_id": image_id, "video_id": "video1", "time": .5, "kind": "guide"}
        state["frame_references"] = [reference]
        self.assertEqual(plan.prepare_plan(state)["state"]["frame_references"][0]["time"], .5)
        for field, value in (("kind", "mask"), ("time", float("nan")), ("video_id", "gone"), ("image_id", "gone")):
            invalid = copy.deepcopy(state)
            invalid["frame_references"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                plan.validate_state(invalid)
        state["frame_references"][0]["time"] = 4
        with self.assertRaisesRegex(ValueError, "超出"):
            plan.prepare_plan(state)

    def test_revision_selected_scope_filters_segments_without_using_trial(self):
        state = self.make_project()
        state["videos"][0]["segments"].append({"id": "segment2", "start": 2, "end": 4, "enabled": True, "role_ids": []})
        state["settings"]["revision"] = {"enabled": True, "video_id": "video1", "base_result_id": "base", "scope": "selected", "segment_ids": ["segment2"]}
        self.assertEqual([segment["id"] for _, segment, _ in plan.active_segments(plan.prepare_plan(state))], ["segment2"])
        state["settings"]["revision"]["segment_ids"] = []
        with self.assertRaisesRegex(ValueError, "勾选"):
            plan.validate_state(state)


class ServerTests(ProjectFixture, unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.open_fixture()
        routes = web.RouteTableDef()
        prompt_server = types.SimpleNamespace(routes=routes, prompt_queue=None)
        server.register_routes(prompt_server)
        app = web.Application()
        app.add_routes(routes)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()
        self.job_patch = patch.object(server, "start_job", return_value={"job_id": "never_run"})
        self.start_job = self.job_patch.start()

    async def asyncTearDown(self):
        self.job_patch.stop()
        await self.client.close()
        self.close_fixture()

    async def test_malformed_workflow_metadata_returns_400_without_starting_a_job(self):
        invalid_role = self.make_project()
        invalid_role["videos"][0]["bindings"][0]["role_id"] = "deleted"
        malformed_state = self.make_project()
        malformed_state["videos"][0]["bindings"][0]["anchors"] = None
        for endpoint in ("analyse",):
            for number, body in enumerate(({}, {"state": [], "video_id": "video1"}, {"state": self.make_project()},
                         {"state": invalid_role, "video_id": "video1"},
                         {"state": malformed_state, "video_id": "video1"},
                         {"state": self.make_project(), "video_id": "deleted"})):
                with self.subTest(endpoint=endpoint, case=number):
                    response = await self.client.post("/csd/" + endpoint, json=body)
                    self.assertEqual(response.status, 400, await response.text())
        self.start_job.assert_not_called()

    async def test_invalid_json_returns_400_without_starting_a_job(self):
        for endpoint in ("analyse",):
            with self.subTest(endpoint=endpoint):
                response = await self.client.post("/csd/" + endpoint, data="{", headers={"Content-Type": "application/json"})
                self.assertEqual(response.status, 400, await response.text())
        self.start_job.assert_not_called()

    async def test_retired_speaker_endpoint_and_audio_assets_never_submit_jobs(self):
        response = await self.client.post("/csd/track", json={})
        self.assertEqual(response.status, 410)
        response = await self.client.post("/csd/speakers", json={"state": self.make_project(), "video_id": "video1"})
        self.assertEqual(response.status, 410)
        self.assertIn("保留原声或静音", (await response.json())["error"])
        response = await self.client.get("/csd/assets?kind=audio")
        self.assertEqual(response.status, 400)
        self.start_job.assert_not_called()

    async def test_valid_analysis_request_is_normalized_before_job_submission(self):
        state = self.make_project()
        state["settings"]["max_segment_seconds"] = "1.25"
        with patch.object(segmentation, "detect_segments", return_value=[]) as detect:
            response = await self.client.post("/csd/analyse", json={"state": state, "video_id": "video1"})
            self.assertEqual(response.status, 200)
            self.assertEqual(await response.json(), {"job_id": "never_run"})
            operation = self.start_job.call_args.args[0]
            operation(lambda *args: None)
            self.assertEqual(detect.call_args.args[1], 1.25)

    async def test_asset_listing_is_lazy_and_selected_video_is_probed(self):
        info = {"duration": 4, "width": 32, "height": 24, "fps_num": 24, "fps_den": 1, "frames": 96, "has_audio": False}
        with patch.object(server, "probe_video", return_value=info) as probe:
            response = await self.client.get("/csd/assets?kind=video")
            self.assertEqual(response.status, 200)
            listed = (await response.json())["assets"]
            self.assertEqual([item["file"] for item in listed], ["source.mp4"])
            self.assertNotIn("info", listed[0])
            probe.assert_not_called()
            response = await self.client.get("/csd/asset", params={"kind": "video", "file": "source.mp4"})
            self.assertEqual(response.status, 200)
            selected = await response.json()
            self.assertEqual(selected["info"], info)
            self.assertEqual(selected["id"], listed[0]["id"])
            self.assertEqual(selected["file"], "source.mp4")
            probe.assert_called_once_with((self.root / "input" / "source.mp4").resolve())

    async def test_invalid_selected_asset_requests_return_400(self):
        cases = (("unknown", "source.mp4"), ("image", "source.mp4"), ("video", ""),
                 ("video", "missing.mp4"), ("video", "../outside.mp4"), ("video", "..\\outside.mp4"))
        with patch.object(server, "probe_video") as probe:
            for kind, file in cases:
                with self.subTest(kind=kind, file=file):
                    response = await self.client.get("/csd/asset", params={"kind": kind, "file": file})
                    self.assertEqual(response.status, 400, await response.text())
            probe.assert_not_called()
        with patch.object(server, "probe_video", side_effect=ValueError("视频流无效")):
            response = await self.client.get("/csd/asset", params={"kind": "video", "file": "source.mp4"})
            self.assertEqual(response.status, 400)
            self.assertIn("视频流无效", await response.text())

    async def test_unknown_job_is_404_and_malformed_job_id_is_400(self):
        response = await self.client.get("/csd/jobs/unknown")
        self.assertEqual(response.status, 404)
        response = await self.client.get("/csd/jobs/!invalid")
        self.assertEqual(response.status, 400, await response.text())

    async def test_frame_endpoint_rejects_foreign_result_and_invalid_time(self):
        state = self.make_project()
        body = {"state": state, "video_id": "video1", "time": float("nan")}
        response = await self.client.post("/csd/frame", json=body)
        self.assertEqual(response.status, 400)
        for result_id in ("../outside", "missing_result"):
            body.update(time=.1, result_id=result_id)
            response = await self.client.post("/csd/frame", json=body)
            self.assertEqual(response.status, 400)
        report = {"source": "different.mp4", "start": 0}
        with patch.object(server, "saved_result", return_value=(self.root / "input/source.mp4", report)):
            body["result_id"] = "valid"
            response = await self.client.post("/csd/frame", json=body)
            self.assertEqual(response.status, 400)
            self.assertIn("不匹配", (await response.json())["error"])

    async def test_dependencies_distinguish_partial_h3_downloads_and_respect_external_model_paths(self):
        model_root = self.root / "models"
        external = self.root / "external-models"
        external.mkdir()
        files = [("diffusion_models", "minimax_h3_ref2va_pruned_int8_convrot.safetensors"),
                 ("text_encoders", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"),
                 ("vae", "minimax_h3_video_vae_int8_convrot.safetensors"), ("vae", "minimax_h3_audio_vae_fp32.safetensors")]
        for index, (category, name) in enumerate(files):
            directory = model_root / category
            directory.mkdir(exist_ok=True, parents=True)
            (directory / (name + ".part" if index == 0 else name)).write_bytes(b"fixture")
        def resolver(category, name):
            return str(external / name if name == files[0][1] else model_root / category / name)
        with patch.object(server.folder_paths, "models_dir", str(model_root)), \
             patch.object(server.folder_paths, "get_full_path", side_effect=resolver, create=True), \
             patch.object(audio, "dependency_status", side_effect=AssertionError("voice models must not be checked"), create=True):
            response = await self.client.get("/csd/dependencies")
            dependencies = (await response.json())["dependencies"]
            self.assertEqual(len(dependencies), 1)
            self.assertFalse(any("换声" in item["name"] for item in dependencies))
            row = next(item for item in dependencies if item["name"].startswith("H3"))
            self.assertFalse(row["available"])
            self.assertIn(files[0][1], row["message"])
            (external / files[0][1]).write_bytes(b"fixture")
            response = await self.client.get("/csd/dependencies")
            row = next(item for item in (await response.json())["dependencies"] if item["name"].startswith("H3"))
            self.assertTrue(row["available"])


if __name__ == "__main__":
    unittest.main()
