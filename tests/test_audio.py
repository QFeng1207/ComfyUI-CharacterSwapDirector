"""Original and mute audio work without a voice runtime or reference assets."""
import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "csd_audio_runtime_tests"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
previous_folder_paths = sys.modules.get("folder_paths")
sys.modules["folder_paths"] = types.ModuleType("folder_paths")
try:
    adapter = importlib.import_module(PACKAGE + ".audio")
finally:
    if previous_folder_paths is None:
        sys.modules.pop("folder_paths", None)
    else:
        sys.modules["folder_paths"] = previous_folder_paths


class AudioPolicyTests(unittest.TestCase):
    def test_original_audio_reads_only_the_source_soundtrack(self):
        expected = {"waveform": "source waveform", "sample_rate": 48000}
        with patch.object(adapter, "input_path", return_value=Path("source.mp4")) as source, \
             patch.object(adapter.media, "extract_audio", return_value=expected) as extract:
            self.assertIs(adapter.prepare_audio({"file": "source.mp4"}, [], [], "original"), expected)
            source.assert_called_once_with("source.mp4")
            extract.assert_called_once_with(Path("source.mp4"))

    def test_mute_never_reads_source_or_voice_assets(self):
        with patch.object(adapter, "input_path", side_effect=AssertionError("must not read files")), \
             patch.object(adapter.media, "extract_audio", side_effect=AssertionError("must not decode audio")):
            self.assertIsNone(adapter.prepare_audio({}, [], [], "mute"))
            self.assertIsNone(adapter.prepare_audio_file({}, [], [], "mute"))

    def test_old_voice_modes_preserve_source_without_any_runtime(self):
        for mode in ("replace", "reference-timbre"):
            with self.subTest(mode=mode), patch.object(adapter, "input_path", return_value=Path("source.mp4")):
                item = adapter.prepare_audio_file({"file": "source.mp4"}, [{"voice_id": "deleted"}],
                                                 [{"file": "missing-reference.wav"}], mode)
                self.assertEqual(item, {"audio_file": str(Path("source.mp4")), "report": {"mode": "original"}})
        self.assertEqual(adapter.dependency_status()["requires_models"], False)

    def test_missing_source_audio_and_unknown_modes(self):
        with patch.object(adapter, "input_path", return_value=Path("silent.mp4")):
            self.assertIsNone(adapter.prepare_audio_file({"file": "silent.mp4", "info": {"has_audio": False}}, [], [], "original"))
        with self.assertRaisesRegex(ValueError, "保留原声或静音"):
            adapter.prepare_audio({}, [], [], "unknown")


if __name__ == "__main__":
    unittest.main()
