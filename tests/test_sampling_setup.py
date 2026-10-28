import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("csd_sampling_setup_tests", Path(__file__).resolve().parents[1] / "sampling_setup.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SamplingSetupTests(unittest.TestCase):
    def graph(self, lora=False):
        prompt = {
            "4": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_ref2va.safetensors"}},
            "6": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["4", 0]}},
            "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
            "13": {"class_type": "BasicScheduler", "inputs": {"model": ["6", 0], "scheduler": "simple", "steps": 8}},
            "14": {"class_type": "CSD_RenderReferenceBatch", "inputs": {"model": ["6", 0], "sampler": ["12", 0], "sigmas": ["13", 0]}},
        }
        if lora:
            prompt["18"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["4", 0],
                "lora_name": "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors", "strength_model": 1.}}
            prompt["6"]["inputs"]["model"] = ["18", 0]
        return prompt

    def test_base_eight_steps_warns_without_mutating_workflow(self):
        graph = self.graph()
        original = copy.deepcopy(graph)
        info, warnings = module.sampling_setup(graph, 14, 8, "max")
        self.assertEqual(info["steps"], 8)
        self.assertEqual(info["loras"], [])
        self.assertEqual(info["model_file"], "minimax_h3_ref2va.safetensors")
        self.assertEqual(len(warnings), 1)
        self.assertIn("未启用加速 LoRA", warnings[0])
        self.assertEqual(graph, original)

    def test_base_twenty_steps_and_matching_turbo_setup_have_no_warning(self):
        self.assertEqual(module.sampling_setup(self.graph(), "14", 20, "max")[1], [])
        info, warnings = module.sampling_setup(self.graph(True), "14", 8, "match")
        self.assertEqual(warnings, [])
        self.assertEqual(len(info["loras"]), 1)

    def test_bypassed_and_disabled_lora_do_not_hide_low_step_warning(self):
        graph = self.graph(True)
        graph["18"]["inputs"]["strength_model"] = 0
        self.assertEqual(len(module.sampling_setup(graph, "14", 8, "match")[1]), 1)
        graph["18"]["inputs"]["strength_model"] = 1
        graph["6"]["inputs"]["model"] = ["4", 0]
        self.assertEqual(len(module.sampling_setup(graph, "14", 8, "match")[1]), 1)

    def test_turbo_mismatched_sampler_and_reference_size_are_reported(self):
        graph = self.graph(True)
        graph["12"]["inputs"]["sampler_name"] = "er_sde"
        info, warnings = module.sampling_setup(graph, "14", 8, "max")
        self.assertEqual(info["sampler_name"], "er_sde")
        self.assertEqual(len(warnings), 2)
        self.assertTrue(any("euler" in message for message in warnings))
        self.assertTrue(any("match" in message for message in warnings))

    def test_no_prompt_does_not_invent_loader_configuration(self):
        info, warnings = module.sampling_setup(None, "14", 8, "max")
        self.assertNotIn("loras", info)
        self.assertEqual(warnings, [])

    def test_linked_lora_widget_is_reported_without_guessing_its_filename(self):
        graph = self.graph(True)
        graph["18"]["inputs"]["lora_name"] = ["primitive", 0]
        info, warnings = module.sampling_setup(graph, "14", 8, "max")
        self.assertEqual(info["loras"][0]["name"], ["primitive", 0])
        self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
