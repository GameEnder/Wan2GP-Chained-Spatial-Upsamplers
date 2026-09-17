import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

# Ensure repo root is on sys.path
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from presets import (
    DEFAULT_PRESETS,
    calculate_cumulative_scale,
    get_preset_by_id,
    load_presets,
    parse_scale_from_value,
    sanitize_preset_id,
    save_presets,
    validate_preset_structure,
)
from chained_upsampler import ChainedSpatialUpsampler


class TestPresets(unittest.TestCase):
    def test_parse_scale_from_value(self):
        self.assertEqual(parse_scale_from_value("flashvsr*2"), 2.0)
        self.assertEqual(parse_scale_from_value("lanczos*1.5"), 1.5)
        self.assertEqual(parse_scale_from_value("coz*4.0"), 4.0)
        self.assertEqual(parse_scale_from_value("h3_face_refiner"), 1.0)
        self.assertEqual(parse_scale_from_value(""), 1.0)

    def test_calculate_cumulative_scale(self):
        stages = [
            {"method_value": "flashvsr*2"},
            {"method_value": "lanczos*1.5"},
        ]
        self.assertEqual(calculate_cumulative_scale(stages), 3.0)

        stages_with_refiner = [
            {"method_value": "flashvsr*2"},
            {"method_value": "h3_face_refiner"},
            {"method_value": "coz*2"},
        ]
        self.assertEqual(calculate_cumulative_scale(stages_with_refiner), 4.0)

    def test_sanitize_preset_id(self):
        self.assertEqual(sanitize_preset_id("FlashVSR 2x -> Lanczos 1.5x"), "flashvsr_2x_lanczos_1_5x")
        self.assertEqual(sanitize_preset_id("Special #@! Chain"), "special_chain")
        self.assertEqual(sanitize_preset_id(""), "chain")

    def test_load_and_save_presets(self, tmp_path=None):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_presets.json"
            presets = [
                {
                    "id": "test_chain",
                    "name": "Test Chain",
                    "description": "A test chain",
                    "enabled": True,
                    "stages": [{"method_value": "lanczos*2", "parameters": {}}],
                }
            ]
            save_presets(presets, file_path)
            loaded = load_presets(file_path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["id"], "test_chain")

    def test_validate_preset_structure(self):
        valid = {
            "id": "valid_id",
            "name": "Valid Name",
            "stages": [{"method_value": "lanczos*2"}],
        }
        self.assertEqual(len(validate_preset_structure(valid)), 0)

        invalid = {"id": "", "name": "", "stages": []}
        errors = validate_preset_structure(invalid)
        self.assertTrue(len(errors) >= 3)


class TestChainedSpatialUpsampler(unittest.TestCase):
    def setUp(self):
        self.handler = ChainedSpatialUpsampler()

    def test_query_upsampler_def(self):
        udef = self.handler.query_upsampler_def()
        self.assertEqual(udef["name"], "Chained Upsamplers")
        self.assertIn("postprocessing", udef["upsampler_types"])
        self.assertIn("methods", udef)
        self.assertIn("multipliers", udef)
        self.assertTrue(len(udef["methods"]) > 0)

    def test_is_upsampling(self):
        self.assertTrue(self.handler.is_upsampling("chain_flashvsr2x_lanczos15x"))
        self.assertTrue(self.handler.is_upsampling("chain_custom*3"))
        self.assertFalse(self.handler.is_upsampling("lanczos*2"))
        self.assertFalse(self.handler.is_upsampling(""))

    def test_split_value(self):
        method, scale = self.handler.split_value("chain_flashvsr2x_lanczos15x*3.0")
        self.assertEqual(method, "chain_flashvsr2x_lanczos15x")
        self.assertEqual(scale, 3.0)

    def test_upscale_sequential_execution(self):
        # Create dummy stages
        mock_stage1_handler = MagicMock()
        mock_stage2_handler = MagicMock()

        # Stage 1 doubles tensor values
        def stage1_upscale(sample, val, **kwargs):
            return sample * 2, None

        # Stage 2 adds 10 to tensor values
        def stage2_upscale(sample, val, **kwargs):
            return sample + 10, None

        mock_stage1_handler.upscale.side_effect = stage1_upscale
        mock_stage2_handler.upscale.side_effect = stage2_upscale
        mock_stage1_handler.query_upsampler_def.return_value = {"name": "MockStage1"}
        mock_stage2_handler.query_upsampler_def.return_value = {"name": "MockStage2"}

        def mock_find_postprocessing(val):
            if "stage1" in val:
                return mock_stage1_handler
            if "stage2" in val:
                return mock_stage2_handler
            return None

        test_preset = {
            "id": "mock_test_chain",
            "name": "Mock Test Chain",
            "enabled": True,
            "stages": [
                {"method_value": "stage1*2", "parameters": {"custom_param": 42}},
                {"method_value": "stage2*1", "parameters": {}},
            ],
        }

        progress_calls = []

        # Mimic Wan2GP's actual progress_callback signature: upsampler_progress(phase, current_step=None, total_steps=None)
        def mock_progress(phase, current_step=None, total_steps=None):
            phase_text = str(phase)
            step_int = int(current_step) if current_step is not None else -1
            progress_calls.append((phase_text, step_int, total_steps))

        with patch("chained_upsampler.get_preset_by_id", return_value=test_preset), \
             patch("chained_upsampler.find_postprocessing_upsampler", side_effect=mock_find_postprocessing), \
             patch("chained_upsampler.upscale_postprocessing", None):

            input_sample = torch.ones((3, 1, 16, 16))
            result, cache = self.handler.upscale(
                input_sample,
                "chain_mock_test_chain",
                progress_callback=mock_progress,
            )

            # Expected: (1.0 * 2) + 10 = 12.0
            self.assertTrue(torch.allclose(result, torch.full_like(result, 12.0)))
            self.assertEqual(mock_stage1_handler.upscale.call_count, 1)
            self.assertEqual(mock_stage2_handler.upscale.call_count, 1)

            # Verify kwargs passed to stage 1 included custom_param
            call_kwargs = mock_stage1_handler.upscale.call_args[1]
            self.assertEqual(call_kwargs.get("custom_param"), 42)

            # Verify progress callback was invoked without ValueError on int conversion
            self.assertTrue(len(progress_calls) > 0)
            self.assertTrue(any("Stage 1/2" in call[0] for call in progress_calls))

    def test_validate_upsampling(self):
        mock_handler = MagicMock()
        mock_handler.validate_upsampling.return_value = ""

        def mock_find(val):
            return mock_handler if "valid" in val else None

        test_preset = {
            "id": "val_preset",
            "name": "Validation Preset",
            "stages": [{"method_value": "valid*2"}],
        }

        with patch("chained_upsampler.get_preset_by_id", return_value=test_preset), \
             patch("chained_upsampler.find_postprocessing_upsampler", side_effect=mock_find):
            err = self.handler.validate_upsampling("chain_val_preset", 0)
            self.assertEqual(err, "")

        # Test with missing handler
        with patch("chained_upsampler.get_preset_by_id", return_value=test_preset), \
             patch("chained_upsampler.find_postprocessing_upsampler", return_value=None):
            err = self.handler.validate_upsampling("chain_val_preset", 0)
            self.assertIn("not registered or unavailable", err)

    def test_upscale_abort(self):
        test_preset = {
            "id": "abort_preset",
            "name": "Abort Preset",
            "stages": [{"method_value": "valid*2"}],
        }
        mock_handler = MagicMock()
        with patch("chained_upsampler.get_preset_by_id", return_value=test_preset), \
             patch("chained_upsampler.find_postprocessing_upsampler", return_value=mock_handler), \
             patch("chained_upsampler.upscale_postprocessing", None):
            input_sample = torch.ones((3, 1, 8, 8))
            res, _ = self.handler.upscale(
                input_sample,
                "chain_abort_preset",
                abort_callback=lambda: True,
            )
            self.assertEqual(mock_handler.upscale.call_count, 0)
            self.assertTrue(torch.allclose(res, input_sample))


class TestPluginUI(unittest.TestCase):
    def test_plugin_setup_and_events(self):
        from plugin import ConfigTabPlugin
        plugin = ConfigTabPlugin()
        plugin.setup_ui()
        self.assertIn("refresh_form_trigger", plugin._component_requests if hasattr(plugin, "_component_requests") else [])
        self.assertTrue(len(plugin.on_tab_select({})) > 0)
        self.assertTrue(len(plugin.on_tab_deselect({})) > 0)


if __name__ == "__main__":
    unittest.main()
