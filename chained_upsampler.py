from __future__ import annotations

import copy
import gc
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

try:
    from postprocessing.spatial_upsamplers import (
        PARAMETER_PREFIX,
        UPSAMPLER_PROFILE_VIDEO,
        UPSAMPLER_TYPE_POSTPROCESSING,
        find_postprocessing_upsampler,
        runtime_parameter_kwargs,
        upscale_postprocessing,
    )
except ImportError:
    # Standalone / test fallback
    PARAMETER_PREFIX = "spatial_upsampler_"
    UPSAMPLER_PROFILE_VIDEO = "video"
    UPSAMPLER_TYPE_POSTPROCESSING = "postprocessing"
    find_postprocessing_upsampler = None
    runtime_parameter_kwargs = None
    upscale_postprocessing = None

try:
    from .presets import calculate_cumulative_scale, get_preset_by_id, load_presets
except (ImportError, ValueError):
    from presets import calculate_cumulative_scale, get_preset_by_id, load_presets


class ChainedSpatialUpsampler:
    """Sequential spatial upsampling handler that stages multiple upscalers and refiners in series."""

    batch_image_inputs = True

    def __init__(self, server_config=None, files_locator=None):
        self.server_config = server_config
        self.files_locator = files_locator

    def query_upsampler_def(self) -> Dict[str, Any]:
        presets = load_presets()
        methods = []
        method_descriptions = {}
        multipliers = {}

        for p in presets:
            if not p.get("enabled", True):
                continue
            method_id = f"chain_{p.get('id')}"
            methods.append((p.get("name", method_id), method_id))
            method_descriptions[method_id] = p.get("description", "")
            # Expose cumulative scale as fixed multiplier
            cum_scale = calculate_cumulative_scale(p.get("stages", []))
            multipliers[method_id] = (cum_scale,)

        return {
            "name": "Chained Upsamplers",
            "upsampler_types": (UPSAMPLER_TYPE_POSTPROCESSING,),
            "media": ("video", "image"),
            "profile": UPSAMPLER_PROFILE_VIDEO,
            "config_key": "chained_upsamplers",
            "pos": 45,
            "methods": methods,
            "vae_methods": [],
            "multipliers": multipliers,
            "postprocessing_category": "upsampler",
            "description": "Chain multiple spatial upsamplers and visual refiners sequentially in series.",
            "method_descriptions": method_descriptions,
        }

    def is_upsampling(self, spatial_upsampling: str) -> bool:
        if not spatial_upsampling:
            return False
        val = str(spatial_upsampling).strip()
        if val.startswith("chain_"):
            return True
        method = val.split("*", 1)[0]
        return method.startswith("chain_")

    def split_value(self, spatial_upsampling: str) -> Tuple[str, float]:
        val = str(spatial_upsampling).strip()
        method = val.split("*", 1)[0]
        preset_id = method.replace("chain_", "", 1)
        preset = get_preset_by_id(preset_id)
        if preset is not None:
            scale = calculate_cumulative_scale(preset.get("stages", []))
            return method, scale
        return method, 1.0

    def build_value(self, method: str, scale: float = 1.0) -> str:
        return f"{method}*{scale}"

    def validate_upsampling(self, spatial_upsampling: str, image_mode: int) -> str:
        method, _ = self.split_value(spatial_upsampling)
        preset_id = method.replace("chain_", "", 1)
        preset = get_preset_by_id(preset_id)
        if preset is None:
            return f"Chained upsampler preset '{preset_id}' not found."

        stages = preset.get("stages", [])
        if not stages:
            return f"Chained upsampler preset '{preset.get('name')}' has no stages configured."

        if find_postprocessing_upsampler is None:
            return ""

        for idx, stage in enumerate(stages):
            stage_val = stage.get("method_value", "")
            if not stage_val:
                return f"Stage {idx + 1} of '{preset.get('name')}' is missing method value."
            handler = find_postprocessing_upsampler(stage_val)
            if handler is None:
                return f"Stage {idx + 1} '{stage_val}' handler is not registered or unavailable."
            if hasattr(handler, "validate_upsampling"):
                err = handler.validate_upsampling(stage_val, image_mode)
                if err:
                    return f"Stage {idx + 1} ({stage_val}) error: {err}"

        return ""

    def upscale(
        self,
        sample: torch.Tensor,
        spatial_upsampling: str,
        *,
        abort_callback: Optional[Callable[[], bool]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        loaded_model_context: Any = None,
        main_offloadobj: Any = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Any]:
        method, _ = self.split_value(spatial_upsampling)
        preset_id = method.replace("chain_", "", 1)
        preset = get_preset_by_id(preset_id)

        if preset is None:
            raise ValueError(f"Chained preset '{preset_id}' not found.")

        stages = preset.get("stages", [])
        if not stages:
            return sample, None

        total_stages = len(stages)
        current_sample = sample
        last_continue_cache = None

        for idx, stage in enumerate(stages):
            if abort_callback is not None and abort_callback():
                print("[ChainedUpsamplers] Upscaling aborted by user.")
                return current_sample, None

            stage_val = stage.get("method_value", "")
            stage_params = dict(stage.get("parameters", {}))
            stage_num = idx + 1

            if find_postprocessing_upsampler is None:
                raise RuntimeError("Wan2GP postprocessing registry not available.")

            stage_handler = find_postprocessing_upsampler(stage_val)
            if stage_handler is None:
                raise RuntimeError(f"Upsampler handler for stage {stage_num} ('{stage_val}') not found.")

            handler_def = stage_handler.query_upsampler_def() if hasattr(stage_handler, "query_upsampler_def") else {}
            handler_name = handler_def.get("name", stage_val)

            def stage_progress(phase="Starting...", current_step=None, total_steps=None, *args, **kwargs):
                if progress_callback is None:
                    return
                if isinstance(phase, (int, float)) and not isinstance(phase, bool) and isinstance(current_step, str):
                    phase, current_step = current_step, None

                phase_str = f"Stage {stage_num}/{total_stages} ({handler_name}): {phase}".strip()
                cur = None
                tot = None
                if current_step is not None:
                    try:
                        cur = int(current_step)
                    except (ValueError, TypeError):
                        cur = None
                if total_steps is not None:
                    try:
                        tot = int(total_steps)
                    except (ValueError, TypeError):
                        tot = None

                try:
                    progress_callback(phase_str, cur, tot)
                except TypeError:
                    try:
                        progress_callback(phase_str, cur)
                    except TypeError:
                        progress_callback(phase_str)

            stage_progress("Starting...")

            # Merge kwargs with stage-specific parameter overrides
            stage_kwargs = copy.deepcopy(kwargs)
            stage_kwargs.update(stage_params)
            stage_kwargs["abort_callback"] = abort_callback
            stage_kwargs["progress_callback"] = stage_progress

            # Run stage
            if upscale_postprocessing is not None:
                result = upscale_postprocessing(
                    stage_handler,
                    current_sample,
                    stage_val,
                    main_offloadobj=main_offloadobj,
                    loaded_model_context=loaded_model_context,
                    **stage_kwargs,
                )
            else:
                result = stage_handler.upscale(
                    current_sample,
                    stage_val,
                    loaded_model_context=loaded_model_context,
                    main_offloadobj=main_offloadobj,
                    **stage_kwargs,
                )

            if isinstance(result, tuple):
                current_sample, last_continue_cache = result
            else:
                current_sample = result
                last_continue_cache = None

            stage_progress("Completed.")

            # Memory cleanup between stages
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        return current_sample, last_continue_cache
