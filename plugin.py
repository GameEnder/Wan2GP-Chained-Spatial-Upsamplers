from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

try:
    from shared.utils.plugins import WAN2GPPlugin
except ImportError:
    # Standalone mock for testing outside Wan2GP environment
    class WAN2GPPlugin:
        def __init__(self):
            self.name = "Chained Spatial Upsamplers"
            self.version = "1.0.0"
            self.description = "Stage multiple spatial upsamplers in series."
            self._component_requests = []
            self.tabs = {}

        def add_tab(self, tab_id: str, label: str, component_constructor):
            self.tabs[tab_id] = label

        def request_component(self, name: str):
            if name not in self._component_requests:
                self._component_requests.append(name)

        def request_global(self, name: str):
            pass

try:
    from postprocessing.spatial_upsamplers import (
        UPSAMPLER_TYPE_POSTPROCESSING,
        query_postprocessing_method_choices,
        query_upsampler_defs,
        upsampler_handlers,
    )
except ImportError:
    UPSAMPLER_TYPE_POSTPROCESSING = "postprocessing"
    query_postprocessing_method_choices = None
    query_upsampler_defs = None
    upsampler_handlers = None

try:
    from .presets import (
        DEFAULT_PRESETS,
        calculate_cumulative_scale,
        get_preset_by_id,
        load_presets,
        sanitize_preset_id,
        save_presets,
    )
except (ImportError, ValueError):
    from presets import (
        DEFAULT_PRESETS,
        calculate_cumulative_scale,
        get_preset_by_id,
        load_presets,
        sanitize_preset_id,
        save_presets,
    )

PlugIn_Name = "Chained Upsamplers"
PlugIn_Id = "chained_upsamplers"


def get_available_upsampler_choices() -> List[Tuple[str, str]]:
    """Returns list of (display_label, value) for single-stage postprocessing upsamplers."""
    choices: List[Tuple[str, str]] = []
    if query_upsampler_defs is not None:
        try:
            for udef in query_upsampler_defs(UPSAMPLER_TYPE_POSTPROCESSING, enabled_only=True):
                config_key = udef.get("config_key", "")
                if config_key == "chained_upsamplers":
                    continue
                methods = udef.get("methods", [])
                multipliers = udef.get("multipliers", {})
                for label, method_key in methods:
                    if method_key.startswith("chain_"):
                        continue
                    method_mults = multipliers.get(method_key, ())
                    if method_mults:
                        for m in method_mults:
                            formatted_m = f"{m:g}" if isinstance(m, (int, float)) else str(m)
                            choices.append((f"{label} {formatted_m}x", f"{method_key}*{formatted_m}"))
                    else:
                        choices.append((label, method_key))
        except Exception as e:
            print(f"[ChainedUpsamplers] Error querying upsampler defs: {e}")

    if not choices:
        # Standard built-in fallbacks if registry is not initialized
        choices = [
            ("Lanczos 1.5x", "lanczos*1.5"),
            ("Lanczos 2x", "lanczos*2"),
            ("Lanczos 3x", "lanczos*3"),
            ("Lanczos 4x", "lanczos*4"),
            ("FlashVSR 2x", "flashvsr*2"),
            ("FlashVSR 4x", "flashvsr*4"),
            ("Chain-of-Zoom 2x", "coz*2"),
            ("Chain-of-Zoom 4x", "coz*4"),
            ("H3 Face Refiner", "h3_face_refiner"),
            ("DLSS5 2x", "dlss5*2"),
            ("DLSS5 4x", "dlss5*4"),
            ("SeedVR2 2x", "seedvr2*2"),
        ]
    return choices


class ConfigTabPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = PlugIn_Name
        self.version = "1.0.0"
        self.description = "Configure sequential spatial upsampling chains for Wan2GP post-processing."
        self.refresh_form_trigger = None

    def setup_ui(self):
        self.request_component("state")
        self.request_component("refresh_form_trigger")
        self.add_tab(tab_id=PlugIn_Id, label=PlugIn_Name, component_constructor=self.create_config_ui)

    def on_tab_select(self, state: dict) -> str:
        return str(time.time_ns())

    def on_tab_deselect(self, state: dict) -> str:
        return str(time.time_ns())

    def create_config_ui(self, api_session=None):
        with gr.Blocks() as plugin_ui:
            gr.Markdown(
                """
                ### 🔗 Chained Spatial Upsamplers & Visual Refiners
                Configure multi-stage sequential post-processing chains. Each chain runs its stages in order (e.g. AI upscaling followed by facial refinement or sharp resampling) and appears in Wan2GP's Post Processing dropdowns.
                """
            )

            presets_state = gr.State(value=load_presets())
            selected_preset_id_state = gr.State(value="")
            current_stages_state = gr.State(value=[])

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Presets")
                    preset_selector = gr.Dropdown(
                        label="Select Preset to Edit",
                        choices=[(p["name"], p["id"]) for p in load_presets()] + [("➕ Create New Preset...", "__new__")],
                        value=load_presets()[0]["id"] if load_presets() else "__new__",
                    )
                    preset_name_input = gr.Textbox(label="Preset Display Name", placeholder="e.g. FlashVSR 2x -> Face Refiner")
                    preset_id_input = gr.Textbox(label="Preset ID (Unique)", placeholder="e.g. flashvsr_face_refiner")
                    preset_desc_input = gr.Textbox(label="Description", placeholder="Details about this chain...", lines=2)
                    preset_enabled_cb = gr.Checkbox(label="Enabled (visible in Wan2GP dropdowns)", value=True)

                    with gr.Row():
                        save_btn = gr.Button("💾 Save Preset", variant="primary")
                        delete_btn = gr.Button("🗑️ Delete Preset", variant="stop")

                    reset_defaults_btn = gr.Button("↺ Reset All to Defaults", variant="secondary", size="sm")
                    status_text = gr.Markdown("")

                with gr.Column(scale=2):
                    gr.Markdown("#### Stages in Serial Chain")
                    total_scale_info = gr.Markdown("**Total Cumulative Scale:** 1.0x")

                    stages_display = gr.HTML(value="")

                    gr.Markdown("##### Add Next Stage")
                    with gr.Row():
                        available_choices = get_available_upsampler_choices()
                        stage_method_select = gr.Dropdown(
                            label="Upsampler / Refiner Method",
                            choices=available_choices,
                            value=available_choices[0][1] if available_choices else "",
                            scale=3,
                        )
                        add_stage_btn = gr.Button("➕ Add Stage to Chain", variant="secondary", scale=1)

                    with gr.Row():
                        remove_last_stage_btn = gr.Button("⬅️ Remove Last Stage", size="sm")
                        clear_stages_btn = gr.Button("🧹 Clear All Stages", size="sm")

            def render_stages_html(stages: List[Dict[str, Any]]) -> str:
                if not stages:
                    return "<p style='color: gray; font-style: italic;'>No stages in this chain yet. Add a stage below.</p>"
                items = []
                for idx, stage in enumerate(stages):
                    m_val = stage.get("method_value", "")
                    items.append(f"""
                        <div style='padding: 8px 12px; margin-bottom: 6px; background-color: rgba(255, 255, 255, 0.05); border-left: 4px solid #4CAF50; border-radius: 4px;'>
                            <strong>Stage {idx + 1}:</strong> <code>{m_val}</code>
                        </div>
                    """)
                return "".join(items)

            def update_total_scale_text(stages: List[Dict[str, Any]]) -> str:
                scale = calculate_cumulative_scale(stages)
                return f"**Total Cumulative Scale:** `{scale:g}x` ({len(stages)} stage{'s' if len(stages) != 1 else ''})"

            def on_select_preset(selected_id: str, presets: List[Dict[str, Any]]):
                if selected_id == "__new__" or not selected_id:
                    new_stages = []
                    return (
                        "",
                        "",
                        "",
                        True,
                        new_stages,
                        render_stages_html(new_stages),
                        update_total_scale_text(new_stages),
                        selected_id,
                        "",
                    )
                preset = get_preset_by_id(selected_id, presets)
                if preset is None:
                    new_stages = []
                    return "", "", "", True, new_stages, render_stages_html(new_stages), update_total_scale_text(new_stages), selected_id, "Preset not found"
                stages = preset.get("stages", [])
                return (
                    preset.get("name", ""),
                    preset.get("id", ""),
                    preset.get("description", ""),
                    preset.get("enabled", True),
                    stages,
                    render_stages_html(stages),
                    update_total_scale_text(stages),
                    selected_id,
                    f"Loaded preset '{preset.get('name')}'",
                )

            def on_add_stage(method_val: str, current_stages: List[Dict[str, Any]]):
                if not method_val:
                    return current_stages, render_stages_html(current_stages), update_total_scale_text(current_stages)
                updated = list(current_stages)
                updated.append({"method_value": method_val, "parameters": {}})
                return updated, render_stages_html(updated), update_total_scale_text(updated)

            def on_remove_last_stage(current_stages: List[Dict[str, Any]]):
                if not current_stages:
                    return current_stages, render_stages_html(current_stages), update_total_scale_text(current_stages)
                updated = list(current_stages[:-1])
                return updated, render_stages_html(updated), update_total_scale_text(updated)

            def on_clear_stages():
                empty: List[Dict[str, Any]] = []
                return empty, render_stages_html(empty), update_total_scale_text(empty)

            refresh_trigger = getattr(self, "refresh_form_trigger", None)
            if refresh_trigger is not None:
                self.on_tab_outputs = [refresh_trigger]

            def on_save_preset(p_name: str, p_id: str, p_desc: str, p_enabled: bool, stages: List[Dict[str, Any]], presets: List[Dict[str, Any]]):
                if not p_name.strip():
                    err_ret = (presets, gr.update(), "⚠️ Error: Preset display name is required.", gr.update())
                    return (*err_ret, gr.update()) if refresh_trigger is not None else err_ret
                clean_id = sanitize_preset_id(p_id or p_name)
                if not stages:
                    err_ret = (presets, gr.update(), "⚠️ Error: A chain must have at least one stage.", gr.update())
                    return (*err_ret, gr.update()) if refresh_trigger is not None else err_ret

                updated_presets = [dict(p) for p in presets]
                existing_idx = next((i for i, p in enumerate(updated_presets) if p.get("id") == clean_id), -1)

                new_entry = {
                    "id": clean_id,
                    "name": p_name.strip(),
                    "description": p_desc.strip(),
                    "enabled": bool(p_enabled),
                    "stages": stages,
                }

                if existing_idx >= 0:
                    updated_presets[existing_idx] = new_entry
                else:
                    updated_presets.append(new_entry)

                save_presets(updated_presets)
                choices = [(p["name"], p["id"]) for p in updated_presets] + [("➕ Create New Preset...", "__new__")]
                ret = (
                    updated_presets,
                    gr.update(choices=choices, value=clean_id),
                    f"✅ Saved preset '{new_entry['name']}' ({clean_id})",
                    clean_id,
                )
                return (*ret, str(time.time_ns())) if refresh_trigger is not None else ret

            def on_delete_preset(selected_id: str, presets: List[Dict[str, Any]]):
                if selected_id == "__new__" or not selected_id:
                    ret = (presets, gr.update(), "No preset selected to delete.", "__new__")
                    return (*ret, gr.update()) if refresh_trigger is not None else ret

                updated = [p for p in presets if p.get("id") != selected_id]
                save_presets(updated)
                choices = [(p["name"], p["id"]) for p in updated] + [("➕ Create New Preset...", "__new__")]
                new_val = updated[0]["id"] if updated else "__new__"
                ret = (
                    updated,
                    gr.update(choices=choices, value=new_val),
                    f"🗑️ Deleted preset '{selected_id}'.",
                    new_val,
                )
                return (*ret, str(time.time_ns())) if refresh_trigger is not None else ret

            def on_reset_defaults():
                save_presets(DEFAULT_PRESETS)
                updated = list(DEFAULT_PRESETS)
                choices = [(p["name"], p["id"]) for p in updated] + [("➕ Create New Preset...", "__new__")]
                ret = (
                    updated,
                    gr.update(choices=choices, value=updated[0]["id"] if updated else "__new__"),
                    "↺ Presets reset to default values.",
                )
                return (*ret, str(time.time_ns())) if refresh_trigger is not None else ret

            # Wiring events
            preset_selector.change(
                fn=on_select_preset,
                inputs=[preset_selector, presets_state],
                outputs=[
                    preset_name_input,
                    preset_id_input,
                    preset_desc_input,
                    preset_enabled_cb,
                    current_stages_state,
                    stages_display,
                    total_scale_info,
                    selected_preset_id_state,
                    status_text,
                ],
            )

            add_stage_btn.click(
                fn=on_add_stage,
                inputs=[stage_method_select, current_stages_state],
                outputs=[current_stages_state, stages_display, total_scale_info],
            )

            remove_last_stage_btn.click(
                fn=on_remove_last_stage,
                inputs=[current_stages_state],
                outputs=[current_stages_state, stages_display, total_scale_info],
            )

            clear_stages_btn.click(
                fn=on_clear_stages,
                inputs=[],
                outputs=[current_stages_state, stages_display, total_scale_info],
            )

            save_outputs = [presets_state, preset_selector, status_text, selected_preset_id_state]
            if refresh_trigger is not None:
                save_outputs.append(refresh_trigger)

            delete_outputs = [presets_state, preset_selector, status_text, selected_preset_id_state]
            if refresh_trigger is not None:
                delete_outputs.append(refresh_trigger)

            reset_outputs = [presets_state, preset_selector, status_text]
            if refresh_trigger is not None:
                reset_outputs.append(refresh_trigger)

            save_btn.click(
                fn=on_save_preset,
                inputs=[preset_name_input, preset_id_input, preset_desc_input, preset_enabled_cb, current_stages_state, presets_state],
                outputs=save_outputs,
            )

            delete_btn.click(
                fn=on_delete_preset,
                inputs=[selected_preset_id_state, presets_state],
                outputs=delete_outputs,
            )

            reset_defaults_btn.click(
                fn=on_reset_defaults,
                inputs=[],
                outputs=reset_outputs,
            )

            # Initialize first preset on load
            if load_presets():
                first = load_presets()[0]
                on_select_preset(first["id"], load_presets())

        return plugin_ui
