from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

PRESETS_FILE = Path(__file__).parent / "presets.json"

DEFAULT_PRESETS = [
    {
        "id": "flashvsr2x_lanczos15x",
        "name": "FlashVSR 2x -> Lanczos 1.5x (3x Total)",
        "description": "AI restoration with FlashVSR followed by sharp Lanczos upscaling.",
        "enabled": True,
        "stages": [
            {
                "method_value": "flashvsr*2",
                "parameters": {}
            },
            {
                "method_value": "lanczos*1.5",
                "parameters": {}
            }
        ]
    },
    {
        "id": "flashvsr2x_facerefiner",
        "name": "FlashVSR 2x -> Face Refiner",
        "description": "FlashVSR 2x upscaling followed by H3 Face Refiner detail enhancement.",
        "enabled": True,
        "stages": [
            {
                "method_value": "flashvsr*2",
                "parameters": {}
            },
            {
                "method_value": "h3_face_refiner",
                "parameters": {}
            }
        ]
    },
    {
        "id": "lanczos2x_coz2x",
        "name": "Lanczos 2x -> Chain-of-Zoom 2x (4x Total)",
        "description": "Pre-scale with Lanczos then refine with Chain-of-Zoom.",
        "enabled": True,
        "stages": [
            {
                "method_value": "lanczos*2",
                "parameters": {}
            },
            {
                "method_value": "coz*2",
                "parameters": {}
            }
        ]
    }
]

def sanitize_preset_id(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip().lower())
    cleaned = cleaned.strip("_")
    return cleaned or "chain"

def parse_scale_from_value(value: str) -> float:
    value = str(value or "").strip()
    if "*" in value:
        try:
            return float(value.split("*", 1)[1])
        except (ValueError, TypeError):
            pass
    m = re.search(r"(\d+(?:\.\d+)?)$", value)
    if m:
        try:
            return float(m.group(1))
        except (ValueError, TypeError):
            pass
    return 1.0

def calculate_cumulative_scale(stages: List[Dict[str, Any]]) -> float:
    total_scale = 1.0
    for stage in stages:
        method_val = stage.get("method_value", "")
        scale = parse_scale_from_value(method_val)
        total_scale *= scale
    return total_scale

def load_presets(filepath: Optional[Path | str] = None) -> List[Dict[str, Any]]:
    path = Path(filepath) if filepath else PRESETS_FILE
    if not path.exists():
        save_presets(DEFAULT_PRESETS, path)
        return [dict(p) for p in DEFAULT_PRESETS]
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "presets" in data:
                return data["presets"]
    except Exception as e:
        print(f"[ChainedUpsamplers] Error reading presets from {path}: {e}")
    return [dict(p) for p in DEFAULT_PRESETS]

def save_presets(presets: List[Dict[str, Any]], filepath: Optional[Path | str] = None) -> None:
    path = Path(filepath) if filepath else PRESETS_FILE
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(presets, f, indent=2)
    except Exception as e:
        print(f"[ChainedUpsamplers] Error saving presets to {path}: {e}")

def get_preset_by_id(preset_id: str, presets: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    if presets is None:
        presets = load_presets()
    for p in presets:
        if p.get("id") == preset_id or f"chain_{p.get('id')}" == preset_id:
            return p
    return None

def validate_preset_structure(preset: Dict[str, Any]) -> List[str]:
    errors = []
    if not isinstance(preset, dict):
        return ["Preset must be a dictionary"]
    if not preset.get("id"):
        errors.append("Preset missing 'id'")
    if not preset.get("name"):
        errors.append("Preset missing 'name'")
    stages = preset.get("stages", [])
    if not isinstance(stages, list) or len(stages) == 0:
        errors.append("Preset must have at least one stage in 'stages'")
    else:
        for idx, stage in enumerate(stages):
            if not isinstance(stage, dict):
                errors.append(f"Stage {idx+1} is not a dictionary")
                continue
            if not stage.get("method_value"):
                errors.append(f"Stage {idx+1} is missing 'method_value'")
    return errors
