from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class FallRuleConfig:
    aspect_ratio_threshold: float = 1.25
    center_drop_threshold: float = 0.18
    confidence_threshold: float = 0.45
    min_fall_frames: int = 4
    recovery_frames: int = 6


@dataclass
class PipelineConfig:
    model_path: str = "yolov8n.pt"
    device: str = "cpu"
    input_size: int = 640
    rule: FallRuleConfig = field(default_factory=FallRuleConfig)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    raw = load_yaml(path)
    rule_raw = raw.get("rule", {})
    rule = FallRuleConfig(**{**FallRuleConfig().__dict__, **rule_raw})
    merged = {"model_path": "yolov8n.pt", "device": "cpu", "input_size": 640, **raw, "rule": rule}
    return PipelineConfig(**merged)
