from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np


@dataclass
class CalibrationSample:
    q_rad: List[float]
    position_m: List[float]
    rpy_rad: List[float]


class RobotModelCalibrationStore:
    def __init__(self, path: str = "calibration_data/robot_model_calibration.json"):
        p = Path(path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent / p
        self.path = p
        self.samples: List[CalibrationSample] = []

    def add_sample(self, q_rad: Iterable[float], position_m: Iterable[float], rpy_rad: Iterable[float]) -> None:
        q = list(map(float, q_rad))[:7]
        if len(q) != 7:
            raise ValueError("q_rad must contain 7 arm joints, excluding gripper")
        self.samples.append(CalibrationSample(q, list(map(float, position_m))[:3], list(map(float, rpy_rad))[:3]))

    def save_samples(self, path: Optional[str] = None) -> Path:
        out = self.path if path is None else Path(path)
        if not out.is_absolute():
            out = Path(__file__).resolve().parent / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"samples": [asdict(s) for s in self.samples]}, indent=2), encoding="utf-8")
        return out

    def save_initial_correction(self, path: Optional[str] = None) -> Path:
        out = self.path if path is None else Path(path)
        if not out.is_absolute():
            out = Path(__file__).resolve().parent / out
        data = {
            "joint_zero_offsets_rad": [0.0] * 7,
            "joint_axis_signs": [1.0] * 7,
            "link_lengths_m": {
                "base_height": 0.06,
                "upper_arm": 0.115,
                "forearm": 0.115,
                "tool_a": 0.025,
                "tool_b": 0.025
            },
            "tool_xyz_offset_m": [0.05, 0.0, 0.0],
            "base_to_workspace_transform": np.eye(4).tolist(),
            "mean_calibration_error_m": None,
            "max_calibration_error_m": None,
            "samples": [asdict(s) for s in self.samples],
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return out