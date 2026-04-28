from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

import values as val


_THIS_DIR = Path(__file__).resolve().parent
_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "wrist_pitch",
)
_MOTOR_MAPPING = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_yaw": 5,
    "wrist_roll": 6,
    "wrist_pitch": 7,
    "gripper": 8,
}
_CANDIDATE_PATH = "calibration_data/robot_model_calibration_candidate.json"
_ACTIVE_PATH = "calibration_data/robot_model_calibration.json"


@dataclass
class CalibrationSample:
    q_rad: List[float]
    position_m: List[float]
    rpy_rad: List[float]


class RobotModelCalibrationStore:
    def __init__(self, path: str = _CANDIDATE_PATH):
        self.path = _resolve_path(path)
        self.samples: List[CalibrationSample] = []

    def add_sample(self, q_rad: Iterable[float], position_m: Iterable[float], rpy_rad: Iterable[float]) -> None:
        q = list(map(float, q_rad))[:7]
        if len(q) != 7:
            raise ValueError("q_rad must contain 7 arm joints, excluding gripper")
        self.samples.append(CalibrationSample(q, list(map(float, position_m))[:3], list(map(float, rpy_rad))[:3]))

    def save_samples(self, path: Optional[str] = None) -> Path:
        out = self.path if path is None else _resolve_path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"samples": [asdict(s) for s in self.samples]}, indent=2), encoding="utf-8")
        return out

    def save_initial_correction(self, path: Optional[str] = None) -> Path:
        out = self.path if path is None else _resolve_path(path)
        data = _nominal_model_payload(
            source="robot_model_calibrate.RobotModelCalibrationStore.save_initial_correction",
            learn_results=None,
        )
        data["samples"] = [asdict(s) for s in self.samples]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return out


def _resolve_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = _THIS_DIR / p
    return p


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _nominal_link_lengths() -> dict:
    return {
        "base_height": float(getattr(val, "IK_BASE_HEIGHT_M", 0.06)),
        "upper_arm": float(getattr(val, "IK_LINK1_M", 0.115)),
        "forearm": float(getattr(val, "IK_LINK2_M", 0.115)),
        "wrist_to_yaw": float(getattr(val, "IK_WRIST_TO_YAW_M", 0.0)),
        "tool_a": float(getattr(val, "IK_TOOL_A_M", 0.025)),
        "tool_b": float(getattr(val, "IK_TOOL_B_M", 0.025)),
    }


def _summarize_learn_results(learn_results: Optional[dict]) -> dict:
    if not isinstance(learn_results, dict):
        return {}
    derived = learn_results.get("derived_per_motor", {})
    captures = learn_results.get("captures", {})
    return {
        "created_at_unix": learn_results.get("created_at_unix"),
        "motor_names": learn_results.get("motor_names"),
        "motor_ids": learn_results.get("motor_ids"),
        "capture_count": len(captures) if isinstance(captures, dict) else 0,
        "derived_per_motor": derived if isinstance(derived, dict) else {},
    }


def _nominal_model_payload(source: str, learn_results: Optional[dict]) -> dict:
    link_lengths = _nominal_link_lengths()
    tool_offset = [
        float(link_lengths["tool_a"]) + float(link_lengths["tool_b"]),
        0.0,
        0.0,
    ]
    return {
        "joint_zero_offsets_rad": [0.0] * 7,
        "joint_axis_signs": [1.0] * 7,
        "link_lengths_m": link_lengths,
        "tool_xyz_offset_m": tool_offset,
        "collision_capsule_radius_m": float(getattr(val, "IK_COLLISION_CAPSULE_RADIUS_M", 0.018)),
        "source": source,
        "active": False,
        "samples": [],
        "metadata": {
            "created_at_unix": time.time(),
            "joint_names": list(_JOINT_NAMES),
            "motor_mapping": dict(_MOTOR_MAPPING),
            "learn_results_summary": _summarize_learn_results(learn_results),
            "notes": (
                "Candidate generated from robot_learn.py captures. Numeric model "
                "corrections are conservative defaults until measured pose samples "
                "are added and reviewed."
            ),
        },
    }


def create_candidate(from_learn_results: str, output_path: str = _CANDIDATE_PATH) -> Path:
    learn_path = _resolve_path(from_learn_results)
    learn_results = _load_json(learn_path)
    out = _resolve_path(output_path)
    payload = _nominal_model_payload(
        source=f"robot_learn_results:{learn_path}",
        learn_results=learn_results,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def activate_candidate(candidate_path: str, active_path: str = _ACTIVE_PATH) -> Path:
    src = _resolve_path(candidate_path)
    if not src.exists():
        raise FileNotFoundError(f"candidate file not found: {src}")
    data = _load_json(src)
    required = {
        "joint_zero_offsets_rad",
        "joint_axis_signs",
        "link_lengths_m",
        "tool_xyz_offset_m",
        "collision_capsule_radius_m",
        "source",
        "active",
        "samples",
        "metadata",
    }
    missing = sorted(required - set(data.keys()))
    if missing:
        raise ValueError(f"candidate missing required key(s): {', '.join(missing)}")
    dst = _resolve_path(active_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return dst


def main() -> int:
    parser = argparse.ArgumentParser(description="Candidate robot model calibration workflow.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--from-learn-results",
        metavar="PATH",
        help="Create calibration_data/robot_model_calibration_candidate.json from robot_learn.py output.",
    )
    group.add_argument(
        "--activate",
        metavar="PATH",
        help="Explicitly copy a candidate JSON to calibration_data/robot_model_calibration.json.",
    )
    args = parser.parse_args()

    try:
        if args.from_learn_results:
            out = create_candidate(args.from_learn_results)
            print(f"[robot_model_calibrate] created candidate: {out}")
            print("[robot_model_calibrate] did not activate or overwrite robot_model_calibration.json")
            print(f"[robot_model_calibrate] activation required: python3 robot_model_calibrate.py --activate {out}")
            return 0

        dst = activate_candidate(args.activate)
        print(f"[robot_model_calibrate] activated candidate -> {dst}")
        print("[robot_model_calibrate] robot_model_calibration.json was written only by explicit --activate")
        return 0
    except Exception as exc:
        print(f"[robot_model_calibrate] ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
