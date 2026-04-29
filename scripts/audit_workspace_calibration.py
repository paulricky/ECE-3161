#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean, median

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import values as val
from robot_workspace_mapper import POSE_COORDS, REQUIRED_POSES, RobotWorkspaceMapper


EXPECTED_MOTOR_IDS = [1, 2, 3, 4, 5, 6, 7, 8]


def _resolve(path: str | Path) -> Path:
    p = Path(str(path)).expanduser()
    if not p.is_absolute():
        p = ROOT / p
    return p


def _load_json(path: Path) -> tuple[dict | None, str | None]:
    try:
        if not path.exists():
            return None, "missing"
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None, None if isinstance(data, dict) else "not_object"
    except Exception as exc:
        return None, f"load_failed:{exc}"


def _vec3(x):
    try:
        arr = np.asarray(x, dtype=np.float64).reshape(3)
    except Exception:
        return None
    return arr if np.all(np.isfinite(arr)) else None


def _joints7(x):
    if isinstance(x, dict):
        names = (
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_yaw",
            "wrist_roll",
            "wrist_pitch",
        )
        vals = [x.get(name) for name in names]
    elif isinstance(x, list) and len(x) >= 7:
        vals = x[:7]
    else:
        return None
    out = []
    for v in vals:
        try:
            f = float(v)
        except Exception:
            return None
        if not math.isfinite(f):
            return None
        out.append(float(f))
    return out


def _unit(v):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros(3, dtype=np.float64)


def _cos(a, b):
    au = _unit(a)
    bu = _unit(b)
    if np.linalg.norm(au) < 1e-9 or np.linalg.norm(bu) < 1e-9:
        return 0.0
    return float(np.dot(au, bu))


def _legacy_name(name: str) -> str:
    return name if name == "center" else "mirror_" + name


def _workspace_pose_maps(data: dict | None) -> tuple[dict[str, np.ndarray], dict[str, list[float]], bool]:
    xyz = {}
    joints = {}
    legacy = False
    if not isinstance(data, dict):
        return xyz, joints, legacy
    legacy = data.get("calibration_type") == "robot_mirror_workspace_extrema"
    poses = data.get("poses", {})
    if not isinstance(poses, dict):
        return xyz, joints, legacy
    for raw_name, item in poses.items():
        if not isinstance(item, dict):
            continue
        name = str(raw_name)
        if name.startswith("mirror_"):
            name = name.removeprefix("mirror_")
        v = _vec3(item.get("fk_xyz_m"))
        if v is not None:
            xyz[name] = v
        j = _joints7(item.get("joints_rad"))
        if j is not None:
            joints[name] = j
    return xyz, joints, legacy


def _add_issue(report: dict, level: str, message: str, redo_pose: str | None = None) -> None:
    report[level].append(message)
    if redo_pose:
        report["redo_poses"].setdefault(redo_pose, []).append(message)


def _analyze_schema(report: dict, path: Path, data: dict | None, load_error: str | None, xyz: dict, joints: dict) -> None:
    section = {
        "path": str(path),
        "exists": path.exists(),
        "loaded": isinstance(data, dict),
        "load_error": load_error,
        "calibration_type": None if not isinstance(data, dict) else data.get("calibration_type"),
        "required_present": [p for p in REQUIRED_POSES if p in xyz],
        "required_missing": [p for p in REQUIRED_POSES if p not in xyz],
        "optional_present": [p for p in POSE_COORDS if p not in REQUIRED_POSES and p in xyz],
    }
    report["sections"]["file_presence_schema"] = section
    if not isinstance(data, dict):
        _add_issue(report, "critical", f"Workspace calibration missing or unreadable: {path}")
        return
    if data.get("calibration_type") not in {"robot_workspace_extrema", "robot_mirror_workspace_extrema"}:
        _add_issue(report, "critical", f"Unsupported calibration_type: {data.get('calibration_type')}")
    missing = section["required_missing"]
    if missing:
        _add_issue(report, "critical", "Missing required workspace poses: " + ", ".join(missing))
    motor_ids = data.get("motor_ids", [])
    if list(motor_ids) and list(motor_ids) != EXPECTED_MOTOR_IDS:
        _add_issue(report, "critical", f"Motor IDs are {motor_ids}; expected {EXPECTED_MOTOR_IDS}.")
    for name in REQUIRED_POSES:
        if name in xyz and name not in joints:
            _add_issue(report, "warnings", f"Pose {name} has FK xyz but no valid 7-joint IK seed.", name)


def _analyze_directions(report: dict, xyz: dict) -> None:
    section = {"poses": {}, "opposites": {}, "coupling": {}}
    report["sections"]["workspace_direction_quality"] = section
    if "center" not in xyz:
        return
    center = xyz["center"]
    deltas = {name: v - center for name, v in xyz.items()}
    for name in REQUIRED_POSES:
        if name not in deltas:
            continue
        norm = float(np.linalg.norm(deltas[name]))
        section["poses"][name] = {"delta_xyz_m": deltas[name].tolist(), "delta_norm_m": norm}
        if name != "center" and norm < 0.015:
            _add_issue(report, "warnings", f"Pose {name} is only {norm:.3f} m from center; redo it farther from center.", name)
    pairs = [
        ("horizontal", "left", "right"),
        ("vertical", "down", "up"),
        ("depth", "far", "near"),
    ]
    axes = {}
    for axis, neg, pos in pairs:
        if neg in deltas and pos in deltas:
            c = _cos(deltas[neg], deltas[pos])
            section["opposites"][axis] = {"negative_pose": neg, "positive_pose": pos, "cosine": c}
            axes[axis] = _unit(xyz[pos] - xyz[neg])
            if c > -0.25:
                _add_issue(report, "warnings", f"{neg}/{pos} are not mostly opposite (cos={c:.2f}); redo one or both poses.")
    for axis, neg, pos in pairs:
        for pose in (neg, pos):
            if pose not in deltas:
                continue
            norm = max(float(np.linalg.norm(deltas[pose])), 1e-9)
            coupling = {}
            for other_axis, unit in axes.items():
                if other_axis == axis or np.linalg.norm(unit) < 1e-9:
                    continue
                coupling[other_axis] = abs(float(np.dot(deltas[pose], unit))) / norm
            section["coupling"][pose] = coupling
            if any(v > 0.65 for v in coupling.values()):
                _add_issue(report, "warnings", f"Pose {pose} has strong off-axis coupling; redo if hand mapping feels skewed.", pose)


def _analyze_residuals(report: dict, xyz: dict) -> None:
    section = {"poses": {}, "summary": {}}
    report["sections"]["optional_residual_quality"] = section
    if not all(p in xyz for p in REQUIRED_POSES):
        return
    center = xyz["center"]

    def base(coord):
        h, v, d = coord
        out = center.copy()
        out += abs(h) * ((xyz["left"] if h < 0.0 else xyz["right"]) - center)
        out += abs(v) * ((xyz["down"] if v < 0.0 else xyz["up"]) - center)
        out += abs(d) * ((xyz["far"] if d < 0.0 else xyz["near"]) - center)
        return out

    residual_norms = []
    max_allowed = abs(float(getattr(val, "ROBOT_WORKSPACE_RESIDUAL_MAX_M", 0.030)))
    for name, coord in POSE_COORDS.items():
        if name in REQUIRED_POSES or name not in xyz:
            continue
        residual = xyz[name] - base(np.asarray(coord, dtype=np.float64))
        norm = float(np.linalg.norm(residual))
        residual_norms.append(norm)
        section["poses"][name] = {
            "residual_xyz_m": residual.tolist(),
            "residual_norm_m": norm,
            "exceeds_configured_clamp": bool(norm > max_allowed),
        }
        if norm > 2.0 * max_allowed:
            _add_issue(report, "warnings", f"Optional pose {name} residual {norm:.3f} m exceeds clamp by more than 2x; redo it or fix base extrema first.", name)
    if residual_norms:
        section["summary"] = {
            "mean_residual_m": float(mean(residual_norms)),
            "median_residual_m": float(median(residual_norms)),
            "max_residual_m": float(max(residual_norms)),
            "configured_residual_max_m": max_allowed,
        }


def _analyze_legacy_bounds(report: dict, xyz: dict) -> None:
    if not xyz:
        return
    bounds = {
        "x": (
            float(getattr(val, "HAND_TARGET_X_MIN_M", getattr(val, "WORKSPACE_X_MIN", -0.12))),
            float(getattr(val, "HAND_TARGET_X_MAX_M", getattr(val, "WORKSPACE_X_MAX", 0.12))),
        ),
        "y": (
            float(getattr(val, "HAND_TARGET_Y_MIN_M", getattr(val, "WORKSPACE_Y_MIN", 0.10))),
            float(getattr(val, "HAND_TARGET_Y_MAX_M", getattr(val, "WORKSPACE_Y_MAX", 0.22))),
        ),
        "z": (
            float(getattr(val, "HAND_TARGET_Z_MIN_M", getattr(val, "WORKSPACE_Z_MIN", 0.00))),
            float(getattr(val, "HAND_TARGET_Z_MAX_M", getattr(val, "WORKSPACE_Z_MAX", 0.22))),
        ),
    }
    clipped = []
    for name, v in xyz.items():
        for i, axis in enumerate(("x", "y", "z")):
            lo, hi = bounds[axis]
            if hi < lo:
                lo, hi = hi, lo
            if float(v[i]) < lo or float(v[i]) > hi:
                clipped.append({"pose": name, "axis": axis, "value": float(v[i]), "legacy_min": lo, "legacy_max": hi})
    report["sections"]["runtime_clamp_quality"] = {
        "legacy_bounds": bounds,
        "legacy_bounds_would_clip": clipped,
        "workspace_bounds_clamp_enabled": bool(getattr(val, "ROBOT_WORKSPACE_CLAMP_TO_RECORDED_BOUNDS", True)),
        "workspace_bounds_margin_m": float(getattr(val, "ROBOT_WORKSPACE_CLAMP_MARGIN_M", 0.020)),
    }
    if clipped:
        _add_issue(report, "warnings", f"Legacy HAND_TARGET/WORKSPACE bounds would clip {len(clipped)} recorded pose coordinates; runtime should use calibrated workspace bounds.")


def _analyze_mapper(report: dict, path: Path) -> None:
    mapper = RobotWorkspaceMapper(str(path))
    if not mapper.loaded:
        _add_issue(report, "critical", f"RobotWorkspaceMapper could not load calibration: {mapper.error}")
        return
    anchors = mapper.evaluate_anchor_errors()
    report["sections"]["anchor_reproduction"] = anchors
    warn = 0.015
    crit = 0.030
    for name, item in anchors.items():
        err = float(item.get("final_error_m", 0.0))
        if err > crit:
            _add_issue(report, "critical", f"Anchor {name} reproduces with {err:.3f} m error; redo pose or inspect mapping.", name)
        elif err > warn:
            _add_issue(report, "warnings", f"Anchor {name} reproduces with {err:.3f} m error; verify pose.", name)


def _verdict(report: dict) -> str:
    if report["critical"]:
        return "CRITICAL"
    if len(report["warnings"]) >= 5:
        return "BAD"
    if report["warnings"]:
        return "OK"
    return "GOOD"


def _write_reports(report: dict, json_out: Path, md_out: Path) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    critical_lines = [f"- {x}" for x in report["critical"]] or ["- None"]
    warning_lines = [f"- {x}" for x in report["warnings"]] or ["- None"]
    lines = [
        "# Workspace Calibration Audit",
        "",
        f"Verdict: **{report['verdict']}**",
        "",
        "## Critical",
        *critical_lines,
        "",
        "## Warnings",
        *warning_lines,
        "",
        "## Recommended Redo Poses",
    ]
    if report["redo_poses"]:
        for pose, items in sorted(report["redo_poses"].items()):
            lines.append(f"- {pose}: " + "; ".join(items))
    else:
        lines.append("- None")
    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit robot workspace extrema calibration without hardware.")
    parser.add_argument("--robot-file", default=getattr(val, "ROBOT_WORKSPACE_CALIBRATION_FILE", "calibration_data/robot_workspace_extrema_calibration.json"))
    parser.add_argument("--json-out", default="calibration_data/workspace_calibration_audit.json")
    parser.add_argument("--report-out", default="calibration_data/workspace_calibration_audit.md")
    parser.add_argument("--fail-on-critical", action="store_true")
    args = parser.parse_args()

    path = _resolve(args.robot_file)
    data, load_error = _load_json(path)
    if not isinstance(data, dict) and path == _resolve(getattr(val, "ROBOT_WORKSPACE_CALIBRATION_FILE", "")):
        legacy_path = _resolve(getattr(val, "ROBOT_WORKSPACE_LEGACY_MIRROR_CALIBRATION_FILE", "calibration_data/robot_mirror_workspace_calibration.json"))
        legacy_data, legacy_error = _load_json(legacy_path)
        if isinstance(legacy_data, dict):
            path, data, load_error = legacy_path, legacy_data, legacy_error

    xyz, joints, legacy = _workspace_pose_maps(data)
    report = {
        "verdict": "CRITICAL",
        "critical": [],
        "warnings": [],
        "redo_poses": {},
        "sections": {},
        "inputs": {"robot_file": str(path), "legacy_schema": bool(legacy)},
    }
    _analyze_schema(report, path, data, load_error, xyz, joints)
    _analyze_directions(report, xyz)
    _analyze_residuals(report, xyz)
    _analyze_legacy_bounds(report, xyz)
    _analyze_mapper(report, path)
    report["verdict"] = _verdict(report)

    json_out = _resolve(args.json_out)
    md_out = _resolve(args.report_out)
    _write_reports(report, json_out, md_out)
    print(f"[audit_workspace] verdict={report['verdict']}")
    print(f"[audit_workspace] critical={len(report['critical'])} warnings={len(report['warnings'])}")
    if report["redo_poses"]:
        print("[audit_workspace] redo poses: " + ", ".join(sorted(report["redo_poses"])))
    print(f"[audit_workspace] wrote {json_out}")
    print(f"[audit_workspace] wrote {md_out}")
    return 1 if args.fail_on_critical and report["critical"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
