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
from robot_mirror_mapper import POSE_COORDS, REQUIRED_POSES, RobotMirrorWorkspaceMapper


EXPECTED_MOTOR_IDS = [1, 2, 3, 4, 5, 6, 7, 8]
ARM_JOINT_LIMITS = [
    ("shoulder_pan", -math.pi, math.pi),
    ("shoulder_lift", -math.pi, math.pi),
    ("elbow_flex", -math.pi, math.pi),
    ("wrist_flex", -math.pi, math.pi),
    ("wrist_yaw", -math.pi, math.pi),
    ("wrist_roll", -math.pi, math.pi),
    ("wrist_pitch", -math.pi, math.pi),
]


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


def _finite_float(x, default=None):
    try:
        f = float(x)
    except Exception:
        return default
    return f if math.isfinite(f) else default


def _vec3(x):
    try:
        arr = np.asarray(x, dtype=np.float64).reshape(3)
    except Exception:
        return None
    return arr if np.all(np.isfinite(arr)) else None


def _joints7(x):
    if isinstance(x, dict):
        vals = [x.get(name) for name, _lo, _hi in ARM_JOINT_LIMITS]
    elif isinstance(x, list) and len(x) >= 7:
        vals = x[:7]
    else:
        return None
    out = []
    for v in vals:
        f = _finite_float(v)
        if f is None:
            return None
        out.append(float(f))
    return out


def _unit(v):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros(3, dtype=np.float64)


def _cos(a, b):
    a_u = _unit(a)
    b_u = _unit(b)
    if np.linalg.norm(a_u) < 1e-9 or np.linalg.norm(b_u) < 1e-9:
        return 0.0
    return float(np.dot(a_u, b_u))


def _hand_depth_value(hand: dict) -> tuple[float | None, str]:
    for key in ("depth_norm_raw", "depth_norm", "depth_norm_filtered"):
        f = _finite_float(hand.get(key))
        if f is not None:
            return max(0.0, min(1.0, f)), key
    size = _finite_float(hand.get("hand_size_norm"))
    near_size = _finite_float(getattr(val, "HAND_MONOCULAR_NEAR_SIZE_NORM", None), 0.32)
    far_size = _finite_float(getattr(val, "HAND_MONOCULAR_FAR_SIZE_NORM", None), 0.12)
    if size is not None and near_size is not None and far_size is not None and abs(near_size - far_size) > 1e-6:
        return max(0.0, min(1.0, (size - far_size) / (near_size - far_size))), "hand_size_norm"
    return None, "missing"


def _robot_pose_maps(robot_data: dict | None) -> tuple[dict[str, np.ndarray], dict[str, list[float]]]:
    xyz = {}
    joints = {}
    if not isinstance(robot_data, dict):
        return xyz, joints
    poses = robot_data.get("poses", {})
    if not isinstance(poses, dict):
        return xyz, joints
    for name, item in poses.items():
        if not isinstance(item, dict):
            continue
        v = _vec3(item.get("fk_xyz_m"))
        if v is not None:
            xyz[str(name)] = v
        j = _joints7(item.get("joints_rad"))
        if j is not None:
            joints[str(name)] = j
    return xyz, joints


def _hand_pose_maps(hand_data: dict | None) -> tuple[dict[str, dict], dict[str, np.ndarray], dict[str, str]]:
    raw = {}
    coords = {}
    sources = {}
    if not isinstance(hand_data, dict):
        return raw, coords, sources
    poses = hand_data.get("poses", {})
    if not isinstance(poses, dict):
        return raw, coords, sources
    for name, item in poses.items():
        if not isinstance(item, dict):
            continue
        hand = item.get("hand", item)
        if not isinstance(hand, dict):
            continue
        x = _finite_float(hand.get("x_norm"))
        y = _finite_float(hand.get("y_norm"))
        d, src = _hand_depth_value(hand)
        if x is None or y is None or d is None:
            continue
        raw[str(name)] = hand
        coords[str(name)] = np.array([max(0.0, min(1.0, x)), max(0.0, min(1.0, y)), d], dtype=np.float64)
        sources[str(name)] = src
    return raw, coords, sources


def _add_issue(report: dict, level: str, message: str, redo_pose: str | None = None) -> None:
    report[level].append(message)
    if redo_pose:
        report["redo_poses"].setdefault(redo_pose, []).append(message)


def _analyze_files(report, robot_path, hand_path, depth_path, robot_data, hand_data, depth_data):
    files = {
        "robot": (robot_path, robot_data),
        "hand": (hand_path, hand_data),
        "depth": (depth_path, depth_data),
    }
    section = {}
    for name, (path, data) in files.items():
        section[name] = {"path": str(path), "exists": path.exists(), "loaded": isinstance(data, dict)}
    report["sections"]["file_presence_schema"] = section
    if not isinstance(robot_data, dict):
        _add_issue(report, "critical", f"Robot mirror calibration missing or unreadable: {robot_path}")
    if not isinstance(hand_data, dict):
        _add_issue(report, "warnings", f"Hand mirror calibration missing or unreadable: {hand_path}; runtime will use ideal hand extrema if robot calibration exists.")
    if not isinstance(depth_data, dict):
        _add_issue(report, "warnings", f"Hand depth calibration missing or unreadable: {depth_path}; depth ordering may be less reliable.")


def _analyze_robot(report, robot_data, robot_xyz):
    section = {"poses": {}, "opposites": {}, "coupling": {}}
    missing = [p for p in REQUIRED_POSES if p not in robot_xyz]
    if missing:
        _add_issue(report, "critical", "Robot calibration missing required poses: " + ", ".join(missing))
    motor_ids = robot_data.get("motor_ids", []) if isinstance(robot_data, dict) else []
    if list(motor_ids) != EXPECTED_MOTOR_IDS:
        _add_issue(report, "critical", f"Robot motor IDs are {motor_ids}; expected {EXPECTED_MOTOR_IDS}.")
    if "center" not in robot_xyz:
        report["sections"]["robot_pose_direction_quality"] = section
        return
    center = robot_xyz["center"]
    deltas = {name: xyz - center for name, xyz in robot_xyz.items()}
    for name in REQUIRED_POSES:
        if name in deltas:
            norm = float(np.linalg.norm(deltas[name]))
            section["poses"][name] = {"delta_xyz_m": deltas[name].tolist(), "delta_norm_m": norm}
            if name != "center" and norm < 0.015:
                _add_issue(report, "warnings", f"Robot pose {name} is only {norm:.3f} m from center; redo it farther from center.", name)
    pairs = [
        ("horizontal", "mirror_left", "mirror_right"),
        ("vertical", "mirror_down", "mirror_up"),
        ("depth", "mirror_far", "mirror_near"),
    ]
    axes = {}
    for axis, neg, pos in pairs:
        if neg in robot_xyz and pos in robot_xyz:
            cos = _cos(deltas[neg], deltas[pos])
            section["opposites"][axis] = {"negative_pose": neg, "positive_pose": pos, "cosine": cos}
            axes[axis] = _unit(robot_xyz[pos] - robot_xyz[neg])
            if cos > -0.25:
                _add_issue(report, "warnings", f"Robot {neg}/{pos} are not mostly opposite (cos={cos:.2f}); redo one or both poses.")
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
                _add_issue(report, "warnings", f"Robot pose {pose} has strong motion on non-target axes: {coupling}; redo with other axes closer to center.", pose)
    report["sections"]["robot_pose_direction_quality"] = section


def _analyze_hand(report, hand_data, hand_raw, hand_coords, depth_sources):
    section = {"poses": {}, "ordering": {}, "frame_preprocessing": {}}
    if not isinstance(hand_data, dict):
        report["sections"]["hand_pose_direction_quality"] = section
        return
    missing = [p for p in REQUIRED_POSES if p not in hand_coords]
    if missing:
        _add_issue(report, "critical", "Hand mirror calibration missing required poses: " + ", ".join(missing))
    prep = hand_data.get("frame_preprocessing", {})
    runtime_flip = bool(getattr(val, "HANDTRACKING_FLIP_CAMERA_FRAME", True))
    saved_flip = bool(prep.get("flipped_horizontal", False)) if isinstance(prep, dict) else False
    matches_runtime = bool(prep.get("matches_runtime", False)) if isinstance(prep, dict) else False
    section["frame_preprocessing"] = {
        "saved_flipped_horizontal": saved_flip,
        "runtime_flipped_horizontal": runtime_flip,
        "matches_runtime": matches_runtime,
    }
    if not isinstance(prep, dict) or saved_flip != runtime_flip or not matches_runtime:
        _add_issue(report, "warnings", "Hand mirror calibration frame preprocessing does not match runtime flip setting; re-run camera_calibrate.py --hand-mirror.")
    for name, coord in hand_coords.items():
        hand = hand_raw.get(name, {})
        std = hand_data.get("poses", {}).get(name, {}).get("std", {}) if isinstance(hand_data.get("poses", {}), dict) else {}
        section["poses"][name] = {
            "x_norm": float(coord[0]),
            "y_norm": float(coord[1]),
            "depth_norm": float(coord[2]),
            "depth_source": depth_sources.get(name, "unknown"),
            "hand_size_norm": _finite_float(hand.get("hand_size_norm"), 0.0),
            "std": std if isinstance(std, dict) else {},
        }
        if isinstance(std, dict):
            for key in ("x_norm_std", "y_norm_std", "depth_norm_std"):
                v = _finite_float(std.get(key))
                if v is not None and v > 0.035:
                    _add_issue(report, "warnings", f"Hand pose {name} has unstable {key}={v:.4f}; redo with steadier hand.", name)
    required_present = all(p in hand_coords for p in REQUIRED_POSES)
    if required_present:
        left, center, right = hand_coords["mirror_left"][0], hand_coords["center"][0], hand_coords["mirror_right"][0]
        up, c_y, down = hand_coords["mirror_up"][1], hand_coords["center"][1], hand_coords["mirror_down"][1]
        near, c_d, far = hand_coords["mirror_near"][2], hand_coords["center"][2], hand_coords["mirror_far"][2]
        section["ordering"] = {
            "x_left_center_right": [float(left), float(center), float(right)],
            "y_up_center_down": [float(up), float(c_y), float(down)],
            "depth_near_center_far": [float(near), float(c_d), float(far)],
        }
        if not (left < center < right):
            _add_issue(report, "warnings", "Hand x order is not left < center < right. Re-run hand mirror calibration after frame flip fix.")
        if not (up < c_y < down):
            _add_issue(report, "warnings", "Hand y order is not up/top < center < down/bottom. Redo mirror_up/mirror_down.")
        if not (near > c_d > far):
            _add_issue(report, "warnings", "Hand depth_norm is not near > center > far; redo mirror_near/mirror_far.", "mirror_near")
            _add_issue(report, "warnings", "Hand depth_norm is not near > center > far; redo mirror_near/mirror_far.", "mirror_far")
        if abs(right - left) < 0.20:
            _add_issue(report, "warnings", "Hand left/right range is small; redo mirror_left/mirror_right farther apart.")
        if abs(down - up) < 0.20:
            _add_issue(report, "warnings", "Hand up/down range is small; redo mirror_up/mirror_down farther apart.")
        if abs(near - far) < 0.12:
            _add_issue(report, "warnings", "Hand near/far depth range is small; redo mirror_near/mirror_far.")
        sizes = {p: _finite_float(hand_raw.get(p, {}).get("hand_size_norm")) for p in ("mirror_near", "center", "mirror_far")}
        if all(v is not None for v in sizes.values()) and not (sizes["mirror_near"] > sizes["center"] > sizes["mirror_far"]):
            _add_issue(report, "warnings", "Hand size contradicts near/center/far order; redo mirror_near/mirror_far.")
        if any(src != "depth_norm_raw" for src in depth_sources.values()):
            _add_issue(report, "warnings", "Hand calibration lacks depth_norm_raw for some poses; re-run camera_calibrate.py --hand-mirror to record raw depth anchors.")
    report["sections"]["hand_pose_direction_quality"] = section


def _analyze_pairing(report, robot_xyz, hand_coords):
    required_pairs = [p for p in REQUIRED_POSES if p in robot_xyz and p in hand_coords]
    optional_pairs = [p for p in POSE_COORDS if p not in REQUIRED_POSES and p in robot_xyz and p in hand_coords]
    section = {
        "required_pairs": required_pairs,
        "optional_pairs": optional_pairs,
        "missing_robot_for_hand": sorted(set(hand_coords) - set(robot_xyz)),
        "missing_hand_for_robot": sorted(set(robot_xyz) - set(hand_coords)),
        "paired_axis_blend_ready": len(required_pairs) == len(REQUIRED_POSES),
        "knn_residual_ready": bool(optional_pairs),
        "rbf_residual_ready": len(optional_pairs) + len(required_pairs) >= int(getattr(val, "ROBOT_MIRROR_RBF_MIN_SAMPLES", 8)),
    }
    if len(required_pairs) != len(REQUIRED_POSES):
        _add_issue(report, "critical", "Required robot/hand pose pairs are incomplete.")
    if not optional_pairs:
        _add_issue(report, "warnings", "No optional paired poses available; nonlinear residual correction has no useful optional anchors.")
    report["sections"]["pairing_quality"] = section


def _analyze_mapper(report, robot_path, hand_path):
    section = {"loaded": False, "anchor_errors": {}, "optional_residuals": {}, "bounds": {}}
    mapper = RobotMirrorWorkspaceMapper(str(robot_path), str(hand_path))
    section["loaded"] = bool(mapper.loaded)
    section["robot_error"] = mapper.error
    section["hand_loaded"] = bool(mapper.hand_loaded)
    section["hand_error"] = mapper.hand_error
    if not mapper.loaded:
        _add_issue(report, "critical", f"RobotMirrorWorkspaceMapper could not load: {mapper.error}")
        report["sections"]["anchor_reproduction_test"] = section
        return None
    warn_err = float(getattr(val, "ROBOT_MIRROR_ANCHOR_WARN_ERR_M", 0.015))
    crit_err = float(getattr(val, "ROBOT_MIRROR_ANCHOR_CRITICAL_ERR_M", 0.030))
    anchors = mapper.evaluate_anchor_errors()
    section["anchor_errors"] = anchors
    for name, item in anchors.items():
        err = float(item.get("final_error_m", 0.0))
        if err > crit_err:
            _add_issue(report, "critical", f"Required anchor {name} maps with {err:.3f} m error; redo paired calibration for this pose.", name)
        elif err > warn_err:
            _add_issue(report, "warnings", f"Required anchor {name} maps with {err:.3f} m error; redo this pose if motion feels wrong.", name)
    max_residual = abs(float(getattr(val, "ROBOT_MIRROR_RESIDUAL_MAX_M", 0.030)))
    residuals = []
    for name in sorted(set(mapper.pose_xyz) - set(REQUIRED_POSES)):
        x = mapper._pose_centered_coordinate(name)
        if x is None:
            continue
        base = mapper._axis_blend_from_centered(x)
        recorded = mapper.pose_xyz[name]
        residual = recorded - base
        norm = float(np.linalg.norm(residual))
        residuals.append(norm)
        section["optional_residuals"][name] = {
            "residual_xyz_m": residual.tolist(),
            "residual_norm_m": norm,
            "exceeds_configured_limit": bool(norm > max_residual),
        }
        if norm > 2.0 * max_residual:
            _add_issue(report, "warnings", f"Optional pose {name} residual {norm:.3f} m is more than 2x clamp; redo it or improve required extrema first.", name)
    if residuals:
        section["optional_residual_summary"] = {
            "mean_m": float(mean(residuals)),
            "median_m": float(median(residuals)),
            "max_m": float(max(residuals)),
        }
    section["bounds"] = {
        "calibrated_min_m": mapper.xyz_min.tolist(),
        "calibrated_max_m": mapper.xyz_max.tolist(),
        "clamp_to_calibrated_bounds": bool(getattr(val, "ROBOT_MIRROR_CLAMP_TO_CALIBRATED_BOUNDS", True)),
    }
    report["sections"]["anchor_reproduction_test"] = section
    return mapper


def _analyze_runtime_clamp(report, robot_xyz, mapper):
    legacy_min = np.array([
        float(getattr(val, "HAND_TARGET_X_MIN_M", getattr(val, "WORKSPACE_X_MIN", -0.12))),
        float(getattr(val, "HAND_TARGET_Y_MIN_M", getattr(val, "WORKSPACE_Y_MIN", 0.10))),
        float(getattr(val, "HAND_TARGET_Z_MIN_M", getattr(val, "WORKSPACE_Z_MIN", 0.00))),
    ], dtype=np.float64)
    legacy_max = np.array([
        float(getattr(val, "HAND_TARGET_X_MAX_M", getattr(val, "WORKSPACE_X_MAX", 0.12))),
        float(getattr(val, "HAND_TARGET_Y_MAX_M", getattr(val, "WORKSPACE_Y_MAX", 0.22))),
        float(getattr(val, "HAND_TARGET_Z_MAX_M", getattr(val, "WORKSPACE_Z_MAX", 0.22))),
    ], dtype=np.float64)
    clipped = {}
    for name, xyz in robot_xyz.items():
        below = xyz < legacy_min
        above = xyz > legacy_max
        if bool(np.any(below | above)):
            clipped[name] = {
                "xyz_m": xyz.tolist(),
                "below_axes": [axis for axis, flag in zip(("x", "y", "z"), below, strict=True) if bool(flag)],
                "above_axes": [axis for axis, flag in zip(("x", "y", "z"), above, strict=True) if bool(flag)],
            }
    section = {
        "legacy_bounds_min_m": legacy_min.tolist(),
        "legacy_bounds_max_m": legacy_max.tolist(),
        "legacy_bounds_would_clip": clipped,
        "mapper_bounds_min_m": mapper.xyz_min.tolist() if mapper is not None and mapper.loaded else None,
        "mapper_bounds_max_m": mapper.xyz_max.tolist() if mapper is not None and mapper.loaded else None,
    }
    if clipped:
        _add_issue(report, "warnings", f"Legacy HAND_TARGET/WORKSPACE bounds would clip {len(clipped)} recorded mirror poses; runtime mapper calibrated bounds should be used.")
    report["sections"]["runtime_clamp_quality"] = section


def _analyze_ik_seeds(report, robot_joints):
    section = {}
    for name, joints in robot_joints.items():
        finite = len(joints) == 7 and all(math.isfinite(float(x)) for x in joints)
        near_limits = []
        for value, (joint_name, lo, hi) in zip(joints, ARM_JOINT_LIMITS, strict=True):
            margin = min(abs(float(value) - lo), abs(hi - float(value)))
            if margin < 0.08:
                near_limits.append(joint_name)
        section[name] = {"finite": finite, "length": len(joints), "near_nominal_limits": near_limits}
        if not finite:
            _add_issue(report, "warnings", f"IK seed for {name} is invalid or not length 7.", name)
        if near_limits:
            _add_issue(report, "warnings", f"IK seed for {name} is near nominal joint limits: {near_limits}.", name)
    for a, b in [("mirror_left", "mirror_right"), ("mirror_up", "mirror_down"), ("mirror_near", "mirror_far")]:
        if a in robot_joints and b in robot_joints:
            jump = float(np.linalg.norm(np.asarray(robot_joints[a]) - np.asarray(robot_joints[b])))
            section[f"{a}<->{b}_jump_rad"] = jump
            if jump > 7.0:
                _add_issue(report, "warnings", f"Large IK seed discontinuity between {a} and {b}: {jump:.2f} rad.")
    report["sections"]["ik_seed_quality"] = section


def _finalize(report):
    if report["critical"]:
        grade = "CRITICAL"
    elif len(report["warnings"]) >= 8:
        grade = "BAD"
    elif report["warnings"]:
        grade = "OK"
    else:
        grade = "GOOD"
    report["grade"] = grade
    actions = []
    for pose, issues in sorted(report["redo_poses"].items()):
        actions.append(f"Redo {pose}: " + " ".join(issues[:2]))
    if report["critical"]:
        actions.append("Fix critical schema/pairing issues before relying on runtime mirror mapping.")
    if not actions and report["warnings"]:
        actions.append("Review warnings; calibration is usable but can be improved.")
    if not actions:
        actions.append("Calibration looks ready for runtime use.")
    report["action_items"] = actions


def _markdown(report):
    lines = ["# Mirror Calibration Audit", "", f"Final grade: **{report['grade']}**", ""]
    if report["critical"]:
        lines += ["## Critical", *[f"- {x}" for x in report["critical"]], ""]
    if report["warnings"]:
        lines += ["## Warnings", *[f"- {x}" for x in report["warnings"]], ""]
    lines += ["## Action Items", *[f"- {x}" for x in report["action_items"]], ""]
    anchors = report["sections"].get("anchor_reproduction_test", {}).get("anchor_errors", {})
    if anchors:
        lines += ["## Required Anchor Errors"]
        for name, item in anchors.items():
            lines.append(f"- {name}: base={float(item['base_error_m']):.4f} m, final={float(item['final_error_m']):.4f} m, clamped={item.get('target_clamped', False)}")
        lines.append("")
    residuals = report["sections"].get("anchor_reproduction_test", {}).get("optional_residuals", {})
    if residuals:
        lines += ["## Optional Residuals"]
        for name, item in sorted(residuals.items(), key=lambda kv: kv[1]["residual_norm_m"], reverse=True)[:10]:
            lines.append(f"- {name}: {float(item['residual_norm_m']):.4f} m")
        lines.append("")
    clipped = report["sections"].get("runtime_clamp_quality", {}).get("legacy_bounds_would_clip", {})
    lines += ["## Runtime Clamp Check", f"- Legacy bounds would clip {len(clipped)} recorded mirror poses.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline quality audit for paired hand/robot mirror calibration.")
    parser.add_argument("--robot-file", default=getattr(val, "ROBOT_MIRROR_WORKSPACE_CALIBRATION_FILE", "calibration_data/robot_mirror_workspace_calibration.json"))
    parser.add_argument("--hand-file", default=getattr(val, "HAND_MIRROR_POSITION_CALIBRATION_FILE", "calibration_data/hand_mirror_position_calibration.json"))
    parser.add_argument("--depth-file", default=getattr(val, "HAND_MONOCULAR_DEPTH_CALIBRATION_FILE", "calibration_data/hand_depth_calibration.json"))
    parser.add_argument("--json-out", default=getattr(val, "ROBOT_MIRROR_AUDIT_JSON_FILE", "calibration_data/mirror_calibration_audit.json"))
    parser.add_argument("--report-out", default=getattr(val, "ROBOT_MIRROR_AUDIT_REPORT_FILE", "calibration_data/mirror_calibration_audit.md"))
    parser.add_argument("--fail-on-critical", action="store_true")
    args = parser.parse_args()

    robot_path = _resolve(args.robot_file)
    hand_path = _resolve(args.hand_file)
    depth_path = _resolve(args.depth_file)
    json_out = _resolve(args.json_out)
    report_out = _resolve(args.report_out)
    robot_data, _robot_err = _load_json(robot_path)
    hand_data, _hand_err = _load_json(hand_path)
    depth_data, _depth_err = _load_json(depth_path)
    report = {
        "grade": "UNKNOWN",
        "critical": [],
        "warnings": [],
        "redo_poses": {},
        "action_items": [],
        "sections": {},
        "files": {
            "robot": str(robot_path),
            "hand": str(hand_path),
            "depth": str(depth_path),
        },
    }

    _analyze_files(report, robot_path, hand_path, depth_path, robot_data, hand_data, depth_data)
    robot_xyz, robot_joints = _robot_pose_maps(robot_data)
    hand_raw, hand_coords, depth_sources = _hand_pose_maps(hand_data)
    _analyze_robot(report, robot_data or {}, robot_xyz)
    _analyze_hand(report, hand_data, hand_raw, hand_coords, depth_sources)
    _analyze_pairing(report, robot_xyz, hand_coords)
    mapper = _analyze_mapper(report, robot_path, hand_path)
    _analyze_runtime_clamp(report, robot_xyz, mapper)
    _analyze_ik_seeds(report, robot_joints)
    _finalize(report)

    json_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    with json_out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with report_out.open("w", encoding="utf-8") as f:
        f.write(_markdown(report))

    print(f"[mirror-audit] grade: {report['grade']}")
    print(f"[mirror-audit] critical: {len(report['critical'])} warnings: {len(report['warnings'])}")
    for item in report["action_items"][:8]:
        print(f"[mirror-audit] action: {item}")
    print(f"[mirror-audit] wrote JSON: {json_out}")
    print(f"[mirror-audit] wrote report: {report_out}")
    return 1 if args.fail_on_critical and report["critical"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
