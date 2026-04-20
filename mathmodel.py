from __future__ import annotations

import json
import math
import os
from typing import Any

import numpy as np

import values as val


def dist(a, b) -> float:
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    return math.hypot(ax - bx, ay - by)


def hand_center_xy(hand_lms):
    lm = hand_lms.landmark
    pts = [
        (lm[0].x, lm[0].y),
        (lm[5].x, lm[5].y),
        (lm[9].x, lm[9].y),
        (lm[13].x, lm[13].y),
        (lm[17].x, lm[17].y),
    ]
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return (cx, cy)


def _clip(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _get_value(name: str, default: float) -> float:
    return float(getattr(val, name, default))


def _get_joint_limits():
    return {
        "shoulder_pan": (
            _get_value("BASE_PAN_MIN", -1.2),
            _get_value("BASE_PAN_MAX", 1.2),
        ),
        "shoulder_lift": (
            _get_value("SHOULDER_LIFT_MIN", -0.8),
            _get_value("SHOULDER_LIFT_MAX", 1.0),
        ),
        "elbow_flex": (
            _get_value("ELBOW_MIN", -0.9),
            _get_value("ELBOW_MAX", 1.2),
        ),
        "wrist_flex": (
            _get_value("WRIST_FLEX_MIN", -0.9),
            _get_value("WRIST_FLEX_MAX", 0.9),
        ),
        "wrist_roll": (
            _get_value("WRIST_ROLL_MIN", -1.5),
            _get_value("WRIST_ROLL_MAX", 1.5),
        ),
    }


def _default_lerobot_calibration_path() -> str:
    explicit = getattr(val, "LEROBOT_CALIBRATION_FILE", "").strip()
    if explicit:
        return explicit

    robot_id = getattr(val, "REAL_ROBOT_ID", "my_awesome_follower_arm")
    home = os.path.expanduser("~")
    return os.path.join(
        home,
        ".cache",
        "huggingface",
        "lerobot",
        "calibration",
        "robots",
        "so101_follower",
        f"{robot_id}.json",
    )


def _load_lerobot_calibration_file(path: str | None = None):
    p = path or _default_lerobot_calibration_path()
    if not p or not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _extract_numeric(obj: Any, *keys: str):
    if not isinstance(obj, dict):
        return None
    for k in keys:
        if k in obj and isinstance(obj[k], (int, float)):
            return float(obj[k])
    return None


def _extract_joint_calibration(calibration):
    if not isinstance(calibration, dict):
        return {}

    out = {}
    names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    for name in names:
        node = calibration.get(name)
        if not isinstance(node, dict):
            continue

        range_min = _extract_numeric(node, "range_min", "recorded_min", "min")
        range_max = _extract_numeric(node, "range_max", "recorded_max", "max")
        homing_offset = _extract_numeric(node, "homing_offset")
        drive_mode = _extract_numeric(node, "drive_mode")
        norm_mode = _extract_numeric(node, "norm_mode")
        motor_id = _extract_numeric(node, "id")

        if range_min is None or range_max is None:
            continue

        span = float(range_max) - float(range_min)
        if span <= 0:
            continue

        center = 0.5 * (float(range_min) + float(range_max))

        out[name] = {
            "id": None if motor_id is None else int(motor_id),
            "range_min": float(range_min),
            "range_max": float(range_max),
            "range_center": float(center),
            "range_span": float(span),
            "homing_offset": 0.0 if homing_offset is None else float(homing_offset),
            "drive_mode": 0 if drive_mode is None else int(drive_mode),
            "norm_mode": None if norm_mode is None else int(norm_mode),
        }

    return out


def _merge_limits(base_limits, lerobot_calibration):
    merged = dict(base_limits)

    if lerobot_calibration is None:
        lerobot_calibration = _load_lerobot_calibration_file()

    joint_cal = _extract_joint_calibration(lerobot_calibration)

    # The LeRobot calibration file gives motor-space min/max and homing info,
    # not direct radians. We use it to bias/clamp the solver behavior while
    # keeping the software-side mechanical angle limits from values.py.
    for joint_name in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"):
        if joint_name not in merged or joint_name not in joint_cal:
            continue

        lo, hi = merged[joint_name]
        cal = joint_cal[joint_name]

        # If calibration span is extremely narrow compared to expected full range,
        # avoid shrinking usable range too aggressively.
        span_ratio = _clip(cal["range_span"] / 4095.0, 0.15, 1.0)
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo) * span_ratio

        merged[joint_name] = (mid - half, mid + half)

    return merged, joint_cal


def _joint_mid(lo: float, hi: float) -> float:
    return 0.5 * (lo + hi)


def _joint_span(lo: float, hi: float) -> float:
    return hi - lo


def _clamp_to_joint(name: str, x: float, limits) -> float:
    lo, hi = limits[name]
    return _clip(float(x), float(lo), float(hi))


def _workspace_min_max():
    wmin = np.array(getattr(val, "ARUCO_WORKSPACE_MIN", (-0.18, -0.12, 0.02)), dtype=np.float64)
    wmax = np.array(getattr(val, "ARUCO_WORKSPACE_MAX", (0.18, 0.18, 0.28)), dtype=np.float64)
    return wmin, wmax


def _normalize_workspace_xyz(xyz):
    xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
    wmin, wmax = _workspace_min_max()
    denom = wmax - wmin
    denom = np.where(np.abs(denom) < 1e-9, 1.0, denom)
    z = (xyz - wmin) / denom
    return np.clip(z, 0.0, 1.0)


def _ee_geom():
    return {
        "base_height": _get_value("IK_BASE_HEIGHT_M", 0.06),
        "upper_arm": _get_value("IK_LINK1_M", 0.115),
        "forearm": _get_value("IK_LINK2_M", 0.115),
        "tool": _get_value("IK_TOOL_M", 0.05),
        "r_min": _get_value("IK_RADIAL_MIN_M", 0.04),
    }


def _safe_acos(x: float) -> float:
    return math.acos(_clip(x, -1.0, 1.0))


def _solve_planar_2link(r: float, z: float, l1: float, l2: float):
    rr = float(r)
    zz = float(z)
    d2 = rr * rr + zz * zz
    c2 = (d2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    q2 = _safe_acos(c2)
    k1 = l1 + l2 * math.cos(q2)
    k2 = l2 * math.sin(q2)
    q1 = math.atan2(zz, rr) - math.atan2(k2, k1)
    return q1, q2


def _map_rpy_to_wrist(target_rpy):
    if target_rpy is None:
        return 0.0, 0.0

    rpy = np.asarray(target_rpy, dtype=np.float64).reshape(-1)
    if rpy.size < 3:
        return 0.0, 0.0

    roll = float(rpy[0])
    pitch = float(rpy[1])
    yaw = float(rpy[2])

    wrist_flex = pitch
    wrist_roll = yaw

    alt = 0.5 * (roll + yaw)
    if abs(alt) > abs(wrist_roll):
        wrist_roll = alt

    return wrist_flex, wrist_roll


def _apply_calibration_bias(joints, joint_cal, base_limits):
    if not joint_cal:
        return joints

    out = dict(joints)

    for name in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"):
        if name not in out or name not in joint_cal or name not in base_limits:
            continue

        lo, hi = base_limits[name]
        cal = joint_cal[name]

        # Use the calibrated motor range span to softly compress or expand how much
        # of the software limit range we actually occupy.
        span_ratio = _clip(cal["range_span"] / 4095.0, 0.15, 1.0)

        mid = 0.5 * (lo + hi)
        x = float(out[name])

        if hi - lo > 1e-9:
            norm = (x - lo) / (hi - lo)
        else:
            norm = 0.5

        # Re-center around calibrated midpoint behavior while preserving order.
        calibrated_lo = mid - 0.5 * (hi - lo) * span_ratio
        calibrated_hi = mid + 0.5 * (hi - lo) * span_ratio
        out[name] = calibrated_lo + norm * (calibrated_hi - calibrated_lo)

    return out


def _fallback_joint_mapping_from_workspace(target_xyz, target_rpy, gripper_open01, limits, joint_cal):
    xyz_norm = _normalize_workspace_xyz(target_xyz)

    shoulder_pan = _joint_mid(*limits["shoulder_pan"]) + (0.5 - xyz_norm[0]) * _joint_span(*limits["shoulder_pan"])
    shoulder_lift = limits["shoulder_lift"][0] + xyz_norm[1] * _joint_span(*limits["shoulder_lift"])
    elbow_flex = limits["elbow_flex"][0] + xyz_norm[2] * _joint_span(*limits["elbow_flex"])

    wrist_flex, wrist_roll = _map_rpy_to_wrist(target_rpy)

    out = {
        "shoulder_pan": _clamp_to_joint("shoulder_pan", shoulder_pan, limits),
        "shoulder_lift": _clamp_to_joint("shoulder_lift", shoulder_lift, limits),
        "elbow_flex": _clamp_to_joint("elbow_flex", elbow_flex, limits),
        "wrist_flex": _clamp_to_joint("wrist_flex", wrist_flex, limits),
        "wrist_roll": _clamp_to_joint("wrist_roll", wrist_roll, limits),
        "gripper_open01": _clip(float(gripper_open01), 0.0, 1.0),
    }

    return _apply_calibration_bias(out, joint_cal, limits)


def solve_ik_from_target(
    target_xyz,
    target_rpy=None,
    gripper_open01: float = 1.0,
    lerobot_calibration=None,
):
    if lerobot_calibration is None:
        lerobot_calibration = _load_lerobot_calibration_file()

    base_limits = _get_joint_limits()
    limits, joint_cal = _merge_limits(base_limits, lerobot_calibration)

    try:
        xyz = np.asarray(target_xyz, dtype=np.float64).reshape(3)
    except Exception:
        return _fallback_joint_mapping_from_workspace((0.0, 0.0, 0.1), target_rpy, gripper_open01, limits, joint_cal)

    geom = _ee_geom()

    x = float(xyz[0])
    y = float(xyz[1])
    z = float(xyz[2])

    base_height = geom["base_height"]
    l1 = geom["upper_arm"]
    l2 = geom["forearm"]
    tool = geom["tool"]
    r_min = geom["r_min"]

    wrist_flex_seed, wrist_roll = _map_rpy_to_wrist(target_rpy)

    shoulder_pan = math.atan2(y, x)

    planar_r = math.hypot(x, y)
    wrist_target_r = max(r_min, planar_r - tool)
    wrist_target_z = z - base_height

    max_reach = max(1e-6, l1 + l2 - 1e-4)
    min_reach = max(1e-6, abs(l1 - l2) + 1e-4)

    reach = math.hypot(wrist_target_r, wrist_target_z)
    if reach > max_reach:
        s = max_reach / reach
        wrist_target_r *= s
        wrist_target_z *= s
    elif reach < min_reach:
        s = min_reach / max(reach, 1e-9)
        wrist_target_r *= s
        wrist_target_z *= s

    shoulder_lift, elbow_geom = _solve_planar_2link(wrist_target_r, wrist_target_z, l1, l2)

    elbow_flex = -elbow_geom

    pitch_target = 0.0
    if target_rpy is not None:
        rpy = np.asarray(target_rpy, dtype=np.float64).reshape(-1)
        if rpy.size >= 2:
            pitch_target = float(rpy[1])

    wrist_flex = pitch_target - shoulder_lift - elbow_flex

    solved = {
        "shoulder_pan": _clamp_to_joint("shoulder_pan", shoulder_pan, limits),
        "shoulder_lift": _clamp_to_joint("shoulder_lift", shoulder_lift, limits),
        "elbow_flex": _clamp_to_joint("elbow_flex", elbow_flex, limits),
        "wrist_flex": _clamp_to_joint(
            "wrist_flex",
            wrist_flex if math.isfinite(wrist_flex) else wrist_flex_seed,
            limits,
        ),
        "wrist_roll": _clamp_to_joint("wrist_roll", wrist_roll, limits),
        "gripper_open01": _clip(float(gripper_open01), 0.0, 1.0),
    }

    solved = _apply_calibration_bias(solved, joint_cal, base_limits)
    return solved