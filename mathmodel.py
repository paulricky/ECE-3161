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


def _wrap_angle(x: float) -> float:
    return math.atan2(math.sin(float(x)), math.cos(float(x)))


def _angle_diff(a: float, b: float) -> float:
    return _wrap_angle(float(a) - float(b))


def _rpy_to_matrix(target_rpy):
    if target_rpy is None:
        return np.eye(3, dtype=np.float64)
    rpy = np.asarray(target_rpy, dtype=np.float64).reshape(-1)
    if rpy.size < 3:
        return np.eye(3, dtype=np.float64)
    roll = float(rpy[0])
    pitch = float(rpy[1])
    yaw = float(rpy[2])

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr, cr],
    ], dtype=np.float64)

    ry = np.array([
        [cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp],
    ], dtype=np.float64)

    rz = np.array([
        [cy, -sy, 0.0],
        [sy, cy, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    return rz @ ry @ rx


def _pitch_from_rpy(target_rpy) -> float:
    if target_rpy is None:
        return 0.0
    rpy = np.asarray(target_rpy, dtype=np.float64).reshape(-1)
    if rpy.size < 2:
        return 0.0
    return float(rpy[1])


def _fk_arm_position(joints, geom):
    q0 = float(joints["shoulder_pan"])
    q1 = float(joints["shoulder_lift"])
    q2 = float(joints["elbow_flex"])
    q3 = float(joints["wrist_flex"])

    l1 = float(geom["upper_arm"])
    l2 = float(geom["forearm"])
    tool = float(geom["tool"])
    base_height = float(geom["base_height"])

    a1 = q1
    a2 = q1 + q2
    a3 = q1 + q2 + q3

    radial = (
        l1 * math.cos(a1)
        + l2 * math.cos(a2)
        + tool * math.cos(a3)
    )
    z = (
        base_height
        + l1 * math.sin(a1)
        + l2 * math.sin(a2)
        + tool * math.sin(a3)
    )

    x = math.cos(q0) * radial
    y = math.sin(q0) * radial
    return np.array([x, y, z], dtype=np.float64)


def _fk_pitch(joints) -> float:
    return float(joints["shoulder_lift"] + joints["elbow_flex"] + joints["wrist_flex"])


def _joint_delta_cost(candidate, previous_joints):
    if not previous_joints:
        return 0.0
    names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
    total = 0.0
    for name in names:
        if name not in previous_joints:
            continue
        if name == "wrist_roll":
            total += abs(_angle_diff(candidate[name], float(previous_joints[name])))
        else:
            total += abs(float(candidate[name]) - float(previous_joints[name]))
    return total


def _candidate_cost(candidate, target_xyz, target_pitch, previous_joints, geom):
    pos_err = np.linalg.norm(_fk_arm_position(candidate, geom) - np.asarray(target_xyz, dtype=np.float64))
    pitch_err = abs(_angle_diff(_fk_pitch(candidate), target_pitch))
    continuity = _joint_delta_cost(candidate, previous_joints)
    return pos_err + 0.20 * pitch_err + 0.08 * continuity


def _clamp_joint_dict(joints, limits):
    out = dict(joints)
    for name in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"):
        out[name] = _clamp_to_joint(name, out[name], limits)
    out["gripper_open01"] = _clip(float(out.get("gripper_open01", 1.0)), 0.0, 1.0)
    return out


def _solve_planar_branches(r: float, z: float, l1: float, l2: float):
    rr = float(r)
    zz = float(z)
    d2 = rr * rr + zz * zz
    c2 = _clip((d2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2), -1.0, 1.0)
    elbow_mag = math.acos(c2)

    sols = []
    for q2 in (-elbow_mag, elbow_mag):
        k1 = l1 + l2 * math.cos(q2)
        k2 = l2 * math.sin(q2)
        q1 = math.atan2(zz, rr) - math.atan2(k2, k1)
        sols.append((q1, q2))
    return sols


def _finite_difference_jacobian(joints, target_pitch, geom, var_names, eps=1e-5):
    base_pos = _fk_arm_position(joints, geom)
    base_pitch = _fk_pitch(joints)

    jcols = []
    for name in var_names:
        perturbed = dict(joints)
        perturbed[name] = float(perturbed[name]) + eps
        p_pos = _fk_arm_position(perturbed, geom)
        p_pitch = _fk_pitch(perturbed)

        col = np.array([
            (p_pos[0] - base_pos[0]) / eps,
            (p_pos[1] - base_pos[1]) / eps,
            (p_pos[2] - base_pos[2]) / eps,
            _angle_diff(p_pitch, base_pitch) / eps,
        ], dtype=np.float64)
        jcols.append(col)

    return np.column_stack(jcols)


def _refine_dls(initial_joints, target_xyz, target_pitch, limits, geom, previous_joints):
    q = dict(initial_joints)
    target_xyz = np.asarray(target_xyz, dtype=np.float64).reshape(3)
    var_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]

    damping = float(_get_value("IK_DLS_DAMPING", 0.08))
    pos_gain = float(_get_value("IK_DLS_POSITION_GAIN", 1.0))
    pitch_gain = float(_get_value("IK_DLS_PITCH_GAIN", 0.35))
    max_iters = int(_get_value("IK_DLS_MAX_ITERS", 12))
    step_limit = float(_get_value("IK_DLS_MAX_STEP_RAD", 0.20))

    for _ in range(max_iters):
        pos_err_vec = target_xyz - _fk_arm_position(q, geom)
        pitch_err = _angle_diff(target_pitch, _fk_pitch(q))

        err = np.array([
            pos_gain * pos_err_vec[0],
            pos_gain * pos_err_vec[1],
            pos_gain * pos_err_vec[2],
            pitch_gain * pitch_err,
        ], dtype=np.float64)

        if np.linalg.norm(err[:3]) < 1e-4 and abs(pitch_err) < 1e-3:
            break

        J = _finite_difference_jacobian(q, target_pitch, geom, var_names)
        Jw = J.copy()
        Jw[0:3, :] *= pos_gain
        Jw[3, :] *= pitch_gain

        A = Jw @ Jw.T + (damping * damping) * np.eye(Jw.shape[0], dtype=np.float64)
        delta = Jw.T @ np.linalg.solve(A, err)

        delta = np.clip(delta, -step_limit, step_limit)

        proposal = dict(q)
        for i, name in enumerate(var_names):
            proposal[name] = float(proposal[name]) + float(delta[i])

        proposal = _clamp_joint_dict(proposal, limits)

        current_cost = _candidate_cost(q, target_xyz, target_pitch, previous_joints, geom)
        proposal_cost = _candidate_cost(proposal, target_xyz, target_pitch, previous_joints, geom)

        if proposal_cost <= current_cost:
            q = proposal
        else:
            damping = min(1.0, damping * 1.8)

    return _clamp_joint_dict(q, limits)


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
    previous_joints=None,
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
    base_height = float(geom["base_height"])
    l1 = float(geom["upper_arm"])
    l2 = float(geom["forearm"])
    tool = float(geom["tool"])
    r_min = float(geom["r_min"])

    wrist_flex_seed, wrist_roll = _map_rpy_to_wrist(target_rpy)
    target_pitch = _pitch_from_rpy(target_rpy)

    rmat = _rpy_to_matrix(target_rpy)
    tool_offset_local = np.array([tool, 0.0, 0.0], dtype=np.float64)
    wrist_xyz = xyz - rmat @ tool_offset_local

    wx = float(wrist_xyz[0])
    wy = float(wrist_xyz[1])
    wz = float(wrist_xyz[2])

    planar_r = max(r_min, math.hypot(wx, wy))
    planar_z = wz - base_height

    reach = math.hypot(planar_r, planar_z)
    max_reach = max(1e-6, l1 + l2 - 1e-4)
    min_reach = max(1e-6, abs(l1 - l2) + 1e-4)

    if reach > max_reach:
        s = max_reach / max(reach, 1e-9)
        planar_r *= s
        planar_z *= s
    elif reach < min_reach:
        s = min_reach / max(reach, 1e-9)
        planar_r *= s
        planar_z *= s

    shoulder_pan_seed = math.atan2(wy, wx)

    candidates = []
    for shoulder_lift, elbow_flex in _solve_planar_branches(planar_r, planar_z, l1, l2):
        wrist_flex = target_pitch - shoulder_lift - elbow_flex
        candidate = {
            "shoulder_pan": shoulder_pan_seed,
            "shoulder_lift": shoulder_lift,
            "elbow_flex": elbow_flex,
            "wrist_flex": wrist_flex if math.isfinite(wrist_flex) else wrist_flex_seed,
            "wrist_roll": wrist_roll,
            "gripper_open01": _clip(float(gripper_open01), 0.0, 1.0),
        }
        candidate = _clamp_joint_dict(candidate, limits)
        candidate = _refine_dls(candidate, xyz, target_pitch, limits, geom, previous_joints)
        candidates.append(candidate)

    if not candidates:
        return _fallback_joint_mapping_from_workspace(xyz, target_rpy, gripper_open01, limits, joint_cal)

    best = min(
        candidates,
        key=lambda c: _candidate_cost(c, xyz, target_pitch, previous_joints, geom),
    )

    best["wrist_roll"] = _clamp_to_joint("wrist_roll", wrist_roll, limits)
    best["gripper_open01"] = _clip(float(gripper_open01), 0.0, 1.0)

    best = _apply_calibration_bias(best, joint_cal, base_limits)
    return best