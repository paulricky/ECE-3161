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
        "wrist_yaw": (
            _get_value("WRIST_YAW_MIN", -math.pi),
            _get_value("WRIST_YAW_MAX", math.pi),
        ),
        "wrist_roll": (
            _get_value("WRIST_ROLL_MIN", -math.pi),
            _get_value("WRIST_ROLL_MAX", math.pi),
        ),
        "wrist_pitch": (
            _get_value("WRIST_PITCH_MIN", -2.5),
            _get_value("WRIST_PITCH_MAX", 2.5),
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
    names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_yaw",
        "wrist_roll",
        "wrist_pitch",
        "gripper",
    ]

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

        lo = min(float(range_min), float(range_max))
        hi = max(float(range_min), float(range_max))
        span = hi - lo
        if span <= 0:
            continue

        center = 0.5 * (lo + hi)

        out[name] = {
            "id": None if motor_id is None else int(motor_id),
            "range_min": float(lo),
            "range_max": float(hi),
            "range_center": float(center),
            "range_span": float(span),
            "homing_offset": 0.0 if homing_offset is None else float(homing_offset),
            "drive_mode": 0 if drive_mode is None else int(drive_mode),
            "norm_mode": None if norm_mode is None else int(norm_mode),
        }

    if out:
        return out

    motor_names = calibration.get("motor_names")
    min_pos = calibration.get("min_pos")
    max_pos = calibration.get("max_pos")
    homing_offset = calibration.get("homing_offset")
    drive_mode = calibration.get("drive_mode")

    if not isinstance(motor_names, list) or not isinstance(min_pos, list) or not isinstance(max_pos, list):
        return {}
    if len(motor_names) != len(min_pos) or len(motor_names) != len(max_pos):
        return {}

    for i, name in enumerate(motor_names):
        if name not in names:
            continue
        try:
            lo = min(float(min_pos[i]), float(max_pos[i]))
            hi = max(float(min_pos[i]), float(max_pos[i]))
        except Exception:
            continue
        span = hi - lo
        if span <= 0:
            continue
        out[name] = {
            "id": None,
            "range_min": float(lo),
            "range_max": float(hi),
            "range_center": float(0.5 * (lo + hi)),
            "range_span": float(span),
            "homing_offset": 0.0 if not isinstance(homing_offset, list) or i >= len(homing_offset) else float(homing_offset[i]),
            "drive_mode": 0 if not isinstance(drive_mode, list) or i >= len(drive_mode) else int(drive_mode[i]),
            "norm_mode": None,
        }

    return out


def _merge_limits(base_limits, lerobot_calibration):
    merged = dict(base_limits)

    if lerobot_calibration is None:
        lerobot_calibration = _load_lerobot_calibration_file()

    joint_cal = _extract_joint_calibration(lerobot_calibration)

    for joint_name in (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_yaw",
        "wrist_roll",
        "wrist_pitch",
    ):
        if joint_name not in merged or joint_name not in joint_cal:
            continue

        lo, hi = merged[joint_name]
        cal = joint_cal[joint_name]
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
    # 7-DOF chain: wrist_roll --[tool_a]--> wrist_pitch --[tool_b]--> EE.
    # When wrist_pitch = 0, tool_a + tool_b collapses to the legacy single
    # tool length used for wrist-center math in solve_ik_from_target.
    tool_a = _get_value("IK_TOOL_A_M", 0.025)
    tool_b = _get_value("IK_TOOL_B_M", 0.025)
    # Backwards-compat: if IK_TOOL_M is set and the A/B split isn't, fall
    # back to splitting the old single length evenly.
    legacy = getattr(val, "IK_TOOL_M", None)
    if legacy is not None and not hasattr(val, "IK_TOOL_A_M") and not hasattr(val, "IK_TOOL_B_M"):
        tool_a = float(legacy) / 2.0
        tool_b = float(legacy) / 2.0
    return {
        "base_height": _get_value("IK_BASE_HEIGHT_M", 0.06),
        "upper_arm": _get_value("IK_LINK1_M", 0.115),
        "forearm": _get_value("IK_LINK2_M", 0.115),
        "tool_a": tool_a,
        "tool_b": tool_b,
        # "tool" = distance from wrist_center to EE when wrist_pitch = 0.
        # Existing IK code uses this for the wrist-center projection.
        "tool": tool_a + tool_b,
        "r_min": _get_value("IK_RADIAL_MIN_M", 0.04),
    }


def _apply_calibration_bias(joints, joint_cal, base_limits):
    if not joint_cal:
        return joints

    out = dict(joints)

    for name in (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_yaw",
        "wrist_roll",
        "wrist_pitch",
    ):
        if name not in out or name not in joint_cal or name not in base_limits:
            continue

        lo, hi = base_limits[name]
        cal = joint_cal[name]
        span_ratio = _clip(cal["range_span"] / 4095.0, 0.15, 1.0)
        mid = 0.5 * (lo + hi)
        x = float(out[name])

        if hi - lo > 1e-9:
            norm = (x - lo) / (hi - lo)
        else:
            norm = 0.5

        calibrated_lo = mid - 0.5 * (hi - lo) * span_ratio
        calibrated_hi = mid + 0.5 * (hi - lo) * span_ratio
        out[name] = calibrated_lo + norm * (calibrated_hi - calibrated_lo)

    return out


def _wrap_angle(x: float) -> float:
    return math.atan2(math.sin(float(x)), math.cos(float(x)))


def _angle_diff(a: float, b: float) -> float:
    return _wrap_angle(float(a) - float(b))


def _rot_x(a: float):
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ], dtype=np.float64)


def _rot_y(a: float):
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ], dtype=np.float64)


def _rot_z(a: float):
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _rpy_to_matrix(target_rpy):
    if target_rpy is None:
        return np.eye(3, dtype=np.float64)
    rpy = np.asarray(target_rpy, dtype=np.float64).reshape(-1)
    if rpy.size < 3:
        return np.eye(3, dtype=np.float64)
    roll = float(rpy[0])
    pitch = float(rpy[1])
    yaw = float(rpy[2])
    return _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)


def _matrix_to_rpy(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float64)


def _vee(M):
    return np.array([M[2, 1], M[0, 2], M[1, 0]], dtype=np.float64)


def _orientation_error_vec(R_target, R_current):
    R_err = np.asarray(R_target, dtype=np.float64) @ np.asarray(R_current, dtype=np.float64).T
    cos_theta = _clip((float(np.trace(R_err)) - 1.0) * 0.5, -1.0, 1.0)
    theta = math.acos(cos_theta)
    if theta < 1e-9:
        return 0.5 * _vee(R_err - R_err.T)
    sin_theta = math.sin(theta)
    if abs(sin_theta) < 1e-7:
        diag = np.diag(R_err)
        axis = np.sqrt(np.maximum((diag + 1.0) * 0.5, 0.0))
        if axis[0] < 1e-6 and axis[1] < 1e-6 and axis[2] < 1e-6:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        axis = axis / max(np.linalg.norm(axis), 1e-9)
        return theta * axis
    return (theta / (2.0 * sin_theta)) * _vee(R_err - R_err.T)


def _map_rpy_to_wrist(target_rpy, shoulder_pan_seed=0.0):
    if target_rpy is None:
        return 0.0, 0.0, 0.0

    rpy = np.asarray(target_rpy, dtype=np.float64).reshape(-1)
    if rpy.size < 3:
        return 0.0, 0.0, 0.0

    roll = float(rpy[0])
    pitch = float(rpy[1])
    yaw = float(rpy[2])

    wrist_flex = pitch
    wrist_yaw = _wrap_angle(yaw - shoulder_pan_seed)
    wrist_roll = roll
    return wrist_flex, wrist_yaw, wrist_roll


def _fk_chain_details(joints, geom):
    q0 = float(joints["shoulder_pan"])
    q1 = float(joints["shoulder_lift"])
    q2 = float(joints["elbow_flex"])
    q3 = float(joints["wrist_flex"])
    q4 = float(joints["wrist_yaw"])
    q5 = float(joints["wrist_roll"])
    q6 = float(joints.get("wrist_pitch", 0.0))

    l1 = float(geom["upper_arm"])
    l2 = float(geom["forearm"])
    tool_a = float(geom["tool_a"])
    tool_b = float(geom["tool_b"])
    base_height = float(geom["base_height"])

    p = np.array([0.0, 0.0, base_height], dtype=np.float64)
    R = np.eye(3, dtype=np.float64)

    joint_origins = []
    joint_axes = []

    joint_origins.append(p.copy())
    joint_axes.append(R @ np.array([0.0, 0.0, 1.0], dtype=np.float64))
    R = R @ _rot_z(q0)

    joint_origins.append(p.copy())
    joint_axes.append(R @ np.array([0.0, 1.0, 0.0], dtype=np.float64))
    R = R @ _rot_y(q1)
    p = p + R @ np.array([l1, 0.0, 0.0], dtype=np.float64)

    joint_origins.append(p.copy())
    joint_axes.append(R @ np.array([0.0, 1.0, 0.0], dtype=np.float64))
    R = R @ _rot_y(q2)
    p = p + R @ np.array([l2, 0.0, 0.0], dtype=np.float64)

    joint_origins.append(p.copy())
    joint_axes.append(R @ np.array([0.0, 1.0, 0.0], dtype=np.float64))
    R = R @ _rot_y(q3)

    wrist_center = p.copy()

    joint_origins.append(wrist_center.copy())
    joint_axes.append(R @ np.array([0.0, 0.0, 1.0], dtype=np.float64))
    R = R @ _rot_z(q4)

    joint_origins.append(wrist_center.copy())
    joint_axes.append(R @ np.array([1.0, 0.0, 0.0], dtype=np.float64))
    R = R @ _rot_x(q5)

    # Move tool_a along the current x axis to reach the wrist_pitch joint.
    p = wrist_center + R @ np.array([tool_a, 0.0, 0.0], dtype=np.float64)

    # q6 = wrist_pitch about local y.
    joint_origins.append(p.copy())
    joint_axes.append(R @ np.array([0.0, 1.0, 0.0], dtype=np.float64))
    R = R @ _rot_y(q6)

    ee = p + R @ np.array([tool_b, 0.0, 0.0], dtype=np.float64)
    return ee, R, wrist_center, joint_origins, joint_axes


def _fk_all(joints, geom):
    ee, R, wrist_center, _origins, _axes = _fk_chain_details(joints, geom)
    return ee, R, wrist_center


def _fk_arm_position(joints, geom):
    ee, _R, _wc = _fk_all(joints, geom)
    return ee


def _fk_orientation_matrix(joints, geom):
    _ee, R, _wc = _fk_all(joints, geom)
    return R


def _fk_rpy(joints, geom):
    return _matrix_to_rpy(_fk_orientation_matrix(joints, geom))


def _joint_dict_to_vector(joints):
    return np.array([
        float(joints["shoulder_pan"]),
        float(joints["shoulder_lift"]),
        float(joints["elbow_flex"]),
        float(joints["wrist_flex"]),
        float(joints["wrist_yaw"]),
        float(joints["wrist_roll"]),
        float(joints.get("wrist_pitch", 0.0)),
    ], dtype=np.float64)


def _vector_to_joint_dict(vec, gripper_open01=1.0):
    arr = np.asarray(vec, dtype=np.float64).reshape(7)
    return {
        "shoulder_pan": float(arr[0]),
        "shoulder_lift": float(arr[1]),
        "elbow_flex": float(arr[2]),
        "wrist_flex": float(arr[3]),
        "wrist_yaw": float(arr[4]),
        "wrist_roll": float(arr[5]),
        "wrist_pitch": float(arr[6]),
        "gripper_open01": _clip(float(gripper_open01), 0.0, 1.0),
    }


def _joint_delta_cost(candidate, previous_joints):
    if not previous_joints:
        return 0.0
    names = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_yaw",
        "wrist_roll",
        "wrist_pitch",
    )
    total = 0.0
    for name in names:
        if name not in previous_joints:
            continue
        total += abs(_angle_diff(candidate[name], float(previous_joints[name])))
    return total


def _candidate_cost(candidate, target_xyz, target_R, previous_joints, geom):
    pos_err = np.linalg.norm(_fk_arm_position(candidate, geom) - np.asarray(target_xyz, dtype=np.float64))
    orient_err = np.linalg.norm(_orientation_error_vec(target_R, _fk_orientation_matrix(candidate, geom)))
    continuity = _joint_delta_cost(candidate, previous_joints)
    orient_gain = float(getattr(val, "IK_DLS_ORIENTATION_GAIN", 0.45))
    cont_gain = float(getattr(val, "IK_DLS_CONTINUITY_GAIN", 0.08))
    return pos_err + orient_gain * orient_err + cont_gain * continuity


def _clamp_joint_dict(joints, limits):
    out = dict(joints)
    for name in (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_yaw",
        "wrist_roll",
        "wrist_pitch",
    ):
        out[name] = _clamp_to_joint(name, out.get(name, 0.0), limits)
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


def _analytic_geometric_jacobian(joints, geom, var_names):
    ee, _R, _wrist_center, joint_origins, joint_axes = _fk_chain_details(joints, geom)
    name_to_index = {
        "shoulder_pan": 0,
        "shoulder_lift": 1,
        "elbow_flex": 2,
        "wrist_flex": 3,
        "wrist_yaw": 4,
        "wrist_roll": 5,
        "wrist_pitch": 6,
    }
    cols = []
    for name in var_names:
        idx = name_to_index[name]
        axis = joint_axes[idx]
        origin = joint_origins[idx]
        linear = np.cross(axis, ee - origin)
        angular = axis.copy()
        cols.append(np.concatenate((linear, angular)))
    return np.column_stack(cols)


def _build_orientation_seed(target_R, q0_seed, q1_seed, q2_seed, q3_seed):
    R_pre = _rot_z(q0_seed) @ _rot_y(q1_seed) @ _rot_y(q2_seed) @ _rot_y(q3_seed)
    R_local = R_pre.T @ np.asarray(target_R, dtype=np.float64)
    yaw_seed = math.atan2(float(R_local[1, 0]), float(R_local[0, 0]))
    R_after_yaw = _rot_z(yaw_seed).T @ R_local
    roll_seed = math.atan2(float(R_after_yaw[2, 1]), float(R_after_yaw[1, 1]))
    return _wrap_angle(yaw_seed), _wrap_angle(roll_seed)


def _decompose_local_orientation(target_R, shoulder_pan):
    R_local = _rot_z(-float(shoulder_pan)) @ np.asarray(target_R, dtype=np.float64)
    r11 = float(R_local[0, 0])
    r21 = float(R_local[1, 0])
    r31 = float(R_local[2, 0])
    r22 = float(R_local[1, 1])
    r23 = float(R_local[1, 2])
    cb_abs = math.sqrt(max(0.0, r11 * r11 + r31 * r31))

    branches = []
    q4_a = math.atan2(r21, cb_abs)
    q123_a = math.atan2(-r31, r11)
    q5_a = math.atan2(-r23, r22)
    branches.append((_wrap_angle(q123_a), _wrap_angle(q4_a), _wrap_angle(q5_a)))

    q4_b = math.atan2(r21, -cb_abs)
    q123_b = math.atan2(r31, -r11)
    q5_b = math.atan2(r23, -r22)
    branches.append((_wrap_angle(q123_b), _wrap_angle(q4_b), _wrap_angle(q5_b)))

    out = []
    for q123, q4, q5 in branches:
        keep = True
        for e123, e4, e5 in out:
            if abs(_angle_diff(q123, e123)) < 1e-6 and abs(_angle_diff(q4, e4)) < 1e-6 and abs(_angle_diff(q5, e5)) < 1e-6:
                keep = False
                break
        if keep:
            out.append((q123, q4, q5))
    return out


def _refine_dls(initial_joints, target_xyz, target_R, limits, geom, previous_joints):
    q = _clamp_joint_dict(initial_joints, limits)
    target_xyz = np.asarray(target_xyz, dtype=np.float64).reshape(3)
    target_R = np.asarray(target_R, dtype=np.float64).reshape(3, 3)
    var_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_yaw",
        "wrist_roll",
        "wrist_pitch",
    ]
    n_dof = len(var_names)

    damping = float(_get_value("IK_DLS_DAMPING", 0.08))
    pos_gain = float(_get_value("IK_DLS_POSITION_GAIN", 1.0))
    orient_gain = float(_get_value("IK_DLS_ORIENTATION_GAIN", 0.45))
    cont_gain = float(_get_value("IK_DLS_CONTINUITY_GAIN", 0.08))
    max_iters = int(_get_value("IK_DLS_MAX_ITERS", 20))
    step_limit = float(_get_value("IK_DLS_MAX_STEP_RAD", 0.16))
    pos_tol = float(_get_value("IK_RESIDUAL_THRESH", 1e-4))
    orient_tol = float(_get_value("IK_DLS_ORIENTATION_TOL", 1e-3))
    prev_vec = None if not previous_joints else _joint_dict_to_vector(_clamp_joint_dict(previous_joints, limits))

    q_vec = _joint_dict_to_vector(q)
    best = dict(q)
    best_cost = _candidate_cost(best, target_xyz, target_R, previous_joints, geom)

    for _ in range(max_iters):
        q = _vector_to_joint_dict(q_vec, q.get("gripper_open01", 1.0))
        pos_err_vec = target_xyz - _fk_arm_position(q, geom)
        orient_err = _orientation_error_vec(target_R, _fk_orientation_matrix(q, geom))
        pos_norm = float(np.linalg.norm(pos_err_vec))
        orient_norm = float(np.linalg.norm(orient_err))
        if pos_norm < pos_tol and orient_norm < orient_tol:
            break

        J = _analytic_geometric_jacobian(q, geom, var_names)
        W = np.diag([pos_gain, pos_gain, pos_gain, orient_gain, orient_gain, orient_gain]).astype(np.float64)
        A_top = W @ J
        b_top = W @ np.concatenate((pos_err_vec, orient_err))

        if prev_vec is not None and cont_gain > 0.0:
            A = np.vstack((A_top, math.sqrt(cont_gain) * np.eye(n_dof, dtype=np.float64)))
            b = np.concatenate((b_top, math.sqrt(cont_gain) * (prev_vec - q_vec)))
        else:
            A = A_top
            b = b_top

        lhs = A.T @ A + (damping * damping) * np.eye(n_dof, dtype=np.float64)
        rhs = A.T @ b
        try:
            delta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

        max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
        if max_abs > step_limit:
            delta *= step_limit / max(max_abs, 1e-9)

        current_cost = _candidate_cost(q, target_xyz, target_R, previous_joints, geom)
        accepted = False
        for scale in (1.0, 0.5, 0.25, 0.1):
            trial_vec = q_vec + scale * delta
            trial = _clamp_joint_dict(_vector_to_joint_dict(trial_vec, q.get("gripper_open01", 1.0)), limits)
            trial_vec = _joint_dict_to_vector(trial)
            trial_cost = _candidate_cost(trial, target_xyz, target_R, previous_joints, geom)
            if trial_cost <= current_cost:
                q_vec = trial_vec
                q = trial
                current_cost = trial_cost
                damping = max(1e-4, damping * 0.85)
                accepted = True
                if trial_cost < best_cost:
                    best = dict(trial)
                    best_cost = trial_cost
                break

        if not accepted:
            damping = min(2.0, damping * 2.0)
            if np.linalg.norm(delta) < 1e-6:
                break

    final = _clamp_joint_dict(_vector_to_joint_dict(q_vec, q.get("gripper_open01", 1.0)), limits)
    final_cost = _candidate_cost(final, target_xyz, target_R, previous_joints, geom)
    if final_cost <= best_cost:
        best = final
    return _clamp_joint_dict(best, limits)


def _fallback_joint_mapping_from_workspace(target_xyz, target_rpy, gripper_open01, limits, joint_cal):
    xyz_norm = _normalize_workspace_xyz(target_xyz)
    shoulder_pan = _joint_mid(*limits["shoulder_pan"]) + (0.5 - xyz_norm[0]) * _joint_span(*limits["shoulder_pan"])
    shoulder_lift = limits["shoulder_lift"][0] + xyz_norm[1] * _joint_span(*limits["shoulder_lift"])
    elbow_flex = limits["elbow_flex"][0] + xyz_norm[2] * _joint_span(*limits["elbow_flex"])
    wrist_flex, wrist_yaw, wrist_roll = _map_rpy_to_wrist(target_rpy, shoulder_pan)
    out = {
        "shoulder_pan": _clamp_to_joint("shoulder_pan", shoulder_pan, limits),
        "shoulder_lift": _clamp_to_joint("shoulder_lift", shoulder_lift, limits),
        "elbow_flex": _clamp_to_joint("elbow_flex", elbow_flex, limits),
        "wrist_flex": _clamp_to_joint("wrist_flex", wrist_flex, limits),
        "wrist_yaw": _clamp_to_joint("wrist_yaw", wrist_yaw, limits),
        "wrist_roll": _clamp_to_joint("wrist_roll", wrist_roll, limits),
        "wrist_pitch": _clamp_to_joint("wrist_pitch", 0.0, limits),
        "gripper_open01": _clip(float(gripper_open01), 0.0, 1.0),
    }
    return out


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

    target_R = _rpy_to_matrix(target_rpy)

    geom = _ee_geom()
    base_height = float(geom["base_height"])
    l1 = float(geom["upper_arm"])
    l2 = float(geom["forearm"])
    tool = float(geom["tool"])
    r_min = float(geom["r_min"])

    shoulder_pan_seed = math.atan2(float(xyz[1]), float(xyz[0])) if np.linalg.norm(xyz[:2]) > 1e-9 else 0.0
    wrist_flex_seed, wrist_yaw_seed, wrist_roll_seed = _map_rpy_to_wrist(target_rpy, shoulder_pan_seed)

    wrist_xyz = xyz - target_R @ np.array([tool, 0.0, 0.0], dtype=np.float64)

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

    candidates = []

    # For the analytic seeds we fix wrist_pitch = 0 so the legacy 2-link
    # shoulder/elbow solve and wrist orientation decomposition still apply
    # (with wrist_pitch=0, tool_a + tool_b collapses to the legacy tool length
    # which was already baked into the `tool` key used for wrist_xyz). The
    # DLS refiner then uses all 7 joints, so the redundancy is resolved by
    # continuity (previous value of wrist_pitch, or 0 when there is none).
    if previous_joints:
        seed = {
            "shoulder_pan": float(previous_joints.get("shoulder_pan", shoulder_pan_seed)),
            "shoulder_lift": float(previous_joints.get("shoulder_lift", 0.0)),
            "elbow_flex": float(previous_joints.get("elbow_flex", 0.0)),
            "wrist_flex": float(previous_joints.get("wrist_flex", wrist_flex_seed)),
            "wrist_yaw": float(previous_joints.get("wrist_yaw", wrist_yaw_seed)),
            "wrist_roll": float(previous_joints.get("wrist_roll", wrist_roll_seed)),
            "wrist_pitch": float(previous_joints.get("wrist_pitch", 0.0)),
            "gripper_open01": _clip(float(gripper_open01), 0.0, 1.0),
        }
        candidates.append(_refine_dls(seed, xyz, target_R, limits, geom, previous_joints))

    neutral_seed = {
        "shoulder_pan": shoulder_pan_seed,
        "shoulder_lift": 0.0,
        "elbow_flex": 0.0,
        "wrist_flex": wrist_flex_seed,
        "wrist_yaw": wrist_yaw_seed,
        "wrist_roll": wrist_roll_seed,
        "wrist_pitch": 0.0,
        "gripper_open01": _clip(float(gripper_open01), 0.0, 1.0),
    }
    candidates.append(_refine_dls(neutral_seed, xyz, target_R, limits, geom, previous_joints))

    orientation_branches = _decompose_local_orientation(target_R, shoulder_pan_seed)
    for shoulder_lift, elbow_flex in _solve_planar_branches(planar_r, planar_z, l1, l2):
        for q123, wrist_yaw, wrist_roll in orientation_branches:
            wrist_flex = _wrap_angle(q123 - shoulder_lift - elbow_flex)
            candidate = {
                "shoulder_pan": shoulder_pan_seed,
                "shoulder_lift": shoulder_lift,
                "elbow_flex": elbow_flex,
                "wrist_flex": wrist_flex if math.isfinite(wrist_flex) else wrist_flex_seed,
                "wrist_yaw": wrist_yaw if math.isfinite(wrist_yaw) else wrist_yaw_seed,
                "wrist_roll": wrist_roll if math.isfinite(wrist_roll) else wrist_roll_seed,
                "wrist_pitch": 0.0,
                "gripper_open01": _clip(float(gripper_open01), 0.0, 1.0),
            }
            candidates.append(_refine_dls(candidate, xyz, target_R, limits, geom, previous_joints))

        wrist_yaw, wrist_roll = _build_orientation_seed(target_R, shoulder_pan_seed, shoulder_lift, elbow_flex, 0.0)
        loose_candidate = {
            "shoulder_pan": shoulder_pan_seed,
            "shoulder_lift": shoulder_lift,
            "elbow_flex": elbow_flex,
            "wrist_flex": wrist_flex_seed,
            "wrist_yaw": wrist_yaw if math.isfinite(wrist_yaw) else wrist_yaw_seed,
            "wrist_roll": wrist_roll if math.isfinite(wrist_roll) else wrist_roll_seed,
            "wrist_pitch": 0.0,
            "gripper_open01": _clip(float(gripper_open01), 0.0, 1.0),
        }
        candidates.append(_refine_dls(loose_candidate, xyz, target_R, limits, geom, previous_joints))

    if not candidates:
        return _fallback_joint_mapping_from_workspace(xyz, target_rpy, gripper_open01, limits, joint_cal)

    best = min(candidates, key=lambda c: _candidate_cost(c, xyz, target_R, previous_joints, geom))
    best["gripper_open01"] = _clip(float(gripper_open01), 0.0, 1.0)
    return _clamp_joint_dict(best, limits)