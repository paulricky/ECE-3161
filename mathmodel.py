from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

import values as val

JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "wrist_pitch",
)
GRIPPER_NAME = "gripper_open01"


def dist(a, b) -> float:
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    return math.hypot(ax - bx, ay - by)


def hand_center_xy(hand_lms):
    lm = hand_lms.landmark
    pts = [(lm[i].x, lm[i].y) for i in (0, 5, 9, 13, 17)]
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def _clip(x: float, lo: float, hi: float) -> float:
    return min(max(float(x), float(lo)), float(hi))


def _wrap_angle(x: float) -> float:
    return math.atan2(math.sin(float(x)), math.cos(float(x)))


def _angle_diff(a: float, b: float) -> float:
    return _wrap_angle(float(a) - float(b))


def _get_value(name: str, default: float) -> float:
    try:
        return float(getattr(val, name, default))
    except Exception:
        return float(default)


def _as_np(x, shape=None, default=None) -> np.ndarray:
    try:
        arr = np.asarray(x, dtype=np.float64)
        if shape is not None:
            arr = arr.reshape(shape)
        return arr
    except Exception:
        if default is None:
            raise
        arr = np.asarray(default, dtype=np.float64)
        if shape is not None:
            arr = arr.reshape(shape)
        return arr


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(float(a)), math.sin(float(a))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(float(a)), math.sin(float(a))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(float(a)), math.sin(float(a))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rpy_to_matrix(target_rpy) -> np.ndarray:
    if target_rpy is None:
        return np.eye(3, dtype=np.float64)
    rpy = np.asarray(target_rpy, dtype=np.float64).reshape(-1)
    if rpy.size < 3:
        return np.eye(3, dtype=np.float64)
    roll, pitch, yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])
    return _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)


def _matrix_to_rpy(R) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    sy = math.sqrt(max(0.0, R[0, 0] ** 2 + R[1, 0] ** 2))
    if sy > 1e-8:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float64)


def _orientation_error_vec(R_target, R_current) -> np.ndarray:
    R_err = np.asarray(R_target, dtype=np.float64).reshape(3, 3) @ np.asarray(R_current, dtype=np.float64).reshape(3, 3).T
    return 0.5 * np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ], dtype=np.float64)


def _make_T(R=None, t=None) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    if R is not None:
        T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    if t is not None:
        T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def _T_translate(x=0.0, y=0.0, z=0.0) -> np.ndarray:
    return _make_T(t=np.array([x, y, z], dtype=np.float64))


def _T_rot(axis: str, angle: float) -> np.ndarray:
    if axis == "x":
        return _make_T(R=_rot_x(angle))
    if axis == "y":
        return _make_T(R=_rot_y(angle))
    if axis == "z":
        return _make_T(R=_rot_z(angle))
    raise ValueError(f"Unknown axis {axis!r}")


def _default_lerobot_calibration_path() -> str:
    explicit = str(getattr(val, "LEROBOT_CALIBRATION_FILE", "")).strip()
    if explicit:
        return explicit
    robot_id = getattr(val, "REAL_ROBOT_ID", "my_awesome_follower_arm")
    return os.path.expanduser(f"~/.cache/huggingface/lerobot/calibration/robots/so101_follower/{robot_id}.json")


def _resolve_project_path(raw: str) -> Path:
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    return p


def _load_json_file(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    p = _resolve_project_path(path)
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _load_lerobot_calibration_file(path: str | None = None):
    return _load_json_file(path or _default_lerobot_calibration_path())


def _load_model_calibration_file(path: str | None = None) -> dict:
    candidates = []
    if path:
        candidates.append(path)
    candidates.append(str(getattr(val, "IK_MODEL_CALIBRATION_FILE", "")))
    candidates.append(str(getattr(val, "ROBOT_MODEL_CALIBRATION_FILE", "")))
    candidates.append("calibration_data/robot_model_calibration.json")
    for cand in candidates:
        data = _load_json_file(cand)
        if isinstance(data, dict):
            return data
    return {}


def _extract_numeric(obj: Any, *keys: str):
    if not isinstance(obj, dict):
        return None
    for k in keys:
        v = obj.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _extract_joint_calibration(calibration):
    if not isinstance(calibration, dict):
        return {}
    out = {}
    for name in (*JOINT_NAMES, "gripper"):
        node = calibration.get(name)
        if not isinstance(node, dict):
            continue
        range_min = _extract_numeric(node, "range_min", "recorded_min", "min")
        range_max = _extract_numeric(node, "range_max", "recorded_max", "max")
        homing_offset = _extract_numeric(node, "homing_offset")
        drive_mode = _extract_numeric(node, "drive_mode")
        motor_id = _extract_numeric(node, "id")
        if range_min is None or range_max is None:
            continue
        lo = min(range_min, range_max)
        hi = max(range_min, range_max)
        span = hi - lo
        if span <= 0:
            continue
        out[name] = {
            "id": None if motor_id is None else int(motor_id),
            "range_min": float(lo),
            "range_max": float(hi),
            "range_center": float(0.5 * (lo + hi)),
            "range_span": float(span),
            "homing_offset": 0.0 if homing_offset is None else float(homing_offset),
            "drive_mode": 0 if drive_mode is None else int(drive_mode),
        }
    if out:
        return out

    motor_names = calibration.get("motor_names")
    min_pos = calibration.get("min_pos")
    max_pos = calibration.get("max_pos")
    if not isinstance(motor_names, list) or not isinstance(min_pos, list) or not isinstance(max_pos, list):
        return {}
    for i, name in enumerate(motor_names):
        if name not in (*JOINT_NAMES, "gripper") or i >= len(min_pos) or i >= len(max_pos):
            continue
        try:
            lo = min(float(min_pos[i]), float(max_pos[i]))
            hi = max(float(min_pos[i]), float(max_pos[i]))
        except Exception:
            continue
        if hi <= lo:
            continue
        out[name] = {"id": None, "range_min": lo, "range_max": hi, "range_center": 0.5 * (lo + hi), "range_span": hi - lo, "homing_offset": 0.0, "drive_mode": 0}
    return out


def _get_joint_limits() -> Dict[str, Tuple[float, float]]:
    return {
        "shoulder_pan": (_get_value("BASE_PAN_MIN", -2.5), _get_value("BASE_PAN_MAX", 2.5)),
        "shoulder_lift": (_get_value("SHOULDER_LIFT_MIN", -2.5), _get_value("SHOULDER_LIFT_MAX", 2.5)),
        "elbow_flex": (_get_value("ELBOW_MIN", -2.5), _get_value("ELBOW_MAX", 2.5)),
        "wrist_flex": (_get_value("WRIST_FLEX_MIN", -2.5), _get_value("WRIST_FLEX_MAX", 2.5)),
        "wrist_yaw": (_get_value("WRIST_YAW_MIN", -math.pi), _get_value("WRIST_YAW_MAX", math.pi)),
        "wrist_roll": (_get_value("WRIST_ROLL_MIN", -math.pi), _get_value("WRIST_ROLL_MAX", math.pi)),
        "wrist_pitch": (_get_value("WRIST_PITCH_MIN", -2.5), _get_value("WRIST_PITCH_MAX", 2.5)),
    }


def _merge_limits(base_limits, lerobot_calibration):
    merged = dict(base_limits)
    if lerobot_calibration is None:
        lerobot_calibration = _load_lerobot_calibration_file()
    joint_cal = _extract_joint_calibration(lerobot_calibration)
    for name in JOINT_NAMES:
        if name not in joint_cal or name not in merged:
            continue
        lo, hi = merged[name]
        span_ratio = _clip(float(joint_cal[name]["range_span"]) / 4095.0, 0.15, 1.0)
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo) * span_ratio
        merged[name] = (mid - half, mid + half)
    return merged, joint_cal


def _joint_mid(lo: float, hi: float) -> float:
    return 0.5 * (float(lo) + float(hi))


def _joint_span(lo: float, hi: float) -> float:
    return float(hi) - float(lo)


def _clamp_to_joint(name: str, x: float, limits) -> float:
    lo, hi = limits[name]
    return _clip(x, lo, hi)


def _clamp_joint_dict(joints, limits):
    out = dict(joints)
    for name in JOINT_NAMES:
        out[name] = _clamp_to_joint(name, out.get(name, _joint_mid(*limits[name])), limits)
    out[GRIPPER_NAME] = _clip(out.get(GRIPPER_NAME, 1.0), 0.0, 1.0)
    if "__diagnostics__" in joints:
        out["__diagnostics__"] = joints["__diagnostics__"]
    return out


def _workspace_min_max():
    wmin = np.array(getattr(val, "ARUCO_WORKSPACE_MIN", (-0.18, -0.12, 0.02)), dtype=np.float64)
    wmax = np.array(getattr(val, "ARUCO_WORKSPACE_MAX", (0.18, 0.18, 0.28)), dtype=np.float64)
    return wmin, wmax


def _normalize_workspace_xyz(xyz):
    xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
    wmin, wmax = _workspace_min_max()
    denom = np.where(np.abs(wmax - wmin) < 1e-9, 1.0, wmax - wmin)
    return np.clip((xyz - wmin) / denom, 0.0, 1.0)


def _ee_geom(model_calibration: Optional[dict] = None) -> dict:
    cal = model_calibration if isinstance(model_calibration, dict) else _load_model_calibration_file()
    lengths = cal.get("link_lengths_m", {}) if isinstance(cal.get("link_lengths_m"), dict) else {}
    tool_offset = cal.get("tool_xyz_offset_m", None)
    if tool_offset is None:
        tool_offset = [
            _get_value("IK_TOOL_A_M", 0.025) + _get_value("IK_TOOL_B_M", 0.025),
            0.0,
            0.0,
        ]
    tool_a = float(lengths.get("tool_a", _get_value("IK_TOOL_A_M", 0.025)))
    tool_b = float(lengths.get("tool_b", _get_value("IK_TOOL_B_M", 0.025)))
    legacy = getattr(val, "IK_TOOL_M", None)
    if legacy is not None and not hasattr(val, "IK_TOOL_A_M") and not hasattr(val, "IK_TOOL_B_M"):
        tool_a = float(legacy) / 2.0
        tool_b = float(legacy) / 2.0
    return {
        "base_height": float(lengths.get("base_height", _get_value("IK_BASE_HEIGHT_M", 0.06))),
        "upper_arm": float(lengths.get("upper_arm", _get_value("IK_LINK1_M", 0.115))),
        "forearm": float(lengths.get("forearm", _get_value("IK_LINK2_M", 0.115))),
        "wrist_to_yaw": float(lengths.get("wrist_to_yaw", _get_value("IK_WRIST_TO_YAW_M", 0.0))),
        "tool_a": tool_a,
        "tool_b": tool_b,
        "tool": tool_a + tool_b,
        "tool_xyz_offset_m": np.asarray(tool_offset, dtype=np.float64).reshape(3),
        "r_min": _get_value("IK_RADIAL_MIN_M", 0.04),
        "joint_zero_offsets_rad": np.asarray(cal.get("joint_zero_offsets_rad", [0.0] * 7), dtype=np.float64).reshape(-1)[:7] if len(cal.get("joint_zero_offsets_rad", [0.0] * 7)) >= 7 else np.zeros(7),
        "joint_axis_signs": np.asarray(cal.get("joint_axis_signs", [1.0] * 7), dtype=np.float64).reshape(-1)[:7] if len(cal.get("joint_axis_signs", [1.0] * 7)) >= 7 else np.ones(7),
    }


def _joint_dict_to_vector(joints) -> np.ndarray:
    if isinstance(joints, np.ndarray):
        arr = np.asarray(joints, dtype=np.float64).reshape(-1)
        if arr.size < 7:
            arr = np.pad(arr, (0, 7 - arr.size), mode="constant")
        return arr[:7].astype(np.float64)
    if isinstance(joints, (list, tuple)):
        arr = np.asarray(joints, dtype=np.float64).reshape(-1)
        if arr.size < 7:
            arr = np.pad(arr, (0, 7 - arr.size), mode="constant")
        return arr[:7].astype(np.float64)
    if not isinstance(joints, dict):
        return np.zeros(7, dtype=np.float64)
    return np.array([float(joints.get(name, 0.0)) for name in JOINT_NAMES], dtype=np.float64)


def _vector_to_joint_dict(vec, gripper_open01=1.0) -> Dict[str, float]:
    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    if v.size < 7:
        v = np.pad(v, (0, 7 - v.size), mode="constant")
    out = {name: float(v[i]) for i, name in enumerate(JOINT_NAMES)}
    out[GRIPPER_NAME] = _clip(gripper_open01, 0.0, 1.0)
    return out


def _preferred_posture(limits) -> np.ndarray:
    raw = getattr(val, "IK_PREFERRED_POSTURE_RAD", None)
    if raw is not None:
        try:
            arr = np.asarray(raw, dtype=np.float64).reshape(-1)
            if arr.size >= 7:
                return arr[:7]
        except Exception:
            pass
    pref = np.array([_joint_mid(*limits[n]) for n in JOINT_NAMES], dtype=np.float64)
    # Mildly open elbow and keep the redundant wrist pitch near neutral unless tuned.
    pref[2] = _clip(_get_value("IK_PREFERRED_ELBOW_RAD", 0.65), *limits["elbow_flex"])
    pref[6] = _clip(_get_value("IK_PREFERRED_WRIST_PITCH_RAD", 0.0), *limits["wrist_pitch"])
    return pref


def _limit_cost_vec(q_vec, limits) -> float:
    cost = 0.0
    for i, name in enumerate(JOINT_NAMES):
        lo, hi = limits[name]
        mid = _joint_mid(lo, hi)
        half = max(0.5 * (hi - lo), 1e-6)
        s = (float(q_vec[i]) - mid) / half
        cost += s ** 4
    return float(cost)


def _limit_gradient(q_vec, limits) -> np.ndarray:
    grad = np.zeros(7, dtype=np.float64)
    for i, name in enumerate(JOINT_NAMES):
        lo, hi = limits[name]
        mid = _joint_mid(lo, hi)
        half = max(0.5 * (hi - lo), 1e-6)
        s = (float(q_vec[i]) - mid) / half
        grad[i] = 4.0 * (s ** 3) / half
    return grad


def _posture_cost_vec(q_vec, q_pref, weights=None) -> float:
    d = np.asarray(q_vec, dtype=np.float64) - np.asarray(q_pref, dtype=np.float64)
    if weights is None:
        return float(np.dot(d, d))
    w = np.asarray(weights, dtype=np.float64).reshape(7)
    return float(np.dot(w * d, d))


def _chain_axes_local():
    # The added double of the original wrist joint is represented as wrist_yaw,
    # followed by roll and pitch so the last three wrist joints naturally absorb
    # orientation error while shoulder/elbow stay comfortable.
    return ("z", "y", "y", "y", "z", "x", "y")


def _axis_vector(axis: str) -> np.ndarray:
    if axis == "x":
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if axis == "y":
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if axis == "z":
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    raise ValueError(axis)


def _fk_chain_details(joints, geom=None) -> dict:
    if geom is None:
        geom = _ee_geom()
    q_cmd = _joint_dict_to_vector(joints)
    signs = np.asarray(geom.get("joint_axis_signs", np.ones(7)), dtype=np.float64).reshape(7)
    offsets = np.asarray(geom.get("joint_zero_offsets_rad", np.zeros(7)), dtype=np.float64).reshape(7)
    q_model = signs * q_cmd + offsets

    T = np.eye(4, dtype=np.float64)
    origins = []
    axes_world = []
    frames = []
    link_points = [T[:3, 3].copy()]

    # Base lift to shoulder.
    T = T @ _T_translate(0.0, 0.0, float(geom["base_height"]))
    link_points.append(T[:3, 3].copy())

    axes = _chain_axes_local()
    for i, axis in enumerate(axes):
        origins.append(T[:3, 3].copy())
        axes_world.append((T[:3, :3] @ _axis_vector(axis)) * float(signs[i]))
        frames.append(T.copy())
        T = T @ _T_rot(axis, float(q_model[i]))

        if i == 1:  # shoulder lift -> elbow
            T = T @ _T_translate(float(geom["upper_arm"]), 0.0, 0.0)
            link_points.append(T[:3, 3].copy())
        elif i == 2:  # elbow -> wrist flex
            T = T @ _T_translate(float(geom["forearm"]), 0.0, 0.0)
            link_points.append(T[:3, 3].copy())
        elif i == 3 and abs(float(geom.get("wrist_to_yaw", 0.0))) > 1e-12:
            T = T @ _T_translate(float(geom.get("wrist_to_yaw", 0.0)), 0.0, 0.0)
            link_points.append(T[:3, 3].copy())
        elif i == 5:
            T = T @ _T_translate(float(geom["tool_a"]), 0.0, 0.0)
            link_points.append(T[:3, 3].copy())

    # Tool-center offset after wrist pitch. The calibrated tool offset takes priority;
    # the nominal offset equals [tool_b, 0, 0] because tool_a was applied after roll.
    tool_offset = np.asarray(geom.get("tool_xyz_offset_m", [float(geom["tool_b"]), 0.0, 0.0]), dtype=np.float64).reshape(3)
    if np.allclose(tool_offset, [float(geom["tool_a"]) + float(geom["tool_b"]), 0.0, 0.0]):
        tool_offset = np.array([float(geom["tool_b"]), 0.0, 0.0], dtype=np.float64)
    T = T @ _T_translate(float(tool_offset[0]), float(tool_offset[1]), float(tool_offset[2]))
    link_points.append(T[:3, 3].copy())

    return {
        "T": T,
        "R": T[:3, :3].copy(),
        "p": T[:3, 3].copy(),
        "origins": np.asarray(origins, dtype=np.float64),
        "axes_world": np.asarray(axes_world, dtype=np.float64),
        "frames": frames,
        "link_points": np.asarray(link_points, dtype=np.float64),
        "q_model": q_model,
        "q_cmd": q_cmd,
    }


def _fk_all(joints, geom=None):
    return _fk_chain_details(joints, geom)


def _fk_arm_position(joints, geom=None) -> np.ndarray:
    return _fk_chain_details(joints, geom)["p"]


def _fk_orientation_matrix(joints, geom=None) -> np.ndarray:
    return _fk_chain_details(joints, geom)["R"]


def _fk_rpy(joints, geom=None) -> np.ndarray:
    return _matrix_to_rpy(_fk_orientation_matrix(joints, geom))


def fk_pose(joints, geom=None) -> dict:
    d = _fk_chain_details(joints, geom)
    return {"position": d["p"], "rotation": d["R"], "rpy": _matrix_to_rpy(d["R"]), "link_points": d["link_points"], "T": d["T"]}


def _analytic_geometric_jacobian(joints, geom=None, var_names: Optional[Sequence[str]] = None) -> np.ndarray:
    if geom is None:
        geom = _ee_geom()
    names = tuple(var_names or JOINT_NAMES)
    details = _fk_chain_details(joints, geom)
    p_e = details["p"]
    J = np.zeros((6, len(names)), dtype=np.float64)
    for col, name in enumerate(names):
        if name not in JOINT_NAMES:
            continue
        i = JOINT_NAMES.index(name)
        z = details["axes_world"][i]
        o = details["origins"][i]
        J[:3, col] = np.cross(z, p_e - o)
        J[3:, col] = z
    return J


def _self_collision_cost(joints, geom=None) -> float:
    enabled = bool(getattr(val, "IK_ENABLE_SELF_COLLISION_COST", False))
    if not enabled:
        return 0.0
    pts = _fk_chain_details(joints, geom)["link_points"]
    safe = _get_value("IK_SELF_COLLISION_SAFE_DIST_M", 0.035)
    cost = 0.0
    # Lightweight point-segment proxy for non-adjacent joint/link points.
    for i in range(len(pts)):
        for j in range(i + 2, len(pts)):
            d = float(np.linalg.norm(pts[i] - pts[j]))
            if d < safe:
                cost += (safe - d) ** 2
    return float(cost)


def _candidate_diagnostics(joints, target_xyz, target_R, previous_joints, limits, geom, iterations=0, converged=False) -> dict:
    q = _joint_dict_to_vector(joints)
    p = _fk_arm_position(joints, geom)
    R = _fk_orientation_matrix(joints, geom)
    pos_err = float(np.linalg.norm(np.asarray(target_xyz, dtype=np.float64).reshape(3) - p))
    orient_err = float(np.linalg.norm(_orientation_error_vec(target_R, R)))
    q_pref = _preferred_posture(limits)
    if previous_joints is not None:
        q_prev = _joint_dict_to_vector(previous_joints)
        continuity = float(np.linalg.norm([_angle_diff(q[i], q_prev[i]) for i in range(7)]))
    else:
        continuity = 0.0
    limit_cost = _limit_cost_vec(q, limits)
    posture_cost = _posture_cost_vec(q, q_pref)
    collision_cost = _self_collision_cost(joints, geom)
    pos_thresh = _get_value("IK_REACHABLE_POSITION_ERR_M", _get_value("IK_ABORT_POSITION_ERR_M", 0.02))
    orient_thresh = _get_value("IK_REACHABLE_ORIENTATION_ERR_RAD", 0.35)
    return {
        "position_error_m": pos_err,
        "orientation_error_rad": orient_err,
        "iterations_used": int(iterations),
        "reachable": bool(pos_err <= pos_thresh and orient_err <= orient_thresh),
        "converged": bool(converged),
        "limit_cost": float(limit_cost),
        "posture_cost": float(posture_cost),
        "continuity_cost": float(continuity),
        "collision_cost": float(collision_cost),
        "fk_position_m": p.tolist(),
        "fk_rpy_rad": _matrix_to_rpy(R).tolist(),
    }


def _candidate_cost(candidate, target_xyz, target_R, previous_joints, geom, limits=None):
    if limits is None:
        limits = _get_joint_limits()
    q = _joint_dict_to_vector(candidate)
    p = _fk_arm_position(candidate, geom)
    R = _fk_orientation_matrix(candidate, geom)
    ep = float(np.linalg.norm(np.asarray(target_xyz, dtype=np.float64).reshape(3) - p))
    eR = float(np.linalg.norm(_orientation_error_vec(target_R, R)))
    w_p = _get_value("IK_COST_POSITION", 1.0)
    w_R = _get_value("IK_COST_ORIENTATION", 0.25)
    w_lim = _get_value("IK_COST_LIMIT", 0.02)
    w_post = _get_value("IK_COST_POSTURE", 0.01)
    w_cont = _get_value("IK_COST_CONTINUITY", 0.05)
    q_pref = _preferred_posture(limits)
    cost = w_p * ep * ep + w_R * eR * eR + w_lim * _limit_cost_vec(q, limits) + w_post * _posture_cost_vec(q, q_pref)
    if previous_joints is not None:
        q_prev = _joint_dict_to_vector(previous_joints)
        cost += w_cont * float(np.dot(q - q_prev, q - q_prev))
    cost += _get_value("IK_COST_COLLISION", 0.0) * _self_collision_cost(candidate, geom)
    return float(cost)


def _make_seed_from_previous(previous_joints, limits, gripper_open01) -> Dict[str, float]:
    if previous_joints is not None:
        d = _vector_to_joint_dict(_joint_dict_to_vector(previous_joints), gripper_open01)
    else:
        d = {name: _joint_mid(*limits[name]) for name in JOINT_NAMES}
        d["elbow_flex"] = _clip(_get_value("IK_PREFERRED_ELBOW_RAD", 0.65), *limits["elbow_flex"])
        d["wrist_pitch"] = _clip(_get_value("IK_PREFERRED_WRIST_PITCH_RAD", 0.0), *limits["wrist_pitch"])
        d[GRIPPER_NAME] = gripper_open01
    return _clamp_joint_dict(d, limits)


def _geometric_seed(target_xyz, target_R, limits, gripper_open01) -> Dict[str, float]:
    geom = _ee_geom()
    xyz = np.asarray(target_xyz, dtype=np.float64).reshape(3)
    tool = float(geom.get("tool", 0.05))
    wrist = xyz - np.asarray(target_R, dtype=np.float64).reshape(3, 3) @ np.array([tool, 0.0, 0.0], dtype=np.float64)
    pan = math.atan2(float(wrist[1]), float(wrist[0])) if np.linalg.norm(wrist[:2]) > 1e-9 else 0.0
    r = max(float(geom.get("r_min", 0.04)), math.hypot(float(wrist[0]), float(wrist[1])))
    z = float(wrist[2]) - float(geom["base_height"])
    l1 = max(float(geom["upper_arm"]), 1e-6)
    l2 = max(float(geom["forearm"]), 1e-6)
    reach = math.hypot(r, z)
    reach = _clip(reach, abs(l1 - l2) + 1e-5, l1 + l2 - 1e-5)
    cos_elbow = _clip((reach * reach - l1 * l1 - l2 * l2) / (2.0 * l1 * l2), -1.0, 1.0)
    elbow_conv = math.acos(cos_elbow)
    shoulder_conv = math.atan2(z, r) - math.atan2(l2 * math.sin(elbow_conv), l1 + l2 * math.cos(elbow_conv))
    shoulder = -shoulder_conv
    elbow = -elbow_conv
    q = {
        "shoulder_pan": pan,
        "shoulder_lift": shoulder,
        "elbow_flex": elbow,
        "wrist_flex": 0.0,
        "wrist_yaw": 0.0,
        "wrist_roll": 0.0,
        "wrist_pitch": 0.0,
        GRIPPER_NAME: gripper_open01,
    }
    return _clamp_joint_dict(q, limits)


def _refine_dls(initial_joints, target_xyz, target_R, limits, geom, previous_joints=None) -> Tuple[Dict[str, float], dict]:
    target_xyz = np.asarray(target_xyz, dtype=np.float64).reshape(3)
    target_R = np.asarray(target_R, dtype=np.float64).reshape(3, 3)
    q = _joint_dict_to_vector(_clamp_joint_dict(initial_joints, limits))
    gripper_open01 = float(initial_joints.get(GRIPPER_NAME, 1.0)) if isinstance(initial_joints, dict) else 1.0

    max_iters = int(_get_value("IK_DLS_MAX_ITERS", _get_value("IK_MAX_ITERS", 80)))
    damping = _get_value("IK_DLS_DAMPING", 0.08)
    max_step = _get_value("IK_DLS_MAX_STEP_RAD", 0.20)
    step_eta = _get_value("IK_DLS_STEP_GAIN", 0.85)
    pos_tol = _get_value("IK_RESIDUAL_THRESH", 1e-4)
    orient_tol = _get_value("IK_DLS_ORIENTATION_TOL", 2e-3)
    w_xyz = _get_value("IK_DLS_POSITION_GAIN", 1.0)
    w_ori = _get_value("IK_DLS_ORIENTATION_GAIN", 0.45)
    W = np.diag([w_xyz, w_xyz, w_xyz, w_ori, w_ori, w_ori]).astype(np.float64)

    q_prev = None if previous_joints is None else _joint_dict_to_vector(previous_joints)
    q_pref = _preferred_posture(limits)
    beta_lim = _get_value("IK_NULLSPACE_LIMIT_GAIN", 0.025)
    beta_cont = _get_value("IK_NULLSPACE_CONTINUITY_GAIN", 0.035)
    beta_post = _get_value("IK_NULLSPACE_POSTURE_GAIN", 0.020)
    beta_coll = _get_value("IK_NULLSPACE_COLLISION_GAIN", 0.0)

    best = _clamp_joint_dict(_vector_to_joint_dict(q, gripper_open01), limits)
    best_cost = _candidate_cost(best, target_xyz, target_R, previous_joints, geom, limits)
    converged = False
    it_used = 0

    for it in range(max_iters):
        it_used = it + 1
        joints = _clamp_joint_dict(_vector_to_joint_dict(q, gripper_open01), limits)
        q = _joint_dict_to_vector(joints)
        details = _fk_chain_details(joints, geom)
        pos_err = target_xyz - details["p"]
        ori_err = _orientation_error_vec(target_R, details["R"])
        if float(np.linalg.norm(pos_err)) <= pos_tol and float(np.linalg.norm(ori_err)) <= orient_tol:
            converged = True
            best = joints
            break

        J = _analytic_geometric_jacobian(joints, geom, JOINT_NAMES)
        Jw = W @ J
        ew = W @ np.concatenate([pos_err, ori_err])
        A = Jw @ Jw.T + (damping ** 2) * np.eye(6, dtype=np.float64)
        try:
            y = np.linalg.solve(A, ew)
        except np.linalg.LinAlgError:
            y = np.linalg.lstsq(A, ew, rcond=None)[0]
        dq_task = Jw.T @ y

        # Damped pseudoinverse for null-space projection. This is intentionally
        # separate from the weighted task update so secondary costs do not steal
        # authority from the primary Cartesian pose objective.
        try:
            J_pinv = J.T @ np.linalg.inv(J @ J.T + (damping ** 2) * np.eye(6, dtype=np.float64))
        except np.linalg.LinAlgError:
            J_pinv = np.linalg.pinv(J)
        N = np.eye(7, dtype=np.float64) - J_pinv @ J

        grad = np.zeros(7, dtype=np.float64)
        if beta_lim > 0:
            grad += beta_lim * _limit_gradient(q, limits)
        if beta_cont > 0 and q_prev is not None:
            grad += beta_cont * 2.0 * (q - q_prev)
        if beta_post > 0:
            grad += beta_post * 2.0 * (q - q_pref)
        # Collision gradient is optional and finite-differenced only when enabled.
        if beta_coll > 0 and bool(getattr(val, "IK_ENABLE_SELF_COLLISION_COST", False)):
            eps = 1e-4
            base_c = _self_collision_cost(joints, geom)
            gcol = np.zeros(7, dtype=np.float64)
            for i in range(7):
                q_eps = q.copy(); q_eps[i] += eps
                gcol[i] = (_self_collision_cost(_vector_to_joint_dict(q_eps, gripper_open01), geom) - base_c) / eps
            grad += beta_coll * gcol

        dq = dq_task - N @ grad
        max_abs = float(np.max(np.abs(dq))) if dq.size else 0.0
        if max_abs > max_step:
            dq *= max_step / max(max_abs, 1e-12)
        dq *= step_eta

        current_cost = _candidate_cost(joints, target_xyz, target_R, previous_joints, geom, limits)
        accepted = False
        for scale in (1.0, 0.5, 0.25, 0.1):
            q_trial = q + scale * dq
            trial = _clamp_joint_dict(_vector_to_joint_dict(q_trial, gripper_open01), limits)
            trial_q = _joint_dict_to_vector(trial)
            trial_cost = _candidate_cost(trial, target_xyz, target_R, previous_joints, geom, limits)
            if trial_cost <= current_cost or scale == 0.1:
                q = trial_q
                joints = trial
                accepted = True
                if trial_cost < best_cost:
                    best = trial
                    best_cost = trial_cost
                break
        if not accepted or np.linalg.norm(dq) < 1e-8:
            break

    diag = _candidate_diagnostics(best, target_xyz, target_R, previous_joints, limits, geom, iterations=it_used, converged=converged)
    best["__diagnostics__"] = diag
    return best, diag


def solve_ik_pose(target_xyz, target_R=None, gripper_open01: float = 1.0, lerobot_calibration=None, previous_joints=None, model_calibration=None) -> dict:
    if target_R is None:
        target_R = np.eye(3, dtype=np.float64)
    target_R = np.asarray(target_R, dtype=np.float64).reshape(3, 3)
    return solve_ik_from_target(target_xyz, _matrix_to_rpy(target_R), gripper_open01, lerobot_calibration, previous_joints, target_R=target_R, model_calibration=model_calibration)


def solve_ik_from_target(
    target_xyz,
    target_rpy=None,
    gripper_open01: float = 1.0,
    lerobot_calibration=None,
    previous_joints=None,
    *,
    target_R=None,
    return_diagnostics: bool = False,
    model_calibration: Optional[dict] = None,
):
    if lerobot_calibration is None:
        lerobot_calibration = _load_lerobot_calibration_file()
    base_limits = _get_joint_limits()
    limits, _joint_cal = _merge_limits(base_limits, lerobot_calibration)
    geom = _ee_geom(model_calibration)

    try:
        xyz = np.asarray(target_xyz, dtype=np.float64).reshape(3)
    except Exception:
        xyz = np.array([0.0, 0.0, 0.12], dtype=np.float64)
    R = np.asarray(target_R, dtype=np.float64).reshape(3, 3) if target_R is not None else _rpy_to_matrix(target_rpy)

    seeds = []
    seeds.append(_make_seed_from_previous(previous_joints, limits, gripper_open01))
    seeds.append(_geometric_seed(xyz, R, limits, gripper_open01))
    neutral = {name: _joint_mid(*limits[name]) for name in JOINT_NAMES}
    neutral["elbow_flex"] = _clip(_get_value("IK_PREFERRED_ELBOW_RAD", 0.65), *limits["elbow_flex"])
    neutral[GRIPPER_NAME] = gripper_open01
    seeds.append(_clamp_joint_dict(neutral, limits))

    # Add mirrored elbow branch and a few wrist-pitch redundancy seeds.
    geo = seeds[1]
    mirrored = dict(geo)
    mirrored["elbow_flex"] = _clamp_to_joint("elbow_flex", -float(geo["elbow_flex"]), limits)
    seeds.append(_clamp_joint_dict(mirrored, limits))
    for w_pitch in (0.0, 0.35, -0.35):
        s = dict(seeds[0])
        s["wrist_pitch"] = _clamp_to_joint("wrist_pitch", w_pitch, limits)
        seeds.append(s)

    best = None
    best_diag = None
    best_cost = float("inf")
    seen = set()
    for seed in seeds:
        key = tuple(round(float(seed[n]), 4) for n in JOINT_NAMES)
        if key in seen:
            continue
        seen.add(key)
        sol, diag = _refine_dls(seed, xyz, R, limits, geom, previous_joints)
        cost = _candidate_cost(sol, xyz, R, previous_joints, geom, limits)
        if cost < best_cost:
            best = sol
            best_diag = diag
            best_cost = cost

    if best is None:
        best = _make_seed_from_previous(previous_joints, limits, gripper_open01)
        best_diag = _candidate_diagnostics(best, xyz, R, previous_joints, limits, geom, iterations=0, converged=False)
    best[GRIPPER_NAME] = _clip(gripper_open01, 0.0, 1.0)
    best["__diagnostics__"] = best_diag
    if return_diagnostics:
        return best, best_diag
    return best