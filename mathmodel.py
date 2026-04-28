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


def _skew_to_vec(S) -> np.ndarray:
    S = np.asarray(S, dtype=np.float64).reshape(3, 3)
    return np.array([S[2, 1], S[0, 2], S[1, 0]], dtype=np.float64)


def _project_to_rotation(R) -> np.ndarray:
    """Project a noisy 3x3 matrix onto SO(3)."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    try:
        U, _, Vt = np.linalg.svd(R)
        Rp = U @ Vt
        if np.linalg.det(Rp) < 0.0:
            U[:, -1] *= -1.0
            Rp = U @ Vt
        return Rp.astype(np.float64)
    except Exception:
        return np.eye(3, dtype=np.float64)


def _so3_log(R) -> np.ndarray:
    """Robust SO(3) logarithmic orientation error vector."""
    R = _project_to_rotation(R)
    cos_theta = _clip((float(np.trace(R)) - 1.0) * 0.5, -1.0, 1.0)
    theta = math.acos(cos_theta)
    if theta < 1e-7:
        return 0.5 * _skew_to_vec(R - R.T)
    if math.pi - theta < 1e-5:
        axis = np.empty(3, dtype=np.float64)
        axis[0] = math.sqrt(max(0.0, (R[0, 0] + 1.0) * 0.5))
        axis[1] = math.sqrt(max(0.0, (R[1, 1] + 1.0) * 0.5))
        axis[2] = math.sqrt(max(0.0, (R[2, 2] + 1.0) * 0.5))
        if R[2, 1] - R[1, 2] < 0.0:
            axis[0] = -axis[0]
        if R[0, 2] - R[2, 0] < 0.0:
            axis[1] = -axis[1]
        if R[1, 0] - R[0, 1] < 0.0:
            axis[2] = -axis[2]
        n = float(np.linalg.norm(axis))
        if n < 1e-9:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            axis /= n
        return theta * axis
    return (theta / (2.0 * math.sin(theta))) * _skew_to_vec(R - R.T)


def _orientation_error_vec(R_target, R_current) -> np.ndarray:
    R_err = _project_to_rotation(R_target) @ _project_to_rotation(R_current).T
    return _so3_log(R_err)

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
    """Load an optional measured kinematic model.

    The arm works with the nominal hardcoded SO100-style chain today.  When a
    URDF/CAD-derived calibration artifact is added later, place a JSON file in
    calibration_artifacts/ or point IK_MODEL_CALIBRATION_FILE at it. Supported
    keys include link_lengths_m, tool_xyz_offset_m, joint_zero_offsets_rad,
    joint_axis_signs, joint_axes_local, and collision_capsule_radius_m.
    """
    candidates = []
    if path:
        candidates.append(path)
    candidates.extend([
        str(getattr(val, "IK_MODEL_CALIBRATION_FILE", "")),
        str(getattr(val, "ROBOT_MODEL_CALIBRATION_FILE", "")),
        "calibration_artifacts/robot_model_calibration.json",
        "calibration_artifacts/kinematic_model.json",
        "calibration_data/robot_model_calibration.json",
        "calibration_data/kinematic_model.json",
    ])
    for cand in candidates:
        data = _load_json_file(cand)
        if isinstance(data, dict):
            data = dict(data)
            data["__source_path__"] = str(_resolve_project_path(cand))
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


def _as_len7_array(value, default) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size >= 7:
            return arr[:7].astype(np.float64)
    except Exception:
        pass
    return np.asarray(default, dtype=np.float64).reshape(7)


def _as_axis_tuple(value) -> Tuple[str, ...]:
    default = ("z", "y", "y", "y", "z", "x", "y")
    if isinstance(value, (list, tuple)) and len(value) >= 7:
        out = []
        for x in value[:7]:
            sx = str(x).strip().lower()
            out.append(sx if sx in {"x", "y", "z"} else default[len(out)])
        return tuple(out)
    return default


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
        "joint_zero_offsets_rad": _as_len7_array(cal.get("joint_zero_offsets_rad", [0.0] * 7), np.zeros(7)),
        "joint_axis_signs": _as_len7_array(cal.get("joint_axis_signs", [1.0] * 7), np.ones(7)),
        "joint_axes_local": _as_axis_tuple(cal.get("joint_axes_local", getattr(val, "IK_JOINT_AXES_LOCAL", None))),
        "collision_capsule_radius_m": float(cal.get("collision_capsule_radius_m", _get_value("IK_COLLISION_CAPSULE_RADIUS_M", 0.018))),
        "model_source_path": str(cal.get("__source_path__", "nominal_hardcoded_chain")),
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


def _chain_axes_local(geom=None):
    # The nominal added double of the original wrist joint is represented as
    # wrist_yaw, followed by roll and pitch. A future calibration artifact can
    # override this through joint_axes_local without changing solver callers.
    if isinstance(geom, dict):
        axes = geom.get("joint_axes_local")
        if isinstance(axes, (list, tuple)) and len(axes) >= 7:
            return tuple(str(a).lower() for a in axes[:7])
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

    axes = _chain_axes_local(geom)
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


def _segment_segment_distance(a0, a1, b0, b1) -> float:
    """Minimum distance between two 3D line segments."""
    a0 = np.asarray(a0, dtype=np.float64).reshape(3)
    a1 = np.asarray(a1, dtype=np.float64).reshape(3)
    b0 = np.asarray(b0, dtype=np.float64).reshape(3)
    b1 = np.asarray(b1, dtype=np.float64).reshape(3)
    u = a1 - a0
    v = b1 - b0
    w = a0 - b0
    A = float(np.dot(u, u))
    B = float(np.dot(u, v))
    C = float(np.dot(v, v))
    D = float(np.dot(u, w))
    E = float(np.dot(v, w))
    den = A * C - B * B
    if A < 1e-12 and C < 1e-12:
        return float(np.linalg.norm(a0 - b0))
    if A < 1e-12:
        s = 0.0
        t = _clip(E / max(C, 1e-12), 0.0, 1.0)
    elif C < 1e-12:
        t = 0.0
        s = _clip(-D / max(A, 1e-12), 0.0, 1.0)
    elif abs(den) < 1e-12:
        s = 0.0
        t = _clip(E / max(C, 1e-12), 0.0, 1.0)
    else:
        s = _clip((B * E - C * D) / den, 0.0, 1.0)
        t = _clip((A * E - B * D) / den, 0.0, 1.0)
        # Re-project once after clamping for better endpoint cases.
        s = _clip((B * t - D) / max(A, 1e-12), 0.0, 1.0)
        t = _clip((B * s + E) / max(C, 1e-12), 0.0, 1.0)
    pa = a0 + s * u
    pb = b0 + t * v
    return float(np.linalg.norm(pa - pb))


def _link_segments_from_points(pts) -> list[tuple[np.ndarray, np.ndarray]]:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    segs = []
    for i in range(max(0, len(pts) - 1)):
        if float(np.linalg.norm(pts[i + 1] - pts[i])) > 1e-8:
            segs.append((pts[i], pts[i + 1]))
    return segs


def _self_collision_cost(joints, geom=None) -> float:
    if not bool(getattr(val, "IK_ENABLE_SELF_COLLISION_COST", False)):
        return 0.0
    if geom is None:
        geom = _ee_geom()
    details = _fk_chain_details(joints, geom)
    segs = _link_segments_from_points(details["link_points"])
    radius = float(geom.get("collision_capsule_radius_m", _get_value("IK_COLLISION_CAPSULE_RADIUS_M", 0.018)))
    safe = _get_value("IK_SELF_COLLISION_SAFE_DIST_M", 0.035)
    min_clearance = max(safe, 2.0 * radius)
    cost = 0.0
    for i, (a0, a1) in enumerate(segs):
        for j, (b0, b1) in enumerate(segs):
            if j <= i + 1:
                continue
            d = _segment_segment_distance(a0, a1, b0, b1)
            if d < min_clearance:
                cost += (min_clearance - d) ** 2
    floor_z = _get_value("IK_COLLISION_FLOOR_Z_M", -999.0)
    if floor_z > -10.0:
        for p in details["link_points"]:
            clearance = float(p[2]) - floor_z
            if clearance < radius:
                cost += (radius - clearance) ** 2
    return float(cost)


def _elbow_cost(joints, geom=None) -> float:
    gain_enabled = bool(getattr(val, "IK_ENABLE_ELBOW_REGION_COST", True))
    if not gain_enabled:
        return 0.0
    details = _fk_chain_details(joints, geom)
    pts = details["link_points"]
    if len(pts) < 3:
        return 0.0
    elbow = np.asarray(pts[2], dtype=np.float64)
    cost = 0.0
    pref = getattr(val, "IK_ELBOW_PREFERRED_WORLD_M", None)
    if pref is not None:
        try:
            pref_v = np.asarray(pref, dtype=np.float64).reshape(3)
            d = elbow - pref_v
            cost += float(np.dot(d, d))
        except Exception:
            pass
    min_z = _get_value("IK_ELBOW_MIN_Z_M", -999.0)
    if min_z > -10.0 and elbow[2] < min_z:
        cost += float((min_z - elbow[2]) ** 2)
    min_radial = _get_value("IK_ELBOW_MIN_RADIAL_M", 0.03)
    radial = float(np.linalg.norm(elbow[:2]))
    if radial < min_radial:
        cost += float((min_radial - radial) ** 2)
    return float(cost)


def _wrist_awkwardness_cost(q_vec) -> float:
    q = np.asarray(q_vec, dtype=np.float64).reshape(7)
    # Extra wrist pitch/yaw/roll are redundancy; keep them useful but avoid
    # extreme corkscrew poses unless needed for the Cartesian task.
    weights = np.asarray(getattr(val, "IK_WRIST_AWKWARDNESS_WEIGHTS", [0, 0, 0, 0.25, 0.45, 0.25, 0.45]), dtype=np.float64).reshape(-1)
    if weights.size < 7:
        weights = np.pad(weights, (0, 7 - weights.size), mode="constant")
    return float(np.dot(weights[:7] * q, q))


def _manipulability_score(joints, geom=None) -> float:
    J = _analytic_geometric_jacobian(joints, geom, JOINT_NAMES)
    # Normalize orientation rows slightly so the determinant is not dominated by
    # mixed meter/radian units.  This score is only a secondary objective.
    scale = np.diag([1.0, 1.0, 1.0, 0.20, 0.20, 0.20])
    JJ = (scale @ J) @ (scale @ J).T
    try:
        eig = np.linalg.eigvalsh(JJ)
        return float(math.sqrt(max(0.0, float(np.prod(np.maximum(eig, 1e-12))))))
    except Exception:
        return 0.0


def _secondary_cost(candidate, previous_joints, geom, limits) -> dict:
    q = _joint_dict_to_vector(candidate)
    q_pref = _preferred_posture(limits)
    continuity = 0.0
    if previous_joints is not None:
        q_prev = _joint_dict_to_vector(previous_joints)
        continuity = float(np.dot(q - q_prev, q - q_prev))
    manip = _manipulability_score(candidate, geom)
    manip_cost = 1.0 / max(manip, 1e-6)
    return {
        "limit": _limit_cost_vec(q, limits),
        "posture": _posture_cost_vec(q, q_pref),
        "continuity": continuity,
        "collision": _self_collision_cost(candidate, geom),
        "elbow": _elbow_cost(candidate, geom),
        "wrist": _wrist_awkwardness_cost(q),
        "manipulability": manip,
        "manipulability_cost": manip_cost,
    }


def _ik_mode_config(mode: Optional[str], previous_joints=None) -> dict:
    requested = str(mode or getattr(val, "IK_DEFAULT_MODE", "auto")).strip().lower()
    if requested not in {"auto", "teleop", "planning"}:
        requested = "auto"
    resolved = "teleop" if requested == "auto" and previous_joints is not None and bool(getattr(val, "IK_FAST_PREVIOUS_ONLY", False)) else requested
    if resolved == "auto":
        resolved = "planning"
    prefix = "IK_TELEOP" if resolved == "teleop" else "IK_PLANNING"
    return {
        "mode": resolved,
        "max_iters": int(_get_value(f"{prefix}_DLS_MAX_ITERS", _get_value("IK_DLS_MAX_ITERS", _get_value("IK_MAX_ITERS", 80)))),
        "damping": _get_value(f"{prefix}_DLS_DAMPING", _get_value("IK_DLS_DAMPING", 0.08)),
        "max_step": _get_value(f"{prefix}_DLS_MAX_STEP_RAD", _get_value("IK_DLS_MAX_STEP_RAD", 0.20)),
        "step_eta": _get_value(f"{prefix}_DLS_STEP_GAIN", _get_value("IK_DLS_STEP_GAIN", 0.85)),
        "pos_tol": _get_value(f"{prefix}_POSITION_TOL_M", _get_value("IK_RESIDUAL_THRESH", 1e-4)),
        "orient_tol": _get_value(f"{prefix}_ORIENTATION_TOL_RAD", _get_value("IK_DLS_ORIENTATION_TOL", 2e-3)),
        "w_xyz": _get_value(f"{prefix}_POSITION_GAIN", _get_value("IK_DLS_POSITION_GAIN", 1.0)),
        "w_ori": _get_value(f"{prefix}_ORIENTATION_GAIN", _get_value("IK_DLS_ORIENTATION_GAIN", 0.45)),
        "include_many_seeds": resolved == "planning",
        "strict": bool(getattr(val, f"{prefix}_STRICT_REACHABILITY", resolved == "planning")),
    }


def _candidate_diagnostics(joints, target_xyz, target_R, previous_joints, limits, geom, iterations=0, converged=False, mode="auto") -> dict:
    q = _joint_dict_to_vector(joints)
    p = _fk_arm_position(joints, geom)
    R = _fk_orientation_matrix(joints, geom)
    pos_vec = np.asarray(target_xyz, dtype=np.float64).reshape(3) - p
    ori_vec = _orientation_error_vec(target_R, R)
    pos_err = float(np.linalg.norm(pos_vec))
    orient_err = float(np.linalg.norm(ori_vec))
    sec = _secondary_cost(joints, previous_joints, geom, limits)
    pos_thresh = _get_value("IK_REACHABLE_POSITION_ERR_M", _get_value("IK_ABORT_POSITION_ERR_M", 0.02))
    orient_thresh = _get_value("IK_REACHABLE_ORIENTATION_ERR_RAD", 0.35)
    return {
        "success": bool(pos_err <= pos_thresh and orient_err <= orient_thresh),
        "failure_reason": "" if (pos_err <= pos_thresh and orient_err <= orient_thresh) else "position_or_orientation_error_above_threshold",
        "q_rad": q.tolist(),
        "achieved_xyz_m": p.tolist(),
        "achieved_rpy_rad": _matrix_to_rpy(R).tolist(),
        "position_error_m": pos_err,
        "position_error_xyz_m": pos_vec.tolist(),
        "orientation_error_rad": orient_err,
        "orientation_error_vec_rad": ori_vec.tolist(),
        "iterations_used": int(iterations),
        "reachable": bool(pos_err <= pos_thresh and orient_err <= orient_thresh),
        "converged": bool(converged),
        "ik_mode": str(mode),
        "limit_cost": float(sec["limit"]),
        "posture_cost": float(sec["posture"]),
        "continuity_cost": float(sec["continuity"]),
        "collision_cost": float(sec["collision"]),
        "elbow_cost": float(sec["elbow"]),
        "wrist_awkwardness_cost": float(sec["wrist"]),
        "manipulability": float(sec["manipulability"]),
        "fk_position_m": p.tolist(),
        "fk_rpy_rad": _matrix_to_rpy(R).tolist(),
        "model_source_path": str(geom.get("model_source_path", "nominal_hardcoded_chain")),
    }


def _candidate_cost(candidate, target_xyz, target_R, previous_joints, geom, limits=None, mode_cfg: Optional[dict] = None):
    if limits is None:
        limits = _get_joint_limits()
    p = _fk_arm_position(candidate, geom)
    R = _fk_orientation_matrix(candidate, geom)
    ep = float(np.linalg.norm(np.asarray(target_xyz, dtype=np.float64).reshape(3) - p))
    eR = float(np.linalg.norm(_orientation_error_vec(target_R, R)))
    mode_name = str((mode_cfg or {}).get("mode", ""))
    prefix = "IK_TELEOP" if mode_name == "teleop" else "IK_PLANNING" if mode_name == "planning" else "IK"
    w_p = _get_value(f"{prefix}_COST_POSITION", _get_value("IK_COST_POSITION", 1.0))
    w_R = _get_value(f"{prefix}_COST_ORIENTATION", _get_value("IK_COST_ORIENTATION", 0.25))
    w_lim = _get_value(f"{prefix}_COST_LIMIT", _get_value("IK_COST_LIMIT", 0.02))
    w_post = _get_value(f"{prefix}_COST_POSTURE", _get_value("IK_COST_POSTURE", 0.01))
    w_cont = _get_value(f"{prefix}_COST_CONTINUITY", _get_value("IK_COST_CONTINUITY", 0.05))
    w_coll = _get_value(f"{prefix}_COST_COLLISION", _get_value("IK_COST_COLLISION", 0.0))
    w_elbow = _get_value(f"{prefix}_COST_ELBOW", _get_value("IK_COST_ELBOW", 0.02))
    w_wrist = _get_value(f"{prefix}_COST_WRIST", _get_value("IK_COST_WRIST", 0.005))
    w_manip = _get_value(f"{prefix}_COST_MANIPULABILITY", _get_value("IK_COST_MANIPULABILITY", 0.0))
    sec = _secondary_cost(candidate, previous_joints, geom, limits)
    cost = w_p * ep * ep + w_R * eR * eR
    cost += w_lim * sec["limit"] + w_post * sec["posture"] + w_cont * sec["continuity"]
    cost += w_coll * sec["collision"] + w_elbow * sec["elbow"] + w_wrist * sec["wrist"]
    cost += w_manip * sec["manipulability_cost"]
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


def _geometric_seed(target_xyz, target_R, limits, gripper_open01, geom=None) -> Dict[str, float]:
    geom = geom if isinstance(geom, dict) else _ee_geom()
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


def _finite_difference_gradient_cost(q, gripper_open01, geom, limits, previous_joints, cost_name: str) -> np.ndarray:
    eps = _get_value("IK_NULLSPACE_FD_EPS_RAD", 1e-4)
    base = _clamp_joint_dict(_vector_to_joint_dict(q, gripper_open01), limits)

    def eval_cost(qv):
        j = _clamp_joint_dict(_vector_to_joint_dict(qv, gripper_open01), limits)
        if cost_name == "collision":
            return _self_collision_cost(j, geom)
        if cost_name == "elbow":
            return _elbow_cost(j, geom)
        if cost_name == "manipulability":
            return 1.0 / max(_manipulability_score(j, geom), 1e-6)
        if cost_name == "wrist":
            return _wrist_awkwardness_cost(_joint_dict_to_vector(j))
        return 0.0

    base_c = eval_cost(q)
    grad = np.zeros(7, dtype=np.float64)
    for i in range(7):
        q_eps = np.asarray(q, dtype=np.float64).copy()
        q_eps[i] += eps
        grad[i] = (eval_cost(q_eps) - base_c) / max(eps, 1e-12)
    return grad


def _refine_dls(initial_joints, target_xyz, target_R, limits, geom, previous_joints=None, mode_cfg: Optional[dict] = None) -> Tuple[Dict[str, float], dict]:
    target_xyz = np.asarray(target_xyz, dtype=np.float64).reshape(3)
    target_R = _project_to_rotation(target_R)
    mode_cfg = mode_cfg or _ik_mode_config(None, previous_joints)
    mode_name = str(mode_cfg.get("mode", "auto"))
    prefix = "IK_TELEOP" if mode_name == "teleop" else "IK_PLANNING" if mode_name == "planning" else "IK"

    q = _joint_dict_to_vector(_clamp_joint_dict(initial_joints, limits))
    gripper_open01 = float(initial_joints.get(GRIPPER_NAME, 1.0)) if isinstance(initial_joints, dict) else 1.0

    max_iters = int(mode_cfg.get("max_iters", _get_value("IK_DLS_MAX_ITERS", _get_value("IK_MAX_ITERS", 80))))
    damping = float(mode_cfg.get("damping", _get_value("IK_DLS_DAMPING", 0.08)))
    max_step = float(mode_cfg.get("max_step", _get_value("IK_DLS_MAX_STEP_RAD", 0.20)))
    step_eta = float(mode_cfg.get("step_eta", _get_value("IK_DLS_STEP_GAIN", 0.85)))
    pos_tol = float(mode_cfg.get("pos_tol", _get_value("IK_RESIDUAL_THRESH", 1e-4)))
    orient_tol = float(mode_cfg.get("orient_tol", _get_value("IK_DLS_ORIENTATION_TOL", 2e-3)))
    w_xyz = float(mode_cfg.get("w_xyz", _get_value("IK_DLS_POSITION_GAIN", 1.0)))
    w_ori = float(mode_cfg.get("w_ori", _get_value("IK_DLS_ORIENTATION_GAIN", 0.45)))
    W = np.diag([w_xyz, w_xyz, w_xyz, w_ori, w_ori, w_ori]).astype(np.float64)

    q_prev = None if previous_joints is None else _joint_dict_to_vector(previous_joints)
    q_pref = _preferred_posture(limits)
    beta_lim = _get_value(f"{prefix}_NULLSPACE_LIMIT_GAIN", _get_value("IK_NULLSPACE_LIMIT_GAIN", 0.025))
    beta_cont = _get_value(f"{prefix}_NULLSPACE_CONTINUITY_GAIN", _get_value("IK_NULLSPACE_CONTINUITY_GAIN", 0.035))
    beta_post = _get_value(f"{prefix}_NULLSPACE_POSTURE_GAIN", _get_value("IK_NULLSPACE_POSTURE_GAIN", 0.020))
    beta_coll = _get_value(f"{prefix}_NULLSPACE_COLLISION_GAIN", _get_value("IK_NULLSPACE_COLLISION_GAIN", 0.0))
    beta_elbow = _get_value(f"{prefix}_NULLSPACE_ELBOW_GAIN", _get_value("IK_NULLSPACE_ELBOW_GAIN", 0.0))
    beta_wrist = _get_value(f"{prefix}_NULLSPACE_WRIST_GAIN", _get_value("IK_NULLSPACE_WRIST_GAIN", 0.0))
    beta_manip = _get_value(f"{prefix}_NULLSPACE_MANIPULABILITY_GAIN", _get_value("IK_NULLSPACE_MANIPULABILITY_GAIN", 0.0))

    best = _clamp_joint_dict(_vector_to_joint_dict(q, gripper_open01), limits)
    best_cost = _candidate_cost(best, target_xyz, target_R, previous_joints, geom, limits, mode_cfg)
    converged = False
    it_used = 0
    last_cost = best_cost

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
        # Damped least squares: dq = Jw^T (Jw Jw^T + lambda^2 I)^-1 e.
        A = Jw @ Jw.T + (damping ** 2) * np.eye(6, dtype=np.float64)
        try:
            y = np.linalg.solve(A, ew)
        except np.linalg.LinAlgError:
            y = np.linalg.lstsq(A, ew, rcond=None)[0]
        dq_task = Jw.T @ y

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
        if beta_wrist > 0:
            grad += beta_wrist * _finite_difference_gradient_cost(q, gripper_open01, geom, limits, previous_joints, "wrist")
        if beta_elbow > 0:
            grad += beta_elbow * _finite_difference_gradient_cost(q, gripper_open01, geom, limits, previous_joints, "elbow")
        if beta_coll > 0 and bool(getattr(val, "IK_ENABLE_SELF_COLLISION_COST", False)):
            grad += beta_coll * _finite_difference_gradient_cost(q, gripper_open01, geom, limits, previous_joints, "collision")
        if beta_manip > 0:
            grad += beta_manip * _finite_difference_gradient_cost(q, gripper_open01, geom, limits, previous_joints, "manipulability")

        dq = dq_task - N @ grad
        max_abs = float(np.max(np.abs(dq))) if dq.size else 0.0
        if max_abs > max_step:
            dq *= max_step / max(max_abs, 1e-12)
        dq *= step_eta

        current_cost = _candidate_cost(joints, target_xyz, target_R, previous_joints, geom, limits, mode_cfg)
        accepted = False
        for scale in (1.0, 0.5, 0.25, 0.1, 0.05):
            q_trial = q + scale * dq
            trial = _clamp_joint_dict(_vector_to_joint_dict(q_trial, gripper_open01), limits)
            trial_q = _joint_dict_to_vector(trial)
            trial_cost = _candidate_cost(trial, target_xyz, target_R, previous_joints, geom, limits, mode_cfg)
            if trial_cost <= current_cost or scale == 0.05:
                q = trial_q
                joints = trial
                accepted = True
                last_cost = trial_cost
                if trial_cost < best_cost:
                    best = trial
                    best_cost = trial_cost
                break
        if not accepted:
            break
        if np.linalg.norm(dq) < 1e-8 or abs(current_cost - last_cost) < 1e-12:
            break

    diag = _candidate_diagnostics(best, target_xyz, target_R, previous_joints, limits, geom, iterations=it_used, converged=converged, mode=mode_name)
    best["__diagnostics__"] = diag
    return best, diag


def solve_ik_pose(target_xyz, target_R=None, gripper_open01: float = 1.0, lerobot_calibration=None, previous_joints=None, model_calibration=None, ik_mode: Optional[str] = None, return_diagnostics: bool = False) -> dict:
    if target_R is None:
        target_R = np.eye(3, dtype=np.float64)
    target_R = _project_to_rotation(target_R)
    return solve_ik_from_target(
        target_xyz,
        _matrix_to_rpy(target_R),
        gripper_open01,
        lerobot_calibration,
        previous_joints,
        target_R=target_R,
        model_calibration=model_calibration,
        ik_mode=ik_mode,
        return_diagnostics=return_diagnostics,
    )


def _build_ik_seeds(xyz, R, limits, gripper_open01, previous_joints, geom, mode_cfg) -> list[Dict[str, float]]:
    seeds: list[Dict[str, float]] = []
    prev_seed = _make_seed_from_previous(previous_joints, limits, gripper_open01)
    geo_seed = _geometric_seed(xyz, R, limits, gripper_open01, geom)
    seeds.append(prev_seed)
    include_many = bool(mode_cfg.get("include_many_seeds", False))
    if include_many or bool(_get_value("IK_FAST_INCLUDE_GEOMETRIC_SEED", True)):
        seeds.append(geo_seed)
    if include_many:
        neutral = {name: _joint_mid(*limits[name]) for name in JOINT_NAMES}
        neutral["elbow_flex"] = _clip(_get_value("IK_PREFERRED_ELBOW_RAD", 0.65), *limits["elbow_flex"])
        neutral[GRIPPER_NAME] = gripper_open01
        seeds.append(_clamp_joint_dict(neutral, limits))
        mirrored = dict(geo_seed)
        mirrored["elbow_flex"] = _clamp_to_joint("elbow_flex", -float(geo_seed["elbow_flex"]), limits)
        seeds.append(_clamp_joint_dict(mirrored, limits))
        for w_yaw in (0.0, 0.45, -0.45):
            for w_roll in (0.0, 0.55, -0.55):
                for w_pitch in (0.0, 0.35, -0.35):
                    s = dict(prev_seed)
                    s["wrist_yaw"] = _clamp_to_joint("wrist_yaw", w_yaw, limits)
                    s["wrist_roll"] = _clamp_to_joint("wrist_roll", w_roll, limits)
                    s["wrist_pitch"] = _clamp_to_joint("wrist_pitch", w_pitch, limits)
                    seeds.append(_clamp_joint_dict(s, limits))
    return seeds


def solve_ik_from_target(
    target_xyz,
    target_rpy=None,
    gripper_open01: float = 1.0,
    lerobot_calibration=None,
    previous_joints=None,
    *,
    q_seed=None,
    target_R=None,
    return_diagnostics: bool = False,
    model_calibration: Optional[dict] = None,
    ik_mode: Optional[str] = None,
    strict_reachability: Optional[bool] = None,
):
    if previous_joints is None and q_seed is not None:
        previous_joints = q_seed
    if lerobot_calibration is None:
        lerobot_calibration = _load_lerobot_calibration_file()
    base_limits = _get_joint_limits()
    limits, _joint_cal = _merge_limits(base_limits, lerobot_calibration)
    geom = _ee_geom(model_calibration)
    mode_cfg = _ik_mode_config(ik_mode, previous_joints)

    try:
        xyz = np.asarray(target_xyz, dtype=np.float64).reshape(3)
    except Exception:
        xyz = np.array([0.0, 0.0, 0.12], dtype=np.float64)
    R = _project_to_rotation(target_R) if target_R is not None else _project_to_rotation(_rpy_to_matrix(target_rpy))

    seeds = _build_ik_seeds(xyz, R, limits, gripper_open01, previous_joints, geom, mode_cfg)
    best = None
    best_diag = None
    best_cost = float("inf")
    seen = set()
    total_iters = 0
    for seed in seeds:
        key = tuple(round(float(seed[n]), 4) for n in JOINT_NAMES)
        if key in seen:
            continue
        seen.add(key)
        sol, diag = _refine_dls(seed, xyz, R, limits, geom, previous_joints, mode_cfg)
        total_iters += int(diag.get("iterations_used", 0)) if isinstance(diag, dict) else 0
        cost = _candidate_cost(sol, xyz, R, previous_joints, geom, limits, mode_cfg)
        if cost < best_cost:
            best = sol
            best_diag = diag
            best_cost = cost

    if best is None:
        best = _make_seed_from_previous(previous_joints, limits, gripper_open01)
        best_diag = _candidate_diagnostics(best, xyz, R, previous_joints, limits, geom, iterations=0, converged=False, mode=mode_cfg.get("mode", "auto"))

    best[GRIPPER_NAME] = _clip(gripper_open01, 0.0, 1.0)
    if isinstance(best_diag, dict):
        best_diag = dict(best_diag)
        best_diag["seed_count"] = int(len(seen))
        best_diag["total_iterations_all_seeds"] = int(total_iters)
        best_diag["strict_reachability"] = bool(mode_cfg.get("strict", False) if strict_reachability is None else strict_reachability)
        best_diag["model_source_path"] = str(geom.get("model_source_path", "nominal_hardcoded_chain"))
    else:
        best_diag = {}
    best["__diagnostics__"] = best_diag

    # Backward compatible behavior: always return the best clamped solution.  If
    # strict_reachability=True, the caller can still inspect diagnostics and will
    # also get a clear flag that this command should not be executed.
    strict = bool(mode_cfg.get("strict", False) if strict_reachability is None else strict_reachability)
    if strict and not bool(best_diag.get("reachable", False)):
        best["__strict_unreachable__"] = True

    if return_diagnostics:
        return best, best_diag
    return best
