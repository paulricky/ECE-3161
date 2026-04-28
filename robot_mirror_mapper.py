from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

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

REQUIRED_POSES = (
    "center",
    "mirror_left",
    "mirror_right",
    "mirror_up",
    "mirror_down",
    "mirror_near",
    "mirror_far",
)

POSE_COORDS = {
    "center": (0.0, 0.0, 0.0),
    "mirror_left": (-1.0, 0.0, 0.0),
    "mirror_right": (1.0, 0.0, 0.0),
    "mirror_up": (0.0, 1.0, 0.0),
    "mirror_down": (0.0, -1.0, 0.0),
    "mirror_near": (0.0, 0.0, 1.0),
    "mirror_far": (0.0, 0.0, -1.0),
    "mirror_up_left": (-1.0, 1.0, 0.0),
    "mirror_up_right": (1.0, 1.0, 0.0),
    "mirror_down_left": (-1.0, -1.0, 0.0),
    "mirror_down_right": (1.0, -1.0, 0.0),
    "mirror_near_left": (-1.0, 0.0, 1.0),
    "mirror_near_right": (1.0, 0.0, 1.0),
    "mirror_far_left": (-1.0, 0.0, -1.0),
    "mirror_far_right": (1.0, 0.0, -1.0),
    "mirror_near_up": (0.0, 1.0, 1.0),
    "mirror_near_down": (0.0, -1.0, 1.0),
    "mirror_far_up": (0.0, 1.0, -1.0),
    "mirror_far_down": (0.0, -1.0, -1.0),
    "mirror_near_up_left": (-1.0, 1.0, 1.0),
    "mirror_near_up_right": (1.0, 1.0, 1.0),
    "mirror_far_down_left": (-1.0, -1.0, -1.0),
    "mirror_far_down_right": (1.0, -1.0, -1.0),
}


def _resolve_path(path: str) -> Path:
    p = Path(str(path)).expanduser()
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    return p


def _finite_float(x, default=None):
    try:
        f = float(x)
    except Exception:
        return default
    return f if math.isfinite(f) else default


def _clip(x: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(x)))


def _sat01(x: float) -> float:
    return _clip(float(x), 0.0, 1.0)


def _norm3(x) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(x, dtype=np.float64).reshape(3)
    except Exception:
        return None
    return arr if np.all(np.isfinite(arr)) else None


class RobotMirrorWorkspaceMapper:
    """Map normalized hand mirror coordinates to robot FK workspace targets.

    The calibration stores robot extrema only. Saved joint poses are returned
    only as optional IK seeds; runtime never commands them directly.
    """

    def __init__(self, path: Optional[str] = None):
        self.path = _resolve_path(path or getattr(val, "ROBOT_MIRROR_WORKSPACE_CALIBRATION_FILE", "calibration_data/robot_mirror_workspace_calibration.json"))
        self.loaded = False
        self.error = ""
        self.data: dict = {}
        self.poses: dict = {}
        self.pose_xyz: dict[str, np.ndarray] = {}
        self.pose_joints: dict[str, Optional[dict[str, float]]] = {}
        self.samples_x = np.zeros((0, 3), dtype=np.float64)
        self.residual_y = np.zeros((0, 3), dtype=np.float64)
        self.sample_names: list[str] = []
        self.rbf_weights: Optional[np.ndarray] = None
        self.rbf_ready = False
        self.xyz_min = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
        self.xyz_max = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
        self.load(self.path)

    def reset(self) -> None:
        self.__init__(str(self.path))

    def is_loaded(self) -> bool:
        return bool(self.loaded)

    def load(self, path: Optional[str] = None) -> bool:
        if path is not None:
            self.path = _resolve_path(str(path))
        self.loaded = False
        self.error = ""
        self.data = {}
        self.poses = {}
        self.pose_xyz = {}
        self.pose_joints = {}
        self.samples_x = np.zeros((0, 3), dtype=np.float64)
        self.residual_y = np.zeros((0, 3), dtype=np.float64)
        self.sample_names = []
        self.rbf_weights = None
        self.rbf_ready = False
        if not self.path.exists():
            self.error = "missing"
            return False
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            self.error = f"load_failed:{exc}"
            return False
        if not isinstance(data, dict) or data.get("calibration_type") != "robot_mirror_workspace_extrema":
            self.error = "bad_schema"
            return False
        poses = data.get("poses", {})
        if not isinstance(poses, dict):
            self.error = "poses_missing"
            return False
        for name, item in poses.items():
            if not isinstance(item, dict):
                continue
            xyz = _norm3(item.get("fk_xyz_m"))
            if xyz is None:
                continue
            self.poses[str(name)] = item
            self.pose_xyz[str(name)] = xyz
            self.pose_joints[str(name)] = self._parse_joints(item.get("joints_rad"))
        missing = [name for name in REQUIRED_POSES if name not in self.pose_xyz]
        if missing:
            self.error = "missing_required:" + ",".join(missing)
            return False
        self._build_sample_residuals()
        self._fit_rbf()
        all_xyz = np.stack(list(self.pose_xyz.values()), axis=0)
        margin = abs(float(getattr(val, "ROBOT_MIRROR_RESIDUAL_MAX_M", 0.030)))
        self.xyz_min = np.min(all_xyz, axis=0) - margin
        self.xyz_max = np.max(all_xyz, axis=0) + margin
        self.data = data
        self.loaded = True
        return True

    def _parse_joints(self, raw) -> Optional[dict[str, float]]:
        if isinstance(raw, dict):
            vals = [raw.get(name) for name in JOINT_NAMES]
        elif isinstance(raw, (list, tuple)) and len(raw) >= 7:
            vals = list(raw[:7])
        else:
            return None
        out = {}
        for name, value in zip(JOINT_NAMES, vals, strict=True):
            f = _finite_float(value)
            if f is None:
                return None
            out[name] = float(f)
        return out

    def _centered_inputs(self, horizontal_norm, vertical_norm, depth_norm) -> np.ndarray:
        h = 2.0 * (_sat01(horizontal_norm) - float(getattr(val, "HAND_MIRROR_CENTER_X_NORM", 0.5)))
        v = 2.0 * (_sat01(vertical_norm) - float(getattr(val, "HAND_MIRROR_CENTER_Y_NORM", 0.5)))
        d = 2.0 * (_sat01(depth_norm) - float(getattr(val, "HAND_MIRROR_CENTER_DEPTH_NORM", 0.5)))
        if bool(getattr(val, "HAND_MIRROR_HORIZONTAL_FLIP", False)):
            h = -h
        if bool(getattr(val, "HAND_MIRROR_VERTICAL_FLIP", True)):
            v = -v
        if bool(getattr(val, "HAND_MIRROR_DEPTH_FLIP", False)):
            d = -d
        if bool(getattr(val, "HAND_MIRROR_CLAMP_INPUTS", True)):
            h = _clip(h, -1.0, 1.0)
            v = _clip(v, -1.0, 1.0)
            d = _clip(d, -1.0, 1.0)
        return np.array([h, v, d], dtype=np.float64)

    def _axis_blend_from_centered(self, x: np.ndarray) -> np.ndarray:
        center = self.pose_xyz["center"]
        h, v, d = [float(a) for a in np.asarray(x, dtype=np.float64).reshape(3)]
        out = center.copy()
        out += abs(h) * ((self.pose_xyz["mirror_left"] if h < 0.0 else self.pose_xyz["mirror_right"]) - center)
        out += abs(v) * ((self.pose_xyz["mirror_down"] if v < 0.0 else self.pose_xyz["mirror_up"]) - center)
        out += abs(d) * ((self.pose_xyz["mirror_far"] if d < 0.0 else self.pose_xyz["mirror_near"]) - center)
        return self._clamp_xyz(out)

    def _build_sample_residuals(self) -> None:
        xs = []
        residuals = []
        names = []
        for name, coord in POSE_COORDS.items():
            if name not in self.pose_xyz:
                continue
            x = np.asarray(coord, dtype=np.float64).reshape(3)
            base = self._axis_blend_from_centered(x)
            xs.append(x)
            residuals.append(np.asarray(self.pose_xyz[name], dtype=np.float64).reshape(3) - base)
            names.append(name)
        if xs:
            self.samples_x = np.stack(xs, axis=0)
            self.residual_y = np.stack(residuals, axis=0)
            self.sample_names = names

    def _kernel(self, r) -> np.ndarray:
        r = np.asarray(r, dtype=np.float64)
        kind = str(getattr(val, "ROBOT_MIRROR_RBF_KERNEL", "thin_plate")).strip().lower()
        if kind in {"gaussian", "gauss"}:
            return np.exp(-(r ** 2))
        if kind in {"multiquadric", "mq"}:
            return np.sqrt(1.0 + r ** 2)
        safe = np.maximum(r, 1e-12)
        out = (safe ** 2) * np.log(safe)
        out[r < 1e-12] = 0.0
        return out

    def _fit_rbf(self) -> None:
        if not bool(getattr(val, "ROBOT_MIRROR_RBF_ENABLED", True)):
            return
        n = int(self.samples_x.shape[0])
        min_n = int(getattr(val, "ROBOT_MIRROR_RBF_MIN_SAMPLES", 8))
        if n < max(3, min_n):
            return
        try:
            d = np.linalg.norm(self.samples_x[:, None, :] - self.samples_x[None, :, :], axis=2)
            K = self._kernel(d)
            smooth = max(0.0, float(getattr(val, "ROBOT_MIRROR_RBF_SMOOTHING", 1e-4)))
            self.rbf_weights = np.linalg.solve(K + smooth * np.eye(n, dtype=np.float64), self.residual_y)
            self.rbf_ready = bool(np.all(np.isfinite(self.rbf_weights)))
        except Exception as exc:
            self.error = f"rbf_fit_failed:{exc}"
            self.rbf_weights = None
            self.rbf_ready = False

    def _rbf_residual(self, x: np.ndarray) -> Optional[np.ndarray]:
        if not bool(getattr(val, "ROBOT_MIRROR_RBF_ENABLED", True)):
            return None
        if not self.rbf_ready or self.rbf_weights is None or self.samples_x.size == 0:
            return None
        try:
            r = np.linalg.norm(self.samples_x - x.reshape(1, 3), axis=1)
            residual = self._kernel(r) @ self.rbf_weights
            return residual if np.all(np.isfinite(residual)) else None
        except Exception:
            return None

    def _knn_residual(self, x: np.ndarray) -> tuple[np.ndarray, Optional[str]]:
        if not bool(getattr(val, "ROBOT_MIRROR_KNN_ENABLED", True)):
            return np.zeros(3, dtype=np.float64), None
        if self.samples_x.size == 0 or self.residual_y.size == 0:
            return np.zeros(3, dtype=np.float64), None
        d = np.linalg.norm(self.samples_x - x.reshape(1, 3), axis=1)
        order = np.argsort(d)
        k = max(1, min(int(getattr(val, "ROBOT_MIRROR_KNN_K", 4)), len(order)))
        idx = order[:k]
        weights = 1.0 / np.maximum(d[idx], 1e-6)
        weights /= max(float(np.sum(weights)), 1e-12)
        residual = np.sum(self.residual_y[idx] * weights[:, None], axis=0)
        nearest = self.sample_names[int(order[0])] if len(order) else None
        return residual, nearest

    def _clamp_residual(self, residual) -> np.ndarray:
        r = np.asarray(residual, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(r)):
            return np.zeros(3, dtype=np.float64)
        max_m = abs(float(getattr(val, "ROBOT_MIRROR_RESIDUAL_MAX_M", 0.030)))
        r = np.clip(r, -max_m, max_m)
        n = float(np.linalg.norm(r))
        if n > max_m > 0.0:
            r *= max_m / n
        return r

    def _clamp_xyz(self, xyz) -> np.ndarray:
        arr = np.asarray(xyz, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(arr)):
            return self.pose_xyz.get("center", np.zeros(3, dtype=np.float64)).copy()
        if np.all(np.isfinite(self.xyz_min)) and np.all(np.isfinite(self.xyz_max)):
            arr = np.minimum(np.maximum(arr, self.xyz_min), self.xyz_max)
        return arr

    def map_hand_to_robot_target(self, horizontal_norm, vertical_norm, depth_norm):
        if not self.loaded:
            return None, {
                "mirror_mapping_source": f"fallback_no_mirror_calibration:{self.error or 'missing'}",
                "mirror_method": "none",
            }
        x = self._centered_inputs(horizontal_norm, vertical_norm, depth_norm)
        base = self._axis_blend_from_centered(x)
        method = str(getattr(val, "ROBOT_MIRROR_MAPPING_METHOD", "axis_blend_knn_residual")).strip().lower()
        residual = np.zeros(3, dtype=np.float64)
        nearest = self._nearest_pose_name(x)
        residual_source = "none"
        used_method = "axis_blend"
        if method in {"axis_blend_knn_residual", "knn_residual"}:
            residual, nearest = self._knn_residual(x)
            residual_source = "knn_residual" if nearest is not None else "none"
            used_method = "axis_blend_knn_residual" if nearest is not None else "axis_blend"
        elif method in {"axis_blend_rbf_residual", "rbf_residual"}:
            rbf = self._rbf_residual(x)
            if rbf is not None:
                residual = rbf
                residual_source = "bounded_rbf_residual"
                used_method = "axis_blend_rbf_residual"
            else:
                residual, nearest = self._knn_residual(x)
                residual_source = "knn_residual_fallback"
                used_method = "axis_blend_knn_residual"
        residual = self._clamp_residual(residual)
        final = self._clamp_xyz(base + residual)
        debug = {
            "mirror_mapping_source": "robot_mirror_workspace_calibration",
            "mirror_method": used_method,
            "mirror_residual_source": residual_source,
            "mirror_horizontal_norm": float(_sat01(horizontal_norm)),
            "mirror_vertical_norm": float(_sat01(vertical_norm)),
            "mirror_depth_norm": float(_sat01(depth_norm)),
            "mirror_h_centered": float(x[0]),
            "mirror_v_centered": float(x[1]),
            "mirror_d_centered": float(x[2]),
            "mirror_nearest_pose": nearest,
            "target_xyz_base_m": base.tolist(),
            "target_xyz_residual_m": residual.tolist(),
            "target_xyz_final_m": final.tolist(),
            "robot_mirror_calibration_loaded": True,
            "robot_mirror_calibration_path": str(self.path),
            "robot_mirror_direct_joint_learning_enabled": bool(getattr(val, "ROBOT_MIRROR_DIRECT_JOINT_LEARNING_ENABLED", False)),
        }
        return final, debug

    def _nearest_pose_name(self, x: np.ndarray) -> Optional[str]:
        if self.samples_x.size == 0:
            return None
        d = np.linalg.norm(self.samples_x - np.asarray(x, dtype=np.float64).reshape(1, 3), axis=1)
        return self.sample_names[int(np.argmin(d))] if d.size else None

    def choose_ik_seed(self, horizontal_norm, vertical_norm, depth_norm, previous_q=None):
        if previous_q is not None or not bool(getattr(val, "ROBOT_MIRROR_USE_JOINT_SEED_EXAMPLES", True)):
            return previous_q, {"ik_seed_source": "previous" if previous_q is not None else "none"}
        if not self.loaded:
            return None, {"ik_seed_source": "none"}
        x = self._centered_inputs(horizontal_norm, vertical_norm, depth_norm)
        ordered = sorted(POSE_COORDS.items(), key=lambda kv: float(np.linalg.norm(np.asarray(kv[1], dtype=np.float64) - x)))
        for name, _coord in ordered:
            seed = self.pose_joints.get(name)
            if isinstance(seed, dict):
                return dict(seed), {"ik_seed_source": "robot_mirror_nearest_pose", "mirror_nearest_pose": name}
        return None, {"ik_seed_source": "none"}
