from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

import values as val


AXES = ("robot_x", "robot_y", "robot_z")
JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "wrist_pitch",
)


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
    lo = float(lo)
    hi = float(hi)
    if hi < lo:
        lo, hi = hi, lo
    return max(lo, min(hi, float(x)))


def _sat01(x: float) -> float:
    return _clip(float(x), 0.0, 1.0)


def _lerp(a: float, b: float, t: float) -> float:
    return float(a) + (float(b) - float(a)) * _sat01(t)


def _norm3(x) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(x, dtype=np.float64).reshape(3)
    except Exception:
        return None
    return arr if np.all(np.isfinite(arr)) else None


def _default_bounds() -> dict[str, tuple[float, float]]:
    return {
        "robot_x": (
            float(getattr(val, "HAND_TARGET_X_MIN_M", getattr(val, "WORKSPACE_X_MIN", -0.12))),
            float(getattr(val, "HAND_TARGET_X_MAX_M", getattr(val, "WORKSPACE_X_MAX", 0.12))),
        ),
        "robot_y": (
            float(getattr(val, "HAND_TARGET_Y_MIN_M", getattr(val, "WORKSPACE_Y_MIN", 0.10))),
            float(getattr(val, "HAND_TARGET_Y_MAX_M", getattr(val, "WORKSPACE_Y_MAX", 0.22))),
        ),
        "robot_z": (
            float(getattr(val, "HAND_TARGET_Z_MIN_M", getattr(val, "WORKSPACE_Z_MIN", 0.00))),
            float(getattr(val, "HAND_TARGET_Z_MAX_M", getattr(val, "WORKSPACE_Z_MAX", 0.22))),
        ),
    }


def _axis_map() -> dict:
    axis_map = getattr(val, "HAND_CAMERA_TO_ROBOT_AXIS_MAP", {"image_x": "robot_x", "image_y": "robot_z", "depth": "robot_y"})
    if not isinstance(axis_map, dict):
        axis_map = {"image_x": "robot_x", "image_y": "robot_z", "depth": "robot_y"}
    return dict(axis_map)


class HandWorkspaceMapper:
    """Non-neural hand-coordinate to robot-workspace mapper.

    The learned methods only correct workspace xyz. Stored joint examples are
    exposed as optional IK seeds and are never returned as direct motor commands.
    """

    def __init__(self, path: Optional[str] = None):
        self.path = _resolve_path(path or getattr(val, "HAND_WORKSPACE_CALIBRATION_FILE", "calibration_data/hand_workspace_calibration.json"))
        self.loaded = False
        self.error = ""
        self.data: dict = {}
        self.poses: dict = {}
        self.samples_x = np.zeros((0, 3), dtype=np.float64)
        self.samples_y = np.zeros((0, 3), dtype=np.float64)
        self.residual_y = np.zeros((0, 3), dtype=np.float64)
        self.sample_names: list[str] = []
        self.sample_joints: list[Optional[dict[str, float]]] = []
        self.bounds = _default_bounds()
        self.rbf_weights: Optional[np.ndarray] = None
        self.rbf_ready = False
        self.load(self.path)

    def reset(self) -> None:
        self.__init__(str(self.path))

    def load(self, path: Optional[str] = None) -> bool:
        if path is not None:
            self.path = _resolve_path(str(path))
        self.loaded = False
        self.error = ""
        self.data = {}
        self.poses = {}
        self.samples_x = np.zeros((0, 3), dtype=np.float64)
        self.samples_y = np.zeros((0, 3), dtype=np.float64)
        self.residual_y = np.zeros((0, 3), dtype=np.float64)
        self.sample_names = []
        self.sample_joints = []
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
        if not isinstance(data, dict) or data.get("calibration_type") != "hand_to_robot_workspace":
            self.error = "bad_schema"
            return False

        bounds_node = data.get("workspace_bounds", {})
        if isinstance(bounds_node, dict):
            self.bounds = {
                "robot_x": (
                    _finite_float(bounds_node.get("x_min_m"), self.bounds["robot_x"][0]),
                    _finite_float(bounds_node.get("x_max_m"), self.bounds["robot_x"][1]),
                ),
                "robot_y": (
                    _finite_float(bounds_node.get("y_min_m"), self.bounds["robot_y"][0]),
                    _finite_float(bounds_node.get("y_max_m"), self.bounds["robot_y"][1]),
                ),
                "robot_z": (
                    _finite_float(bounds_node.get("z_min_m"), self.bounds["robot_z"][0]),
                    _finite_float(bounds_node.get("z_max_m"), self.bounds["robot_z"][1]),
                ),
            }

        poses = data.get("poses", {})
        if not isinstance(poses, dict):
            self.error = "poses_missing"
            return False
        xs = []
        ys = []
        names = []
        joints = []
        for name, item in poses.items():
            if not isinstance(item, dict):
                continue
            hand = item.get("hand", {})
            robot = item.get("robot", {})
            if not isinstance(hand, dict) or not isinstance(robot, dict):
                continue
            x = _finite_float(hand.get("x_norm"))
            y = _finite_float(hand.get("y_norm"))
            d = _finite_float(hand.get("depth_norm"))
            rx = _finite_float(robot.get("x_m"))
            ry = _finite_float(robot.get("y_m"))
            rz = _finite_float(robot.get("z_m"))
            if None in (x, y, d, rx, ry, rz):
                continue
            inp = np.array([_sat01(x), _sat01(y), _sat01(d)], dtype=np.float64)
            out = self._clamp_xyz(np.array([rx, ry, rz], dtype=np.float64))
            xs.append(inp)
            ys.append(out)
            names.append(str(name))
            joints.append(self._parse_joints(item.get("joints_rad")))
            self.poses[str(name)] = item
        if xs:
            self.samples_x = np.stack(xs, axis=0)
            self.samples_y = np.stack(ys, axis=0)
            self.sample_names = names
            self.sample_joints = joints
            base = np.stack([self._piecewise_affine_raw(x[0], x[1], x[2]) for x in self.samples_x], axis=0)
            self.residual_y = self.samples_y - base
            self._fit_rbf()
        self.data = data
        self.loaded = True
        return True

    def _parse_joints(self, raw) -> Optional[dict[str, float]]:
        if raw is None:
            return None
        vals = None
        if isinstance(raw, dict):
            vals = [raw.get(name) for name in JOINT_NAMES]
        elif isinstance(raw, (list, tuple)) and len(raw) >= 7:
            vals = list(raw[:7])
        if vals is None:
            return None
        out = {}
        for name, value in zip(JOINT_NAMES, vals, strict=True):
            f = _finite_float(value)
            if f is None:
                return None
            out[name] = float(f)
        return out

    def _clamp_xyz(self, xyz) -> np.ndarray:
        arr = np.asarray(xyz, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(arr)):
            arr = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        for i, axis in enumerate(AXES):
            lo, hi = self.bounds[axis]
            arr[i] = _clip(arr[i], lo, hi)
        return arr

    def _axis_value(self, axis: str, norm: float) -> float:
        lo, hi = self.bounds.get(axis, (0.0, 1.0))
        return _lerp(lo, hi, norm)

    def _depth_value(self, axis: str, depth_norm: float) -> float:
        lo, hi = self.bounds.get(axis, (0.0, 1.0))
        near = _finite_float(getattr(val, "HAND_DEPTH_TARGET_NEAR_M", None), None)
        far = _finite_float(getattr(val, "HAND_DEPTH_TARGET_FAR_M", None), None)
        if near is None:
            near = lo
        if far is None:
            far = hi
        return _lerp(far, near, depth_norm)

    def _piecewise_affine_raw(self, x_norm: float, y_norm: float, depth_norm: float) -> np.ndarray:
        axis_map = _axis_map()
        depth_axis = str(getattr(val, "HAND_DEPTH_AXIS", axis_map.get("depth", "robot_y")))
        xyz_by_axis = {
            "robot_x": 0.5 * sum(self.bounds["robot_x"]),
            "robot_y": 0.5 * sum(self.bounds["robot_y"]),
            "robot_z": 0.5 * sum(self.bounds["robot_z"]),
        }
        ix_axis = str(axis_map.get("image_x", "robot_x"))
        iy_axis = str(axis_map.get("image_y", "robot_z"))
        if ix_axis in xyz_by_axis:
            xyz_by_axis[ix_axis] = self._axis_value(ix_axis, x_norm)
        if iy_axis in xyz_by_axis:
            xyz_by_axis[iy_axis] = self._axis_value(iy_axis, y_norm)
        if depth_axis in xyz_by_axis:
            xyz_by_axis[depth_axis] = self._depth_value(depth_axis, depth_norm)
        return np.array([xyz_by_axis["robot_x"], xyz_by_axis["robot_y"], xyz_by_axis["robot_z"]], dtype=np.float64)

    def _kernel(self, r) -> np.ndarray:
        r = np.asarray(r, dtype=np.float64)
        kind = str(getattr(val, "HAND_WORKSPACE_RBF_KERNEL", "thin_plate")).strip().lower()
        if kind in {"gaussian", "gauss"}:
            eps = 1.0
            return np.exp(-((eps * r) ** 2))
        if kind in {"multiquadric", "mq"}:
            eps = 1.0
            return np.sqrt(1.0 + (eps * r) ** 2)
        safe = np.maximum(r, 1e-12)
        out = (safe ** 2) * np.log(safe)
        out[r < 1e-12] = 0.0
        return out

    def _fit_rbf(self) -> None:
        n = int(self.samples_x.shape[0])
        min_n = int(getattr(val, "HAND_WORKSPACE_MIN_EXAMPLES_FOR_RBF", 8))
        if n < max(3, min_n):
            return
        try:
            d = np.linalg.norm(self.samples_x[:, None, :] - self.samples_x[None, :, :], axis=2)
            K = self._kernel(d)
            smooth = max(0.0, float(getattr(val, "HAND_WORKSPACE_RBF_SMOOTHING", 1e-4)))
            A = K + smooth * np.eye(n, dtype=np.float64)
            self.rbf_weights = np.linalg.solve(A, self.residual_y)
            self.rbf_ready = bool(np.all(np.isfinite(self.rbf_weights)))
        except Exception as exc:
            self.error = f"rbf_fit_failed:{exc}"
            self.rbf_weights = None
            self.rbf_ready = False

    def _rbf_residual(self, x: np.ndarray) -> Optional[np.ndarray]:
        if not self.rbf_ready or self.rbf_weights is None or self.samples_x.size == 0:
            return None
        try:
            r = np.linalg.norm(self.samples_x - x.reshape(1, 3), axis=1)
            phi = self._kernel(r)
            residual = phi @ self.rbf_weights
            return residual if np.all(np.isfinite(residual)) else None
        except Exception:
            return None

    def _knn_residual(self, x: np.ndarray) -> tuple[np.ndarray, Optional[str]]:
        if self.samples_x.size == 0 or self.residual_y.size == 0:
            return np.zeros(3, dtype=np.float64), None
        d = np.linalg.norm(self.samples_x - x.reshape(1, 3), axis=1)
        order = np.argsort(d)
        k = max(1, min(int(getattr(val, "HAND_WORKSPACE_KNN_K", 4)), len(order)))
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
        max_m = abs(float(getattr(val, "HAND_WORKSPACE_RESIDUAL_MAX_M", 0.030)))
        r = np.clip(r, -max_m, max_m)
        norm = float(np.linalg.norm(r))
        if norm > max_m > 0.0:
            r *= max_m / norm
        return r

    def map_hand_to_workspace(self, x_norm, y_norm, depth_norm, hand_size_norm=None):
        del hand_size_norm
        x = np.array([_sat01(x_norm), _sat01(y_norm), _sat01(depth_norm)], dtype=np.float64)
        base = self._clamp_xyz(self._piecewise_affine_raw(x[0], x[1], x[2]))
        method = str(getattr(val, "HAND_WORKSPACE_MAPPING_METHOD", "rbf_residual")).strip().lower()
        if method == "thin_plate_rbf":
            method = "rbf_residual"
        residual = np.zeros(3, dtype=np.float64)
        nearest = None
        source = "values_piecewise_affine"
        used_method = "piecewise_affine"

        if self.loaded and self.samples_x.size > 0 and bool(getattr(val, "HAND_WORKSPACE_LEARNING_ENABLED", True)):
            d = np.linalg.norm(self.samples_x - x.reshape(1, 3), axis=1)
            nearest = self.sample_names[int(np.argmin(d))] if d.size else None
            if method == "piecewise_affine":
                source = "calibrated_piecewise_affine"
                used_method = "piecewise_affine"
            elif method == "knn_weighted":
                residual, nearest = self._knn_residual(x)
                source = "knn_weighted"
                used_method = "knn_weighted"
            elif method == "rbf_residual":
                rbf_res = self._rbf_residual(x)
                if rbf_res is not None:
                    residual = rbf_res
                    source = "rbf_residual"
                    used_method = "rbf_residual"
                else:
                    residual, nearest = self._knn_residual(x)
                    source = "knn_weighted_fallback"
                    used_method = "knn_weighted"

        residual = self._clamp_residual(residual)
        final = self._clamp_xyz(base + residual)
        debug = {
            "workspace_mapping_source": source if self.loaded else f"fallback_no_calibration:{self.error or 'missing'}",
            "workspace_learning_method": used_method,
            "nearest_calibration_pose": nearest,
            "target_xyz_base_m": base.tolist(),
            "target_xyz_residual_m": residual.tolist(),
            "target_xyz_final_m": final.tolist(),
            "hand_workspace_calibration_loaded": bool(self.loaded),
            "hand_workspace_calibration_path": str(self.path),
            "hand_workspace_direct_joint_learning_enabled": bool(getattr(val, "HAND_WORKSPACE_DIRECT_JOINT_LEARNING_ENABLED", False)),
        }
        return final, debug

    def choose_ik_seed(self, x_norm, y_norm, depth_norm, previous_q=None):
        if previous_q is not None or not bool(getattr(val, "HAND_WORKSPACE_USE_JOINT_SEED_EXAMPLES", True)):
            return previous_q, {"ik_seed_source": "previous" if previous_q is not None else "none"}
        if not self.loaded or self.samples_x.size == 0:
            return None, {"ik_seed_source": "none"}
        x = np.array([_sat01(x_norm), _sat01(y_norm), _sat01(depth_norm)], dtype=np.float64)
        d = np.linalg.norm(self.samples_x - x.reshape(1, 3), axis=1)
        if not d.size:
            return None, {"ik_seed_source": "none"}
        for idx in np.argsort(d):
            seed = self.sample_joints[int(idx)]
            if isinstance(seed, dict):
                return dict(seed), {
                    "ik_seed_source": "hand_workspace_nearest_pose",
                    "nearest_calibration_pose": self.sample_names[int(idx)],
                }
        return None, {"ik_seed_source": "none"}
