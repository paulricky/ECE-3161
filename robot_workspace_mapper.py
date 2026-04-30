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

REQUIRED_POSES = tuple(getattr(val, "ROBOT_WORKSPACE_REQUIRED_POSES", (
    "center",
    "left",
    "right",
    "up",
    "down",
    "near",
    "far",
)))

OPTIONAL_POSES = tuple(getattr(val, "ROBOT_WORKSPACE_OPTIONAL_POSES", (
    "up_left",
    "up_right",
    "down_left",
    "down_right",
    "near_left",
    "near_right",
    "far_left",
    "far_right",
    "near_up",
    "near_down",
    "far_up",
    "far_down",
    "near_up_left",
    "near_up_right",
    "far_down_left",
    "far_down_right",
)))

POSE_COORDS = {
    "center": (0.0, 0.0, 0.0),
    "left": (-1.0, 0.0, 0.0),
    "right": (1.0, 0.0, 0.0),
    "up": (0.0, 1.0, 0.0),
    "down": (0.0, -1.0, 0.0),
    "near": (0.0, 0.0, 1.0),
    "far": (0.0, 0.0, -1.0),
    "up_left": (-1.0, 1.0, 0.0),
    "up_right": (1.0, 1.0, 0.0),
    "down_left": (-1.0, -1.0, 0.0),
    "down_right": (1.0, -1.0, 0.0),
    "near_left": (-1.0, 0.0, 1.0),
    "near_right": (1.0, 0.0, 1.0),
    "far_left": (-1.0, 0.0, -1.0),
    "far_right": (1.0, 0.0, -1.0),
    "near_up": (0.0, 1.0, 1.0),
    "near_down": (0.0, -1.0, 1.0),
    "far_up": (0.0, 1.0, -1.0),
    "far_down": (0.0, -1.0, -1.0),
    "near_up_left": (-1.0, 1.0, 1.0),
    "near_up_right": (1.0, 1.0, 1.0),
    "far_down_left": (-1.0, -1.0, -1.0),
    "far_down_right": (1.0, -1.0, -1.0),
}

LEGACY_MIRROR_TO_WORKSPACE = {
    "mirror_left": "left",
    "mirror_right": "right",
    "mirror_up": "up",
    "mirror_down": "down",
    "mirror_near": "near",
    "mirror_far": "far",
    "mirror_up_left": "up_left",
    "mirror_up_right": "up_right",
    "mirror_down_left": "down_left",
    "mirror_down_right": "down_right",
    "mirror_near_left": "near_left",
    "mirror_near_right": "near_right",
    "mirror_far_left": "far_left",
    "mirror_far_right": "far_right",
    "mirror_near_up": "near_up",
    "mirror_near_down": "near_down",
    "mirror_far_up": "far_up",
    "mirror_far_down": "far_down",
    "mirror_near_up_left": "near_up_left",
    "mirror_near_up_right": "near_up_right",
    "mirror_far_down_left": "far_down_left",
    "mirror_far_down_right": "far_down_right",
}


def _resolve_path(path: str | Path) -> Path:
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


def _vec3(x) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(x, dtype=np.float64).reshape(3)
    except Exception:
        return None
    return arr if np.all(np.isfinite(arr)) else None


class RobotWorkspaceMapper:
    """Build a calibrated reachable workspace from recorded robot FK extrema.

    Recorded motor/joint poses are never commanded directly. They define FK
    boundary vectors and optional IK seeds; runtime still sends xyz/rpy targets
    through the 7-DOF IK solver.
    """

    def __init__(self, path: Optional[str] = None):
        self.path = _resolve_path(path or getattr(val, "ROBOT_WORKSPACE_CALIBRATION_FILE", "calibration_data/robot_workspace_extrema_calibration.json"))
        self.loaded = False
        self.legacy_loaded = False
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
        self.load(self.path)

    def is_loaded(self) -> bool:
        return bool(self.loaded)

    def _candidate_paths(self, path: Optional[str | Path]) -> list[Path]:
        first = _resolve_path(path or self.path)
        paths = [first]
        legacy = _resolve_path(getattr(val, "ROBOT_WORKSPACE_LEGACY_MIRROR_CALIBRATION_FILE", "calibration_data/robot_mirror_workspace_calibration.json"))
        if legacy not in paths:
            paths.append(legacy)
        return paths

    def load(self, path: Optional[str | Path] = None) -> bool:
        self.loaded = False
        self.legacy_loaded = False
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
        self.xyz_min = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
        self.xyz_max = np.array([np.inf, np.inf, np.inf], dtype=np.float64)

        last_error = "missing"
        for candidate in self._candidate_paths(path):
            if not candidate.exists():
                last_error = "missing"
                continue
            ok = self._load_one(candidate)
            if ok:
                return True
            last_error = self.error
        self.error = last_error
        return False

    def _load_one(self, path: Path) -> bool:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            self.error = f"load_failed:{exc}"
            return False
        if not isinstance(data, dict):
            self.error = "bad_schema"
            return False
        ctype = str(data.get("calibration_type", ""))
        if ctype == "robot_workspace_extrema":
            legacy = False
        elif ctype == "robot_mirror_workspace_extrema":
            legacy = True
        else:
            self.error = "bad_schema"
            return False
        poses = data.get("poses", {})
        if not isinstance(poses, dict):
            self.error = "poses_missing"
            return False
        for raw_name, item in poses.items():
            if not isinstance(item, dict):
                continue
            name = LEGACY_MIRROR_TO_WORKSPACE.get(str(raw_name), str(raw_name))
            xyz = _vec3(item.get("fk_xyz_m"))
            if xyz is None:
                continue
            self.poses[name] = item
            self.pose_xyz[name] = xyz
            self.pose_joints[name] = self._parse_joints(item.get("joints_rad"))
        missing = [name for name in REQUIRED_POSES if name not in self.pose_xyz]
        if missing:
            self.error = "missing_required:" + ",".join(missing)
            return False

        all_xyz = np.stack(list(self.pose_xyz.values()), axis=0)
        margin = abs(float(getattr(val, "ROBOT_WORKSPACE_CLAMP_MARGIN_M", 0.020)))
        self.xyz_min = np.min(all_xyz, axis=0) - margin
        self.xyz_max = np.max(all_xyz, axis=0) + margin
        self._build_sample_residuals()
        self._fit_rbf()
        self.path = path
        self.data = data
        self.legacy_loaded = bool(legacy)
        self.loaded = True
        self.error = ""
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
        h = 2.0 * (_sat01(horizontal_norm) - float(getattr(val, "HAND_WORKSPACE_CENTER_X_NORM", 0.5)))
        v = 2.0 * (_sat01(vertical_norm) - float(getattr(val, "HAND_WORKSPACE_CENTER_Y_NORM", 0.5)))
        d = 2.0 * (_sat01(depth_norm) - float(getattr(val, "HAND_WORKSPACE_CENTER_DEPTH_NORM", 0.5)))
        if bool(getattr(val, "HAND_WORKSPACE_HORIZONTAL_FLIP", False)):
            h = -h
        if bool(getattr(val, "HAND_WORKSPACE_VERTICAL_FLIP", True)):
            v = -v
        if bool(getattr(val, "HAND_WORKSPACE_DEPTH_FLIP", False)):
            d = -d
        if bool(getattr(val, "HAND_WORKSPACE_CLAMP_INPUTS", True)):
            h = _clip(h, -1.0, 1.0)
            v = _clip(v, -1.0, 1.0)
            d = _clip(d, -1.0, 1.0)
        return np.array([h, v, d], dtype=np.float64)

    @staticmethod
    def _signed_gamma(x: float, gamma: float) -> float:
        if not math.isfinite(float(gamma)) or float(gamma) <= 0.0:
            gamma = 1.0
        x = _clip(float(x), -1.0, 1.0)
        return math.copysign(abs(x) ** float(gamma), x)

    def _shape_centered_inputs(self, x: np.ndarray) -> np.ndarray:
        raw = np.asarray(x, dtype=np.float64).reshape(3)
        shaped = raw.copy()
        # Horizontal is intentionally identity. Current left/right behavior is
        # the known-good baseline and must not be changed by extension shaping.
        shaped[0] = raw[0]
        if bool(getattr(val, "ROBOT_WORKSPACE_VERTICAL_ENDPOINT_BOOST_ENABLED", True)):
            shaped[1] = self._signed_gamma(raw[1], float(getattr(val, "ROBOT_WORKSPACE_VERTICAL_RESPONSE_GAMMA", 1.0)))
        if bool(getattr(val, "ROBOT_WORKSPACE_DEPTH_ENDPOINT_BOOST_ENABLED", True)):
            shaped[2] = self._signed_gamma(raw[2], float(getattr(val, "ROBOT_WORKSPACE_DEPTH_RESPONSE_GAMMA", 1.0)))
        if bool(getattr(val, "ROBOT_WORKSPACE_EXTENSION_SHAPING_CLAMP", True)):
            shaped = np.clip(shaped, -1.0, 1.0)
            shaped[0] = raw[0]
        return shaped

    def _axis_vector_from_centered(self, x: np.ndarray) -> np.ndarray:
        center = self.pose_xyz["center"]
        h, v, d = [float(a) for a in np.asarray(x, dtype=np.float64).reshape(3)]
        out = center.copy()
        out += abs(h) * ((self.pose_xyz["left"] if h < 0.0 else self.pose_xyz["right"]) - center)
        out += abs(v) * ((self.pose_xyz["down"] if v < 0.0 else self.pose_xyz["up"]) - center)
        out += abs(d) * ((self.pose_xyz["far"] if d < 0.0 else self.pose_xyz["near"]) - center)
        return out

    def _pose_centered_coordinate(self, name: str) -> Optional[np.ndarray]:
        coord = POSE_COORDS.get(str(name))
        if coord is None:
            return None
        return np.asarray(coord, dtype=np.float64).reshape(3)

    def _build_sample_residuals(self) -> None:
        xs = []
        residuals = []
        names = []
        for name, xyz in self.pose_xyz.items():
            coord = self._pose_centered_coordinate(name)
            if coord is None:
                continue
            base = self._axis_vector_from_centered(coord)
            xs.append(coord)
            residuals.append(np.asarray(xyz, dtype=np.float64).reshape(3) - base)
            names.append(name)
        if xs:
            self.samples_x = np.stack(xs, axis=0)
            self.residual_y = np.stack(residuals, axis=0)
            self.sample_names = names

    def _kernel(self, r) -> np.ndarray:
        r = np.asarray(r, dtype=np.float64)
        kind = str(getattr(val, "ROBOT_WORKSPACE_RBF_KERNEL", "thin_plate")).strip().lower()
        if kind in {"gaussian", "gauss"}:
            return np.exp(-(r ** 2))
        if kind in {"multiquadric", "mq"}:
            return np.sqrt(1.0 + r ** 2)
        safe = np.maximum(r, 1e-12)
        out = (safe ** 2) * np.log(safe)
        out[r < 1e-12] = 0.0
        return out

    def _fit_rbf(self) -> None:
        if not bool(getattr(val, "ROBOT_WORKSPACE_RBF_ENABLED", True)):
            return
        n = int(self.samples_x.shape[0])
        min_n = int(getattr(val, "ROBOT_WORKSPACE_RBF_MIN_SAMPLES", 8))
        if n < max(3, min_n):
            return
        try:
            d = np.linalg.norm(self.samples_x[:, None, :] - self.samples_x[None, :, :], axis=2)
            K = self._kernel(d)
            smooth = max(0.0, float(getattr(val, "ROBOT_WORKSPACE_RBF_SMOOTHING", 1e-4)))
            self.rbf_weights = np.linalg.solve(K + smooth * np.eye(n, dtype=np.float64), self.residual_y)
            self.rbf_ready = bool(np.all(np.isfinite(self.rbf_weights)))
        except Exception as exc:
            self.error = f"rbf_fit_failed:{exc}"
            self.rbf_weights = None
            self.rbf_ready = False

    def _rbf_residual(self, x: np.ndarray) -> Optional[np.ndarray]:
        if not bool(getattr(val, "ROBOT_WORKSPACE_RBF_ENABLED", True)):
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
        if not bool(getattr(val, "ROBOT_WORKSPACE_KNN_ENABLED", True)):
            return np.zeros(3, dtype=np.float64), None
        if self.samples_x.size == 0 or self.residual_y.size == 0:
            return np.zeros(3, dtype=np.float64), None
        d = np.linalg.norm(self.samples_x - x.reshape(1, 3), axis=1)
        order = np.argsort(d)
        k = max(1, min(int(getattr(val, "ROBOT_WORKSPACE_KNN_K", 4)), len(order)))
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
        max_m = abs(float(getattr(val, "ROBOT_WORKSPACE_RESIDUAL_MAX_M", 0.030)))
        r = np.clip(r, -max_m, max_m)
        n = float(np.linalg.norm(r))
        if n > max_m > 0.0:
            r *= max_m / n
        return r

    def _clamp_xyz_debug(self, xyz) -> tuple[np.ndarray, bool, np.ndarray]:
        before = np.asarray(xyz, dtype=np.float64).reshape(3)
        after = before.copy()
        if bool(getattr(val, "ROBOT_WORKSPACE_CLAMP_TO_RECORDED_BOUNDS", True)):
            after = np.minimum(np.maximum(after, self.xyz_min), self.xyz_max)
        clamped = bool(np.linalg.norm(after - before) > 1e-12)
        return after, clamped, before

    def get_calibrated_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.xyz_min.copy(), self.xyz_max.copy()

    def _nearest_pose_name(self, x: np.ndarray) -> Optional[str]:
        if self.samples_x.size == 0:
            return None
        d = np.linalg.norm(self.samples_x - np.asarray(x, dtype=np.float64).reshape(1, 3), axis=1)
        return self.sample_names[int(np.argmin(d))] if d.size else None

    def map_hand_to_workspace(self, horizontal_norm, vertical_norm, depth_norm):
        if not self.loaded:
            return None, {
                "workspace_mapping_source": f"fallback_no_robot_workspace_calibration:{self.error or 'missing'}",
                "workspace_method": "none",
            }
        x_raw = self._centered_inputs(horizontal_norm, vertical_norm, depth_norm)
        x = self._shape_centered_inputs(x_raw)
        base = self._axis_vector_from_centered(x)
        method = str(getattr(val, "ROBOT_WORKSPACE_MAPPING_METHOD", "axis_vector_knn_residual")).strip().lower()
        residual = np.zeros(3, dtype=np.float64)
        nearest = self._nearest_pose_name(x)
        residual_source = "none"
        used_method = "axis_vector"
        if method in {"axis_vector_knn_residual", "knn_residual"}:
            residual, nearest = self._knn_residual(x)
            residual_source = "knn_residual" if nearest is not None else "none"
            used_method = "axis_vector_knn_residual" if nearest is not None else "axis_vector"
        elif method in {"axis_vector_rbf_residual", "rbf_residual"}:
            rbf = self._rbf_residual(x)
            if rbf is not None:
                residual = rbf
                residual_source = "bounded_rbf_residual"
                used_method = "axis_vector_rbf_residual"
            else:
                residual, nearest = self._knn_residual(x)
                residual_source = "knn_residual_fallback" if nearest is not None else "none"
                used_method = "axis_vector_knn_residual" if nearest is not None else "axis_vector"
        residual = self._clamp_residual(residual)
        before_clamp = base + residual
        final, clamped, before_clamp = self._clamp_xyz_debug(before_clamp)
        debug = {
            "workspace_mapping_source": "legacy_robot_mirror_workspace_calibration" if self.legacy_loaded else "robot_workspace_extrema_calibration",
            "workspace_method": used_method,
            "workspace_residual_source": residual_source,
            "hand_h_norm": float(_sat01(horizontal_norm)),
            "hand_v_norm": float(_sat01(vertical_norm)),
            "hand_depth_norm": float(_sat01(depth_norm)),
            "centered_hvd": x.tolist(),
            "workspace_h_centered": float(x[0]),
            "workspace_v_centered": float(x[1]),
            "workspace_d_centered": float(x[2]),
            "workspace_h_centered_raw": float(x_raw[0]),
            "workspace_h_centered_shaped": float(x[0]),
            "workspace_v_centered_raw": float(x_raw[1]),
            "workspace_v_centered_shaped": float(x[1]),
            "workspace_d_centered_raw": float(x_raw[2]),
            "workspace_d_centered_shaped": float(x[2]),
            "workspace_extension_shaping_enabled": bool(
                bool(getattr(val, "ROBOT_WORKSPACE_VERTICAL_ENDPOINT_BOOST_ENABLED", True))
                or bool(getattr(val, "ROBOT_WORKSPACE_DEPTH_ENDPOINT_BOOST_ENABLED", True))
            ),
            "nearest_pose": nearest,
            "target_xyz_base_m": base.tolist(),
            "target_xyz_residual_m": residual.tolist(),
            "target_xyz_before_clamp_m": before_clamp.tolist(),
            "target_xyz_final_m": final.tolist(),
            "workspace_center_xyz_m": self.pose_xyz["center"].tolist(),
            "target_clamped": bool(clamped),
            "workspace_bounds_min_m": self.xyz_min.tolist(),
            "workspace_bounds_max_m": self.xyz_max.tolist(),
            "robot_workspace_calibration_loaded": True,
            "robot_workspace_calibration_path": str(self.path),
            "robot_workspace_legacy_loaded": bool(self.legacy_loaded),
            "direct_joint_learning_enabled": bool(getattr(val, "ROBOT_WORKSPACE_DIRECT_JOINT_LEARNING_ENABLED", False)),
        }
        return final, debug

    def choose_ik_seed(self, horizontal_norm, vertical_norm, depth_norm, previous_q=None):
        if previous_q is not None or not bool(getattr(val, "ROBOT_WORKSPACE_USE_JOINT_SEED_EXAMPLES", True)):
            return previous_q, {"ik_seed_source": "previous" if previous_q is not None else "none"}
        if not self.loaded:
            return None, {"ik_seed_source": "none"}
        x = self._shape_centered_inputs(self._centered_inputs(horizontal_norm, vertical_norm, depth_norm))
        if self.samples_x.size == 0:
            return None, {"ik_seed_source": "none"}
        order = np.argsort(np.linalg.norm(self.samples_x - x.reshape(1, 3), axis=1))
        for idx in order:
            name = self.sample_names[int(idx)]
            seed = self.pose_joints.get(name)
            if isinstance(seed, dict):
                return dict(seed), {"ik_seed_source": "robot_workspace_nearest_pose", "nearest_pose": name}
        return None, {"ik_seed_source": "none"}

    def project_toward_center_until_reachable(self, target_xyz, ik_check_fn=None):
        target = np.asarray(target_xyz, dtype=np.float64).reshape(3)
        center = self.pose_xyz.get("center")
        if center is None or not np.all(np.isfinite(target)):
            return target, {"target_projected": False}
        steps = max(1, int(getattr(val, "ROBOT_WORKSPACE_PROJECTION_STEPS", 8)))
        if ik_check_fn is None:
            return target, {
                "target_projected": False,
                "projection_center_xyz_m": center.tolist(),
                "projection_steps": steps,
            }
        for i in range(steps + 1):
            alpha = i / float(steps)
            candidate = (1.0 - alpha) * target + alpha * center
            try:
                if bool(ik_check_fn(candidate)):
                    return candidate, {
                        "target_projected": bool(i > 0),
                        "projection_alpha": float(alpha),
                        "original_target_xyz_m": target.tolist(),
                        "final_target_xyz_m": candidate.tolist(),
                    }
            except Exception:
                break
        return center.copy(), {
            "target_projected": True,
            "projection_alpha": 1.0,
            "original_target_xyz_m": target.tolist(),
            "final_target_xyz_m": center.tolist(),
        }

    def evaluate_anchor_errors(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        if not self.loaded:
            return out
        for name in REQUIRED_POSES:
            recorded = self.pose_xyz.get(name)
            coord = self._pose_centered_coordinate(name)
            if recorded is None or coord is None:
                continue
            base = self._axis_vector_from_centered(coord)
            final, clamped, _ = self._clamp_xyz_debug(base)
            out[name] = {
                "centered_hvd": coord.tolist(),
                "recorded_xyz_m": recorded.tolist(),
                "base_xyz_m": base.tolist(),
                "final_xyz_m": final.tolist(),
                "base_error_m": float(np.linalg.norm(base - recorded)),
                "final_error_m": float(np.linalg.norm(final - recorded)),
                "target_clamped": bool(clamped),
            }
        return out