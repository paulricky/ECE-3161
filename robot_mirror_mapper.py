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

REQUIRED_POSES = tuple(getattr(val, "ROBOT_MIRROR_REQUIRED_POSES", (
    "center",
    "mirror_left",
    "mirror_right",
    "mirror_up",
    "mirror_down",
    "mirror_near",
    "mirror_far",
)))

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
    """Map live hand mirror coordinates to robot FK workspace targets.

    Two calibration files can be used together:
    - robot_mirror_workspace_calibration.json: robot FK extrema and joint seeds.
    - hand_mirror_position_calibration.json: matching hand positions by pose name.

    If the hand calibration is missing, ideal normalized coordinates are used.
    Saved robot joints are returned only as IK seeds; runtime does not directly
    command learned/saved joint positions.
    """

    def __init__(self, path: Optional[str] = None, hand_path: Optional[str] = None):
        self.path = _resolve_path(path or getattr(val, "ROBOT_MIRROR_WORKSPACE_CALIBRATION_FILE", "calibration_data/robot_mirror_workspace_calibration.json"))
        self.hand_path = _resolve_path(hand_path or getattr(val, "HAND_MIRROR_POSITION_CALIBRATION_FILE", "calibration_data/hand_mirror_position_calibration.json"))
        self.loaded = False
        self.hand_loaded = False
        self.error = ""
        self.hand_error = ""
        self.data: dict = {}
        self.hand_data: dict = {}
        self.poses: dict = {}
        self.hand_poses: dict = {}
        self.pose_xyz: dict[str, np.ndarray] = {}
        self.pose_joints: dict[str, Optional[dict[str, float]]] = {}
        self.hand_pose_input: dict[str, np.ndarray] = {}
        self.samples_x = np.zeros((0, 3), dtype=np.float64)
        self.residual_y = np.zeros((0, 3), dtype=np.float64)
        self.sample_names: list[str] = []
        self.rbf_weights: Optional[np.ndarray] = None
        self.rbf_ready = False
        self.xyz_min = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
        self.xyz_max = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
        self.hand_depth_pairing_source = "ideal"
        self.load(self.path, self.hand_path)

    def reset(self) -> None:
        self.__init__(str(self.path), str(self.hand_path))

    def is_loaded(self) -> bool:
        return bool(self.loaded)

    def load(self, path: Optional[str] = None, hand_path: Optional[str] = None) -> bool:
        if path is not None:
            self.path = _resolve_path(str(path))
        if hand_path is not None:
            self.hand_path = _resolve_path(str(hand_path))
        self.loaded = False
        self.hand_loaded = False
        self.error = ""
        self.hand_error = ""
        self.data = {}
        self.hand_data = {}
        self.poses = {}
        self.hand_poses = {}
        self.pose_xyz = {}
        self.pose_joints = {}
        self.hand_pose_input = {}
        self.samples_x = np.zeros((0, 3), dtype=np.float64)
        self.residual_y = np.zeros((0, 3), dtype=np.float64)
        self.sample_names = []
        self.rbf_weights = None
        self.rbf_ready = False
        self.xyz_min = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
        self.xyz_max = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
        self.hand_depth_pairing_source = "ideal"

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

        self._load_hand_calibration(self.hand_path)
        self._build_sample_residuals()
        self._fit_rbf()
        all_xyz = np.stack(list(self.pose_xyz.values()), axis=0)
        margin = abs(float(getattr(val, "ROBOT_MIRROR_CLAMP_MARGIN_M", 0.020)))
        self.xyz_min = np.min(all_xyz, axis=0) - margin
        self.xyz_max = np.max(all_xyz, axis=0) + margin
        self.data = data
        self.loaded = True
        return True

    def _depth_from_hand_size(self, hand: dict) -> tuple[Optional[float], str]:
        size = _finite_float(hand.get("hand_size_norm"))
        if size is None or size <= 0.0:
            return None, "unavailable"
        near_size = _finite_float(getattr(val, "HAND_MONOCULAR_NEAR_SIZE_NORM", None), 0.32)
        far_size = _finite_float(getattr(val, "HAND_MONOCULAR_FAR_SIZE_NORM", None), 0.12)
        if near_size is None or far_size is None or abs(float(near_size) - float(far_size)) < 1e-6:
            return None, "unavailable"
        depth_norm = (float(size) - float(far_size)) / (float(near_size) - float(far_size))
        return _sat01(depth_norm), "hand_size_norm"

    def _extract_hand_depth(self, hand: dict) -> tuple[Optional[float], str]:
        preferred = str(getattr(val, "HAND_MIRROR_DEPTH_PAIRING_SOURCE", "raw")).strip().lower()
        if preferred == "raw":
            order = [("depth_norm_raw", "depth_norm_raw"), ("depth_norm", "depth_norm"), ("depth_norm_filtered", "depth_norm_filtered")]
        elif preferred == "filtered":
            order = [("depth_norm", "depth_norm"), ("depth_norm_filtered", "depth_norm_filtered"), ("depth_norm_raw", "depth_norm_raw")]
        elif preferred == "hand_size":
            size_depth, source = self._depth_from_hand_size(hand)
            if size_depth is not None:
                return size_depth, source
            order = [("depth_norm_raw", "depth_norm_raw"), ("depth_norm", "depth_norm"), ("depth_norm_filtered", "depth_norm_filtered")]
        else:
            order = [("depth_norm_raw", "depth_norm_raw"), ("depth_norm", "depth_norm"), ("depth_norm_filtered", "depth_norm_filtered")]
        for key, source in order:
            d = _finite_float(hand.get(key))
            if d is not None:
                return _sat01(d), source
        return self._depth_from_hand_size(hand)

    def _load_hand_calibration(self, hand_path: Path) -> bool:
        if not bool(getattr(val, "ROBOT_MIRROR_PAIRED_CALIBRATION_ENABLED", True)):
            self.hand_error = "disabled"
            return False
        if not hand_path.exists():
            self.hand_error = "missing"
            return False
        try:
            with hand_path.open("r", encoding="utf-8") as f:
                hand_data = json.load(f)
        except Exception as exc:
            self.hand_error = f"load_failed:{exc}"
            return False
        if not isinstance(hand_data, dict) or hand_data.get("calibration_type") not in {
            "hand_mirror_position_extrema",
            "hand_to_robot_workspace",
        }:
            self.hand_error = "bad_schema"
            return False
        poses = hand_data.get("poses", {})
        if not isinstance(poses, dict):
            self.hand_error = "poses_missing"
            return False
        depth_sources: list[str] = []
        for name, item in poses.items():
            if not isinstance(item, dict):
                continue
            hand = item.get("hand", item)
            if not isinstance(hand, dict):
                continue
            x = _finite_float(hand.get("x_norm"))
            y = _finite_float(hand.get("y_norm"))
            d, depth_source = self._extract_hand_depth(hand)
            if x is None or y is None or d is None:
                continue
            self.hand_poses[str(name)] = item
            self.hand_pose_input[str(name)] = np.array([_sat01(x), _sat01(y), _sat01(d)], dtype=np.float64)
            depth_sources.append(depth_source)
        required_hand = [name for name in REQUIRED_POSES if name not in self.hand_pose_input]
        if required_hand:
            self.hand_error = "missing_required:" + ",".join(required_hand)
            return False
        self.hand_data = hand_data
        self.hand_loaded = True
        self.hand_error = ""
        unique_sources = sorted(set(depth_sources))
        self.hand_depth_pairing_source = unique_sources[0] if len(unique_sources) == 1 else "mixed:" + ",".join(unique_sources)
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

    def _scale_axis_from_hand(self, value: float, center: float, negative: float, positive: float) -> float:
        value = _sat01(value)
        center = _sat01(center)
        negative = _sat01(negative)
        positive = _sat01(positive)
        delta = value - center
        pos_span = positive - center
        neg_span = negative - center
        candidates = []
        if abs(pos_span) > 1e-6 and delta * pos_span >= -1e-12:
            candidates.append(delta / pos_span)
        if abs(neg_span) > 1e-6 and delta * neg_span >= -1e-12:
            candidates.append(-delta / neg_span)
        if candidates:
            # Prefer the candidate with lower magnitude when both axes are noisy.
            return float(_clip(min(candidates, key=lambda z: abs(z)), -1.0, 1.0))
        # Outside the calibrated segment: extrapolate toward the closer span.
        spans = [(abs(pos_span), pos_span, 1.0), (abs(neg_span), neg_span, -1.0)]
        spans = [s for s in spans if s[0] > 1e-6]
        if not spans:
            return 0.0
        _mag, span, sign = min(spans, key=lambda s: abs(delta - s[1]))
        if sign > 0:
            return float(_clip(delta / span, -1.0, 1.0))
        return float(_clip(-delta / span, -1.0, 1.0))

    def _centered_inputs_from_hand_calibration(self, horizontal_norm, vertical_norm, depth_norm) -> np.ndarray:
        hp = self.hand_pose_input
        c = hp["center"]
        h = self._scale_axis_from_hand(horizontal_norm, c[0], hp["mirror_left"][0], hp["mirror_right"][0])
        v = self._scale_axis_from_hand(vertical_norm, c[1], hp["mirror_down"][1], hp["mirror_up"][1])
        d = self._scale_axis_from_hand(depth_norm, c[2], hp["mirror_far"][2], hp["mirror_near"][2])
        x = np.array([h, v, d], dtype=np.float64)
        if bool(getattr(val, "HAND_MIRROR_APPLY_FLIPS_TO_PAIRED_INPUTS", False)):
            if bool(getattr(val, "HAND_MIRROR_HORIZONTAL_FLIP", False)):
                x[0] = -x[0]
            if bool(getattr(val, "HAND_MIRROR_VERTICAL_FLIP", False)):
                x[1] = -x[1]
            if bool(getattr(val, "HAND_MIRROR_DEPTH_FLIP", False)):
                x[2] = -x[2]
        if bool(getattr(val, "HAND_MIRROR_CLAMP_INPUTS", True)):
            x = np.clip(x, -1.0, 1.0)
        return x

    def _centered_inputs(self, horizontal_norm, vertical_norm, depth_norm) -> np.ndarray:
        if self.hand_loaded:
            raw = np.array([_sat01(horizontal_norm), _sat01(vertical_norm), _sat01(depth_norm)], dtype=np.float64)
            for name in REQUIRED_POSES:
                recorded = self.hand_pose_input.get(name)
                coord = POSE_COORDS.get(name)
                if recorded is not None and coord is not None and float(np.linalg.norm(raw - recorded)) <= 1e-7:
                    return np.asarray(coord, dtype=np.float64).reshape(3)
            return self._centered_inputs_from_hand_calibration(horizontal_norm, vertical_norm, depth_norm)
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

    @staticmethod
    def _signed_gamma(x: float, gamma: float) -> float:
        if not math.isfinite(float(gamma)) or float(gamma) <= 0.0:
            gamma = 1.0
        x = _clip(float(x), -1.0, 1.0)
        return math.copysign(abs(x) ** float(gamma), x)

    def _shape_centered_inputs(self, x: np.ndarray) -> np.ndarray:
        raw = np.asarray(x, dtype=np.float64).reshape(3)
        shaped = raw.copy()
        # Horizontal is intentionally identity so left/right behavior is unchanged.
        shaped[0] = raw[0]
        if bool(getattr(val, "ROBOT_WORKSPACE_VERTICAL_ENDPOINT_BOOST_ENABLED", True)):
            shaped[1] = self._signed_gamma(raw[1], float(getattr(val, "ROBOT_WORKSPACE_VERTICAL_RESPONSE_GAMMA", 1.0)))
        if bool(getattr(val, "ROBOT_WORKSPACE_DEPTH_ENDPOINT_BOOST_ENABLED", True)):
            shaped[2] = self._signed_gamma(raw[2], float(getattr(val, "ROBOT_WORKSPACE_DEPTH_RESPONSE_GAMMA", 1.0)))
        if bool(getattr(val, "ROBOT_WORKSPACE_EXTENSION_SHAPING_CLAMP", True)):
            shaped = np.clip(shaped, -1.0, 1.0)
            shaped[0] = raw[0]
        return shaped

    def _axis_blend_from_centered(self, x: np.ndarray) -> np.ndarray:
        center = self.pose_xyz["center"]
        h, v, d = [float(a) for a in np.asarray(x, dtype=np.float64).reshape(3)]
        out = center.copy()
        out += abs(h) * ((self.pose_xyz["mirror_left"] if h < 0.0 else self.pose_xyz["mirror_right"]) - center)
        out += abs(v) * ((self.pose_xyz["mirror_down"] if v < 0.0 else self.pose_xyz["mirror_up"]) - center)
        out += abs(d) * ((self.pose_xyz["mirror_far"] if d < 0.0 else self.pose_xyz["mirror_near"]) - center)
        return out

    def _pose_centered_coordinate(self, name: str) -> Optional[np.ndarray]:
        if name in REQUIRED_POSES and name in POSE_COORDS:
            return np.asarray(POSE_COORDS[name], dtype=np.float64).reshape(3)
        if self.hand_loaded and name in self.hand_pose_input:
            raw = self.hand_pose_input[name]
            return self._centered_inputs_from_hand_calibration(raw[0], raw[1], raw[2])
        coord = POSE_COORDS.get(name)
        if coord is None:
            return None
        return np.asarray(coord, dtype=np.float64).reshape(3)

    def _build_sample_residuals(self) -> None:
        xs = []
        residuals = []
        names = []
        for name in self.pose_xyz:
            x = self._pose_centered_coordinate(name)
            if x is None:
                continue
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
        if (
            bool(getattr(val, "ROBOT_MIRROR_CLAMP_TO_CALIBRATED_BOUNDS", True))
            and np.all(np.isfinite(self.xyz_min))
            and np.all(np.isfinite(self.xyz_max))
        ):
            arr = np.minimum(np.maximum(arr, self.xyz_min), self.xyz_max)
        return arr

    def _clamp_xyz_debug(self, xyz) -> tuple[np.ndarray, bool, np.ndarray]:
        before = np.asarray(xyz, dtype=np.float64).reshape(3)
        after = self._clamp_xyz(before)
        clamped = bool(np.linalg.norm(after - before) > 1e-12)
        return after, clamped, before

    def _exact_required_anchor_name(self, x: np.ndarray, tol: float = 1e-7) -> Optional[str]:
        x = np.asarray(x, dtype=np.float64).reshape(3)
        for name in REQUIRED_POSES:
            coord = self._pose_centered_coordinate(name)
            if coord is not None and float(np.linalg.norm(x - coord)) <= tol:
                return name
        return None

    def map_hand_to_robot_target(self, horizontal_norm, vertical_norm, depth_norm):
        if not self.loaded:
            return None, {
                "mirror_mapping_source": f"fallback_no_mirror_calibration:{self.error or 'missing'}",
                "mirror_method": "none",
            }
        x_raw = self._centered_inputs(horizontal_norm, vertical_norm, depth_norm)
        x = self._shape_centered_inputs(x_raw)
        base = self._axis_blend_from_centered(x)
        method = str(getattr(val, "ROBOT_MIRROR_MAPPING_METHOD", "paired_axis_blend_knn_residual")).strip().lower()
        residual = np.zeros(3, dtype=np.float64)
        nearest = self._nearest_pose_name(x)
        residual_source = "none"
        used_method = "paired_axis_blend" if self.hand_loaded else "axis_blend"
        exact_anchor = self._exact_required_anchor_name(x)
        if exact_anchor is not None:
            nearest = exact_anchor
            residual_source = "required_anchor_zero"
        elif method in {"axis_blend_knn_residual", "paired_axis_blend_knn_residual", "knn_residual"}:
            residual, nearest = self._knn_residual(x)
            residual_source = "knn_residual" if nearest is not None else "none"
            used_method = ("paired_axis_blend_knn_residual" if self.hand_loaded else "axis_blend_knn_residual") if nearest is not None else used_method
        elif method in {"axis_blend_rbf_residual", "paired_axis_blend_rbf_residual", "rbf_residual"}:
            rbf = self._rbf_residual(x)
            if rbf is not None:
                residual = rbf
                residual_source = "bounded_rbf_residual"
                used_method = "paired_axis_blend_rbf_residual" if self.hand_loaded else "axis_blend_rbf_residual"
            else:
                residual, nearest = self._knn_residual(x)
                residual_source = "knn_residual_fallback"
                used_method = "paired_axis_blend_knn_residual" if self.hand_loaded else "axis_blend_knn_residual"
        residual = self._clamp_residual(residual)
        target_before_clamp = base + residual
        final, target_clamped, target_before_clamp = self._clamp_xyz_debug(target_before_clamp)
        debug = {
            "mirror_mapping_source": "paired_robot_hand_mirror_calibration" if self.hand_loaded else "robot_mirror_workspace_calibration",
            "mirror_method": used_method,
            "mirror_residual_source": residual_source,
            "mirror_horizontal_norm": float(_sat01(horizontal_norm)),
            "mirror_vertical_norm": float(_sat01(vertical_norm)),
            "mirror_depth_norm": float(_sat01(depth_norm)),
            "mirror_h_centered": float(x[0]),
            "mirror_v_centered": float(x[1]),
            "mirror_d_centered": float(x[2]),
            "mirror_h_centered_raw": float(x_raw[0]),
            "mirror_h_centered_shaped": float(x[0]),
            "mirror_v_centered_raw": float(x_raw[1]),
            "mirror_v_centered_shaped": float(x[1]),
            "mirror_d_centered_raw": float(x_raw[2]),
            "mirror_d_centered_shaped": float(x[2]),
            "mirror_extension_shaping_enabled": bool(
                bool(getattr(val, "ROBOT_WORKSPACE_VERTICAL_ENDPOINT_BOOST_ENABLED", True))
                or bool(getattr(val, "ROBOT_WORKSPACE_DEPTH_ENDPOINT_BOOST_ENABLED", True))
            ),
            "mirror_nearest_pose": nearest,
            "target_xyz_base_m": base.tolist(),
            "target_xyz_residual_m": residual.tolist(),
            "mirror_target_before_clamp_m": target_before_clamp.tolist(),
            "mirror_target_after_clamp_m": final.tolist(),
            "mirror_target_clamped": bool(target_clamped),
            "mirror_calibrated_bounds_min_m": self.xyz_min.tolist(),
            "mirror_calibrated_bounds_max_m": self.xyz_max.tolist(),
            "target_xyz_final_m": final.tolist(),
            "robot_mirror_calibration_loaded": True,
            "paired_hand_calibration_loaded": bool(self.hand_loaded),
            "hand_depth_pairing_source": self.hand_depth_pairing_source,
            "hand_mirror_calibration_path": str(self.hand_path),
            "hand_mirror_calibration_error": self.hand_error,
            "robot_mirror_calibration_path": str(self.path),
            "robot_mirror_direct_joint_learning_enabled": bool(getattr(val, "ROBOT_MIRROR_DIRECT_JOINT_LEARNING_ENABLED", False)),
        }
        return final, debug

    def evaluate_anchor_errors(self) -> dict[str, dict]:
        """Evaluate how well required paired anchors reproduce recorded robot poses."""
        out: dict[str, dict] = {}
        if not self.loaded:
            return out
        for name in REQUIRED_POSES:
            recorded = self.pose_xyz.get(name)
            x = self._pose_centered_coordinate(name)
            if recorded is None or x is None:
                continue
            base = self._axis_blend_from_centered(x)
            if self.hand_loaded and name in self.hand_pose_input:
                final, debug = self.map_hand_to_robot_target(
                    self.hand_pose_input[name][0],
                    self.hand_pose_input[name][1],
                    self.hand_pose_input[name][2],
                )
                if final is None:
                    final = base
                    debug = {}
            else:
                final, clamped, _before = self._clamp_xyz_debug(base)
                debug = {"mirror_target_clamped": clamped}
            out[name] = {
                "centered_hvd": np.asarray(x, dtype=np.float64).reshape(3).tolist(),
                "recorded_xyz_m": recorded.tolist(),
                "base_xyz_m": base.tolist(),
                "final_xyz_m": np.asarray(final, dtype=np.float64).reshape(3).tolist(),
                "base_error_m": float(np.linalg.norm(base - recorded)),
                "final_error_m": float(np.linalg.norm(np.asarray(final, dtype=np.float64).reshape(3) - recorded)),
                "target_clamped": bool(debug.get("mirror_target_clamped", False)),
            }
        return out

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
        x = self._shape_centered_inputs(self._centered_inputs(horizontal_norm, vertical_norm, depth_norm))
        if self.samples_x.size == 0:
            return None, {"ik_seed_source": "none"}
        order = np.argsort(np.linalg.norm(self.samples_x - x.reshape(1, 3), axis=1))
        for idx in order:
            name = self.sample_names[int(idx)]
            seed = self.pose_joints.get(name)
            if isinstance(seed, dict):
                return dict(seed), {"ik_seed_source": "robot_mirror_nearest_pose", "mirror_nearest_pose": name}
        return None, {"ik_seed_source": "none"}