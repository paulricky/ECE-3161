from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path
from typing import Optional

import values as val


def _finite_float(x, default=None):
    try:
        f = float(x)
    except Exception:
        return default
    return f if math.isfinite(f) else default


def _clamp(x: float, lo: float, hi: float) -> float:
    x = float(x)
    lo = float(lo)
    hi = float(hi)
    if hi < lo:
        lo, hi = hi, lo
    return max(lo, min(hi, x))


def _sat01(x: float) -> float:
    return _clamp(float(x), 0.0, 1.0)


def _ema(prev, x: float, alpha: float) -> float:
    x = float(x)
    if prev is None or not math.isfinite(float(prev)):
        return x
    a = _sat01(alpha)
    return (1.0 - a) * float(prev) + a * x


def _dist2(a, b) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _dist_px(a, b, frame_w: int, frame_h: int) -> float:
    return math.hypot((float(a[0]) - float(b[0])) * float(frame_w), (float(a[1]) - float(b[1])) * float(frame_h))


def _landmark_xy(lm, idx: int) -> tuple[float, float]:
    p = lm[idx]
    return float(p.x), float(p.y)


class HandDepthEstimator:
    """RGB-only MediaPipe hand depth estimator.

    Apparent hand size provides the absolute monocular estimate. MediaPipe's
    normalized z is used only as a small frame-to-frame trend correction.
    """

    def __init__(self):
        self.calibration = self.load_calibration_file()
        self._last_depth_m: Optional[float] = None
        self._last_size_norm: Optional[float] = None
        self._last_relative_z: Optional[float] = None
        self._last_timestamp: Optional[float] = None
        self._history: list[float] = []

    def reset(self) -> None:
        self._last_depth_m = None
        self._last_size_norm = None
        self._last_relative_z = None
        self._last_timestamp = None
        self._history.clear()

    def load_calibration_file(self) -> dict:
        configured = getattr(val, "HAND_MONOCULAR_DEPTH_CALIBRATION_FILE", "calibration_data/hand_depth_calibration.json")
        path = Path(str(configured)).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        try:
            if not path.exists():
                return {}
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _cfg(self, key: str, default):
        return self.calibration.get(key, getattr(val, key.upper(), default))

    def _near_far_m(self) -> tuple[float, float]:
        fit = self.calibration.get("fit", {}) if isinstance(self.calibration.get("fit", {}), dict) else {}
        near = _finite_float(fit.get("near_depth_m", self.calibration.get("near_depth_m")), getattr(val, "HAND_MONOCULAR_NEAR_M", 0.20))
        far = _finite_float(fit.get("far_depth_m", self.calibration.get("far_depth_m")), getattr(val, "HAND_MONOCULAR_FAR_M", 0.70))
        near = 0.20 if near is None else float(near)
        far = 0.70 if far is None else float(far)
        if far < near:
            near, far = far, near
        if abs(far - near) < 1e-6:
            far = near + 1e-3
        return near, far

    def _near_far_size(self) -> tuple[float, float]:
        fit = self.calibration.get("fit", {}) if isinstance(self.calibration.get("fit", {}), dict) else {}
        near = _finite_float(fit.get("near_size_norm", self.calibration.get("near_size_norm")), getattr(val, "HAND_MONOCULAR_NEAR_SIZE_NORM", 0.32))
        far = _finite_float(fit.get("far_size_norm", self.calibration.get("far_size_norm")), getattr(val, "HAND_MONOCULAR_FAR_SIZE_NORM", 0.12))
        near = 0.32 if near is None else float(near)
        far = 0.12 if far is None else float(far)
        if abs(near - far) < 1e-6:
            near = far + 1e-3
        return near, far

    def _size_depth_points(self) -> list[tuple[float, float]]:
        fit = self.calibration.get("fit", {}) if isinstance(self.calibration.get("fit", {}), dict) else {}
        points = [
            (
                _finite_float(fit.get("near_size_norm"), getattr(val, "HAND_MONOCULAR_NEAR_SIZE_NORM", 0.32)),
                _finite_float(fit.get("near_depth_m"), getattr(val, "HAND_MONOCULAR_NEAR_M", 0.20)),
            ),
            (
                _finite_float(fit.get("center_size_norm"), getattr(val, "HAND_MONOCULAR_CENTER_SIZE_NORM", 0.20)),
                _finite_float(fit.get("center_depth_m"), getattr(val, "HAND_MONOCULAR_CENTER_M", 0.45)),
            ),
            (
                _finite_float(fit.get("far_size_norm"), getattr(val, "HAND_MONOCULAR_FAR_SIZE_NORM", 0.12)),
                _finite_float(fit.get("far_depth_m"), getattr(val, "HAND_MONOCULAR_FAR_M", 0.70)),
            ),
        ]
        clean = [(float(s), float(d)) for s, d in points if s is not None and d is not None and s > 0.0 and d > 0.0]
        clean.sort(key=lambda item: item[0])
        return clean

    def _calibrated_depth_from_size(self, size_norm: float) -> Optional[float]:
        pts = self._size_depth_points()
        if len(pts) < 2 or not math.isfinite(float(size_norm)) or float(size_norm) <= 0.0:
            return None
        size = float(size_norm)
        near_m, far_m = self._near_far_m()
        if size <= pts[0][0]:
            return _clamp(pts[0][1], near_m, far_m)
        if size >= pts[-1][0]:
            return _clamp(pts[-1][1], near_m, far_m)
        for (s0, d0), (s1, d1) in zip(pts[:-1], pts[1:]):
            if s0 <= size <= s1 and abs(s1 - s0) > 1e-9:
                t = (size - s0) / (s1 - s0)
                return _clamp(d0 + t * (d1 - d0), near_m, far_m)
        return None

    def _safe_depth(self) -> float:
        near, far = self._near_far_m()
        if self._last_depth_m is not None and math.isfinite(float(self._last_depth_m)):
            return _clamp(float(self._last_depth_m), near, far)
        default = _finite_float(getattr(val, "HAND_DEPTH_DEFAULT_M", 0.45), 0.45)
        return _clamp(default, near, far)

    def _metrics(self, hand_lms, frame_w: int, frame_h: int) -> dict:
        lm = hand_lms.landmark
        pts = [_landmark_xy(lm, i) for i in range(len(lm))]
        wrist = _landmark_xy(lm, 0)
        thumb_tip = _landmark_xy(lm, 4)
        index_mcp = _landmark_xy(lm, 5)
        index_tip = _landmark_xy(lm, 8)
        middle_mcp = _landmark_xy(lm, 9)
        middle_pip = _landmark_xy(lm, 10)
        pinky_mcp = _landmark_xy(lm, 17)
        palm_width = _dist2(index_mcp, pinky_mcp)
        wrist_to_middle = _dist2(wrist, middle_mcp)
        palm_height = wrist_to_middle if wrist_to_middle > 1e-5 else _dist2(wrist, middle_pip)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bbox_w = max(xs) - min(xs)
        bbox_h = max(ys) - min(ys)
        bbox_size = math.sqrt(max(0.0, bbox_w * bbox_h))
        thumb_index_span = _dist2(thumb_tip, index_tip)
        return {
            "palm_width_norm": float(palm_width),
            "wrist_to_middle_mcp_norm": float(wrist_to_middle),
            "palm_height_norm": float(palm_height),
            "bbox_size_norm": float(bbox_size),
            "thumb_index_span_norm": float(thumb_index_span),
            "palm_width_px": float(_dist_px(index_mcp, pinky_mcp, frame_w, frame_h)),
            "wrist_to_middle_mcp_px": float(_dist_px(wrist, middle_mcp, frame_w, frame_h)),
            "bbox_size_px": float(math.sqrt(max(0.0, (bbox_w * frame_w) * (bbox_h * frame_h)))),
        }

    def _fused_size(self, metrics: dict) -> float:
        vals = [
            metrics.get("palm_width_norm"),
            metrics.get("wrist_to_middle_mcp_norm"),
            metrics.get("palm_height_norm"),
            metrics.get("bbox_size_norm"),
        ]
        clean = [float(x) for x in vals if _finite_float(x) is not None and float(x) > 1e-6]
        if not clean:
            return 0.0
        mode = str(getattr(val, "HAND_MONOCULAR_SIZE_FUSION", "median")).strip().lower()
        if mode == "weighted" and len(clean) >= 4:
            weights = [0.35, 0.30, 0.25, 0.10]
            return float(sum(weights[i] * clean[i] for i in range(4)) / sum(weights))
        return float(statistics.median(clean))

    def _camera_focal_px(self, frame_w: int, frame_h: int) -> Optional[float]:
        if not bool(getattr(val, "HAND_MONOCULAR_USE_CAMERA_INTRINSICS", True)):
            return None
        root = Path(__file__).resolve().parent
        candidates = [
            root / str(getattr(val, "CAMERA_CALIBRATION_JSON", "calibration_data/camera_calibration.json")),
            root / "calibration_data" / "camera_intrinsics.json",
        ]
        for path in candidates:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                K = data.get("camera_matrix", data.get("K"))
                if not (isinstance(K, list) and len(K) >= 2):
                    continue
                fx = _finite_float(K[0][0])
                fy = _finite_float(K[1][1])
                src_w = _finite_float(data.get("image_width", data.get("image_w")), frame_w)
                src_h = _finite_float(data.get("image_height", data.get("image_h")), frame_h)
                if fx is None or fy is None or src_w is None or src_h is None or src_w <= 0 or src_h <= 0:
                    continue
                fx *= float(frame_w) / float(src_w)
                fy *= float(frame_h) / float(src_h)
                return 0.5 * (fx + fy)
            except Exception:
                pass
        npz_path = root / str(getattr(val, "CAMERA_CALIBRATION_FILE", "calibration_data/camera_calibration.npz"))
        try:
            import numpy as np
            data = np.load(npz_path, allow_pickle=True)
            K = data["camera_matrix"] if "camera_matrix" in data else data["K"]
            if not (isinstance(K, list) and len(K) >= 2):
                K = np.asarray(K)
            fx = _finite_float(K[0][0])
            fy = _finite_float(K[1][1])
            src_w = _finite_float(data["image_width"][0] if "image_width" in data else None, frame_w)
            src_h = _finite_float(data["image_height"][0] if "image_height" in data else None, frame_h)
            if fx is None or fy is None or src_w is None or src_h is None or src_w <= 0 or src_h <= 0:
                return None
            fx *= float(frame_w) / float(src_w)
            fy *= float(frame_h) / float(src_h)
            return 0.5 * (fx + fy)
        except Exception:
            return None

    def estimate_from_monocular_size(self, hand_lms, frame_w: int, frame_h: int) -> dict:
        metrics = self._metrics(hand_lms, frame_w, frame_h)
        size_norm = self._fused_size(metrics)
        near_m, far_m = self._near_far_m()
        near_size, far_size = self._near_far_size()
        candidates: dict[str, float] = {}

        focal = self._camera_focal_px(frame_w, frame_h)
        if focal is not None:
            palm_real = _finite_float(self.calibration.get("real_palm_width_m"), getattr(val, "HAND_MONOCULAR_REAL_PALM_WIDTH_M", 0.085))
            wrist_real = _finite_float(self.calibration.get("real_wrist_to_middle_mcp_m"), getattr(val, "HAND_MONOCULAR_REAL_WRIST_TO_MIDDLE_MCP_M", 0.095))
            if palm_real and metrics["palm_width_px"] > 1.0:
                candidates["palm_width_intrinsics_m"] = _clamp(palm_real * focal / metrics["palm_width_px"], near_m, far_m)
            if wrist_real and metrics["wrist_to_middle_mcp_px"] > 1.0:
                candidates["wrist_to_middle_intrinsics_m"] = _clamp(wrist_real * focal / metrics["wrist_to_middle_mcp_px"], near_m, far_m)

        if size_norm > 1e-6:
            calibrated = self._calibrated_depth_from_size(size_norm)
            if calibrated is not None:
                candidates["hand_depth_calibration_m" if self.calibration else "default_hand_size_m"] = calibrated
            else:
                t = (size_norm - far_size) / (near_size - far_size)
                t = _sat01(t)
                candidates["default_hand_size_m"] = _clamp(far_m + (near_m - far_m) * t, near_m, far_m)

        clean = [float(x) for x in candidates.values() if math.isfinite(float(x))]
        if clean:
            depth_m = float(statistics.median(clean))
            if any("intrinsics" in k for k in candidates) and any("hand_depth_calibration" in k or "default_hand_size" in k for k in candidates):
                source = "intrinsics_fused"
            elif any("hand_depth_calibration" in k for k in candidates):
                source = "hand_depth_calibration"
            elif any("intrinsics" in k for k in candidates):
                source = "intrinsics_fused"
            else:
                source = "default_hand_size"
            confidence = _clamp(0.45 + min(size_norm, 0.35), 0.0, 1.0)
        else:
            depth_m = self._safe_depth()
            source = "fixed"
            confidence = 0.2
        return {
            "depth_m": depth_m,
            "source": source,
            "confidence": confidence,
            "hand_size_norm": size_norm,
            **metrics,
            "raw_candidates": candidates,
        }

    def estimate_from_mediapipe_relative_z(self, hand_lms) -> dict:
        if not bool(getattr(val, "HAND_MEDIAPIPE_RELATIVE_Z_ENABLED", True)):
            return {"delta_m": 0.0, "confidence": 0.0}
        try:
            zs = [float(p.z) for p in hand_lms.landmark]
            rel_z = float(statistics.median(zs))
        except Exception:
            return {"delta_m": 0.0, "confidence": 0.0}
        if not math.isfinite(rel_z):
            return {"delta_m": 0.0, "confidence": 0.0}
        if self._last_relative_z is None:
            self._last_relative_z = rel_z
            return {"delta_m": 0.0, "confidence": 0.2, "relative_z": rel_z}
        dz = rel_z - float(self._last_relative_z)
        self._last_relative_z = rel_z
        max_effect = abs(float(getattr(val, "HAND_MEDIAPIPE_RELATIVE_Z_MAX_EFFECT_M", 0.035)))
        # MediaPipe hand z is relative and sign conventions vary; use only the
        # small frame-to-frame trend. More negative usually means closer.
        delta_m = _clamp(dz, -1.0, 1.0) * max_effect
        return {"delta_m": delta_m, "confidence": 0.35, "relative_z": rel_z}

    def fuse_depth_candidates(self, size_estimate: dict, relz_estimate: dict) -> tuple[float, str, float]:
        depth = float(size_estimate.get("depth_m", self._safe_depth()))
        source = str(size_estimate.get("source", "monocular_size"))
        confidence = float(size_estimate.get("confidence", 0.5))
        rel_conf = float(relz_estimate.get("confidence", 0.0))
        weight = _clamp(float(getattr(val, "HAND_MEDIAPIPE_RELATIVE_Z_WEIGHT", 0.12)), 0.0, 0.5)
        delta = float(relz_estimate.get("delta_m", 0.0)) if rel_conf > 0.0 else 0.0
        if math.isfinite(delta) and abs(delta) > 0.0:
            depth = depth + weight * delta
        return depth, source, confidence

    def depth_m_to_norm(self, depth_m: float) -> float:
        near, far = self._near_far_m()
        return _sat01(1.0 - ((float(depth_m) - near) / (far - near)))

    def depth_norm_to_workspace(self, depth_norm: float, near_target_m: float, far_target_m: float) -> float:
        return float(far_target_m) + (float(near_target_m) - float(far_target_m)) * _sat01(depth_norm)

    def _filter_depth(self, depth_m: float, confidence: float) -> tuple[float, bool]:
        near, far = self._near_far_m()
        depth_m = _clamp(depth_m, near, far)
        valid = math.isfinite(depth_m)
        if not valid:
            return self._safe_depth(), False

        if self._history:
            mean = statistics.mean(self._history)
            stdev = statistics.pstdev(self._history) if len(self._history) > 1 else 0.0
            reject_std = float(getattr(val, "HAND_DEPTH_OUTLIER_REJECT_STD", 2.5))
            if stdev > 1e-4 and abs(depth_m - mean) > reject_std * stdev and confidence < 0.75:
                return self._safe_depth(), False

        if self._last_depth_m is not None:
            max_step = abs(float(getattr(val, "HAND_DEPTH_MAX_STEP_M", 0.06)))
            delta = depth_m - float(self._last_depth_m)
            if abs(delta) > max_step and confidence < 0.8:
                depth_m = float(self._last_depth_m) + math.copysign(max_step, delta)
            alpha = float(getattr(val, "HAND_DEPTH_SMOOTHING_ALPHA", 0.35))
            depth_m = _ema(self._last_depth_m, depth_m, alpha)

        depth_m = _clamp(depth_m, near, far)
        self._last_depth_m = depth_m
        self._history.append(depth_m)
        self._history = self._history[-30:]
        return depth_m, True

    def estimate_depth(self, hand_lms, world_landmarks=None, frame_w: int = 1, frame_h: int = 1, camera_intrinsics=None, timestamp=None) -> dict:
        del world_landmarks, camera_intrinsics
        timestamp = time.time() if timestamp is None else float(timestamp)
        size_est = self.estimate_from_monocular_size(hand_lms, frame_w, frame_h)
        relz_est = self.estimate_from_mediapipe_relative_z(hand_lms)
        depth_m, source, confidence = self.fuse_depth_candidates(size_est, relz_est)
        depth_m, valid = self._filter_depth(depth_m, confidence)
        if not valid and bool(getattr(val, "HAND_DEPTH_HOLD_LAST_ON_INVALID", True)):
            source = "last_valid" if self._last_depth_m is not None else "fixed"
            depth_m = self._safe_depth()
        self._last_timestamp = timestamp
        depth_norm = self.depth_m_to_norm(depth_m)
        raw_candidates = dict(size_est.get("raw_candidates", {}))
        raw_candidates["mediapipe_relative_z_delta_m"] = float(relz_est.get("delta_m", 0.0))
        return {
            "depth_m": float(depth_m),
            "depth_norm": float(depth_norm),
            "source": source,
            "confidence": float(_clamp(confidence, 0.0, 1.0)),
            "raw_candidates": raw_candidates,
            "hand_size_norm": float(size_est.get("hand_size_norm", 0.0)),
            "palm_width_norm": float(size_est.get("palm_width_norm", 0.0)),
            "wrist_to_middle_mcp_norm": float(size_est.get("wrist_to_middle_mcp_norm", 0.0)),
            "palm_height_norm": float(size_est.get("palm_height_norm", 0.0)),
            "bbox_size_norm": float(size_est.get("bbox_size_norm", 0.0)),
            "thumb_index_span_norm": float(size_est.get("thumb_index_span_norm", 0.0)),
            "valid": bool(valid),
            "using_camera_intrinsics": any("intrinsics" in k for k in raw_candidates),
            "using_hand_depth_calibration": bool(self.calibration),
        }


class DepthCalibrator(HandDepthEstimator):
    """Compatibility wrapper for older call sites."""

    def __init__(self, window: int = 240, ema_alpha: float = 0.35):
        del window, ema_alpha
        super().__init__()

    def normalize_hand_size(self, size_norm_or_px: float, *, smooth: bool = False) -> float:
        near_size, far_size = self._near_far_size()
        t = (float(size_norm_or_px) - far_size) / (near_size - far_size)
        norm = _sat01(t)
        if smooth:
            depth_m = self._near_far_m()[1] + (self._near_far_m()[0] - self._near_far_m()[1]) * norm
            depth_m, _ = self._filter_depth(depth_m, 0.5)
            return self.depth_m_to_norm(depth_m)
        return norm

    def update_from_hand_size(self, size_norm_or_px: float) -> float:
        return self.normalize_hand_size(size_norm_or_px, smooth=True)

    def get_smoothed_depth_norm(self) -> float:
        return self.depth_m_to_norm(self._safe_depth())

    def normalize01(self, depth_proxy: float) -> float:
        return self.update_from_hand_size(depth_proxy)

    def value01(self, depth_proxy: float) -> float:
        return self.normalize01(depth_proxy)
