from __future__ import annotations

import math
from collections import deque
from typing import Optional

import values as val


def _sat01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _ema(prev, x: float, alpha: float) -> float:
    x = float(x)
    if prev is None:
        return x
    a = _sat01(alpha)
    return (1.0 - a) * float(prev) + a * x


def _finite_or_none(x) -> Optional[float]:
    try:
        f = float(x)
    except Exception:
        return None
    if not math.isfinite(f):
        return None
    return f


class DepthCalibrator:
    def __init__(self, window: int = 240, ema_alpha: float = 0.15):
        self.buf = deque(maxlen=max(1, int(window)))
        self.ema_alpha = float(ema_alpha)
        self._depth_ema = None
        self._last_valid_norm = None

    def _bounds(self) -> tuple[float, float]:
        lo = _finite_or_none(getattr(val, "HAND_DEPTH_MIN_NORM", 0.0))
        hi = _finite_or_none(getattr(val, "HAND_DEPTH_MAX_NORM", 1.0))
        lo = 0.0 if lo is None else _sat01(lo)
        hi = 1.0 if hi is None else _sat01(hi)
        if hi < lo:
            lo, hi = hi, lo
        if abs(hi - lo) < 1e-9:
            hi = min(1.0, lo + 1e-6)
        return lo, hi

    def _safe_previous(self) -> float:
        if self._last_valid_norm is not None and math.isfinite(float(self._last_valid_norm)):
            return float(self._last_valid_norm)
        return 0.5

    def _clamp_norm(self, x: float) -> float:
        lo, hi = self._bounds()
        return max(lo, min(hi, float(x)))

    def _smooth_norm(self, norm: float) -> float:
        norm = self._clamp_norm(norm)
        self._depth_ema = _ema(self._depth_ema, norm, self.ema_alpha)
        self._depth_ema = self._clamp_norm(self._depth_ema)
        self._last_valid_norm = float(self._depth_ema)
        self.buf.append(float(self._depth_ema))
        return float(self._depth_ema)

    def reset(self) -> None:
        self.buf.clear()
        self._depth_ema = None
        self._last_valid_norm = None

    def update(self, depth_proxy: float) -> None:
        self._smooth_norm(self.normalize_hand_size(depth_proxy, smooth=False))

    def minmax(self):
        if not self.buf:
            return 0.0, 1.0
        mn = float(min(self.buf))
        mx = float(max(self.buf))
        if mx - mn < 1e-9:
            mx = mn + 1e-9
        return mn, mx

    def get_minmax(self):
        return self.minmax()

    def normalize_hand_size(self, size_norm_or_px: float, *, smooth: bool = False) -> float:
        size = _finite_or_none(size_norm_or_px)
        if size is None or size <= 0.0:
            return self._safe_previous()

        near_px = _finite_or_none(getattr(val, "HAND_SIZE_NEAR_PIXELS", None))
        far_px = _finite_or_none(getattr(val, "HAND_SIZE_FAR_PIXELS", None))
        near_norm = _finite_or_none(getattr(val, "HAND_SIZE_NEAR_NORM", 0.32))
        far_norm = _finite_or_none(getattr(val, "HAND_SIZE_FAR_NORM", 0.12))
        near = near_px if near_px is not None and near_px > 0.0 else near_norm
        far = far_px if far_px is not None and far_px > 0.0 else far_norm
        if near is None or far is None or abs(float(near) - float(far)) < 1e-9:
            return self._safe_previous()

        # Larger apparent hand size means closer to the camera, so the
        # normalized depth is 1 near and 0 far unless explicitly inverted.
        norm = (size - float(far)) / (float(near) - float(far))
        if bool(getattr(val, "HAND_SIZE_DEPTH_INVERT", False)):
            norm = 1.0 - norm
        norm = self._clamp_norm(norm)
        return self._smooth_norm(norm) if smooth else norm

    def update_from_hand_size(self, size_norm_or_px: float) -> float:
        return self.normalize_hand_size(size_norm_or_px, smooth=True)

    def depth_m_to_norm(self, depth_m: float, *, smooth: bool = False) -> float:
        depth = _finite_or_none(depth_m)
        if depth is None or depth <= 0.0:
            return self._safe_previous()
        near = _finite_or_none(getattr(val, "HAND_ARUCO_DEPTH_MIN_M", 0.15))
        far = _finite_or_none(getattr(val, "HAND_ARUCO_DEPTH_MAX_M", 0.70))
        if near is None or far is None or abs(far - near) < 1e-9:
            return self._safe_previous()

        # Smaller metric camera depth means closer to the camera.
        norm = 1.0 - ((depth - near) / (far - near))
        norm = self._clamp_norm(norm)
        return self._smooth_norm(norm) if smooth else norm

    def update_from_aruco_depth(self, depth_m: float) -> float:
        return self.depth_m_to_norm(depth_m, smooth=True)

    def get_smoothed_depth_norm(self) -> float:
        return self._safe_previous() if self._depth_ema is None else float(self._depth_ema)

    def normalize01(self, depth_proxy: float) -> float:
        return self.update_from_hand_size(depth_proxy)

    def value01(self, depth_proxy: float) -> float:
        return self.normalize01(depth_proxy)
