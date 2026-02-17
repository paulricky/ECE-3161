from __future__ import annotations

from collections import deque
import numpy as np
import mathmodel as mm


class DepthCalibrator:
    def __init__(self, window: int = 240, ema_alpha: float = 0.15):
        self.buf = deque(maxlen=int(window))
        self.ema_alpha = float(ema_alpha)
        self._depth_ema = None

    def update(self, depth_proxy: float):
        depth_proxy = float(depth_proxy)
        self._depth_ema = mm.ema(self._depth_ema, depth_proxy, self.ema_alpha)
        self.buf.append(float(self._depth_ema))

    def minmax(self):
        if not self.buf:
            return 0.0, 1.0
        mn = float(min(self.buf))
        mx = float(max(self.buf))
        if mx - mn < 1e-9:
            mx = mn + 1e-9
        return mn, mx

    def normalize01(self, depth_proxy: float) -> float:
        self.update(depth_proxy)
        mn, mx = self.minmax()
        return mm.sat01((float(self._depth_ema) - mn) / (mx - mn))
