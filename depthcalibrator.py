from collections import deque

class DepthCalibrator:

    def __init__(self, window=240):
        self.buf = deque(maxlen=window)

    def update(self, depth_proxy):
        self.buf.append(float(depth_proxy))

    def get_minmax(self):
        if not self.buf:
            return (0.0, 1.0)
        mn = min(self.buf)
        mx = max(self.buf)
        if mx - mn < 1e-6:
            mx = mn + 1e-6
        return mn, mx
