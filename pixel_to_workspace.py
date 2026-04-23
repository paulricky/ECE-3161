"""Pixel-to-workspace projection for a top-down (or any PnP-calibrated) camera.

Given intrinsics (K, dist) and extrinsics (R_cam_from_ws, t_cam_from_ws) in the
convention `p_cam = R @ p_ws + t`, this module intersects back-projected rays
with a fixed z = table_z plane in the workspace frame. Pure numpy/OpenCV; no
hardware or file I/O.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


class PixelToWorkspace:
    def __init__(
        self,
        K: np.ndarray,
        dist: np.ndarray,
        R_cam_from_ws: np.ndarray,
        t_cam_from_ws: np.ndarray,
        table_z: float,
    ):
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.dist = np.asarray(dist, dtype=np.float64).reshape(-1, 1)
        self.R_cam_from_ws = np.asarray(R_cam_from_ws, dtype=np.float64).reshape(3, 3)
        self.t_cam_from_ws = np.asarray(t_cam_from_ws, dtype=np.float64).reshape(3)
        self.table_z = float(table_z)

        self.R_ws_cam = self.R_cam_from_ws.T
        self.origin_ws = -self.R_ws_cam @ self.t_cam_from_ws

    def project(self, uv: Sequence[float]) -> Optional[np.ndarray]:
        """Project pixel (u, v) to the workspace plane z = table_z.

        Returns a length-3 array [x_ws, y_ws, table_z] or None if the ray is
        parallel to the plane or points away from it.
        """
        u, v = float(uv[0]), float(uv[1])
        pt = np.array([[[u, v]]], dtype=np.float64)
        pt_u = cv2.undistortPoints(pt, self.K, self.dist)
        nx = float(pt_u[0, 0, 0])
        ny = float(pt_u[0, 0, 1])
        d_cam = np.array([nx, ny, 1.0], dtype=np.float64)
        norm = float(np.linalg.norm(d_cam))
        if norm < 1e-9:
            return None
        d_cam = d_cam / norm

        d_ws = self.R_ws_cam @ d_cam
        if abs(d_ws[2]) < 1e-6:
            return None

        s = (self.table_z - self.origin_ws[2]) / d_ws[2]
        if s <= 0.0:
            return None

        p_ws = self.origin_ws + s * d_ws
        return np.array([p_ws[0], p_ws[1], self.table_z], dtype=np.float64)

    def project_batch(self, uvs: Sequence[Sequence[float]]) -> np.ndarray:
        out = []
        for uv in uvs:
            p = self.project(uv)
            out.append(p if p is not None else np.array([np.nan, np.nan, np.nan]))
        return np.stack(out, axis=0)

    def pixel_angle_to_yaw(
        self,
        center_uv: Sequence[float],
        angle_rad: float,
        pixel_step: float = 20.0,
    ) -> Optional[float]:
        """Convert an in-image angle (radians, rotating CCW in pixel coords,
        with 0 = +u direction) into a workspace yaw by projecting two points
        and taking atan2 on the resulting delta in the workspace plane.
        """
        u, v = float(center_uv[0]), float(center_uv[1])
        du = math.cos(float(angle_rad)) * float(pixel_step)
        dv = math.sin(float(angle_rad)) * float(pixel_step)

        p0 = self.project((u, v))
        p1 = self.project((u + du, v + dv))
        if p0 is None or p1 is None:
            return None

        dx = float(p1[0] - p0[0])
        dy = float(p1[1] - p0[1])
        if math.hypot(dx, dy) < 1e-9:
            return None
        return math.atan2(dy, dx)


def in_bounds_xy(
    xy: Sequence[float],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> bool:
    x, y = float(xy[0]), float(xy[1])
    return (x_min <= x <= x_max) and (y_min <= y <= y_max)


def _self_test() -> None:
    """Synthetic pinhole sanity check. Camera at (0, 0, 0.5) looking down -z.

    For a top-down pinhole with the camera image-plane y-axis aligned with the
    workspace -y axis (standard OpenCV convention where +y_cam points down in
    the image while +y_ws points away from the robot), the extrinsics are a
    pure translation along +z_ws with a rotation that flips y_cam = -y_ws.
    """
    fx = fy = 600.0
    cx = cy = 320.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((5, 1), dtype=np.float64)

    R_cam_from_ws = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    height = 0.5
    t_cam_from_ws = np.array([0.0, 0.0, height], dtype=np.float64)
    table_z = 0.0

    ptw = PixelToWorkspace(K, dist, R_cam_from_ws, t_cam_from_ws, table_z)

    p_center = ptw.project((cx, cy))
    assert p_center is not None, "center pixel should project"
    assert abs(p_center[0]) < 1e-6 and abs(p_center[1]) < 1e-6, p_center

    p_right = ptw.project((cx + fx * 0.1, cy))
    assert p_right is not None
    assert abs(p_right[0] - (0.1 * height)) < 1e-6, p_right

    p_down = ptw.project((cx, cy + fy * 0.1))
    assert p_down is not None
    assert abs(p_down[1] - (0.1 * height)) < 1e-6, p_down

    yaw_x = ptw.pixel_angle_to_yaw((cx, cy), 0.0)
    assert yaw_x is not None and abs(yaw_x) < 1e-6, yaw_x

    yaw_y = ptw.pixel_angle_to_yaw((cx, cy), math.pi / 2.0)
    assert yaw_y is not None and abs(yaw_y - math.pi / 2.0) < 1e-6, yaw_y

    print("pixel_to_workspace self-test OK")


if __name__ == "__main__":
    _self_test()
