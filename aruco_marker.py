"""Shared ArUco detection helpers, extracted to avoid importing handtracking.py
(which has a module-level MediaPipe initialisation). Pure OpenCV.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class MarkerPose:
    marker_id: int
    center_px: Tuple[float, float]
    xyz_camera: np.ndarray
    xyz_workspace: np.ndarray
    rpy_workspace: np.ndarray
    image_corners: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray


def rvec_tvec_to_T(rvec, tvec) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    t = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3:] = t
    return T


def T_inv(T) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def T_apply(T, p) -> np.ndarray:
    ph = np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
    q = T @ ph
    return q[:3]


def rot_to_rpy(R) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float64)


def build_aruco_detector(aruco_dict_id: int):
    dictionary = cv2.aruco.getPredefinedDictionary(int(aruco_dict_id))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 50
    params.cornerRefinementMinAccuracy = 0.01
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 4
    params.adaptiveThreshConstant = 7.0
    params.minMarkerPerimeterRate = 0.01
    params.maxMarkerPerimeterRate = 6.0
    params.polygonalApproxAccuracyRate = 0.08
    params.minCornerDistanceRate = 0.02
    params.minDistanceToBorder = 1
    params.minOtsuStdDev = 3.0
    params.perspectiveRemoveIgnoredMarginPerCell = 0.20
    params.maxErroneousBitsInBorderRate = 0.6
    params.errorCorrectionRate = 0.8
    return cv2.aruco.ArucoDetector(dictionary, params)


def marker_object_points(marker_size_m: float) -> np.ndarray:
    s = float(marker_size_m) / 2.0
    return np.array(
        [
            [-s, s, 0.0],
            [s, s, 0.0],
            [s, -s, 0.0],
            [-s, -s, 0.0],
        ],
        dtype=np.float64,
    )


def aruco_dict_id_from_name(name: str) -> int:
    return int(getattr(cv2.aruco, str(name)))


def detect_marker_corners(detector, frame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _rejected = detector.detectMarkers(gray)
    if ids is None or len(corners) == 0:
        return None, None
    return corners, ids


def solve_single_marker_pose(
    image_corners: np.ndarray,
    marker_size_m: float,
    K: np.ndarray,
    dist: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Solve PnP for a single square marker. Returns (rvec, tvec) or None."""
    img_pts = np.asarray(image_corners, dtype=np.float64).reshape(4, 2)
    obj_pts = marker_object_points(float(marker_size_m))
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts,
        img_pts,
        np.asarray(K, dtype=np.float64).reshape(3, 3),
        np.asarray(dist, dtype=np.float64).reshape(-1, 1),
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        return None
    return (
        np.asarray(rvec, dtype=np.float64).reshape(3),
        np.asarray(tvec, dtype=np.float64).reshape(3),
    )


class TopdownArucoDetector:
    """Detect ArUco markers and report their pose in the workspace frame,
    using a calibrated camera extrinsic `T_cam_from_ws` (rvec/tvec form, where
    `p_cam = R(rvec) @ p_ws + tvec`).
    """

    def __init__(
        self,
        aruco_dict_id: int,
        marker_size_m: float,
        K: np.ndarray,
        dist: np.ndarray,
        R_cam_from_ws: np.ndarray,
        t_cam_from_ws: np.ndarray,
        valid_ids: Optional[Iterable[int]] = None,
    ):
        self.detector = build_aruco_detector(int(aruco_dict_id))
        self.marker_size_m = float(marker_size_m)
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.dist = np.asarray(dist, dtype=np.float64).reshape(-1, 1)

        R_cw = np.asarray(R_cam_from_ws, dtype=np.float64).reshape(3, 3)
        t_cw = np.asarray(t_cam_from_ws, dtype=np.float64).reshape(3)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_cw
        T[:3, 3] = t_cw
        self.T_cam_from_ws = T
        self.T_ws_from_cam = T_inv(T)

        self.valid_ids = None if valid_ids is None else set(int(x) for x in valid_ids)

    def detect(self, frame) -> Dict[int, MarkerPose]:
        corners, ids = detect_marker_corners(self.detector, frame)
        out: Dict[int, MarkerPose] = {}
        if ids is None:
            return out

        ids_list = [int(x) for x in ids.flatten().tolist()]
        for idx, marker_id in enumerate(ids_list):
            if self.valid_ids is not None and marker_id not in self.valid_ids:
                continue

            img_corners = np.asarray(corners[idx], dtype=np.float64).reshape(4, 2)
            solved = solve_single_marker_pose(img_corners, self.marker_size_m, self.K, self.dist)
            if solved is None:
                continue
            rvec, tvec = solved

            T_cam_from_marker = rvec_tvec_to_T(rvec, tvec)
            T_ws_from_marker = self.T_ws_from_cam @ T_cam_from_marker

            xyz_ws = T_ws_from_marker[:3, 3].copy()
            rpy_ws = rot_to_rpy(T_ws_from_marker[:3, :3])
            center_px = (float(img_corners[:, 0].mean()), float(img_corners[:, 1].mean()))

            pose = MarkerPose(
                marker_id=marker_id,
                center_px=center_px,
                xyz_camera=np.asarray(tvec, dtype=np.float64).copy(),
                xyz_workspace=xyz_ws,
                rpy_workspace=rpy_ws,
                image_corners=img_corners,
                rvec=np.asarray(rvec, dtype=np.float64).copy(),
                tvec=np.asarray(tvec, dtype=np.float64).copy(),
            )

            if marker_id in out:
                existing = out[marker_id]
                if np.linalg.norm(pose.xyz_camera) < np.linalg.norm(existing.xyz_camera):
                    out[marker_id] = pose
            else:
                out[marker_id] = pose

        return out

    def draw_overlay(self, frame, pose: MarkerPose, color=(0, 255, 0), label: Optional[str] = None) -> None:
        pts = pose.image_corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)
        if label is None:
            label = f"id={pose.marker_id}"
        cx, cy = pose.center_px
        cv2.putText(frame, label, (int(cx) + 6, int(cy) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
