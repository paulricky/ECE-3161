from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


def _order_quad(pts: Sequence[Sequence[float]]) -> np.ndarray:
    pts_arr = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts_arr.sum(axis=1)
    d = np.diff(pts_arr, axis=1).reshape(-1)
    tl = pts_arr[np.argmin(s)]
    br = pts_arr[np.argmax(s)]
    tr = pts_arr[np.argmin(d)]
    bl = pts_arr[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def detect_largest_rectangle(gray: np.ndarray, min_area_px: float = 5000.0) -> Optional[np.ndarray]:
    gray = np.asarray(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, None, iterations=2)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0.0

    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < float(min_area_px):
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        if area > best_area:
            best_area = area
            best = approx.reshape(4, 2)

    if best is None:
        return None
    return _order_quad(best)


def rect_homography_to_workspace(
    rect_img_4x2: np.ndarray,
    workspace_x_min: float,
    workspace_x_max: float,
    workspace_y_min: float,
    workspace_y_max: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    rect = np.asarray(rect_img_4x2, dtype=np.float32).reshape(4, 2)
    dst = np.array(
        [
            [workspace_x_min, workspace_y_min],
            [workspace_x_max, workspace_y_min],
            [workspace_x_max, workspace_y_max],
            [workspace_x_min, workspace_y_max],
        ],
        dtype=np.float32,
    )
    H, _ = cv2.findHomography(rect, dst, method=0)
    if H is None:
        return None
    try:
        Hinv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    return H.astype(np.float64), Hinv.astype(np.float64)


def maybe_rotate_rect_using_marker(
    rect_img_4x2: np.ndarray,
    marker_center,
    marker_angle: float,
) -> np.ndarray:
    del marker_center
    rect = np.asarray(rect_img_4x2, dtype=np.float32).reshape(4, 2).copy()
    top_edge = rect[1] - rect[0]
    rect_ang = math.atan2(float(top_edge[1]), float(top_edge[0]))

    def angdiff(a, b):
        d = (float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi
        return abs(d)

    best = rect
    best_score = angdiff(marker_angle, rect_ang)
    for k in range(1, 4):
        r = np.roll(rect, -k, axis=0)
        ang = math.atan2(float((r[1] - r[0])[1]), float((r[1] - r[0])[0]))
        score = angdiff(marker_angle, ang)
        if score < best_score:
            best_score = score
            best = r
    return best
