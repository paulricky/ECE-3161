from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

import values as val
from camera_utils import (
    CameraOpenError,
    handtracking_candidate_indices,
    open_handtracking_camera,
    probe_camera_indices,
    read_latest_from_capture,
)
from depthcalibrator import HandDepthEstimator


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_DIR = os.path.join(_THIS_DIR, "calibration_data")
os.makedirs(CALIB_DIR, exist_ok=True)

INTRINSICS_NPZ = os.path.join(CALIB_DIR, "calibration_intrinsics.npz")
WORKSPACE_NPZ = os.path.join(CALIB_DIR, "calibration_workspace.npz")
EXTRINSICS_NPZ = os.path.join(CALIB_DIR, "calibration_extrinsics.npz")
CAMERA_CALIBRATION_NPZ = os.path.join(_THIS_DIR, getattr(val, "CAMERA_CALIBRATION_FILE", "calibration_data/camera_calibration.npz"))
CAMERA_CALIBRATION_JSON = os.path.join(_THIS_DIR, getattr(val, "CAMERA_CALIBRATION_JSON", "calibration_data/camera_calibration.json"))


@dataclass
class PoseSample:
    marker_id: int
    xyz_workspace: np.ndarray
    xyz_camera: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    image_points: np.ndarray


def intrinsics_exists(path: str = INTRINSICS_NPZ) -> bool:
    return os.path.exists(path)


def workspace_exists(path: str = WORKSPACE_NPZ) -> bool:
    return os.path.exists(path)


def extrinsics_exists(path: str = EXTRINSICS_NPZ) -> bool:
    return os.path.exists(path)


def delete_calibration_files() -> None:
    for pth in (WORKSPACE_NPZ, EXTRINSICS_NPZ):
        try:
            if os.path.exists(pth):
                os.remove(pth)
        except Exception:
            pass


def _safe_npz_value(d, keys: Sequence[str], default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def _finite_float(x, default=None):
    try:
        f = float(x)
    except Exception:
        return default
    return f if np.isfinite(f) else default


def load_intrinsics(path: str = INTRINSICS_NPZ):
    candidate_paths = [path]
    if path == INTRINSICS_NPZ:
        candidate_paths.append(CAMERA_CALIBRATION_NPZ)
    data = None
    for candidate in candidate_paths:
        if not candidate or not os.path.exists(candidate):
            continue
        try:
            data = np.load(candidate, allow_pickle=True)
            break
        except Exception:
            data = None
    if data is None:
        return None
    K = _safe_npz_value(data, ["camera_matrix", "K", "mtx"])
    dist = _safe_npz_value(data, ["dist_coeffs", "dist", "distortion_coefficients"])
    if K is None or dist is None:
        return None
    out = {
        "mtx": np.asarray(K, dtype=np.float64).reshape(3, 3),
        "dist": np.asarray(dist, dtype=np.float64).reshape(-1, 1),
    }
    image_size = _safe_npz_value(data, ["image_size"])
    if image_size is not None:
        out["image_size"] = tuple(int(x) for x in np.asarray(image_size).reshape(-1)[:2])
    rms = _safe_npz_value(data, ["rms"])
    if rms is not None:
        out["rms"] = float(np.asarray(rms).reshape(-1)[0])
    reproj_err = _safe_npz_value(data, ["reproj_err"])
    if reproj_err is not None:
        out["reproj_err"] = float(np.asarray(reproj_err).reshape(-1)[0])
    return out


def load_workspace(path: str = WORKSPACE_NPZ):
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path, allow_pickle=True)
    except Exception:
        return None
    out = {}
    for key in (
        "H",
        "workspace_min",
        "workspace_max",
        "neutral_xyz",
        "left_xyz",
        "right_xyz",
        "near_xyz",
        "far_xyz",
        "low_xyz",
        "high_xyz",
        "marker_ids",
        "workspace_mode",
        "aruco_dict",
        "marker_size_m",
    ):
        if key in data:
            out[key] = data[key]
    return out if out else None


def load_extrinsics(path: str = EXTRINSICS_NPZ):
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path, allow_pickle=True)
    except Exception:
        return None
    R = _safe_npz_value(data, ["R"])
    t = _safe_npz_value(data, ["t", "T"])
    if R is None or t is None:
        return None
    out = {
        "R": np.asarray(R, dtype=np.float64).reshape(3, 3),
        "t": np.asarray(t, dtype=np.float64).reshape(3),
    }
    rvec = _safe_npz_value(data, ["rvec"])
    tvec = _safe_npz_value(data, ["tvec"])
    if rvec is not None:
        out["rvec"] = np.asarray(rvec, dtype=np.float64).reshape(3)
    if tvec is not None:
        out["tvec"] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return out


def save_workspace(
    workspace_min_xyz: np.ndarray,
    workspace_max_xyz: np.ndarray,
    neutral_xyz: np.ndarray,
    captures_xyz: Dict[str, np.ndarray],
    marker_ids: Sequence[int],
    aruco_dict_id: int,
    marker_size_m: float,
) -> None:
    kwargs = {
        "H": np.eye(3, dtype=np.float64),
        "workspace_min": np.asarray(workspace_min_xyz, dtype=np.float64).reshape(3),
        "workspace_max": np.asarray(workspace_max_xyz, dtype=np.float64).reshape(3),
        "neutral_xyz": np.asarray(neutral_xyz, dtype=np.float64).reshape(3),
        "marker_ids": np.asarray([int(x) for x in marker_ids], dtype=np.int32),
        "aruco_dict": np.asarray([int(aruco_dict_id)], dtype=np.int32),
        "marker_size_m": np.asarray([float(marker_size_m)], dtype=np.float64),
        "workspace_mode": np.asarray(["aruco_hand_depth"], dtype=object),
    }
    for name, xyz in captures_xyz.items():
        kwargs[f"{name}_xyz"] = np.asarray(xyz, dtype=np.float64).reshape(3)
    np.savez(WORKSPACE_NPZ, **kwargs)


def save_extrinsics(R: np.ndarray, t: np.ndarray, *, rvec=None, tvec=None) -> None:
    np.savez(
        EXTRINSICS_NPZ,
        R=np.asarray(R, dtype=np.float64).reshape(3, 3),
        t=np.asarray(t, dtype=np.float64).reshape(3),
        rvec=np.zeros(3, dtype=np.float64) if rvec is None else np.asarray(rvec, dtype=np.float64).reshape(3),
        tvec=np.zeros(3, dtype=np.float64) if tvec is None else np.asarray(tvec, dtype=np.float64).reshape(3),
    )


def _validate_intrinsics_dict(intr) -> Tuple[bool, str]:
    if intr is None:
        return False, "intr is None"
    mtx = np.asarray(intr.get("mtx", None))
    dist = np.asarray(intr.get("dist", None))
    if mtx.shape != (3, 3):
        return False, f"mtx shape {mtx.shape} != (3,3)"
    if dist.size < 4:
        return False, f"dist size {dist.size} < 4"
    if not np.isfinite(mtx).all() or not np.isfinite(dist).all():
        return False, "non-finite intrinsics"
    return True, "ok"


def _validate_workspace_dict(ws) -> Tuple[bool, str]:
    if ws is None:
        return False, "ws is None"
    mn = ws.get("workspace_min", None)
    mx = ws.get("workspace_max", None)
    if mn is None or mx is None:
        return False, "workspace_min/max missing"
    mn = np.asarray(mn, dtype=np.float64).reshape(-1)
    mx = np.asarray(mx, dtype=np.float64).reshape(-1)
    if mn.size != 3 or mx.size != 3:
        return False, "workspace_min/max must have size 3"
    if not np.isfinite(mn).all() or not np.isfinite(mx).all():
        return False, "non-finite workspace bounds"
    if np.any(mx <= mn):
        return False, f"workspace_max must be greater than workspace_min (min={mn}, max={mx})"
    H = np.asarray(ws.get("H", np.eye(3)), dtype=np.float64)
    if H.shape != (3, 3):
        return False, "H must be 3x3"
    return True, "ok"


def _validate_extrinsics_dict(ext) -> Tuple[bool, str]:
    if ext is None:
        return False, "ext is None"
    R = np.asarray(ext.get("R", None), dtype=np.float64)
    t = np.asarray(ext.get("t", None), dtype=np.float64).reshape(-1)
    if R.shape != (3, 3):
        return False, f"R shape {R.shape} != (3,3)"
    if t.size != 3:
        return False, f"t size {t.size} != 3"
    if not np.isfinite(R).all() or not np.isfinite(t).all():
        return False, "non-finite extrinsics"
    return True, "ok"


def _print_calib_status(prefix: str = "") -> None:
    print(f"{prefix}Calibration folder: {CALIB_DIR}")
    print(f"{prefix}Intrinsics file:  {INTRINSICS_NPZ}  exists={intrinsics_exists()}")
    print(f"{prefix}Workspace file:   {WORKSPACE_NPZ}  exists={workspace_exists()}")
    print(f"{prefix}Extrinsics file:  {EXTRINSICS_NPZ}  exists={extrinsics_exists()}")


def _camera_from_workspace_rotation() -> np.ndarray:
    # Workspace axes used by handtracking command mapping:
    #   x = camera right
    #   y = camera depth away from the lens
    #   z = up
    # Camera axes (OpenCV): x right, y down, z forward.
    # Therefore: p_camera = [x_workspace, -z_workspace, y_workspace]
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )


def _workspace_from_camera_point(p_camera_xyz: np.ndarray) -> np.ndarray:
    p = np.asarray(p_camera_xyz, dtype=np.float64).reshape(3)
    return np.array([p[0], p[2], -p[1]], dtype=np.float64)


def _build_aruco_detector(aruco_dict_id: int):
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


def _marker_object_points(marker_size_m: float) -> np.ndarray:
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


def _draw_text_lines(frame, lines: Sequence[str], x: int = 10, y0: int = 28, dy: int = 26):
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y0 + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y0 + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 1, cv2.LINE_AA)


def _require_aruco():
    aruco = getattr(cv2, "aruco", None)
    if aruco is None:
        raise RuntimeError("OpenCV ArUco/ChArUco support is unavailable. Install opencv-contrib-python.")
    required = ["interpolateCornersCharuco", "calibrateCameraCharuco"]
    if not any(hasattr(aruco, name) for name in ("CharucoBoard", "CharucoBoard_create")):
        required.append("CharucoBoard")
    missing = [name for name in required if not hasattr(aruco, name)]
    if missing:
        raise RuntimeError(
            "OpenCV ChArUco API is unavailable. Install opencv-contrib-python. "
            f"Missing: {missing}"
        )
    return aruco


def _aruco_dictionary_from_name(name: str):
    aruco = _require_aruco()
    dict_name = str(name).strip()
    aliases = {
        "4X4_50": "DICT_4X4_50",
        "5X5_100": "DICT_5X5_100",
        "6X6_250": "DICT_6X6_250",
        "ARUCO_ORIGINAL": "DICT_ARUCO_ORIGINAL",
    }
    dict_name = aliases.get(dict_name.upper(), dict_name)
    if not hasattr(aruco, dict_name):
        valid = sorted(k for k in dir(aruco) if k.startswith("DICT_"))
        raise ValueError(f"Unknown ArUco dictionary '{dict_name}'. Examples: {valid[:8]}")
    return aruco.getPredefinedDictionary(int(getattr(aruco, dict_name)))


def _create_charuco_board(squares_x: int, squares_y: int, square_length_m: float, marker_length_m: float, dictionary):
    aruco = _require_aruco()
    sx = int(squares_x)
    sy = int(squares_y)
    square = float(square_length_m)
    marker = float(marker_length_m)
    if sx < 2 or sy < 2:
        raise ValueError("ChArUco board must have at least 2x2 squares.")
    if square <= 0.0 or marker <= 0.0 or marker >= square:
        raise ValueError("marker_length_m must be positive and smaller than square_length_m.")
    if hasattr(aruco, "CharucoBoard_create"):
        return aruco.CharucoBoard_create(sx, sy, square, marker, dictionary)
    return aruco.CharucoBoard((sx, sy), square, marker, dictionary)


def _draw_charuco_board(board, output_path: str, squares_x: int, squares_y: int) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    width_px = max(900, int(squares_x) * 220)
    height_px = max(700, int(squares_y) * 220)
    if hasattr(board, "generateImage"):
        img = board.generateImage((width_px, height_px), marginSize=40, borderBits=1)
    else:
        img = board.draw((width_px, height_px), marginSize=40, borderBits=1)
    if not cv2.imwrite(output_path, img):
        raise RuntimeError(f"Could not write ChArUco board image: {output_path}")


def _build_charuco_detector(dictionary):
    aruco = _require_aruco()
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    if hasattr(aruco, "ArucoDetector"):
        return aruco.ArucoDetector(dictionary, params)
    return None, dictionary, params


def _detect_markers(detector, gray):
    aruco = _require_aruco()
    if detector is not None and hasattr(detector, "detectMarkers"):
        return detector.detectMarkers(gray)
    _none, dictionary, params = detector
    return aruco.detectMarkers(gray, dictionary, parameters=params)


def _detect_charuco(gray, board, detector, min_markers: int):
    aruco = _require_aruco()
    corners, ids, rejected = _detect_markers(detector, gray)
    marker_count = 0 if ids is None else len(ids)
    if ids is None or len(corners) < int(min_markers):
        return corners, ids, rejected, None, None, marker_count, 0
    try:
        ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(corners, ids, gray, board)
    except Exception:
        ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(corners, ids, gray, board, None, None)
    corner_count = 0 if charuco_ids is None else len(charuco_ids)
    return corners, ids, rejected, charuco_corners, charuco_ids, marker_count, int(ret if ret is not None else corner_count)


def _draw_charuco_detections(frame, corners, ids, charuco_corners, charuco_ids):
    aruco = _require_aruco()
    if ids is not None and corners is not None and len(corners) > 0:
        aruco.drawDetectedMarkers(frame, corners, ids)
    if charuco_ids is not None and charuco_corners is not None and len(charuco_ids) > 0:
        aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)


def _resolve_output(path: str) -> str:
    p = os.path.expanduser(str(path))
    if not os.path.isabs(p):
        p = os.path.join(_THIS_DIR, p)
    return os.path.abspath(p)


def _confirm_overwrite(path: str, overwrite: bool) -> bool:
    if overwrite or not os.path.exists(path):
        return True
    reply = input(f"[calib] Output exists: {path}\nOverwrite? [y/N]: ").strip().lower()
    return reply in ("y", "yes")


def _candidate_path(path: str) -> str:
    root, ext = os.path.splitext(path)
    return f"{root}_candidate{ext or '.npz'}"


def _save_charuco_calibration(result: dict, output_npz: str, output_json: str, overwrite: bool) -> tuple[str, str]:
    npz_path = _resolve_output(output_npz)
    json_path = _resolve_output(output_json)
    if not _confirm_overwrite(npz_path, overwrite):
        npz_path = _candidate_path(npz_path)
        print(f"[charuco] Writing candidate NPZ instead: {npz_path}")
    if not _confirm_overwrite(json_path, overwrite):
        json_path = _candidate_path(json_path)
        print(f"[charuco] Writing candidate JSON instead: {json_path}")
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    camera_matrix = np.asarray(result["camera_matrix"], dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.asarray(result["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    image_w, image_h = result["image_size"]
    charuco = result["charuco"]
    reproj = float(result["reprojection_error"])
    np.savez(
        npz_path,
        camera_matrix=camera_matrix,
        K=camera_matrix,
        mtx=camera_matrix,
        dist_coeffs=dist_coeffs,
        dist=dist_coeffs,
        distortion_coefficients=dist_coeffs,
        image_width=np.asarray([int(image_w)], dtype=np.int32),
        image_height=np.asarray([int(image_h)], dtype=np.int32),
        image_size=np.asarray([int(image_w), int(image_h)], dtype=np.int32),
        reprojection_error=np.asarray([reproj], dtype=np.float64),
        reproj_err=np.asarray([reproj], dtype=np.float64),
        rms=np.asarray([reproj], dtype=np.float64),
        calibration_type=np.asarray(["charuco"], dtype=object),
        charuco_squares_x=np.asarray([int(charuco["squares_x"])], dtype=np.int32),
        charuco_squares_y=np.asarray([int(charuco["squares_y"])], dtype=np.int32),
        charuco_square_length_m=np.asarray([float(charuco["square_length_m"])], dtype=np.float64),
        charuco_marker_length_m=np.asarray([float(charuco["marker_length_m"])], dtype=np.float64),
        charuco_dictionary=np.asarray([str(charuco["dictionary"])], dtype=object),
    )
    json_payload = {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
        "image_width": int(image_w),
        "image_height": int(image_h),
        "reprojection_error": reproj,
        "calibration_type": "charuco",
        "timestamp": result["timestamp"],
        "charuco": dict(charuco),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)
    return npz_path, json_path


def _hand_depth_calibration_path() -> str:
    return _resolve_output(getattr(val, "HAND_MONOCULAR_DEPTH_CALIBRATION_FILE", "calibration_data/hand_depth_calibration.json"))


def _calibration_status() -> int:
    hand_path = _hand_depth_calibration_path()
    cam_npz = _resolve_output(getattr(val, "CAMERA_CALIBRATION_FILE", "calibration_data/camera_calibration.npz"))
    cam_json = _resolve_output(getattr(val, "CAMERA_CALIBRATION_JSON", "calibration_data/camera_calibration.json"))
    legacy_intr = _resolve_output(getattr(val, "CALIB_INTRINSICS_FILE", "calibration_data/calibration_intrinsics.npz"))
    workspace_hand = _hand_workspace_calibration_path()
    hand_mirror = _hand_mirror_position_calibration_path()
    print("For this RGB-only robot control setup, hand-depth calibration is required/recommended.")
    print("Camera intrinsics are optional.")
    print(f"hand-depth calibration: {hand_path} exists={os.path.exists(hand_path)}")
    print(f"camera calibration NPZ: {cam_npz} exists={os.path.exists(cam_npz)}")
    print(f"camera calibration JSON: {cam_json} exists={os.path.exists(cam_json)}")
    print(f"legacy intrinsics NPZ: {legacy_intr} exists={os.path.exists(legacy_intr)}")
    print(f"hand workspace calibration: {workspace_hand} exists={os.path.exists(workspace_hand)}")
    print(f"hand mirror position calibration: {hand_mirror} exists={os.path.exists(hand_mirror)}")
    print("Runtime can proceed with RGB webcam + MediaPipe even if camera intrinsics are missing.")
    return 0


def _hand_landmark_metrics(hand_lms) -> dict:
    lm = hand_lms.landmark
    pts = np.array([[float(p.x), float(p.y)] for p in lm], dtype=np.float64)
    def d(a, b):
        return float(np.linalg.norm(pts[int(a)] - pts[int(b)]))
    palm_width = d(5, 17)
    wrist_to_middle = d(0, 9)
    palm_height = wrist_to_middle if wrist_to_middle > 1e-6 else d(0, 10)
    bbox_w = float(np.max(pts[:, 0]) - np.min(pts[:, 0]))
    bbox_h = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))
    bbox_size = float(np.sqrt(max(0.0, bbox_w * bbox_h)))
    thumb_index_span = d(4, 8)
    cues = [palm_width, wrist_to_middle, palm_height, bbox_size]
    hand_size = float(np.median([x for x in cues if np.isfinite(x) and x > 0.0])) if cues else 0.0
    return {
        "x_norm": float(np.mean(pts[[0, 5, 9, 17], 0])),
        "y_norm": float(np.mean(pts[[0, 5, 9, 17], 1])),
        "hand_size_norm": hand_size,
        "palm_width_norm": palm_width,
        "wrist_to_middle_mcp_norm": wrist_to_middle,
        "palm_height_norm": palm_height,
        "bbox_size_norm": bbox_size,
        "thumb_index_span_norm": thumb_index_span,
    }


def _pose_summary(depth_m: float, samples: Sequence[dict]) -> dict:
    out = {"depth_m": float(depth_m)}
    keys = [
        "hand_size_norm",
        "palm_width_norm",
        "wrist_to_middle_mcp_norm",
        "palm_height_norm",
        "bbox_size_norm",
        "thumb_index_span_norm",
    ]
    for key in keys:
        vals = [float(s[key]) for s in samples if key in s and np.isfinite(float(s[key]))]
        out[f"{key}_mean"] = float(np.mean(vals)) if vals else 0.0
        out[f"{key}_std"] = float(np.std(vals)) if vals else 0.0
    return out


def _save_hand_depth_calibration(payload: dict, path: str, overwrite: bool) -> str:
    out_path = _resolve_output(path)
    if not _confirm_overwrite(out_path, overwrite):
        out_path = _candidate_path(out_path)
        print(f"[hand-depth] Writing candidate JSON instead: {out_path}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path


def _hand_workspace_calibration_path() -> str:
    return _resolve_output(getattr(val, "HAND_WORKSPACE_CALIBRATION_FILE", "calibration_data/hand_workspace_calibration.json"))


def _hand_mirror_position_calibration_path() -> str:
    return _resolve_output(getattr(val, "HAND_MIRROR_POSITION_CALIBRATION_FILE", "calibration_data/hand_mirror_position_calibration.json"))


def _print_hand_depth_camera_config(args, requested_index: int, use_main_defaults: bool) -> None:
    print("[hand-depth] using main.py camera defaults for compatibility" if use_main_defaults else "[hand-depth] using requested camera index with main.py robust camera path")
    print(f"[hand-depth] main camera index config: {getattr(val, 'HANDTRACKING_CAMERA_INDEX', 0)}")
    print(f"[hand-depth] main candidate indices: {handtracking_candidate_indices()}")
    print(f"[hand-depth] hand-depth requested camera index: {requested_index}")
    print(f"[hand-depth] backend config: {getattr(val, 'HANDTRACKING_CAMERA_BACKEND', 'default')}")
    print(
        "[hand-depth] requested frame: "
        f"{int(getattr(val, 'CAMERA_CAPTURE_WIDTH', getattr(val, 'MAIN_CAMERA_FRAME_WIDTH', 640)))}x"
        f"{int(getattr(val, 'CAMERA_CAPTURE_HEIGHT', getattr(val, 'MAIN_CAMERA_FRAME_HEIGHT', 480)))} "
        f"@ {int(getattr(val, 'CAMERA_CAPTURE_FPS', getattr(val, 'MAIN_CAMERA_FPS', 30)))} fps"
    )
    print(
        "[hand-depth] open/read retries: "
        f"open={int(getattr(val, 'MAIN_CAMERA_OPEN_RETRIES', 3))} "
        f"read={int(getattr(val, 'MAIN_CAMERA_READ_RETRIES', 30))} "
        f"warmup={int(getattr(val, 'MAIN_CAMERA_WARMUP_FRAMES', 5))}"
    )


def _open_hand_depth_camera(args):
    explicit_index = "--camera-index" in sys.argv
    use_main_defaults = bool(getattr(args, "use_main_camera_defaults", False)) or not explicit_index
    requested_index = int(
        getattr(val, "HANDTRACKING_CAMERA_INDEX", 0)
        if use_main_defaults
        else getattr(args, "camera_index", getattr(val, "HANDTRACKING_CAMERA_INDEX", 0))
    )
    _print_hand_depth_camera_config(args, requested_index, use_main_defaults)
    indices = None if use_main_defaults else [requested_index, *handtracking_candidate_indices()]
    try:
        camera = open_handtracking_camera(candidate_indices=indices)
    except CameraOpenError as exc:
        print("[hand-depth] ERROR: Could not open/read webcam.")
        tried = sorted({p.index for p in exc.attempts})
        print(f"[hand-depth] Tried indices: {tried}")
        for p in exc.attempts:
            shape = "" if p.frame_shape is None else f" frame_shape={p.frame_shape}"
            print(
                f"[hand-depth] attempt index={p.index} backend={p.backend_name} "
                f"props={'yes' if p.used_props else 'no'} opened={p.opened} read_ok={p.read_ok}{shape}"
            )
        print("[hand-depth] Try changing camera index with --camera-index N.")
        print("[hand-depth] On macOS, check System Settings > Privacy & Security > Camera for PyCharm/Terminal.")
        return None
    print(
        f"[hand-depth] selected camera index={camera.index} backend={camera.backend_name} "
        f"props={'yes' if camera.used_props else 'no'} frame_shape={getattr(camera.frame, 'shape', None)}"
    )
    return camera


def _list_cameras() -> int:
    print("[calib] Probing cameras with the same backend/read logic as main.py.")
    required = max(2, int(getattr(val, "MAIN_CAMERA_STABILITY_FRAMES", 10)))
    probes = probe_camera_indices(range(5), read_retries=required)
    for p in probes:
        shape = "" if p.frame_shape is None else f" frame_shape={p.frame_shape}"
        print(
            f"  index={p.index} backend={p.backend_name} props={'yes' if p.used_props else 'no'} "
            f"opened={p.opened} stable_read={'yes' if p.read_ok else 'no'}{shape}"
        )
    return 0


def _workspace_bounds_for_hand_calibration() -> dict:
    return {
        "x_min_m": float(getattr(val, "HAND_TARGET_X_MIN_M", getattr(val, "WORKSPACE_X_MIN", -0.12))),
        "x_max_m": float(getattr(val, "HAND_TARGET_X_MAX_M", getattr(val, "WORKSPACE_X_MAX", 0.12))),
        "y_min_m": float(getattr(val, "HAND_TARGET_Y_MIN_M", getattr(val, "WORKSPACE_Y_MIN", 0.10))),
        "y_max_m": float(getattr(val, "HAND_TARGET_Y_MAX_M", getattr(val, "WORKSPACE_Y_MAX", 0.22))),
        "z_min_m": float(getattr(val, "HAND_TARGET_Z_MIN_M", getattr(val, "WORKSPACE_Z_MIN", 0.00))),
        "z_max_m": float(getattr(val, "HAND_TARGET_Z_MAX_M", getattr(val, "WORKSPACE_Z_MAX", 0.22))),
    }


def _robot_xyz_for_workspace_pose(name: str, bounds: dict) -> dict:
    cx = 0.5 * (bounds["x_min_m"] + bounds["x_max_m"])
    cy = 0.5 * (bounds["y_min_m"] + bounds["y_max_m"])
    cz = 0.5 * (bounds["z_min_m"] + bounds["z_max_m"])
    near_y = _finite_float(getattr(val, "HAND_DEPTH_TARGET_NEAR_M", None), bounds["y_min_m"])
    far_y = _finite_float(getattr(val, "HAND_DEPTH_TARGET_FAR_M", None), bounds["y_max_m"])
    xyz = {
        "x_m": cx,
        "y_m": cy,
        "z_m": cz,
    }
    if name in {"max_left", "top_left", "bottom_left", "near_left", "far_left"}:
        xyz["x_m"] = bounds["x_min_m"]
    if name in {"max_right", "top_right", "bottom_right", "near_right", "far_right"}:
        xyz["x_m"] = bounds["x_max_m"]
    if name in {"max_up", "top_left", "top_right"}:
        xyz["z_m"] = bounds["z_max_m"]
    if name in {"max_down", "bottom_left", "bottom_right"}:
        xyz["z_m"] = bounds["z_min_m"]
    if name in {"max_near", "near_left", "near_right"}:
        xyz["y_m"] = float(near_y)
    if name in {"max_far", "far_left", "far_right"}:
        xyz["y_m"] = float(far_y)
    return xyz


def _effective_hand_capture_metrics(hand_lms, depth_estimator: HandDepthEstimator, frame_w: int, frame_h: int) -> dict:
    metrics = _hand_landmark_metrics(hand_lms)
    depth = depth_estimator.estimate_depth(hand_lms, frame_w=frame_w, frame_h=frame_h)
    x = float(metrics["x_norm"])
    y = float(metrics["y_norm"])
    if bool(getattr(val, "HAND_IMAGE_X_FLIP", False)):
        x = 1.0 - x
    if bool(getattr(val, "HAND_IMAGE_Y_FLIP", True)):
        y = 1.0 - y
    d = float(depth.get("depth_norm", 0.5))
    if bool(getattr(val, "HAND_DEPTH_FLIP", False)):
        d = 1.0 - d
    return {
        "x_norm": float(np.clip(x, 0.0, 1.0)),
        "y_norm": float(np.clip(y, 0.0, 1.0)),
        "depth_norm": float(np.clip(d, 0.0, 1.0)),
        "hand_size_norm": float(depth.get("hand_size_norm", metrics["hand_size_norm"])),
        "palm_width_norm": float(depth.get("palm_width_norm", metrics["palm_width_norm"])),
        "wrist_to_middle_mcp_norm": float(depth.get("wrist_to_middle_mcp_norm", metrics["wrist_to_middle_mcp_norm"])),
        "palm_height_norm": float(depth.get("palm_height_norm", metrics["palm_height_norm"])),
        "bbox_size_norm": float(depth.get("bbox_size_norm", metrics["bbox_size_norm"])),
        "depth_m": float(depth.get("depth_m", 0.0)),
        "depth_source": str(depth.get("source", "unknown")),
        "depth_confidence": float(depth.get("confidence", 0.0)),
    }


def _mean_metrics(samples: Sequence[dict]) -> dict:
    if not samples:
        return {}
    keys = sorted({k for s in samples for k in s.keys() if isinstance(s.get(k), (int, float))})
    out = {}
    for key in keys:
        vals = [float(s[key]) for s in samples if key in s and np.isfinite(float(s[key]))]
        if vals:
            out[key] = float(np.mean(vals))
            out[f"{key}_std"] = float(np.std(vals))
    return out


def _hand_depth_pose_summary(depth_m: float, samples: Sequence[dict]) -> dict:
    summary = _pose_summary(depth_m, samples)
    for key in ("x_norm", "y_norm"):
        vals = [float(s[key]) for s in samples if key in s and np.isfinite(float(s[key]))]
        if vals:
            summary[f"{key}_mean"] = float(np.mean(vals))
            summary[f"{key}_std"] = float(np.std(vals))
    return summary


def _print_pose_capture_summary(prefix: str, poses: dict) -> None:
    print(f"[{prefix}] Captured poses:")
    for name, item in poses.items():
        if not isinstance(item, dict):
            continue
        if "hand" in item:
            hand = item.get("hand", {})
            robot = item.get("robot")
            if isinstance(robot, dict):
                print(
                    f"[{prefix}] {name}: hand=({float(hand.get('x_norm', 0.0)):.3f},"
                    f"{float(hand.get('y_norm', 0.0)):.3f},{float(hand.get('depth_norm', 0.0)):.3f}) "
                    f"size={float(hand.get('hand_size_norm', 0.0)):.4f} "
                    f"robot=({float(robot.get('x_m', 0.0)):+.3f},{float(robot.get('y_m', 0.0)):+.3f},{float(robot.get('z_m', 0.0)):+.3f})"
                )
            else:
                print(
                    f"[{prefix}] {name}: hand=({float(hand.get('x_norm', 0.0)):.3f},"
                    f"{float(hand.get('y_norm', 0.0)):.3f},{float(hand.get('depth_norm', 0.0)):.3f}) "
                    f"size={float(hand.get('hand_size_norm', 0.0)):.4f}"
                )
        else:
            print(
                f"[{prefix}] {name}: depth={float(item.get('depth_m', 0.0)):.3f}m "
                f"size={float(item.get('hand_size_norm_mean', 0.0)):.4f} "
                f"std={float(item.get('hand_size_norm_std', 0.0)):.4f}"
            )


def _confirm_final_save(prefix: str) -> bool:
    prompt = "Save hand workspace calibration?" if prefix == "hand-workspace" else "Save calibration?"
    reply = input(f"[{prefix}] {prompt} [y/N]: ").strip().lower()
    return reply in ("y", "yes")


def _workspace_pose_optional(name: str) -> bool:
    required = {"center", "max_left", "max_right", "max_up", "max_down", "max_near", "max_far"}
    return str(name) not in required


def _workspace_required_poses() -> list[str]:
    return ["center", "max_left", "max_right", "max_up", "max_down", "max_near", "max_far"]


def _workspace_optional_poses() -> list[str]:
    return ["top_left", "top_right", "bottom_left", "bottom_right", "near_left", "near_right", "far_left", "far_right"]


def _workspace_capture_pose_order() -> list[str]:
    configured = [str(x) for x in getattr(val, "HAND_WORKSPACE_CAPTURE_POSES", []) if str(x).strip()]
    out: list[str] = []
    for name in _workspace_required_poses():
        if name not in out:
            out.append(name)
    for name in configured:
        if name not in out:
            out.append(name)
    for name in _workspace_optional_poses():
        if name not in out:
            out.append(name)
    return out


def _workspace_pose_instruction(name: str) -> str:
    text = {
        "center": "Hand centered in the camera frame at comfortable middle depth; robot target is workspace center.",
        "max_left": "Move your hand to the far LEFT edge of the usable camera frame; maps to the robot horizontal left extreme.",
        "max_right": "Move your hand to the far RIGHT edge of the usable camera frame; maps to the opposite robot horizontal extreme.",
        "max_up": "Move your hand to the TOP usable camera position; maps to robot vertical max z.",
        "max_down": "Move your hand to the BOTTOM usable camera position; maps to robot vertical min z.",
        "max_near": "Move your hand as close/front as intended for runtime control; maps to near/front robot depth/reach.",
        "max_far": "Move your hand as far/back as intended for runtime control; maps to far/back robot depth/reach.",
        "top_left": "Optional corner: hand at top-left usable camera position.",
        "top_right": "Optional corner: hand at top-right usable camera position.",
        "bottom_left": "Optional corner: hand at bottom-left usable camera position.",
        "bottom_right": "Optional corner: hand at bottom-right usable camera position.",
        "near_left": "Optional depth edge: hand near/front and left.",
        "near_right": "Optional depth edge: hand near/front and right.",
        "far_left": "Optional depth edge: hand far/back and left.",
        "far_right": "Optional depth edge: hand far/back and right.",
    }
    return text.get(str(name), f"Place hand at {name} calibration position.")


def _print_workspace_pose_instruction(name: str, target: dict, index: int, total: int) -> None:
    print("\n[hand-workspace] Pose instruction")
    print(f"[hand-workspace] Pose: {name} ({index}/{total})")
    print(f"[hand-workspace] {_workspace_pose_instruction(name)}")
    print(
        "[hand-workspace] Robot/end-effector target meaning: "
        f"x={target['x_m']:+.3f} m, y={target['y_m']:+.3f} m, z={target['z_m']:+.3f} m"
    )
    print("[hand-workspace] If recording robot joint seeds, move the robot/end-effector to the corresponding workspace target.")
    print("[hand-workspace] Press ENTER to open camera preview for this pose. Sampling still waits for SPACE.")


def _maybe_connect_robot_for_workspace_calibration():
    reply = input("[hand-workspace] Optionally connect robot to record current joints as IK seeds? [y/N]: ").strip().lower()
    if reply not in ("y", "yes"):
        return None
    try:
        from robot_controller import SOArmHardwareController

        robot = SOArmHardwareController()
        robot.connect()
        print("[hand-workspace] Robot connected for read-only joint seed capture.")
        return robot
    except Exception as exc:
        print(f"[hand-workspace] Robot seed capture unavailable: {exc}")
        return None


def _read_robot_seed_joints(robot) -> Optional[list[float]]:
    if robot is None:
        return None
    try:
        present = robot.read_present_joints_rad()
    except Exception:
        return None
    if not isinstance(present, dict):
        return None
    names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "wrist_pitch"]
    vals = []
    for name in names:
        try:
            v = float(present[name])
        except Exception:
            return None
        if not np.isfinite(v):
            return None
        vals.append(v)
    return vals


def _run_hand_workspace_calibration(args) -> int:
    try:
        import mediapipe as mp
    except Exception as exc:
        print(f"[hand-workspace] ERROR: MediaPipe is required: {exc}")
        return 1

    hands = None
    camera = None
    cap = None
    pending_first_frame = None
    depth_estimator = HandDepthEstimator()
    poses = _workspace_capture_pose_order()
    bounds = _workspace_bounds_for_hand_calibration()
    required = max(3, int(getattr(val, "HAND_DEPTH_CALIBRATION_REQUIRED_STABLE_FRAMES", 20)))
    samples_per_pose = max(required, int(getattr(val, "HAND_DEPTH_CALIBRATION_SAMPLES_PER_POSE", 40)))
    stability_max = float(getattr(val, "HAND_DEPTH_CALIBRATION_STABILITY_STD_MAX", 0.015))
    robot = None
    pose_results = {}
    pose_index = 0
    samples: list[dict] = []
    print("[hand-workspace] Legacy camera-based hand workspace calibration.")
    print("[hand-workspace] Robot mirror workspace extrema are recorded separately by robot_calibrate.py mirror_workspace.")
    print("[hand-workspace] Captures RGB MediaPipe hand positions for nonlinear workspace correction.")
    print("[hand-workspace] This does not learn direct motor commands; joint samples are IK seeds only.")
    print("[hand-workspace] SPACE=accept, R=reset current pose, Q/ESC=quit")
    if bool(getattr(val, "HAND_WORKSPACE_USE_JOINT_SEED_EXAMPLES", True)):
        robot = _maybe_connect_robot_for_workspace_calibration()
    try:
        while pose_index < len(poses):
            state = "waiting_for_user_ready"
            review_summary = None
            pose_name = str(poses[pose_index])
            target = _robot_xyz_for_workspace_pose(pose_name, bounds)
            samples.clear()
            optional = _workspace_pose_optional(pose_name)
            _print_workspace_pose_instruction(pose_name, target, pose_index + 1, len(poses))
            try:
                reply = input("[hand-workspace] Press ENTER when ready for camera preview, or Q to quit: ").strip().lower()
            except EOFError:
                reply = ""
            if reply in {"q", "quit", "esc", "exit"}:
                print("[hand-workspace] Quit before sampling; not saved.")
                return 1
            if hands is None:
                hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    model_complexity=int(getattr(val, "HANDTRACKING_MODEL_COMPLEXITY", 0)),
                    min_detection_confidence=float(getattr(val, "HANDTRACKING_MIN_DETECTION_CONFIDENCE", 0.55)),
                    min_tracking_confidence=float(getattr(val, "HANDTRACKING_MIN_TRACKING_CONFIDENCE", 0.55)),
                )
            if camera is None:
                camera = _open_hand_depth_camera(args)
                if camera is None:
                    return 1
                cap = camera.cap
                pending_first_frame = camera.frame

            while True:
                if cap is None or hands is None:
                    print("[hand-workspace] ERROR: Camera/MediaPipe initialization failed.")
                    return 1
                if pending_first_frame is not None:
                    ok, frame = True, pending_first_frame
                    pending_first_frame = None
                else:
                    ok, frame = read_latest_from_capture(cap)
                if not ok or frame is None:
                    print("[hand-workspace] ERROR: Could not open/read webcam.")
                    return 1
                frame_h, frame_w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                metrics = None
                if results.multi_hand_landmarks:
                    hand_lms = results.multi_hand_landmarks[0]
                    metrics = _effective_hand_capture_metrics(hand_lms, depth_estimator, frame_w, frame_h)
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand_lms, mp.solutions.hands.HAND_CONNECTIONS)

                std = float("nan")
                stable = False
                if state == "sampling" and metrics is not None:
                    samples.append(metrics)
                    samples = samples[-samples_per_pose:]
                    recent = samples[-required:]
                    if len(recent) >= required:
                        std = float(np.std([s["hand_size_norm"] for s in recent]))
                        stable = std <= stability_max
                        if stable:
                            review_summary = _mean_metrics(recent)
                            state = "review"
                elif state == "sampling" and metrics is None:
                    # Hand disappeared: sampling pauses by not appending.
                    pass

                lines = [
                    f"Hand workspace calibration: {pose_name} ({pose_index + 1}/{len(poses)})",
                    f"Place hand at {pose_name} position",
                    f"Target robot xyz=({target['x_m']:+.3f},{target['y_m']:+.3f},{target['z_m']:+.3f}) m",
                    "Required: one visible MediaPipe hand, held steady",
                ]
                if state == "waiting_for_user_ready":
                    lines += [
                        "Press SPACE when ready to start sampling",
                        "R = reset current pose",
                        "S = skip optional pose",
                        "Q/ESC = quit without saving",
                    ]
                elif state == "sampling":
                    std_text = "n/a" if not np.isfinite(std) else f"{std:.5f}"
                    lines += [
                        f"Sampling: {len(samples)}/{required} valid samples",
                        f"stability std: {std_text} limit={stability_max:.5f}",
                        "No hand detected: sampling paused" if metrics is None else "Hold steady",
                        "R = reset current pose, Q/ESC = quit without saving",
                    ]
                else:
                    s = review_summary or {}
                    lines += [
                        "Review summary",
                        f"mean=({float(s.get('x_norm', 0.0)):.3f},{float(s.get('y_norm', 0.0)):.3f},{float(s.get('depth_norm', 0.0)):.3f}) size={float(s.get('hand_size_norm', 0.0)):.4f}",
                        f"std=({float(s.get('x_norm_std', 0.0)):.4f},{float(s.get('y_norm_std', 0.0)):.4f},{float(s.get('depth_norm_std', 0.0)):.4f}) size_std={float(s.get('hand_size_norm_std', 0.0)):.5f}",
                        "A = accept pose, R = resample, Q/ESC = quit",
                    ]
                if metrics is not None:
                    lines.append(f"live hand=({metrics['x_norm']:.2f},{metrics['y_norm']:.2f},{metrics['depth_norm']:.2f}) size={metrics['hand_size_norm']:.3f}")
                else:
                    lines.append("live hand: not detected")
                _draw_text_lines(frame, lines)
                cv2.imshow("Hand workspace calibration", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    return 1
                if key == ord("r"):
                    samples.clear()
                    review_summary = None
                    state = "waiting_for_user_ready"
                    continue
                if state == "waiting_for_user_ready" and key == ord("s") and optional:
                    print(f"[hand-workspace] skipped optional pose {pose_name}")
                    pose_index += 1
                    break
                if state == "waiting_for_user_ready" and key == ord(" "):
                    samples.clear()
                    review_summary = None
                    state = "sampling"
                    continue
                if state == "review" and key == ord("a") and review_summary:
                    joints_rad = _read_robot_seed_joints(robot)
                    pose_results[pose_name] = {
                        "hand": {
                            "x_norm": float(review_summary.get("x_norm", 0.5)),
                            "y_norm": float(review_summary.get("y_norm", 0.5)),
                            "depth_norm": float(review_summary.get("depth_norm", 0.5)),
                            "hand_size_norm": float(review_summary.get("hand_size_norm", 0.0)),
                        },
                        "robot": target,
                        "joints_rad": joints_rad,
                        "metrics": review_summary,
                    }
                    print(
                        f"[hand-workspace] accepted {pose_name}: "
                        f"hand=({review_summary.get('x_norm', 0.5):.3f},{review_summary.get('y_norm', 0.5):.3f},{review_summary.get('depth_norm', 0.5):.3f}) "
                        f"robot=({target['x_m']:+.3f},{target['y_m']:+.3f},{target['z_m']:+.3f})"
                    )
                    pose_index += 1
                    break

        payload = {
            "calibration_type": "hand_to_robot_workspace",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "source": str(getattr(args, "source_name", "camera_calibrate.py")),
            "camera": {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) if cap is not None else 0,
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) if cap is not None else 0,
                "index": int(camera.index) if camera is not None else int(getattr(val, "HANDTRACKING_CAMERA_INDEX", 0)),
            },
            "poses": pose_results,
            "workspace_bounds": bounds,
            "required_poses": _workspace_required_poses(),
            "optional_poses": _workspace_optional_poses(),
            "mapping": {
                "method": str(getattr(val, "HAND_WORKSPACE_MAPPING_METHOD", "rbf_residual")),
                "fallback": str(getattr(val, "HAND_WORKSPACE_FALLBACK_METHOD", "piecewise_affine")),
                "axis_map": dict(getattr(val, "HAND_CAMERA_TO_ROBOT_AXIS_MAP", {"image_x": "robot_x", "image_y": "robot_z", "depth": "robot_y"})),
                "flips": {
                    "x": bool(getattr(val, "HAND_IMAGE_X_FLIP", False)),
                    "y": bool(getattr(val, "HAND_IMAGE_Y_FLIP", True)),
                    "depth": bool(getattr(val, "HAND_DEPTH_FLIP", False)),
                },
            },
            "notes": "Complements robot_joint_calibration.json. Joints are IK seed examples only; runtime still uses IK.",
        }
        _print_pose_capture_summary("hand-workspace", pose_results)
        if not _confirm_final_save("hand-workspace"):
            print("[hand-workspace] Not saved.")
            return 0
        out_path = _save_hand_depth_calibration(payload, _hand_workspace_calibration_path(), bool(args.overwrite))
        print(f"[hand-workspace] Saved calibration: {out_path}")
        return 0
    finally:
        try:
            if hands is not None:
                hands.close()
        except Exception:
            pass
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        try:
            if robot is not None:
                robot.disconnect()
        except Exception:
            pass
        cv2.destroyAllWindows()



def _mirror_required_poses() -> list[str]:
    configured = list(getattr(val, "ROBOT_MIRROR_REQUIRED_POSES", []))
    return [str(x) for x in configured] if configured else [
        "center",
        "mirror_left",
        "mirror_right",
        "mirror_up",
        "mirror_down",
        "mirror_near",
        "mirror_far",
    ]


def _mirror_optional_poses() -> list[str]:
    configured = list(getattr(val, "ROBOT_MIRROR_OPTIONAL_POSES", []))
    if configured:
        return [str(x) for x in configured]
    return [
        "mirror_up_left",
        "mirror_up_right",
        "mirror_down_left",
        "mirror_down_right",
        "mirror_near_left",
        "mirror_near_right",
        "mirror_far_left",
        "mirror_far_right",
        "mirror_near_up",
        "mirror_near_down",
        "mirror_far_up",
        "mirror_far_down",
        "mirror_near_up_left",
        "mirror_near_up_right",
        "mirror_far_down_left",
        "mirror_far_down_right",
    ]


def _mirror_capture_pose_order() -> list[str]:
    out: list[str] = []
    for name in _mirror_required_poses() + _mirror_optional_poses():
        if name not in out:
            out.append(name)
    return out


def _mirror_pose_optional(name: str) -> bool:
    return str(name) not in set(_mirror_required_poses())


def _hand_mirror_pose_instruction(name: str) -> str:
    text = {
        "center": "Place your hand centered in the camera frame at normal runtime depth.",
        "mirror_left": "Place your hand at the leftmost usable part of the camera frame before it leaves view.",
        "mirror_right": "Place your hand at the rightmost usable part of the camera frame before it leaves view.",
        "mirror_up": "Place your hand near the top of the usable camera frame before it leaves view.",
        "mirror_down": "Place your hand near the bottom of the usable camera frame before it leaves view.",
        "mirror_near": "Move your hand close to the camera at the closest intended runtime distance.",
        "mirror_far": "Move your hand away from the camera at the farthest intended runtime distance.",
        "mirror_up_left": "Optional: place your hand at the upper-left usable screen corner at center depth.",
        "mirror_up_right": "Optional: place your hand at the upper-right usable screen corner at center depth.",
        "mirror_down_left": "Optional: place your hand at the lower-left usable screen corner at center depth.",
        "mirror_down_right": "Optional: place your hand at the lower-right usable screen corner at center depth.",
        "mirror_near_left": "Optional: place your hand close to the camera and left in the image.",
        "mirror_near_right": "Optional: place your hand close to the camera and right in the image.",
        "mirror_far_left": "Optional: place your hand far from the camera and left in the image.",
        "mirror_far_right": "Optional: place your hand far from the camera and right in the image.",
        "mirror_near_up": "Optional: place your hand close to the camera and high in the image.",
        "mirror_near_down": "Optional: place your hand close to the camera and low in the image.",
        "mirror_far_up": "Optional: place your hand far from the camera and high in the image.",
        "mirror_far_down": "Optional: place your hand far from the camera and low in the image.",
        "mirror_near_up_left": "Optional: place your hand close, high, and left. Skip if tracking is unreliable.",
        "mirror_near_up_right": "Optional: place your hand close, high, and right. Skip if tracking is unreliable.",
        "mirror_far_down_left": "Optional: place your hand far, low, and left. Skip if tracking is unreliable.",
        "mirror_far_down_right": "Optional: place your hand far, low, and right. Skip if tracking is unreliable.",
    }
    return text.get(str(name), f"Place your hand at {name}.")


def _print_hand_mirror_pose_instruction(name: str, index: int, total: int, optional: bool) -> None:
    print("\n[hand-mirror] Pose instruction")
    print(f"[hand-mirror] Pose: {name} ({index}/{total})")
    print(f"[hand-mirror] {_hand_mirror_pose_instruction(name)}")
    print("[hand-mirror] This pose will be paired by name with the robot pose recorded in robot_calibrate.py.")
    print("[hand-mirror] Press ENTER to open/continue camera preview. Sampling still waits for SPACE.")
    if optional:
        print("[hand-mirror] Optional pose: press S in the preview to skip if tracking is unreliable.")


def _estimate_simple_wrist_rpy_from_landmarks(hand_lms) -> list[float]:
    try:
        lm = hand_lms.landmark
        index = np.array([float(lm[5].x), float(lm[5].y)], dtype=np.float64)
        pinky = np.array([float(lm[17].x), float(lm[17].y)], dtype=np.float64)
        wrist = np.array([float(lm[0].x), float(lm[0].y)], dtype=np.float64)
        middle = np.array([float(lm[9].x), float(lm[9].y)], dtype=np.float64)
        lateral = pinky - index
        forward = middle - wrist
        roll = float(np.arctan2(lateral[1], lateral[0])) if np.linalg.norm(lateral) > 1e-6 else 0.0
        yaw = float(np.arctan2(forward[0], max(abs(float(forward[1])), 1e-6))) if np.linalg.norm(forward) > 1e-6 else 0.0
        pitch = -float(np.arctan2(forward[1], max(abs(float(forward[0])), 1e-6))) if np.linalg.norm(forward) > 1e-6 else 0.0
        return [roll, pitch, yaw]
    except Exception:
        return [0.0, 0.0, 0.0]


def _effective_hand_mirror_metrics(hand_lms, depth_estimator: HandDepthEstimator, frame_w: int, frame_h: int) -> dict:
    # Use the raw normalized values that handtracking.py passes into RobotMirrorWorkspaceMapper.
    # Pose names define direction, so this paired calibration should not apply HAND_IMAGE_* flips here.
    metrics = _hand_landmark_metrics(hand_lms)
    depth = depth_estimator.estimate_depth(hand_lms, frame_w=frame_w, frame_h=frame_h)
    return {
        "x_norm": float(np.clip(metrics["x_norm"], 0.0, 1.0)),
        "y_norm": float(np.clip(metrics["y_norm"], 0.0, 1.0)),
        "depth_norm": float(np.clip(float(depth.get("depth_norm", 0.5)), 0.0, 1.0)),
        "hand_size_norm": float(depth.get("hand_size_norm", metrics["hand_size_norm"])),
        "palm_width_norm": float(depth.get("palm_width_norm", metrics["palm_width_norm"])),
        "wrist_to_middle_mcp_norm": float(depth.get("wrist_to_middle_mcp_norm", metrics["wrist_to_middle_mcp_norm"])),
        "palm_height_norm": float(depth.get("palm_height_norm", metrics["palm_height_norm"])),
        "bbox_size_norm": float(depth.get("bbox_size_norm", metrics["bbox_size_norm"])),
        "thumb_index_span_norm": float(metrics.get("thumb_index_span_norm", 0.0)),
        "depth_m": float(depth.get("depth_m", 0.0)),
        "depth_source": str(depth.get("source", "unknown")),
        "depth_confidence": float(depth.get("confidence", 0.0)),
        "wrist_rpy_rad": _estimate_simple_wrist_rpy_from_landmarks(hand_lms),
    }


def _run_hand_mirror_position_calibration(args) -> int:
    try:
        import mediapipe as mp
    except Exception as exc:
        print(f"[hand-mirror] ERROR: MediaPipe is required: {exc}")
        return 1

    hands = None
    camera = None
    cap = None
    pending_first_frame = None
    depth_estimator = HandDepthEstimator()
    poses = _mirror_capture_pose_order()
    required = max(3, int(getattr(val, "HAND_DEPTH_CALIBRATION_REQUIRED_STABLE_FRAMES", 20)))
    samples_per_pose = max(required, int(getattr(val, "HAND_DEPTH_CALIBRATION_SAMPLES_PER_POSE", 40)))
    stability_max = float(getattr(val, "HAND_DEPTH_CALIBRATION_STABILITY_STD_MAX", 0.015))
    pose_results = {}
    pose_index = 0
    samples: list[dict] = []
    print("[hand-mirror] Hand-position mirror calibration.")
    print("[hand-mirror] This records camera/MediaPipe hand positions only; it does not connect to or move the robot.")
    print("[hand-mirror] Run robot_calibrate.py mirror_workspace separately to record matching robot poses.")
    try:
        while pose_index < len(poses):
            state = "waiting_for_user_ready"
            review_summary = None
            pose_name = str(poses[pose_index])
            samples.clear()
            optional = _mirror_pose_optional(pose_name)
            _print_hand_mirror_pose_instruction(pose_name, pose_index + 1, len(poses), optional)
            try:
                reply = input("[hand-mirror] Press ENTER when ready for camera preview, or Q to quit: ").strip().lower()
            except EOFError:
                reply = ""
            if reply in {"q", "quit", "esc", "exit"}:
                print("[hand-mirror] Quit before sampling; not saved.")
                return 1
            if hands is None:
                hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    model_complexity=int(getattr(val, "HANDTRACKING_MODEL_COMPLEXITY", 0)),
                    min_detection_confidence=float(getattr(val, "HANDTRACKING_MIN_DETECTION_CONFIDENCE", 0.55)),
                    min_tracking_confidence=float(getattr(val, "HANDTRACKING_MIN_TRACKING_CONFIDENCE", 0.55)),
                )
            if camera is None:
                camera = _open_hand_depth_camera(args)
                if camera is None:
                    return 1
                cap = camera.cap
                pending_first_frame = camera.frame

            while True:
                if cap is None or hands is None:
                    print("[hand-mirror] ERROR: Camera/MediaPipe initialization failed.")
                    return 1
                if pending_first_frame is not None:
                    ok, frame = True, pending_first_frame
                    pending_first_frame = None
                else:
                    ok, frame = read_latest_from_capture(cap)
                if not ok or frame is None:
                    print("[hand-mirror] ERROR: Could not open/read webcam.")
                    return 1
                frame_h, frame_w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                metrics = None
                if results.multi_hand_landmarks:
                    hand_lms = results.multi_hand_landmarks[0]
                    metrics = _effective_hand_mirror_metrics(hand_lms, depth_estimator, frame_w, frame_h)
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand_lms, mp.solutions.hands.HAND_CONNECTIONS)

                std = float("nan")
                if state == "sampling" and metrics is not None:
                    samples.append(metrics)
                    samples = samples[-samples_per_pose:]
                    recent = samples[-required:]
                    if len(recent) >= required:
                        std = float(np.std([s["hand_size_norm"] for s in recent]))
                        if std <= stability_max:
                            review_summary = _mean_metrics(recent)
                            # average wrist_rpy separately because it is a list.
                            rpy_samples = [s.get("wrist_rpy_rad") for s in recent if isinstance(s.get("wrist_rpy_rad"), list)]
                            if rpy_samples:
                                review_summary["wrist_rpy_rad"] = np.mean(np.asarray(rpy_samples, dtype=np.float64), axis=0).tolist()
                            state = "review"

                lines = [
                    f"Hand mirror calibration: {pose_name} ({pose_index + 1}/{len(poses)})",
                    _hand_mirror_pose_instruction(pose_name),
                    "Required: one visible MediaPipe hand, held steady",
                ]
                if state == "waiting_for_user_ready":
                    lines += [
                        "Press SPACE when ready to start sampling",
                        "R = reset current pose",
                        "S = skip optional pose",
                        "Q/ESC = quit without saving",
                    ]
                elif state == "sampling":
                    std_text = "n/a" if not np.isfinite(std) else f"{std:.5f}"
                    lines += [
                        f"Sampling: {len(samples)}/{required} valid samples",
                        f"stability std: {std_text} limit={stability_max:.5f}",
                        "No hand detected: sampling paused" if metrics is None else "Hold steady",
                        "R = reset current pose, Q/ESC = quit without saving",
                    ]
                else:
                    s = review_summary or {}
                    lines += [
                        "Review summary",
                        f"mean=({float(s.get('x_norm', 0.0)):.3f},{float(s.get('y_norm', 0.0)):.3f},{float(s.get('depth_norm', 0.0)):.3f}) size={float(s.get('hand_size_norm', 0.0)):.4f}",
                        f"std=({float(s.get('x_norm_std', 0.0)):.4f},{float(s.get('y_norm_std', 0.0)):.4f},{float(s.get('depth_norm_std', 0.0)):.4f}) size_std={float(s.get('hand_size_norm_std', 0.0)):.5f}",
                        "A = accept pose, R = resample, Q/ESC = quit",
                    ]
                if metrics is not None:
                    lines.append(f"live hand=({metrics['x_norm']:.2f},{metrics['y_norm']:.2f},{metrics['depth_norm']:.2f}) size={metrics['hand_size_norm']:.3f}")
                else:
                    lines.append("live hand: not detected")
                _draw_text_lines(frame, lines)
                cv2.imshow("Hand mirror position calibration", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    return 1
                if key == ord("r"):
                    samples.clear()
                    review_summary = None
                    state = "waiting_for_user_ready"
                    continue
                if state == "waiting_for_user_ready" and key == ord("s") and optional:
                    print(f"[hand-mirror] skipped optional pose {pose_name}")
                    pose_index += 1
                    break
                if state == "waiting_for_user_ready" and key == ord(" "):
                    samples.clear()
                    review_summary = None
                    state = "sampling"
                    continue
                if state == "review" and key == ord("a") and review_summary:
                    pose_results[pose_name] = {
                        "hand": {
                            "x_norm": float(review_summary.get("x_norm", 0.5)),
                            "y_norm": float(review_summary.get("y_norm", 0.5)),
                            "depth_norm": float(review_summary.get("depth_norm", 0.5)),
                            "hand_size_norm": float(review_summary.get("hand_size_norm", 0.0)),
                            "wrist_rpy_rad": list(review_summary.get("wrist_rpy_rad", [0.0, 0.0, 0.0])),
                        },
                        "std": {
                            k: float(v) for k, v in review_summary.items()
                            if k.endswith("_std") and isinstance(v, (int, float)) and np.isfinite(float(v))
                        },
                        "metrics": review_summary,
                        "samples": int(len(samples)),
                    }
                    print(
                        f"[hand-mirror] accepted {pose_name}: "
                        f"hand=({review_summary.get('x_norm', 0.5):.3f},{review_summary.get('y_norm', 0.5):.3f},{review_summary.get('depth_norm', 0.5):.3f})"
                    )
                    pose_index += 1
                    break
        missing = [name for name in _mirror_required_poses() if name not in pose_results]
        if missing:
            print("[hand-mirror] Missing required poses; not saving: " + ", ".join(missing))
            return 1
        payload = {
            "calibration_type": "hand_mirror_position_extrema",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "source": "camera_calibrate.py",
            "camera": {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) if cap is not None else 0,
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) if cap is not None else 0,
                "index": int(camera.index) if camera is not None else int(getattr(val, "HANDTRACKING_CAMERA_INDEX", 0)),
            },
            "required_poses": _mirror_required_poses(),
            "optional_poses": _mirror_optional_poses(),
            "poses": pose_results,
            "notes": "Pairs by pose name with robot_mirror_workspace_calibration.json; no robot movement is performed here.",
        }
        _print_pose_capture_summary("hand-mirror", pose_results)
        if not _confirm_final_save("hand-mirror"):
            print("[hand-mirror] Not saved.")
            return 0
        out_path = _save_hand_depth_calibration(payload, _hand_mirror_position_calibration_path(), bool(args.overwrite))
        print(f"[hand-mirror] Saved calibration: {out_path}")
        return 0
    finally:
        try:
            if hands is not None:
                hands.close()
        except Exception:
            pass
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()


def _run_hand_depth_calibration(args) -> int:
    try:
        import mediapipe as mp
    except Exception as exc:
        print(f"[hand-depth] ERROR: MediaPipe is required for hand-depth calibration: {exc}")
        return 1

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=int(getattr(val, "HANDTRACKING_MODEL_COMPLEXITY", 0)),
        min_detection_confidence=float(getattr(val, "HANDTRACKING_MIN_DETECTION_CONFIDENCE", 0.55)),
        min_tracking_confidence=float(getattr(val, "HANDTRACKING_MIN_TRACKING_CONFIDENCE", 0.55)),
    )
    camera = _open_hand_depth_camera(args)
    if camera is None:
        try:
            hands.close()
        except Exception:
            pass
        return 1
    cap = camera.cap
    pending_first_frame = camera.frame
    camera_index = int(camera.index)

    poses_cfg = list(getattr(val, "HAND_DEPTH_CALIBRATION_POSES", ["near", "center", "far"]))
    depth_by_pose = {
        "near": float(getattr(val, "HAND_MONOCULAR_NEAR_M", 0.20)),
        "center": float(getattr(val, "HAND_MONOCULAR_CENTER_M", 0.45)),
        "far": float(getattr(val, "HAND_MONOCULAR_FAR_M", 0.70)),
    }
    samples_per_pose = max(3, int(getattr(val, "HAND_DEPTH_CALIBRATION_SAMPLES_PER_POSE", 40)))
    required_stable = max(3, int(getattr(val, "HAND_DEPTH_CALIBRATION_REQUIRED_STABLE_FRAMES", 20)))
    stability_max = float(getattr(val, "HAND_DEPTH_CALIBRATION_STABILITY_STD_MAX", 0.015))
    pose_results: dict[str, dict] = {}
    pose_index = 0
    pose_samples: list[dict] = []
    print("For this RGB-only robot control setup, hand-depth calibration is required/recommended.")
    print("[hand-depth] Hold an open hand at each requested measured distance.")
    print("[hand-depth] Sampling starts only after SPACE. Acceptance requires A.")
    try:
        while pose_index < len(poses_cfg):
            pose_name = str(poses_cfg[pose_index])
            target_depth = float(depth_by_pose.get(pose_name, getattr(val, "HAND_DEPTH_DEFAULT_M", 0.45)))
            state = "waiting_for_user_ready"
            review_summary = None
            pose_samples.clear()

            while True:
                if pending_first_frame is not None:
                    ok, frame = True, pending_first_frame
                    pending_first_frame = None
                else:
                    ok, frame = read_latest_from_capture(cap)
                if not ok or frame is None:
                    print("[hand-depth] ERROR: Could not open/read webcam.")
                    print("[hand-depth] Try changing camera index with --camera-index N.")
                    print("[hand-depth] On macOS, check System Settings > Privacy & Security > Camera for PyCharm/Terminal.")
                    return 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                metrics = None
                if results.multi_hand_landmarks:
                    hand_lms = results.multi_hand_landmarks[0]
                    metrics = _hand_landmark_metrics(hand_lms)
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand_lms, mp.solutions.hands.HAND_CONNECTIONS)

                std = float("nan")
                stable = False
                if state == "sampling" and metrics is not None:
                    pose_samples.append(metrics)
                    pose_samples = pose_samples[-samples_per_pose:]
                    recent = pose_samples[-required_stable:]
                    if len(recent) >= required_stable:
                        std = float(np.std([s["hand_size_norm"] for s in recent]))
                        stable = std <= stability_max
                        if stable:
                            review_summary = _hand_depth_pose_summary(target_depth, recent)
                            state = "review"
                elif state == "sampling" and metrics is None:
                    # Hand disappeared: sampling pauses by not appending.
                    pass

                lines = [
                    f"Hand-depth calibration: {pose_name.upper()} ({pose_index + 1}/{len(poses_cfg)})",
                    f"Place open hand at {pose_name.upper()} distance: {target_depth:.2f} m",
                    "Required: one visible open MediaPipe hand, held steady",
                ]
                if state == "waiting_for_user_ready":
                    lines += [
                        "Press SPACE when ready to start sampling",
                        "R = reset current pose",
                        "S = skip optional pose",
                        "Q/ESC = quit without saving",
                    ]
                elif state == "sampling":
                    std_text = "n/a" if not np.isfinite(std) else f"{std:.5f}"
                    lines += [
                        f"Sampling: {len(pose_samples)}/{required_stable} valid samples",
                        f"stability std: {std_text} limit={stability_max:.5f}",
                        "No hand detected: sampling paused" if metrics is None else "Hold steady",
                        "R = reset current pose, Q/ESC = quit without saving",
                    ]
                else:
                    s = review_summary or {}
                    lines += [
                        "Review summary",
                        f"mean size={float(s.get('hand_size_norm_mean', 0.0)):.4f} depth={float(s.get('depth_m', 0.0)):.2f}m",
                        f"std size={float(s.get('hand_size_norm_std', 0.0)):.5f} x={float(s.get('x_norm_std', 0.0)):.4f} y={float(s.get('y_norm_std', 0.0)):.4f}",
                        "A = accept pose, R = resample, Q/ESC = quit",
                    ]
                if metrics:
                    lines.append(f"live hand_size_norm: {metrics['hand_size_norm']:.4f}")
                else:
                    lines.append("live hand: not detected")
                _draw_text_lines(frame, lines)
                cv2.imshow("RGB hand-depth calibration", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    return 1
                if key == ord("r"):
                    pose_samples.clear()
                    review_summary = None
                    state = "waiting_for_user_ready"
                    continue
                if state == "waiting_for_user_ready" and key == ord(" "):
                    pose_samples.clear()
                    review_summary = None
                    state = "sampling"
                    continue
                if state == "review" and key == ord("a") and review_summary:
                    pose_results[pose_name] = review_summary
                    print(
                        f"[hand-depth] accepted {pose_name}: depth={target_depth:.2f}m "
                        f"size={review_summary['hand_size_norm_mean']:.4f} std={review_summary['hand_size_norm_std']:.4f}"
                    )
                    pose_index += 1
                    break

        near = pose_results.get("near", {})
        center = pose_results.get("center", {})
        far = pose_results.get("far", {})
        payload = {
            "calibration_type": "monocular_hand_depth",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "camera_index": camera_index,
            "image_width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "image_height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            "poses": pose_results,
            "fit": {
                "method": "inverse_or_piecewise",
                "near_depth_m": float(near.get("depth_m", getattr(val, "HAND_MONOCULAR_NEAR_M", 0.20))),
                "center_depth_m": float(center.get("depth_m", getattr(val, "HAND_MONOCULAR_CENTER_M", 0.45))),
                "far_depth_m": float(far.get("depth_m", getattr(val, "HAND_MONOCULAR_FAR_M", 0.70))),
                "near_size_norm": float(near.get("hand_size_norm_mean", getattr(val, "HAND_MONOCULAR_NEAR_SIZE_NORM", 0.32))),
                "center_size_norm": float(center.get("hand_size_norm_mean", getattr(val, "HAND_MONOCULAR_CENTER_SIZE_NORM", 0.20))),
                "far_size_norm": float(far.get("hand_size_norm_mean", getattr(val, "HAND_MONOCULAR_FAR_SIZE_NORM", 0.12))),
            },
            "notes": "Runtime uses RGB MediaPipe hand size only; no markers required.",
        }
        _print_pose_capture_summary("hand-depth", pose_results)
        if not _confirm_final_save("hand-depth"):
            print("[hand-depth] Not saved.")
            return 0
        out_path = _save_hand_depth_calibration(
            payload,
            getattr(val, "HAND_MONOCULAR_DEPTH_CALIBRATION_FILE", "calibration_data/hand_depth_calibration.json"),
            bool(args.overwrite),
        )
        print(f"[hand-depth] Saved calibration: {out_path}")
        return 0
    finally:
        try:
            hands.close()
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()


def _run_charuco_calibration(args) -> int:
    aruco = _require_aruco()
    dictionary_name = str(args.dictionary)
    dictionary = _aruco_dictionary_from_name(dictionary_name)
    board = _create_charuco_board(
        args.squares_x,
        args.squares_y,
        args.square_length_m,
        args.marker_length_m,
        dictionary,
    )

    if args.print_board:
        _draw_charuco_board(board, _resolve_output(args.print_board), args.squares_x, args.squares_y)
        print(f"[charuco] Wrote board image: {_resolve_output(args.print_board)}")
        print(f"[charuco] squares: {args.squares_x} x {args.squares_y}")
        print(f"[charuco] square length: {args.square_length_m:.4f} m")
        print(f"[charuco] marker length: {args.marker_length_m:.4f} m")
        print("[charuco] Print/display without scaling. Measure square_length_m after printing if possible.")
        if not args.charuco:
            return 0

    print("\n[charuco] Optional camera intrinsics calibration")
    print("[charuco] ChArUco camera calibration")
    print("[charuco] Use a printed/displayed ChArUco board at 100% scale.")
    print("[charuco] Move the board through image center/corners and several tilts.")
    print("[charuco] Avoid motion blur and glare.")
    print("[charuco] Controls: SPACE accept | A auto on/off | R reset | C calibrate | Q/ESC quit")

    cap = cv2.VideoCapture(int(args.camera_index))
    if not cap.isOpened():
        print(f"[charuco] ERROR: could not open camera index {args.camera_index}")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(getattr(args, "frame_width", getattr(val, "CHARUCO_FRAME_WIDTH", 1280))))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(getattr(args, "frame_height", getattr(val, "CHARUCO_FRAME_HEIGHT", 720))))
    cap.set(cv2.CAP_PROP_FPS, int(getattr(args, "fps", getattr(val, "CHARUCO_FPS", 30))))

    detector = _build_charuco_detector(dictionary)
    all_corners = []
    all_ids = []
    image_size = None
    auto_capture = True
    last_capture = 0.0
    latest_valid = None
    calibrated = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                _draw_text_lines(np.zeros((480, 640, 3), dtype=np.uint8), ["Camera read failed"])
                continue
            image_size = (int(frame.shape[1]), int(frame.shape[0]))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected, ch_corners, ch_ids, marker_count, corner_count = _detect_charuco(
                gray,
                board,
                detector,
                args.min_markers,
            )
            valid = ch_corners is not None and ch_ids is not None and marker_count >= args.min_markers and corner_count >= args.min_corners
            if valid:
                latest_valid = (ch_corners.copy(), ch_ids.copy())
            preview = frame.copy()
            if args.show_detections or marker_count > 0 or valid:
                _draw_charuco_detections(preview, corners, ids, ch_corners, ch_ids)

            now = time.time()
            if valid and auto_capture and (now - last_capture) >= float(args.capture_delay_s) and len(all_corners) < args.frames:
                all_corners.append(ch_corners.copy())
                all_ids.append(ch_ids.copy())
                last_capture = now

            lines = [
                "ChArUco camera calibration",
                f"Markers detected: {marker_count}",
                f"ChArUco corners: {corner_count}",
                f"Accepted frames: {len(all_corners)}/{args.frames} auto={'ON' if auto_capture else 'OFF'}",
                "SPACE=accept, A=auto, C=calibrate, R=reset, Q/ESC=quit",
            ]
            if marker_count == 0:
                lines.append("Detected no markers. Verify the board dictionary matches CHARUCO_DICTIONARY.")
            elif not valid:
                lines.append(f"Need >= {args.min_markers} markers and >= {args.min_corners} ChArUco corners.")
            _draw_text_lines(preview, lines)
            cv2.imshow("ChArUco camera calibration", preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                return 1 if calibrated is None else 0
            if key == ord("a"):
                auto_capture = not auto_capture
            elif key == ord("r"):
                all_corners.clear()
                all_ids.clear()
                calibrated = None
                print("[charuco] Captures reset.")
            elif key == ord(" ") and latest_valid is not None:
                all_corners.append(latest_valid[0].copy())
                all_ids.append(latest_valid[1].copy())
                print(f"[charuco] Accepted manual frame {len(all_corners)}/{args.frames}.")
            should_calibrate = key == ord("c") or len(all_corners) >= args.frames
            if should_calibrate:
                if len(all_corners) < max(3, int(args.min_frames_for_calibration)):
                    print(f"[charuco] Need at least {args.min_frames_for_calibration} valid frames before calibration.")
                    continue
                if image_size is None:
                    print("[charuco] No image size available yet.")
                    continue
                try:
                    retval, camera_matrix, dist_coeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
                        all_corners,
                        all_ids,
                        board,
                        image_size,
                        None,
                        None,
                    )
                except AttributeError:
                    print("[charuco] ERROR: calibrateCameraCharuco unavailable. Install opencv-contrib-python.")
                    return 1
                except Exception as exc:
                    print(f"[charuco] Calibration failed: {exc}")
                    continue
                calibrated = {
                    "camera_matrix": np.asarray(camera_matrix, dtype=np.float64),
                    "dist_coeffs": np.asarray(dist_coeffs, dtype=np.float64),
                    "reprojection_error": float(retval),
                    "image_size": image_size,
                    "rvecs": rvecs,
                    "tvecs": tvecs,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "charuco": {
                        "squares_x": int(args.squares_x),
                        "squares_y": int(args.squares_y),
                        "square_length_m": float(args.square_length_m),
                        "marker_length_m": float(args.marker_length_m),
                        "dictionary": dictionary_name,
                    },
                }
                npz_path, json_path = _save_charuco_calibration(calibrated, args.output_npz, args.output_json, bool(args.overwrite))
                print(f"[charuco] Saved NPZ:  {npz_path}")
                print(f"[charuco] Saved JSON: {json_path}")
                print(f"[charuco] Reprojection error: {float(retval):.3f} px")
                if float(retval) > float(getattr(val, "CHARUCO_REPROJECTION_ERROR_WARN", 1.0)):
                    print("[charuco] WARNING: reprojection error is high; collect sharper/more varied views.")
                return 0
    finally:
        cap.release()
        cv2.destroyAllWindows()


def _choose_marker(ids_list: Sequence[int], allowed_ids: Sequence[int]) -> Optional[Tuple[int, int]]:
    for desired in allowed_ids:
        if desired in ids_list:
            return ids_list.index(desired), int(desired)
    return None


def _solve_marker_pose(frame, detector, K, dist, marker_size_m: float, allowed_ids: Sequence[int]) -> Optional[PoseSample]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _rejected = detector.detectMarkers(gray)
    if ids is None or len(corners) == 0:
        return None

    ids_list = [int(x) for x in ids.flatten().tolist()]
    chosen = _choose_marker(ids_list, allowed_ids)
    if chosen is None:
        return None
    chosen_idx, chosen_id = chosen

    image_points = np.asarray(corners[chosen_idx], dtype=np.float64).reshape(4, 2)
    ok, rvec, tvec = cv2.solvePnP(
        _marker_object_points(marker_size_m),
        image_points,
        np.asarray(K, dtype=np.float64).reshape(3, 3),
        np.asarray(dist, dtype=np.float64).reshape(-1, 1),
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        return None

    xyz_camera = np.asarray(tvec, dtype=np.float64).reshape(3)
    xyz_workspace = _workspace_from_camera_point(xyz_camera)
    return PoseSample(
        marker_id=int(chosen_id),
        xyz_workspace=xyz_workspace,
        xyz_camera=xyz_camera,
        rvec=np.asarray(rvec, dtype=np.float64).reshape(3),
        tvec=np.asarray(tvec, dtype=np.float64).reshape(3),
        image_points=image_points,
    )


def _median_pose_from_samples(samples: Sequence[PoseSample]) -> PoseSample:
    xyz_workspace = np.median(np.stack([s.xyz_workspace for s in samples], axis=0), axis=0)
    xyz_camera = np.median(np.stack([s.xyz_camera for s in samples], axis=0), axis=0)
    rvec = np.median(np.stack([s.rvec for s in samples], axis=0), axis=0)
    tvec = np.median(np.stack([s.tvec for s in samples], axis=0), axis=0)
    image_points = np.median(np.stack([s.image_points for s in samples], axis=0), axis=0)
    marker_id = int(samples[-1].marker_id)
    return PoseSample(marker_id, xyz_workspace, xyz_camera, rvec, tvec, image_points)


def _capture_pose_interactive(
    cap,
    detector,
    K,
    dist,
    marker_size_m: float,
    allowed_ids: Sequence[int],
    label: str,
    instruction: str,
    hold_frames: int = 20,
    flip_preview: bool = True,
) -> Optional[PoseSample]:
    samples = []

    while True:
        ok, frame = cap.read()
        if not ok:
            return None

        raw = frame.copy()
        preview = cv2.flip(frame, 1) if flip_preview else frame.copy()
        pose = _solve_marker_pose(raw, detector, K, dist, marker_size_m, allowed_ids)

        lines = [
            f"Capture: {label}",
            instruction,
            f"SPACE = record {hold_frames} stable frames | R = reset | Q = quit",
        ]

        if pose is None:
            lines.append("Marker status: not found / wrong id")
        else:
            xyz = pose.xyz_workspace
            lines.append(
                f"Marker id={pose.marker_id}  workspace xyz = ({xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f})"
            )

        if samples:
            lines.append(f"Sampling: {len(samples)}/{hold_frames}")

        _draw_text_lines(preview, lines)
        cv2.imshow("calib.py - Hand ArUco calibration", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return None
        if key == ord("r"):
            samples.clear()
            continue
        if key != ord(" "):
            continue

        samples.clear()
        while len(samples) < int(hold_frames):
            ok2, frame2 = cap.read()
            if not ok2:
                return None
            raw2 = frame2.copy()
            preview2 = cv2.flip(frame2, 1) if flip_preview else frame2.copy()
            pose2 = _solve_marker_pose(raw2, detector, K, dist, marker_size_m, allowed_ids)
            if pose2 is not None:
                samples.append(pose2)

            lines2 = [
                f"Capture: {label}",
                instruction,
                f"Hold marker steady... {len(samples)}/{hold_frames}",
            ]
            if pose2 is None:
                lines2.append("Marker status: not found / wrong id")
            else:
                xyz2 = pose2.xyz_workspace
                lines2.append(
                    f"workspace xyz = ({xyz2[0]:.4f}, {xyz2[1]:.4f}, {xyz2[2]:.4f})"
                )
            _draw_text_lines(preview2, lines2)
            cv2.imshow("calib.py - Hand ArUco calibration", preview2)
            key2 = cv2.waitKey(1) & 0xFF
            if key2 == ord("q"):
                return None
            if key2 == ord("r"):
                samples.clear()

        return _median_pose_from_samples(samples)


def run_hand_aruco_depth_calibration(
    cap,
    *,
    allowed_ids: Sequence[int],
    aruco_dict_id: int,
    marker_size_m: float,
    hold_frames: int = 20,
) -> Tuple[Optional[dict], Optional[dict], Optional[dict]]:
    intr = load_intrinsics(INTRINSICS_NPZ)
    ok_i, msg_i = _validate_intrinsics_dict(intr)
    if not ok_i:
        raise RuntimeError(
            "Camera intrinsics are required before hand-marker depth calibration can run.\n"
            f"Expected: {INTRINSICS_NPZ}\n"
            f"Validation result: {msg_i}"
        )

    detector = _build_aruco_detector(int(aruco_dict_id))
    K = intr["mtx"]
    dist = intr["dist"]

    steps = [
        ("neutral", "Place the hand marker at your neutral center pose and press SPACE."),
        ("left", "Move the hand marker to the LEFT limit you want to allow, then press SPACE."),
        ("right", "Move the hand marker to the RIGHT limit you want to allow, then press SPACE."),
        ("near", "Move the hand marker CLOSE to the camera (minimum reach depth), then press SPACE."),
        ("far", "Move the hand marker FAR from the camera (maximum reach depth), then press SPACE."),
        ("low", "Move the hand marker to the LOWEST allowed height, then press SPACE."),
        ("high", "Move the hand marker to the HIGHEST allowed height, then press SPACE."),
    ]

    captures: Dict[str, PoseSample] = {}
    for name, instruction in steps:
        pose = _capture_pose_interactive(
            cap,
            detector,
            K,
            dist,
            marker_size_m=float(marker_size_m),
            allowed_ids=allowed_ids,
            label=name,
            instruction=instruction,
            hold_frames=int(hold_frames),
            flip_preview=True,
        )
        if pose is None:
            print(f"[calib] Calibration cancelled during step '{name}'.")
            return None, None, None
        captures[name] = pose
        xyz = pose.xyz_workspace
        print(f"[calib] Captured {name}: marker={pose.marker_id} xyz=({xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f})")

    neutral = captures["neutral"].xyz_workspace
    workspace_min = np.array(
        [
            min(captures["left"].xyz_workspace[0], neutral[0]),
            min(captures["near"].xyz_workspace[1], neutral[1]),
            min(captures["low"].xyz_workspace[2], neutral[2]),
        ],
        dtype=np.float64,
    )
    workspace_max = np.array(
        [
            max(captures["right"].xyz_workspace[0], neutral[0]),
            max(captures["far"].xyz_workspace[1], neutral[1]),
            max(captures["high"].xyz_workspace[2], neutral[2]),
        ],
        dtype=np.float64,
    )

    span = workspace_max - workspace_min
    min_span = np.array([0.04, 0.04, 0.04], dtype=np.float64)
    center = 0.5 * (workspace_min + workspace_max)
    span = np.maximum(span, min_span)
    workspace_min = center - 0.5 * span
    workspace_max = center + 0.5 * span

    R_camera_from_workspace = _camera_from_workspace_rotation()
    t_camera_from_workspace = np.zeros(3, dtype=np.float64)
    save_extrinsics(R_camera_from_workspace, t_camera_from_workspace)

    capture_xyz = {name: pose.xyz_workspace for name, pose in captures.items()}
    save_workspace(
        workspace_min_xyz=workspace_min,
        workspace_max_xyz=workspace_max,
        neutral_xyz=neutral,
        captures_xyz=capture_xyz,
        marker_ids=allowed_ids,
        aruco_dict_id=int(aruco_dict_id),
        marker_size_m=float(marker_size_m),
    )

    intr2 = load_intrinsics(INTRINSICS_NPZ)
    ws2 = load_workspace(WORKSPACE_NPZ)
    ext2 = load_extrinsics(EXTRINSICS_NPZ)

    ok_w, msg_w = _validate_workspace_dict(ws2)
    ok_e, msg_e = _validate_extrinsics_dict(ext2)

    print("\n[calib] Saved workspace:", WORKSPACE_NPZ)
    print("[calib] Workspace validation:", "OK" if ok_w else f"FAIL ({msg_w})")
    print("[calib] Saved extrinsics:", EXTRINSICS_NPZ)
    print("[calib] Extrinsics validation:", "OK" if ok_e else f"FAIL ({msg_e})")
    print(f"[calib] workspace_min = {np.asarray(ws2['workspace_min']).reshape(3)}")
    print(f"[calib] workspace_max = {np.asarray(ws2['workspace_max']).reshape(3)}")

    return intr2, ws2, ext2


def ensure_calibration(cap=None, verbose: bool = True, **_ignored_kwargs):
    intr = load_intrinsics(INTRINSICS_NPZ)
    ws = load_workspace(WORKSPACE_NPZ)
    ext = load_extrinsics(EXTRINSICS_NPZ)

    ok_i, msg_i = _validate_intrinsics_dict(intr)
    ok_w, msg_w = _validate_workspace_dict(ws)
    ok_e, msg_e = _validate_extrinsics_dict(ext)

    if verbose:
        _print_calib_status(prefix="[calib] ")
        print(f"[calib] Intrinsics validation: {'OK' if ok_i else f'FAIL ({msg_i})'}")
        print(f"[calib] Workspace  validation: {'OK' if ok_w else f'FAIL ({msg_w})'}")
        print(f"[calib] Extrinsics validation: {'OK' if ok_e else f'FAIL ({msg_e})'}")

    if not ok_i or not ok_w or not ok_e:
        raise RuntimeError(
            "Calibration data not found or invalid.\n"
            f"  intrinsics: {INTRINSICS_NPZ} -> {'OK' if ok_i else 'FAIL'} ({msg_i})\n"
            f"  workspace:  {WORKSPACE_NPZ} -> {'OK' if ok_w else 'FAIL'} ({msg_w})\n"
            f"  extrinsics: {EXTRINSICS_NPZ} -> {'OK' if ok_e else 'FAIL'} ({msg_e})\n"
            "Run calib.py directly to regenerate calibration_data/*.npz."
        )

    return intr, ws, ext


def force_recalibration(
    camera_index: int = 0,
    *,
    allowed_ids: Optional[Iterable[int]] = None,
    aruco_dict_id: Optional[int] = None,
    marker_size_m: Optional[float] = None,
    hold_frames: int = 20,
):
    if allowed_ids is None:
        allowed_ids = [
            int(getattr(val, "ARUCO_GLOVE_FRONT_ID", 1)),
            int(getattr(val, "ARUCO_GLOVE_BACK_ID", 5)),
        ]
    if aruco_dict_id is None:
        dict_name = getattr(val, "ARUCO_DICT_NAME", "DICT_4X4_50")
        aruco_dict_id = int(getattr(cv2.aruco, dict_name))
    if marker_size_m is None:
        marker_size_m = float(getattr(val, "ARUCO_MARKER_SIZE_M", 0.03))

    delete_calibration_files()

    cap = cv2.VideoCapture(int(camera_index))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {camera_index}.")

    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(getattr(val, "CAM_W", 720)))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(getattr(val, "CAM_H", 720)))
    except Exception:
        pass

    try:
        return run_hand_aruco_depth_calibration(
            cap,
            allowed_ids=[int(x) for x in allowed_ids],
            aruco_dict_id=int(aruco_dict_id),
            marker_size_m=float(marker_size_m),
            hold_frames=int(hold_frames),
        )
    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def _parse_ids(s: str) -> Sequence[int]:
    if not s.strip():
        return []
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _no_arg_menu(args) -> int:
    print("\nCamera calibration / hand-depth setup\n")
    print("Recommended for this RGB-only robot:")
    print("1. Run hand-depth calibration now")
    print("2. Show calibration status")
    print("3. Optional ChArUco camera intrinsics calibration")
    print("4. Optional chessboard camera intrinsics calibration")
    print("5. Quit")
    print("6. Legacy hand-to-workspace calibration")
    print("7. Hand mirror position calibration")
    print("\nTip: In PyCharm, add --hand-depth to Run Configuration > Parameters to launch calibration directly.")
    try:
        choice = input("\nSelection [1/2/3/4/5/6/7]: ").strip()
    except EOFError:
        choice = ""
    if choice == "":
        choice = "1"
    if choice == "1":
        return _run_hand_depth_calibration(args)
    if choice == "2":
        return _calibration_status()
    if choice == "3":
        args.charuco = True
        return _run_charuco_calibration(args)
    if choice == "4":
        args.chessboard = True
        return _run_charuco_calibration(args)
    if choice == "5":
        print("[calib] Quit.")
        return 0
    if choice == "6":
        return _run_hand_workspace_calibration(args)
    if choice == "7":
        return _run_hand_mirror_position_calibration(args)
    print(f"[calib] Unknown selection: {choice}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "RGB-only calibration utilities. Use --hand-depth for the recommended "
            "MediaPipe hand-size depth calibration; camera intrinsics are optional."
        )
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0).")
    parser.add_argument("--status", action="store_true", help="Print calibration status and runtime readiness.")
    parser.add_argument("--list-cameras", action="store_true", help="Probe camera indices using the same backend/read logic as main.py.")
    parser.add_argument("--hand-depth", action="store_true", help="Run RGB MediaPipe hand-size depth calibration.")
    parser.add_argument("--hand-workspace", action="store_true", help="Run legacy nonlinear non-neural hand-to-workspace calibration.")
    parser.add_argument("--hand-mirror", action="store_true", help="Run hand-side mirror position calibration paired by pose name with robot_calibrate.py mirror_workspace.")
    parser.add_argument("--hand-positions", action="store_true", help="Alias for --hand-mirror.")
    parser.add_argument(
        "--use-main-camera-defaults",
        action="store_true",
        help="Use the same hand-tracking camera index/backend/properties/retry path as main.py.",
    )
    parser.add_argument("--charuco", action="store_true", help="Run optional ChArUco camera intrinsics calibration.")
    parser.add_argument("--chessboard", action="store_true", help="Alias for optional ChArUco/ArUco chessboard intrinsics calibration.")
    parser.add_argument("--hand-marker", action="store_true", help="Run legacy single hand-marker workspace calibration.")
    parser.add_argument("--camera-index", type=int, default=int(getattr(val, "CHARUCO_CAMERA_INDEX", 0)), help="Camera index for ChArUco mode.")
    parser.add_argument("--squares-x", type=int, default=int(getattr(val, "CHARUCO_SQUARES_X", 7)))
    parser.add_argument("--squares-y", type=int, default=int(getattr(val, "CHARUCO_SQUARES_Y", 5)))
    parser.add_argument("--square-length-m", type=float, default=float(getattr(val, "CHARUCO_SQUARE_LENGTH_M", 0.030)))
    parser.add_argument("--marker-length-m", type=float, default=float(getattr(val, "CHARUCO_MARKER_LENGTH_M", 0.022)))
    parser.add_argument("--dictionary", type=str, default=str(getattr(val, "CHARUCO_DICTIONARY", "DICT_4X4_50")))
    parser.add_argument("--frames", type=int, default=int(getattr(val, "CHARUCO_REQUIRED_FRAMES", 25)))
    parser.add_argument("--output-npz", type=str, default=getattr(val, "CAMERA_CALIBRATION_FILE", "calibration_data/camera_calibration.npz"))
    parser.add_argument("--output-json", type=str, default=getattr(val, "CAMERA_CALIBRATION_JSON", "calibration_data/camera_calibration.json"))
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files.")
    parser.add_argument("--print-board", type=str, default="", help="Write a printable ChArUco board image and exit unless --charuco is also set.")
    parser.add_argument("--show-detections", action="store_true", help="Always draw marker/corner detections.")
    parser.add_argument("--min-markers", type=int, default=int(getattr(val, "CHARUCO_MIN_MARKERS", 4)))
    parser.add_argument("--min-corners", type=int, default=int(getattr(val, "CHARUCO_MIN_CORNERS", 8)))
    parser.add_argument("--capture-delay-s", type=float, default=float(getattr(val, "CHARUCO_CAPTURE_DELAY_S", 0.4)))
    parser.add_argument("--frame-width", type=int, default=int(getattr(val, "CHARUCO_FRAME_WIDTH", 1280)))
    parser.add_argument("--frame-height", type=int, default=int(getattr(val, "CHARUCO_FRAME_HEIGHT", 720)))
    parser.add_argument("--fps", type=int, default=int(getattr(val, "CHARUCO_FPS", 30)))
    parser.add_argument("--min-frames-for-calibration", type=int, default=5)
    parser.add_argument("--ids", type=str, default="", help="Allowed glove marker ids, e.g. '1,5'.")
    parser.add_argument("--aruco", type=int, default=-1, help="cv2.aruco dictionary id. Default uses values.ARUCO_DICT_NAME.")
    parser.add_argument("--marker-size", type=float, default=-1.0, help="Marker size in meters. Default uses values.ARUCO_MARKER_SIZE_M.")
    parser.add_argument("--hold-frames", type=int, default=20, help="Stable frames to average per capture.")
    args = parser.parse_args()

    if len(sys.argv) == 1:
        return _no_arg_menu(args)

    if args.list_cameras:
        return _list_cameras()
    if args.status:
        return _calibration_status()
    if args.hand_depth:
        return _run_hand_depth_calibration(args)
    if args.hand_workspace:
        return _run_hand_workspace_calibration(args)
    if args.hand_mirror or args.hand_positions:
        return _run_hand_mirror_position_calibration(args)
    if args.charuco or args.chessboard or args.print_board:
        return _run_charuco_calibration(args)
    if not args.hand_marker:
        print("For this RGB-only robot control setup, hand-depth calibration is required/recommended.")
        print("Camera intrinsics are optional.")
        print("Run: python3 camera_calibrate.py --hand-depth")
        print("Use --status to check calibration files, or --charuco for optional intrinsics.")
        return _calibration_status()

    allowed_ids = list(_parse_ids(args.ids))
    if not allowed_ids:
        allowed_ids = [
            int(getattr(val, "ARUCO_GLOVE_FRONT_ID", 1)),
            int(getattr(val, "ARUCO_GLOVE_BACK_ID", 5)),
        ]

    if args.aruco >= 0:
        aruco_dict_id = int(args.aruco)
    else:
        dict_name = getattr(val, "ARUCO_DICT_NAME", "DICT_4X4_50")
        aruco_dict_id = int(getattr(cv2.aruco, dict_name))

    marker_size_m = float(args.marker_size) if args.marker_size > 0.0 else float(getattr(val, "ARUCO_MARKER_SIZE_M", 0.03))

    print("\n[calib] Hand-marker calibration")
    print("[calib] This script calibrates workspace bounds using the ArUco tag mounted on the user's hand.")
    print(f"[calib] Allowed marker ids: {allowed_ids}")
    print(f"[calib] Marker size (m): {marker_size_m}")
    print(f"[calib] ArUco dictionary id: {aruco_dict_id}")
    print(f"[calib] Intrinsics file: {INTRINSICS_NPZ}")
    print(f"[calib] Workspace file:  {WORKSPACE_NPZ}")
    print(f"[calib] Extrinsics file: {EXTRINSICS_NPZ}\n")

    if not intrinsics_exists():
        print("[calib] ERROR: camera intrinsics file does not exist.")
        print("[calib] Create calibration_intrinsics.npz first, then rerun calib.py.")
        return 1

    force_recalibration(
        camera_index=int(args.camera),
        allowed_ids=allowed_ids,
        aruco_dict_id=int(aruco_dict_id),
        marker_size_m=float(marker_size_m),
        hold_frames=int(args.hold_frames),
    )

    print("\n[calib] Calibration complete.")
    print(f"[calib] Folder: {CALIB_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())