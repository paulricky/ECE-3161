"""Two-phase calibration for the top-down pick-and-place camera.

Phase 1 (--phase intrinsics):
    Capture a set of checkerboard images and solve for the camera matrix K
    and distortion coefficients. Saves to calibration_data/topdown_intrinsics.npz.

Phase 2 (--phase extrinsics):
    With intrinsics known, detect a set of ArUco anchor markers placed at
    known workspace (x, y, z) positions. Solve a single PnP on the full set
    of 16 detected corners to recover `T_cam_from_ws` (rotation + translation
    such that `p_cam = R @ p_ws + t`). Saves to
    calibration_data/topdown_extrinsics.npz.

All anchor markers must be physically oriented so their printed "up" edge
points along the workspace +y axis (away from the robot base). The anchor
map is read from values.PICKPLACE_CALIB_ANCHOR_MAP.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import values as val
from aruco_marker import (
    aruco_dict_id_from_name,
    build_aruco_detector,
    marker_object_points,
)


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_path(configured: str, default_subpath: str) -> str:
    raw = str(configured or default_subpath)
    if os.path.isabs(raw):
        return raw
    return os.path.join(_THIS_DIR, raw)


def _intrinsics_path() -> str:
    return _resolve_path(
        getattr(val, "PICKPLACE_TOPDOWN_INTRINSICS_FILE", "calibration_data/topdown_intrinsics.npz"),
        "calibration_data/topdown_intrinsics.npz",
    )


def _extrinsics_path() -> str:
    return _resolve_path(
        getattr(val, "PICKPLACE_TOPDOWN_EXTRINSICS_FILE", "calibration_data/topdown_extrinsics.npz"),
        "calibration_data/topdown_extrinsics.npz",
    )


def _open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(index))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {index}")
    return cap


def _draw_hud(frame, lines: Sequence[str]) -> None:
    for i, line in enumerate(lines):
        y = 30 + 26 * i
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (255, 255, 255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Phase 1: intrinsics (checkerboard)
# ---------------------------------------------------------------------------


def _board_object_points(pattern_size: Tuple[int, int], square_size: float) -> np.ndarray:
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), dtype=np.float64)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid * float(square_size)
    return objp


def phase1_intrinsics(
    camera_index: int,
    pattern_size: Tuple[int, int],
    square_size: float,
    save_path: str,
    min_captures: int,
) -> int:
    print(f"[topdown_calibrate] phase 1 (intrinsics)")
    print(f"  camera index = {camera_index}")
    print(f"  pattern inner corners = {pattern_size}  square = {square_size} m")
    print(f"  minimum captures = {min_captures}")
    print(f"  save path = {save_path}")
    print()
    print("  SPACE = accept current detection   D = done (calibrate)   Q = quit")
    print()

    cap = _open_camera(camera_index)
    objp = _board_object_points(pattern_size, square_size)
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objpoints: List[np.ndarray] = []
    imgpoints: List[np.ndarray] = []
    image_size: Optional[Tuple[int, int]] = None

    window = "topdown_calibrate - intrinsics"
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[topdown_calibrate] camera read failed")
                return 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if image_size is None:
                image_size = (gray.shape[1], gray.shape[0])

            found, corners = cv2.findChessboardCorners(
                gray, pattern_size,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )

            preview = frame.copy()
            corners_refined: Optional[np.ndarray] = None
            if found:
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
                cv2.drawChessboardCorners(preview, pattern_size, corners_refined, True)

            status = "FOUND" if found else "-- searching --"
            _draw_hud(preview, [
                f"captures: {len(objpoints)} / {min_captures} (more = better)",
                f"status: {status}",
                "SPACE = accept   D = done   Q = quit",
            ])
            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("[topdown_calibrate] intrinsics cancelled by user")
                return 1
            if key == ord(' ') and found and corners_refined is not None:
                objpoints.append(objp.copy())
                imgpoints.append(corners_refined.copy())
                print(f"[topdown_calibrate] captured {len(objpoints)}")
            if key == ord('d'):
                if len(objpoints) < min_captures:
                    print(f"[topdown_calibrate] need at least {min_captures} captures "
                          f"(have {len(objpoints)})")
                    continue
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"[topdown_calibrate] calibrating from {len(objpoints)} captures ...")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None,
    )

    # Per-view reprojection error.
    total_err = 0.0
    total_pts = 0
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        err = float(np.linalg.norm(imgpoints[i] - proj.reshape(-1, 2)))
        total_err += err * err
        total_pts += len(objpoints[i])
    reproj_err = math.sqrt(total_err / max(total_pts, 1))

    print(f"[topdown_calibrate] rms = {rms:.4f}   mean reproj err = {reproj_err:.4f} px")
    if rms > 1.0:
        print(f"[topdown_calibrate] WARNING: rms > 1.0 px. Consider recapturing "
              f"with more varied angles / better lighting.")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(
        save_path,
        camera_matrix=np.asarray(K, dtype=np.float64),
        dist_coeffs=np.asarray(dist, dtype=np.float64),
        image_size=np.asarray(image_size, dtype=np.int32),
        rms=np.asarray([rms], dtype=np.float64),
        reproj_err=np.asarray([reproj_err], dtype=np.float64),
    )
    print(f"[topdown_calibrate] saved intrinsics -> {save_path}")
    return 0


# ---------------------------------------------------------------------------
# Phase 2: extrinsics (ArUco anchors)
# ---------------------------------------------------------------------------


@dataclass
class AnchorObservation:
    marker_id: int
    workspace_center: np.ndarray
    image_corners: np.ndarray


def _anchor_corners_workspace(center: np.ndarray, marker_size_m: float) -> np.ndarray:
    s = float(marker_size_m) / 2.0
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    return np.array(
        [
            [cx - s, cy + s, cz],  # TL
            [cx + s, cy + s, cz],  # TR
            [cx + s, cy - s, cz],  # BR
            [cx - s, cy - s, cz],  # BL
        ],
        dtype=np.float64,
    )


def _detect_anchor_corners(
    detector, frame, anchor_ids: Sequence[int]
) -> Optional[Dict[int, np.ndarray]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _rej = detector.detectMarkers(gray)
    if ids is None:
        return None
    ids_list = [int(x) for x in ids.flatten().tolist()]
    out: Dict[int, np.ndarray] = {}
    for want in anchor_ids:
        if want not in ids_list:
            return None
        out[int(want)] = np.asarray(
            corners[ids_list.index(int(want))], dtype=np.float64
        ).reshape(4, 2)
    return out


def _capture_stable_anchor_corners(
    cap,
    detector,
    anchor_ids: Sequence[int],
    stable_frames: int,
    window: str,
) -> Optional[Dict[int, np.ndarray]]:
    accumulator: Dict[int, List[np.ndarray]] = {int(i): [] for i in anchor_ids}
    captured = 0
    print(f"[topdown_calibrate] hold the camera and anchors steady. "
          f"collecting {stable_frames} frames where ALL anchors are detected ...")
    while captured < stable_frames:
        ok, frame = cap.read()
        if not ok:
            print("[topdown_calibrate] camera read failed during capture")
            return None
        preview = frame.copy()
        detections = _detect_anchor_corners(detector, frame, anchor_ids)
        if detections is not None:
            for mid, pts in detections.items():
                accumulator[mid].append(pts)
                cv2.polylines(preview, [pts.astype(np.int32).reshape(-1, 1, 2)], True,
                              (0, 255, 0), 2, cv2.LINE_AA)
                cx = float(pts[:, 0].mean())
                cy = float(pts[:, 1].mean())
                cv2.putText(preview, f"id={mid}", (int(cx) + 6, int(cy) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
            captured += 1
        status = f"stable frames: {captured}/{stable_frames}"
        missing = []
        if detections is None:
            missing = [i for i in anchor_ids]
        _draw_hud(preview, [
            status,
            f"required anchor ids: {list(anchor_ids)}",
            f"missing: {missing}" if missing else "all anchors visible",
            "Q = quit",
        ])
        cv2.imshow(window, preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[topdown_calibrate] extrinsics capture cancelled")
            return None

    # Median across captured frames gives robustness to jitter.
    out: Dict[int, np.ndarray] = {}
    for mid, frames_list in accumulator.items():
        stack = np.stack(frames_list, axis=0)  # (N, 4, 2)
        out[mid] = np.median(stack, axis=0)
    return out


def _compute_table_z_from_anchors(
    anchor_map: Sequence[Tuple[int, np.ndarray]],
    anchor_corners: Dict[int, np.ndarray],
    marker_size_m: float,
    K: np.ndarray,
    dist: np.ndarray,
    R_cam_from_ws: np.ndarray,
    t_cam_from_ws: np.ndarray,
) -> float:
    """Re-pose each anchor individually and transform its center into workspace
    coordinates; the mean of the z components gives the observed table plane.
    """
    R_ws_cam = R_cam_from_ws.T
    origin_ws = -R_ws_cam @ t_cam_from_ws
    zs: List[float] = []
    for marker_id, _center in anchor_map:
        img_pts = anchor_corners[int(marker_id)]
        ok, rvec, tvec = cv2.solvePnP(
            marker_object_points(marker_size_m),
            img_pts,
            K, dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            continue
        t_cam_marker = np.asarray(tvec, dtype=np.float64).reshape(3)
        p_ws = origin_ws + R_ws_cam @ t_cam_marker
        zs.append(float(p_ws[2]))
    if not zs:
        return float("nan")
    return float(np.mean(zs))


def phase2_extrinsics(
    camera_index: int,
    intrinsics_path: str,
    extrinsics_path: str,
    anchor_map: Sequence[Tuple[int, np.ndarray]],
    marker_size_m: float,
    aruco_dict_id: int,
    stable_frames: int,
    reproj_warn_px: float,
) -> int:
    print(f"[topdown_calibrate] phase 2 (extrinsics)")
    print(f"  camera index = {camera_index}")
    print(f"  intrinsics   = {intrinsics_path}")
    print(f"  save path    = {extrinsics_path}")
    print(f"  anchor ids   = {[a[0] for a in anchor_map]}")
    print(f"  marker size  = {marker_size_m} m")
    print(f"  stable frames= {stable_frames}")
    print()

    if not os.path.exists(intrinsics_path):
        print(f"[topdown_calibrate] intrinsics file not found: {intrinsics_path}")
        print("  Run --phase intrinsics first.")
        return 1

    data = np.load(intrinsics_path, allow_pickle=True)
    K = None
    for key in ("camera_matrix", "mtx", "K"):
        if key in data:
            K = np.asarray(data[key], dtype=np.float64).reshape(3, 3)
            break
    if K is None:
        print(f"[topdown_calibrate] intrinsics file missing camera matrix")
        return 1
    dist = None
    for key in ("dist_coeffs", "dist", "distortion_coefficients"):
        if key in data:
            dist = np.asarray(data[key], dtype=np.float64).reshape(-1, 1)
            break
    if dist is None:
        print(f"[topdown_calibrate] intrinsics file missing distortion coefficients")
        return 1

    detector = build_aruco_detector(aruco_dict_id)
    anchor_ids = [int(a[0]) for a in anchor_map]

    cap = _open_camera(camera_index)
    window = "topdown_calibrate - extrinsics"
    try:
        corners_per_anchor = _capture_stable_anchor_corners(
            cap, detector, anchor_ids, stable_frames, window,
        )
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if corners_per_anchor is None:
        return 1

    # Build concatenated (obj_pts, img_pts) for a single PnP solve.
    obj_pts_list = []
    img_pts_list = []
    for mid, center_xyz in anchor_map:
        corners_ws = _anchor_corners_workspace(np.asarray(center_xyz, dtype=np.float64),
                                                marker_size_m)
        obj_pts_list.append(corners_ws)
        img_pts_list.append(corners_per_anchor[int(mid)])
    obj_pts = np.concatenate(obj_pts_list, axis=0).astype(np.float64)
    img_pts = np.concatenate(img_pts_list, axis=0).astype(np.float64)

    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        print("[topdown_calibrate] solvePnP (ITERATIVE) failed, trying IPPE")
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        print("[topdown_calibrate] solvePnP failed for the aggregate anchors")
        return 1

    R_cam_from_ws, _ = cv2.Rodrigues(rvec)
    t_cam_from_ws = np.asarray(tvec, dtype=np.float64).reshape(3)

    # Per-anchor reprojection error.
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    per_pt_err = np.linalg.norm(img_pts - proj, axis=1)
    per_anchor_err = per_pt_err.reshape(-1, 4).mean(axis=1)
    max_err = float(per_pt_err.max())
    mean_err = float(per_pt_err.mean())

    print(f"[topdown_calibrate] mean reproj err = {mean_err:.3f} px   "
          f"max = {max_err:.3f} px")
    for (mid, _), e in zip(anchor_map, per_anchor_err):
        flag = "  OK" if e < reproj_warn_px else "  WARN"
        print(f"  anchor id={mid}: reproj err = {e:.3f} px{flag}")
    if max_err > reproj_warn_px:
        print(f"[topdown_calibrate] WARNING: max reproj err > {reproj_warn_px:.1f} px. "
              f"Check anchor position measurements and orientation consistency.")

    table_z_fitted = _compute_table_z_from_anchors(
        anchor_map, corners_per_anchor, marker_size_m,
        K, dist, R_cam_from_ws, t_cam_from_ws,
    )
    print(f"[topdown_calibrate] table_z fitted from anchors = {table_z_fitted:.4f} m "
          f"(configured PICKPLACE_TABLE_Z_M = "
          f"{getattr(val, 'PICKPLACE_TABLE_Z_M', 0.02):.4f})")

    anchor_ids_arr = np.asarray([int(a[0]) for a in anchor_map], dtype=np.int32)
    anchor_xyz_arr = np.asarray([np.asarray(a[1], dtype=np.float64).reshape(3)
                                 for a in anchor_map], dtype=np.float64)

    os.makedirs(os.path.dirname(extrinsics_path), exist_ok=True)
    np.savez(
        extrinsics_path,
        R=np.asarray(R_cam_from_ws, dtype=np.float64),
        t=np.asarray(t_cam_from_ws, dtype=np.float64),
        rvec=np.asarray(rvec, dtype=np.float64).reshape(3),
        tvec=np.asarray(tvec, dtype=np.float64).reshape(3),
        anchor_ids=anchor_ids_arr,
        anchor_xyz_workspace=anchor_xyz_arr,
        per_anchor_reproj_err_px=per_anchor_err.astype(np.float64),
        mean_reproj_err_px=np.asarray([mean_err], dtype=np.float64),
        max_reproj_err_px=np.asarray([max_err], dtype=np.float64),
        table_z_fitted=np.asarray([table_z_fitted], dtype=np.float64),
        marker_size_m=np.asarray([float(marker_size_m)], dtype=np.float64),
        aruco_dict_id=np.asarray([int(aruco_dict_id)], dtype=np.int32),
    )
    print(f"[topdown_calibrate] saved extrinsics -> {extrinsics_path}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_anchor_map() -> List[Tuple[int, np.ndarray]]:
    raw = getattr(val, "PICKPLACE_CALIB_ANCHOR_MAP", None)
    if not raw:
        raise RuntimeError("values.PICKPLACE_CALIB_ANCHOR_MAP is empty. "
                           "Configure at least 3 anchors before running extrinsics.")
    out: List[Tuple[int, np.ndarray]] = []
    for entry in raw:
        mid, xyz = entry[0], entry[1]
        out.append((int(mid), np.asarray(xyz, dtype=np.float64).reshape(3)))
    if len(out) < 3:
        raise RuntimeError(f"Need at least 3 anchors; got {len(out)}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Two-phase calibration for the top-down pick-and-place camera."
    )
    parser.add_argument("--phase", choices=("intrinsics", "extrinsics", "both"),
                        default="both", help="Which calibration phase to run.")
    parser.add_argument("--camera-index", type=int,
                        default=int(getattr(val, "PICKPLACE_CAMERA_INDEX", 1)),
                        help="OpenCV VideoCapture index.")
    parser.add_argument("--board-rows", type=int,
                        default=int(getattr(val, "PICKPLACE_CALIB_CHESSBOARD_ROWS", 7)),
                        help="Number of inner-corner rows on the checkerboard.")
    parser.add_argument("--board-cols", type=int,
                        default=int(getattr(val, "PICKPLACE_CALIB_CHESSBOARD_COLS", 9)),
                        help="Number of inner-corner columns on the checkerboard.")
    parser.add_argument("--square-size", type=float,
                        default=float(getattr(val, "PICKPLACE_CALIB_SQUARE_SIZE_M", 0.025)),
                        help="Checkerboard square size in meters.")
    parser.add_argument("--min-captures", type=int,
                        default=int(getattr(val, "PICKPLACE_CALIB_MIN_INTRINSICS_CAPTURES", 12)),
                        help="Minimum accepted captures for intrinsics.")
    parser.add_argument("--marker-size", type=float,
                        default=float(getattr(val, "PICKPLACE_CALIB_ANCHOR_MARKER_SIZE_M", 0.04)),
                        help="Anchor marker side length in meters.")
    parser.add_argument("--aruco-dict",
                        default=str(getattr(val, "PICKPLACE_CALIB_ANCHOR_MARKER_DICT",
                                            "DICT_6X6_250")),
                        help="OpenCV ArUco dictionary name for anchor markers.")
    parser.add_argument("--stable-frames", type=int,
                        default=int(getattr(val, "PICKPLACE_CALIB_STABLE_FRAMES", 20)),
                        help="Frames to average for extrinsic corner positions.")
    parser.add_argument("--reproj-warn-px", type=float,
                        default=float(getattr(val, "PICKPLACE_CALIB_REPROJ_WARN_PX", 3.0)),
                        help="Warn threshold for per-anchor reprojection error.")
    args = parser.parse_args()

    pattern_size = (int(args.board_cols), int(args.board_rows))
    intrinsics_path = _intrinsics_path()
    extrinsics_path = _extrinsics_path()

    rc = 0
    if args.phase in ("intrinsics", "both"):
        rc = phase1_intrinsics(
            camera_index=int(args.camera_index),
            pattern_size=pattern_size,
            square_size=float(args.square_size),
            save_path=intrinsics_path,
            min_captures=int(args.min_captures),
        )
        if rc != 0:
            return rc

    if args.phase in ("extrinsics", "both"):
        aruco_dict_id = aruco_dict_id_from_name(args.aruco_dict)
        anchor_map = _resolve_anchor_map()
        rc = phase2_extrinsics(
            camera_index=int(args.camera_index),
            intrinsics_path=intrinsics_path,
            extrinsics_path=extrinsics_path,
            anchor_map=anchor_map,
            marker_size_m=float(args.marker_size),
            aruco_dict_id=int(aruco_dict_id),
            stable_frames=int(args.stable_frames),
            reproj_warn_px=float(args.reproj_warn_px),
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
