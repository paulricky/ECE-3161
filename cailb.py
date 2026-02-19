from __future__ import annotations

import os
import time
import argparse
from typing import Dict, Tuple, Optional, Iterable, List

import cv2
import numpy as np


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_DIR = os.path.join(_THIS_DIR, "calibration_data")
os.makedirs(CALIB_DIR, exist_ok=True)

INTRINSICS_NPZ = os.path.join(CALIB_DIR, "calibration_intrinsics.npz")
WORKSPACE_NPZ = os.path.join(CALIB_DIR, "calibration_workspace.npz")
EXTRINSICS_NPZ = os.path.join(CALIB_DIR, "calibration_extrinsics.npz")

ARTIFACTS_DIR = os.path.join(CALIB_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def intrinsics_exists(path: str = INTRINSICS_NPZ) -> bool:
    return os.path.exists(path)


def workspace_exists(path: str = WORKSPACE_NPZ) -> bool:
    return os.path.exists(path)


def extrinsics_exists(path: str = EXTRINSICS_NPZ) -> bool:
    return os.path.exists(path)


def delete_calibration_files() -> None:
    for pth in (INTRINSICS_NPZ, WORKSPACE_NPZ, EXTRINSICS_NPZ):
        try:
            if os.path.exists(pth):
                os.remove(pth)
        except Exception:
            pass


def _validate_intrinsics_dict(intr) -> Tuple[bool, str]:
    if intr is None:
        return False, "intr is None"
    if "mtx" not in intr or "dist" not in intr:
        return False, "missing mtx/dist"
    mtx = np.asarray(intr["mtx"])
    dist = np.asarray(intr["dist"])
    if mtx.shape != (3, 3):
        return False, f"mtx shape {mtx.shape} != (3,3)"
    if dist.size < 4:
        return False, f"dist size {dist.size} < 4"
    if not np.isfinite(mtx).all() or not np.isfinite(dist).all():
        return False, "non-finite values in mtx/dist"
    fx = float(mtx[0, 0])
    fy = float(mtx[1, 1])
    if fx <= 0 or fy <= 0:
        return False, f"fx/fy not positive (fx={fx}, fy={fy})"
    return True, "ok"


def _validate_workspace_dict(ws) -> Tuple[bool, str]:
    if ws is None:
        return False, "ws is None"
    if "H" not in ws:
        return False, "missing H"
    H = np.asarray(ws["H"])
    if H.shape != (3, 3):
        return False, f"H shape {H.shape} != (3,3)"
    if not np.isfinite(H).all():
        return False, "non-finite values in H"
    if abs(float(H[2, 2])) < 1e-9:
        return False, "H[2,2] too small"
    return True, "ok"


def _validate_extrinsics_dict(ext) -> Tuple[bool, str]:
    if ext is None:
        return False, "ext is None"
    if "R" not in ext or "t" not in ext:
        return False, "missing R/t"
    R = np.asarray(ext["R"])
    t = np.asarray(ext["t"])
    if R.shape != (3, 3):
        return False, f"R shape {R.shape} != (3,3)"
    if t.reshape(-1).size != 3:
        return False, f"t size {t.reshape(-1).size} != 3"
    if not np.isfinite(R).all() or not np.isfinite(t).all():
        return False, "non-finite values in R/t"
    return True, "ok"


def _print_calib_status(prefix: str = "") -> None:
    print(f"{prefix}Calibration folder: {CALIB_DIR}")
    print(f"{prefix}Intrinsics file:  {INTRINSICS_NPZ}  exists={intrinsics_exists()}")
    print(f"{prefix}Workspace file:   {WORKSPACE_NPZ}  exists={workspace_exists()}")
    print(f"{prefix}Extrinsics file:  {EXTRINSICS_NPZ}  exists={extrinsics_exists()}")


def load_intrinsics(path: str = INTRINSICS_NPZ):
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path, allow_pickle=True)
    except Exception:
        return None
    if "mtx" not in data or "dist" not in data:
        return None
    out = {
        "mtx": data["mtx"].astype(np.float64),
        "dist": data["dist"].astype(np.float64),
    }
    if "image_size" in data:
        out["image_size"] = tuple(int(x) for x in data["image_size"].tolist())
    if "rms" in data:
        out["rms"] = float(np.array(data["rms"]).reshape(-1)[0])
    if "reproj_err" in data:
        out["reproj_err"] = float(np.array(data["reproj_err"]).reshape(-1)[0])
    return out


def load_workspace(path: str = WORKSPACE_NPZ):
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path, allow_pickle=True)
    except Exception:
        return None
    if "H" not in data:
        return None
    out = {"H": data["H"].astype(np.float64)}
    if "marker_world_mm" in data:
        out["marker_world_mm"] = data["marker_world_mm"]
    if "aruco_dict" in data:
        out["aruco_dict"] = int(np.array(data["aruco_dict"]).reshape(-1)[0])
    if "workspace_mode" in data:
        out["workspace_mode"] = str(np.array(data["workspace_mode"]).reshape(-1)[0])
    return out


def load_extrinsics(path: str = EXTRINSICS_NPZ):
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path, allow_pickle=True)
    except Exception:
        return None
    if "R" not in data or "t" not in data:
        return None
    out = {
        "R": data["R"].astype(np.float64),
        "t": data["t"].astype(np.float64).reshape(3),
    }
    if "camera_height_mm" in data:
        out["camera_height_mm"] = float(np.array(data["camera_height_mm"]).reshape(-1)[0])
    if "rvec" in data:
        out["rvec"] = data["rvec"].astype(np.float64).reshape(3)
    if "tvec" in data:
        out["tvec"] = data["tvec"].astype(np.float64).reshape(3)
    return out


def save_intrinsics(mtx, dist, image_size, rms=None, reproj_err=None):
    np.savez(
        INTRINSICS_NPZ,
        mtx=np.asarray(mtx, dtype=np.float64),
        dist=np.asarray(dist, dtype=np.float64),
        image_size=np.asarray([image_size[0], image_size[1]], dtype=np.int32),
        rms=np.asarray([0.0 if rms is None else float(rms)], dtype=np.float64),
        reproj_err=np.asarray([0.0 if reproj_err is None else float(reproj_err)], dtype=np.float64),
    )


def save_workspace(H, workspace_mode: str, aruco_dict_id: int, marker_world_mm_arr=None):
    kwargs = {
        "H": np.asarray(H, dtype=np.float64),
        "aruco_dict": np.asarray([int(aruco_dict_id)], dtype=np.int32),
        "workspace_mode": np.asarray([str(workspace_mode)], dtype=object),
    }
    if marker_world_mm_arr is not None:
        kwargs["marker_world_mm"] = np.asarray(marker_world_mm_arr, dtype=np.float64)
    np.savez(WORKSPACE_NPZ, **kwargs)


def save_extrinsics(R, t, camera_height_mm=None, rvec=None, tvec=None):
    np.savez(
        EXTRINSICS_NPZ,
        R=np.asarray(R, dtype=np.float64),
        t=np.asarray(t, dtype=np.float64).reshape(3),
        camera_height_mm=np.asarray([0.0 if camera_height_mm is None else float(camera_height_mm)], dtype=np.float64),
        rvec=np.asarray([0.0, 0.0, 0.0] if rvec is None else np.asarray(rvec, dtype=np.float64).reshape(3), dtype=np.float64),
        tvec=np.asarray([0.0, 0.0, 0.0] if tvec is None else np.asarray(tvec, dtype=np.float64).reshape(3), dtype=np.float64),
    )


def undistort_frame(frame, mtx, dist):
    h, w = frame.shape[:2]
    new_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    return cv2.undistort(frame, mtx, dist, None, new_mtx)


def pixel_to_world_mm(H, u, v):
    pt = np.array([[[float(u), float(v)]]], dtype=np.float64)
    out = cv2.perspectiveTransform(pt, H)[0, 0]
    return float(out[0]), float(out[1])


def _aruco_detector(dict_id: int):
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 10
    params.minMarkerPerimeterRate = 0.02
    params.maxMarkerPerimeterRate = 4.0
    params.polygonalApproxAccuracyRate = 0.03
    params.minCornerDistanceRate = 0.02
    params.minDistanceToBorder = 2
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    return aruco_dict, detector


def _make_charuco(board_cols=5, board_rows=7, square_len=0.030, marker_len=0.022, dict_id=cv2.aruco.DICT_5X5_250):
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    board = cv2.aruco.CharucoBoard((board_cols, board_rows), float(square_len), float(marker_len), aruco_dict)
    return aruco_dict, board


def generate_aruco_marker_png(marker_id: int, dict_id: int, size_px: int = 600) -> str:
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    img = cv2.aruco.generateImageMarker(aruco_dict, int(marker_id), int(size_px))
    out_path = os.path.join(ARTIFACTS_DIR, f"aruco_{dict_id}_id{int(marker_id)}_{int(size_px)}px.png")
    cv2.imwrite(out_path, img)
    return out_path


def ensure_aruco_markers_exist(marker_ids: Iterable[int], dict_id: int, size_px: int = 600) -> Dict[int, str]:
    paths: Dict[int, str] = {}
    for mid in marker_ids:
        out_path = os.path.join(ARTIFACTS_DIR, f"aruco_{dict_id}_id{int(mid)}_{int(size_px)}px.png")
        if not os.path.exists(out_path):
            try:
                out_path = generate_aruco_marker_png(int(mid), int(dict_id), int(size_px))
            except Exception:
                pass
        paths[int(mid)] = out_path
    return paths


def generate_marker_sheet_png(
    marker_ids: List[int],
    dict_id: int,
    marker_px: int = 520,
    pad_px: int = 60,
    gap_px: int = 140,
) -> str:
    marker_ids = [int(x) for x in marker_ids]
    if len(marker_ids) != 4:
        raise ValueError("marker_ids must have exactly 4 ids for a 2x2 sheet")

    aruco_dict = cv2.aruco.getPredefinedDictionary(int(dict_id))
    markers = [cv2.aruco.generateImageMarker(aruco_dict, int(mid), int(marker_px)) for mid in marker_ids]

    sheet_w = int(pad_px) * 2 + int(marker_px) * 2 + int(gap_px)
    sheet_h = int(pad_px) * 2 + int(marker_px) * 2 + int(gap_px)
    sheet = np.full((sheet_h, sheet_w), 255, dtype=np.uint8)

    x0 = int(pad_px)
    y0 = int(pad_px)
    x1 = int(pad_px) + int(marker_px) + int(gap_px)
    y1 = int(pad_px) + int(marker_px) + int(gap_px)

    sheet[y0:y0 + marker_px, x0:x0 + marker_px] = markers[0]
    sheet[y0:y0 + marker_px, x1:x1 + marker_px] = markers[1]
    sheet[y1:y1 + marker_px, x0:x0 + marker_px] = markers[2]
    sheet[y1:y1 + marker_px, x1:x1 + marker_px] = markers[3]

    out_path = os.path.join(ARTIFACTS_DIR, f"aruco_sheet_{dict_id}_{marker_ids[0]}_{marker_ids[1]}_{marker_ids[2]}_{marker_ids[3]}_{marker_px}px.png")
    cv2.imwrite(out_path, sheet)
    return out_path


def generate_charuco_board_png(
    dict_id: int = cv2.aruco.DICT_5X5_250,
    cols: int = 5,
    rows: int = 7,
    square_px: int = 140,
    margin_px: int = 20,
    marker_to_square_ratio: float = 0.7333333333,
) -> str:
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    board = cv2.aruco.CharucoBoard((int(cols), int(rows)), 1.0, float(marker_to_square_ratio), aruco_dict)
    w = int(cols) * int(square_px) + 2 * int(margin_px)
    h = int(rows) * int(square_px) + 2 * int(margin_px)
    img = board.generateImage((w, h), marginSize=int(margin_px), borderBits=1)
    out_path = os.path.join(ARTIFACTS_DIR, f"charuco_{dict_id}_{cols}x{rows}_{square_px}px.png")
    cv2.imwrite(out_path, img)
    return out_path


def _enhance_gray(gray: np.ndarray) -> np.ndarray:
    g = gray
    if g.dtype != np.uint8:
        g = np.clip(g, 0, 255).astype(np.uint8)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    g = cv2.equalizeHist(g)
    return g


def run_intrinsics_calibration(
    cap,
    target_frames=25,
    cooldown_s=0.15,
    aruco_dict_id=cv2.aruco.DICT_5X5_250,
    charuco_cols=5,
    charuco_rows=7,
    square_len_m=0.030,
    marker_len_m=0.022,
    min_charuco_corners=10,
):
    print("\n=== Intrinsics Calibration (CHARUCO) ===")
    print("SPACE=capture  a=auto  r=reset  q=quit")
    print("After enough captures, intrinsics will be solved and saved.\n")

    try:
        cb_path = generate_charuco_board_png(
            dict_id=int(aruco_dict_id),
            cols=int(charuco_cols),
            rows=int(charuco_rows),
            marker_to_square_ratio=float(marker_len_m / square_len_m),
        )
        print("Charuco board image:", cb_path)
    except Exception:
        pass

    aruco_dict, board = _make_charuco(
        board_cols=int(charuco_cols),
        board_rows=int(charuco_rows),
        square_len=float(square_len_m),
        marker_len=float(marker_len_m),
        dict_id=int(aruco_dict_id),
    )
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    all_charuco_corners = []
    all_charuco_ids = []
    image_size = None

    good = 0
    last_capture_t = 0.0
    auto_capture = False

    while True:
        ok, frame = cap.read()
        if not ok:
            return False

        vis = frame.copy()
        gray = _enhance_gray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        h, w = gray.shape[:2]
        image_size = (w, h)

        corners, ids, _ = detector.detectMarkers(gray)
        ok_i = False
        status = "ARUCO: none"

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            ok_i, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                markerCorners=corners,
                markerIds=ids,
                image=gray,
                board=board,
            )
            if ok_i and charuco_ids is not None:
                if len(charuco_ids) >= int(min_charuco_corners):
                    cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)
                    status = f"CHARUCO OK ({len(charuco_ids)} corners)"
                else:
                    ok_i = False
                    status = f"CHARUCO low ({len(charuco_ids)}/{min_charuco_corners})"
            else:
                ok_i = False
                status = "CHARUCO: interpolate failed"

        cv2.putText(vis, f"Frames: {good}/{target_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, status, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if ok_i else (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, f"auto: {'ON' if auto_capture else 'OFF'}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("calib.py - Intrinsics (Charuco)", vis)
        key = cv2.waitKey(1) & 0xFF
        now = time.time()

        if key == ord("q"):
            return False
        if key == ord("a"):
            auto_capture = not auto_capture
        if key == ord("r"):
            all_charuco_corners.clear()
            all_charuco_ids.clear()
            good = 0

        capture_now = (key == ord(" ")) or (auto_capture and ok_i)
        if capture_now and ok_i:
            if now - last_capture_t >= float(cooldown_s):
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
                good += 1
                last_capture_t = now

        if good >= int(target_frames) and len(all_charuco_corners) >= 8:
            try:
                ret, mtx, dist, _rvecs, _tvecs = cv2.aruco.calibrateCameraCharuco(
                    charucoCorners=all_charuco_corners,
                    charucoIds=all_charuco_ids,
                    board=board,
                    imageSize=image_size,
                    cameraMatrix=None,
                    distCoeffs=None,
                )
            except Exception:
                continue

            save_intrinsics(mtx, dist, image_size, rms=float(ret), reproj_err=None)
            intr = load_intrinsics(INTRINSICS_NPZ)
            okv, msg = _validate_intrinsics_dict(intr)
            print("\n[calib] Saved intrinsics:", INTRINSICS_NPZ)
            print("[calib] Intrinsics validation:", "OK" if okv else f"FAIL ({msg})")
            return okv


def run_workspace_calibration_markers(
    cap,
    marker_world_mm: Dict[int, Tuple[float, float]],
    aruco_dict_id=cv2.aruco.DICT_5X5_250,
    use_undistort=True,
):
    intr = load_intrinsics(INTRINSICS_NPZ) if use_undistort else None
    mtx = intr["mtx"] if intr is not None else None
    dist = intr["dist"] if intr is not None else None

    if len(marker_world_mm) < 4:
        print("[calib] Need at least 4 markers for workspace calibration.")
        return False

    marker_ids = sorted(int(k) for k in marker_world_mm.keys())
    if len(marker_ids) != 4:
        print("[calib] This phone-sheet workflow expects exactly 4 markers (2x2).")
        return False

    ensure_aruco_markers_exist(marker_ids, int(aruco_dict_id), size_px=600)
    sheet_path = generate_marker_sheet_png(marker_ids, int(aruco_dict_id), marker_px=520, pad_px=60, gap_px=140)

    print("\n=== Workspace Calibration (4 markers on ONE sheet) ===")
    print("After intrinsics are saved, DISPLAY THIS IMAGE FULLSCREEN on your phone (no zoom/stretch):")
    print("  ", sheet_path)
    print("Hold the phone flat on the workspace plane; keep the whole sheet visible to the camera.")
    print("Press 'c' to compute+save homography (pixel->mm). Press 'q' to quit.\n")

    _aruco_dict, detector = _aruco_detector(int(aruco_dict_id))

    best_H = None
    best_inliers = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            return False

        if (mtx is not None) and (dist is not None) and use_undistort:
            frame_u = undistort_frame(frame, mtx, dist)
        else:
            frame_u = frame

        vis = frame_u.copy()
        gray = _enhance_gray(cv2.cvtColor(frame_u, cv2.COLOR_BGR2GRAY))

        corners, ids, _ = detector.detectMarkers(gray)

        found_centers: Dict[int, Tuple[float, float]] = {}
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            ids_list = ids.flatten().tolist()
            for c, mid in zip(corners, ids_list):
                pts = c[0]
                center = pts.mean(axis=0)
                found_centers[int(mid)] = (float(center[0]), float(center[1]))

        img_pts = []
        world_pts = []
        for mid in marker_ids:
            if mid in found_centers:
                cx, cy = found_centers[mid]
                wx, wy = marker_world_mm[mid]
                img_pts.append([cx, cy])
                world_pts.append([wx, wy])
                cv2.circle(vis, (int(cx), int(cy)), 4, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.putText(vis, f"ID{mid}", (int(cx) + 6, int(cy) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        inliers = 0
        if len(img_pts) >= 4:
            img_pts_np = np.array(img_pts, dtype=np.float64)
            world_pts_np = np.array(world_pts, dtype=np.float64)
            H, mask = cv2.findHomography(img_pts_np, world_pts_np, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            inliers = int(mask.sum()) if mask is not None else 0
            if H is not None and inliers > best_inliers:
                best_inliers = inliers
                best_H = H

        cv2.putText(vis, f"Markers seen: {len(img_pts)}/4", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, f"Best inliers: {max(best_inliers, 0)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, "c=save  q=quit", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("calib.py - Workspace (marker sheet)", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            return False

        if key == ord("c"):
            if best_H is None:
                print("[calib] No homography yet. Make sure all 4 markers are visible.")
                continue

            marker_arr = np.array(
                [[int(mid), float(marker_world_mm[mid][0]), float(marker_world_mm[mid][1])] for mid in marker_ids],
                dtype=np.float64,
            )
            save_workspace(best_H, workspace_mode="markers_sheet", aruco_dict_id=int(aruco_dict_id), marker_world_mm_arr=marker_arr)

            ws = load_workspace(WORKSPACE_NPZ)
            okv, msg = _validate_workspace_dict(ws)
            print("\n[calib] Saved workspace:", WORKSPACE_NPZ)
            print("[calib] Workspace validation:", "OK" if okv else f"FAIL ({msg})")
            return okv


def ensure_calibration(
    cap=None,
    marker_world_mm: Optional[Dict[int, Tuple[float, float]]] = None,
    aruco_dict_id: int = cv2.aruco.DICT_5X5_250,
    verbose: bool = True,
    **_ignored_kwargs,
):
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

    if not ok_i or not ok_w:
        raise RuntimeError(
            "Calibration data not found or invalid.\n"
            f"  intrinsics: {INTRINSICS_NPZ} -> {'OK' if ok_i else 'FAIL'} ({msg_i})\n"
            f"  workspace:  {WORKSPACE_NPZ} -> {'OK' if ok_w else 'FAIL'} ({msg_w})\n"
            "Run cailb.py directly to regenerate calibration_data/*.npz."
        )

    return intr, ws, (ext if ok_e else None)


def force_recalibration(
    camera_index: int = 0,
    marker_world_mm: Optional[Dict[int, Tuple[float, float]]] = None,
    aruco_dict_id: int = cv2.aruco.DICT_5X5_250,
    intrinsics_target_frames: int = 25,
):
    if marker_world_mm is None:
        marker_world_mm = {}

    delete_calibration_files()

    cap = cv2.VideoCapture(int(camera_index))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {camera_index}.")

    try:
        ok_intr = run_intrinsics_calibration(
            cap,
            target_frames=int(intrinsics_target_frames),
            aruco_dict_id=int(aruco_dict_id),
        )
        if not ok_intr:
            return None, None, None

        marker_ids = sorted(int(k) for k in marker_world_mm.keys())
        if len(marker_ids) == 4:
            sheet_path = generate_marker_sheet_png(marker_ids, int(aruco_dict_id), marker_px=520, pad_px=60, gap_px=140)
            print("\n[calib] Marker sheet generated (show this on your phone next):")
            print("  ", sheet_path)

        ok_ws = run_workspace_calibration_markers(
            cap,
            marker_world_mm=marker_world_mm,
            aruco_dict_id=int(aruco_dict_id),
            use_undistort=True,
        )

        intr = load_intrinsics(INTRINSICS_NPZ)
        ws = load_workspace(WORKSPACE_NPZ) if ok_ws else None
        ext = load_extrinsics(EXTRINSICS_NPZ)

        ok_i, msg_i = _validate_intrinsics_dict(intr)
        ok_w, msg_w = _validate_workspace_dict(ws)
        ok_e, msg_e = _validate_extrinsics_dict(ext)

        print("\n[calib] FINAL SAVE CHECK")
        _print_calib_status(prefix="[calib] ")
        print(f"[calib] Intrinsics validation: {'OK' if ok_i else f'FAIL ({msg_i})'}")
        print(f"[calib] Workspace  validation: {'OK' if ok_w else f'FAIL ({msg_w})'}")
        print(f"[calib] Extrinsics validation: {'OK' if ok_e else f'FAIL ({msg_e})'}")

        return intr, ws, (ext if ok_e else None)
    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def _parse_marker_world(s: str) -> Dict[int, Tuple[float, float]]:
    out: Dict[int, Tuple[float, float]] = {}
    if not s:
        return out
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        a = p.split(":")
        if len(a) != 2:
            continue
        mid = int(a[0].strip())
        xy = a[1].split("|")
        if len(xy) != 2:
            continue
        out[mid] = (float(xy[0]), float(xy[1]))
    return out


def _default_marker_world_mm() -> Dict[int, Tuple[float, float]]:
    return {
        0: (0.0, 0.0),
        1: (300.0, 0.0),
        2: (0.0, 300.0),
        3: (300.0, 300.0),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibration tool (writes into ./calibration_data/).")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0).")
    parser.add_argument("--aruco", type=int, default=int(cv2.aruco.DICT_5X5_250), help="cv2.aruco dictionary id.")
    parser.add_argument("--frames", type=int, default=25, help="Target captures for intrinsics.")
    parser.add_argument("--markers", type=str, default="", help="Mapping like: 0:0|0,1:300|0,2:0|300,3:300|300 (mm).")
    parser.add_argument("--gen_sheet_only", action="store_true", help="Generate the 2x2 marker sheet PNG and exit.")
    args = parser.parse_args()

    marker_world_mm = _parse_marker_world(args.markers)
    if not marker_world_mm:
        marker_world_mm = _default_marker_world_mm()

    marker_ids = sorted(int(k) for k in marker_world_mm.keys())
    if len(marker_ids) != 4:
        raise RuntimeError("This workflow expects exactly 4 markers in --markers (2x2 marker sheet).")

    ensure_aruco_markers_exist(marker_ids, int(args.aruco), size_px=600)
    sheet_path = generate_marker_sheet_png(marker_ids, int(args.aruco), marker_px=520, pad_px=60, gap_px=140)
    print("\n[calib] Marker sheet image (display this fullscreen on your phone; no zoom/stretch):")
    print("  ", sheet_path)

    if args.gen_sheet_only:
        print("\n[calib] Done (gen_sheet_only).")
    else:
        intr, ws, ext = force_recalibration(
            camera_index=int(args.camera),
            marker_world_mm=marker_world_mm,
            aruco_dict_id=int(args.aruco),
            intrinsics_target_frames=int(args.frames),
        )
        print("\n[calib] Calibration complete.")
        print("Folder:", CALIB_DIR)
