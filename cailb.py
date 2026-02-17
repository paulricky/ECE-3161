from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import cv2
import numpy as np

import values as val

CALIB_DIR = os.path.join(os.path.dirname(__file__), "calibration_data")
INTR_FILE = os.path.join(CALIB_DIR, "camera_intrinsics.json")
WS_FILE = os.path.join(CALIB_DIR, "workspace.json")
EXT_FILE = os.path.join(CALIB_DIR, "camera_extrinsics.json")


def _ensure_dir() -> None:
    os.makedirs(CALIB_DIR, exist_ok=True)


def _np_to_list(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [_np_to_list(v) for v in x]
    if isinstance(x, dict):
        return {k: _np_to_list(v) for k, v in x.items()}
    return x


def _list_to_np(x: Any, dtype=np.float64) -> Any:
    if isinstance(x, list):
        try:
            arr = np.array(x, dtype=dtype)
            return arr
        except Exception:
            return [_list_to_np(v, dtype=dtype) for v in x]
    if isinstance(x, dict):
        return {k: _list_to_np(v, dtype=dtype) for k, v in x.items()}
    return x


def save_intrinsics(intr: Dict[str, Any]) -> None:
    _ensure_dir()
    with open(INTR_FILE, "w") as f:
        json.dump(_np_to_list(intr), f, indent=2)


def save_workspace(ws: Dict[str, Any]) -> None:
    _ensure_dir()
    with open(WS_FILE, "w") as f:
        json.dump(_np_to_list(ws), f, indent=2)


def save_extrinsics(ext: Dict[str, Any]) -> None:
    _ensure_dir()
    with open(EXT_FILE, "w") as f:
        json.dump(_np_to_list(ext), f, indent=2)


def load_intrinsics() -> Optional[Dict[str, Any]]:
    if not os.path.exists(INTR_FILE):
        return None
    with open(INTR_FILE, "r") as f:
        data = json.load(f)
    data = _list_to_np(data)
    if "K" in data:
        data["K"] = np.array(data["K"], dtype=np.float64)
    if "dist" in data:
        data["dist"] = np.array(data["dist"], dtype=np.float64).reshape(-1, 1)
    return data


def load_workspace() -> Optional[Dict[str, Any]]:
    if not os.path.exists(WS_FILE):
        return None
    with open(WS_FILE, "r") as f:
        data = json.load(f)
    data = _list_to_np(data)
    if "H_img_to_ws" in data:
        data["H_img_to_ws"] = np.array(data["H_img_to_ws"], dtype=np.float64)
    if "H_ws_to_img" in data:
        data["H_ws_to_img"] = np.array(data["H_ws_to_img"], dtype=np.float64)
    if "rect_img_corners" in data:
        data["rect_img_corners"] = np.array(data["rect_img_corners"], dtype=np.float64).reshape(4, 2)
    return data


def load_extrinsics() -> Optional[Dict[str, Any]]:
    if not os.path.exists(EXT_FILE):
        return None
    with open(EXT_FILE, "r") as f:
        data = json.load(f)
    data = _list_to_np(data)
    if "R" in data:
        data["R"] = np.array(data["R"], dtype=np.float64).reshape(3, 3)
    if "t" in data:
        data["t"] = np.array(data["t"], dtype=np.float64).reshape(3, 1)
    return data


def generate_aruco_marker_png(
    out_png: str = "aruco_marker_id0.png",
    marker_id: int = 0,
    dict_name: int = cv2.aruco.DICT_5X5_250,
    size_px: int = 900,
    border_bits: int = 1,
) -> str:
    d = cv2.aruco.getPredefinedDictionary(dict_name)
    img = cv2.aruco.generateImageMarker(d, marker_id, size_px, borderBits=border_bits)
    cv2.imwrite(out_png, img)
    return os.path.abspath(out_png)


def generate_charuco_board_png(
    out_png: str = "charuco_board.png",
    dict_name: int = cv2.aruco.DICT_5X5_250,
    squares_x: int = 7,
    squares_y: int = 5,
    square_length: float = 0.04,
    marker_length: float = 0.02,
    dpi_px: int = 1600,
) -> str:
    d = cv2.aruco.getPredefinedDictionary(dict_name)
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, d)
    img = board.generateImage((dpi_px, int(dpi_px * squares_y / squares_x)))
    cv2.imwrite(out_png, img)
    return os.path.abspath(out_png)



def _order_quad(pts: np.ndarray) -> np.ndarray:
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def detect_largest_rectangle(gray: np.ndarray) -> Optional[np.ndarray]:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, None, iterations=2)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0.0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 6000:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue
        if area > best_area:
            best_area = area
            best = approx.reshape(4, 2)

    if best is None:
        return None
    return _order_quad(best)


def detect_any_aruco(gray: np.ndarray, dict_name: int = cv2.aruco.DICT_5X5_250) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    d = cv2.aruco.getPredefinedDictionary(dict_name)
    params = cv2.aruco.DetectorParameters()
    det = cv2.aruco.ArucoDetector(d, params)
    corners, ids, _rej = det.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None
    return corners, ids


def marker_center_and_angle(marker_corners: np.ndarray) -> Tuple[np.ndarray, float]:
    pts = np.array(marker_corners, dtype=np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    v = pts[1] - pts[0]
    ang = float(np.arctan2(v[1], v[0]))
    return c, ang


def _ang_diff(a: float, b: float) -> float:
    d = (a - b + np.pi) % (2 * np.pi) - np.pi
    return float(abs(d))


def align_rect_corner_order_to_marker(rect_img: np.ndarray, marker_ang: float) -> np.ndarray:
    rect = rect_img.copy()
    top_edge = rect[1] - rect[0]
    rect_ang = float(np.arctan2(top_edge[1], top_edge[0]))

    best = rect
    best_score = _ang_diff(marker_ang, rect_ang)

    for k in range(1, 4):
        r = np.roll(rect, -k, axis=0)
        e = r[1] - r[0]
        a = float(np.arctan2(e[1], e[0]))
        s = _ang_diff(marker_ang, a)
        if s < best_score:
            best_score = s
            best = r

    return best


def rect_homography_to_workspace(rect_img_4x2: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    xmin, xmax = float(val.WORKSPACE_X_MIN), float(val.WORKSPACE_X_MAX)
    ymin, ymax = float(val.WORKSPACE_Y_MIN), float(val.WORKSPACE_Y_MAX)

    dst = np.array(
        [
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax],
        ],
        dtype=np.float32,
    )

    H, _ = cv2.findHomography(rect_img_4x2.astype(np.float32), dst, method=0)
    if H is None:
        return None
    Hinv = np.linalg.inv(H)
    return H, Hinv


def pose_from_world_to_image_homography(K: np.ndarray, H_w2i: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    K = np.array(K, dtype=np.float64).reshape(3, 3)
    H = np.array(H_w2i, dtype=np.float64).reshape(3, 3)

    invK = np.linalg.inv(K)
    B = invK @ H
    h1 = B[:, 0]
    h2 = B[:, 1]
    h3 = B[:, 2]

    lam = 1.0 / (np.linalg.norm(h1) + 1e-12)
    r1 = lam * h1
    r2 = lam * h2
    r3 = np.cross(r1, r2)
    t = lam * h3.reshape(3, 1)

    R_approx = np.stack([r1, r2, r3], axis=1)
    U, _, Vt = np.linalg.svd(R_approx)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    return R.astype(np.float64), t.astype(np.float64)



def calibrate_intrinsics_charuco(
    cap: cv2.VideoCapture,
    target_frames: int = 25,
    dict_name: int = cv2.aruco.DICT_5X5_250,
    squares_x: int = 7,
    squares_y: int = 5,
    square_length: float = 0.04,
    marker_length: float = 0.02,
    timeout_s: float = 90.0,
) -> Dict[str, Any]:
    d = cv2.aruco.getPredefinedDictionary(dict_name)
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, d)

    all_corners: List[np.ndarray] = []
    all_ids: List[np.ndarray] = []
    img_size = None

    params = cv2.aruco.DetectorParameters()
    det = cv2.aruco.ArucoDetector(d, params)

    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])

        corners, ids, _rej = det.detectMarkers(gray)
        vis = frame.copy()

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            ret, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            if ret is not None and ch_corners is not None and ch_ids is not None and len(ch_ids) >= 12:
                all_corners.append(ch_corners)
                all_ids.append(ch_ids)
                cv2.putText(vis, f"ChArUco accepted ({len(all_corners)}/{target_frames})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(
            vis,
            "Intrinsics: show a ChArUco board (print or display). Press 'q' to abort.",
            (10, vis.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.imshow("Calibration - Intrinsics (ChArUco)", vis)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if len(all_corners) >= target_frames:
            break
        if time.time() - t0 > timeout_s and len(all_corners) >= max(10, target_frames // 2):
            break

    cv2.destroyWindow("Calibration - Intrinsics (ChArUco)")

    if img_size is None or len(all_corners) < 8:
        raise RuntimeError("Intrinsics calibration failed: not enough valid ChArUco frames.")

    flags = 0
    ret, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_corners,
        charucoIds=all_ids,
        board=board,
        imageSize=img_size,
        cameraMatrix=None,
        distCoeffs=None,
        flags=flags,
    )

    intr = {
        "K": np.array(K, dtype=np.float64),
        "dist": np.array(dist, dtype=np.float64).reshape(-1, 1),
        "image_w": int(img_size[0]),
        "image_h": int(img_size[1]),
        "reproj_error": float(ret),
    }
    return intr



def calibrate_workspace_from_rectangle_and_aruco(
    cap: cv2.VideoCapture,
    intr: Optional[Dict[str, Any]],
    dict_name: int = cv2.aruco.DICT_5X5_250,
    required_frames: int = 12,
    timeout_s: float = 60.0,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    Hs_img2ws: List[np.ndarray] = []
    rects: List[np.ndarray] = []
    marker_ids_seen: List[int] = []

    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ar = detect_any_aruco(gray, dict_name=dict_name)
        rect = detect_largest_rectangle(gray)

        vis = frame.copy()

        if rect is not None:
            cv2.polylines(vis, [rect.astype(np.int32)], True, (0, 255, 255), 2)

        if ar is not None:
            corners, ids = ar
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)

        accepted = False
        if rect is not None and ar is not None:
            corners, ids = ar
            mc, mang = marker_center_and_angle(corners[0])
            rid = int(ids.flatten()[0])
            rect_aligned = align_rect_corner_order_to_marker(rect, mang)
            out = rect_homography_to_workspace(rect_aligned)
            if out is not None:
                H, Hinv = out
                Hs_img2ws.append(H)
                rects.append(rect_aligned)
                marker_ids_seen.append(rid)
                accepted = True

        if accepted:
            cv2.putText(vis, f"Workspace accepted ({len(Hs_img2ws)}/{required_frames})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(
            vis,
            "Workspace: place ANY rectangle on table with ANY visible ArUco tag. Press 'q' to abort.",
            (10, vis.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.imshow("Calibration - Workspace (Rectangle + ArUco)", vis)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break

        if len(Hs_img2ws) >= required_frames:
            break

        if time.time() - t0 > timeout_s and len(Hs_img2ws) >= max(5, required_frames // 2):
            break

    cv2.destroyWindow("Calibration - Workspace (Rectangle + ArUco)")

    if len(Hs_img2ws) < 5:
        raise RuntimeError("Workspace calibration failed: not enough frames with both rectangle and ArUco.")

    H_avg = np.mean(np.stack(Hs_img2ws, axis=0), axis=0)
    rect_avg = np.mean(np.stack(rects, axis=0), axis=0)

    out = rect_homography_to_workspace(_order_quad(rect_avg))
    if out is None:
        raise RuntimeError("Workspace calibration failed: homography could not be computed.")
    H_img_to_ws, H_ws_to_img = out

    ws = {
        "H_img_to_ws": np.array(H_img_to_ws, dtype=np.float64),
        "H_ws_to_img": np.array(H_ws_to_img, dtype=np.float64),
        "rect_img_corners": np.array(_order_quad(rect_avg), dtype=np.float64),
        "ws_bounds": [
            float(val.WORKSPACE_X_MIN),
            float(val.WORKSPACE_X_MAX),
            float(val.WORKSPACE_Y_MIN),
            float(val.WORKSPACE_Y_MAX),
        ],
        "table_z": float(val.WORKSPACE_Z_MIN),
        "marker_ids_seen": marker_ids_seen[-20:],
    }

    ext = None
    if intr is not None and "K" in intr and ws.get("H_ws_to_img", None) is not None:
        R, t = pose_from_world_to_image_homography(np.array(intr["K"], dtype=np.float64), np.array(ws["H_ws_to_img"], dtype=np.float64))
        ext = {
            "R": R,
            "t": t,
            "world_frame": "workspace_plane_z0",
            "note": "Recovered from homography: world plane (X,Y,1) -> image, assuming Z=0 plane",
        }

    return ws, ext



def ensure_calibration(
    cap: cv2.VideoCapture,
    use_charuco_intrinsics: bool = True,
    intrinsics_target_frames: int = 25,
    phone_ids: Tuple[int, int, int, int] = (10, 11, 12, 13),
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    _ensure_dir()

    intr = load_intrinsics()
    ws = load_workspace()
    ext = load_extrinsics()

    if intr is None and use_charuco_intrinsics:
        try:
            generate_charuco_board_png(out_png=os.path.join(CALIB_DIR, "charuco_board.png"))
        except Exception:
            pass
        intr = calibrate_intrinsics_charuco(
            cap=cap,
            target_frames=int(intrinsics_target_frames),
        )
        save_intrinsics(intr)

    if ws is None or (ext is None and intr is not None):
        ws, ext2 = calibrate_workspace_from_rectangle_and_aruco(
            cap=cap,
            intr=intr,
            required_frames=12,
        )
        save_workspace(ws)
        if ext2 is not None:
            ext = ext2
            save_extrinsics(ext)

    return intr, ws, ext
