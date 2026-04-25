from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from config import values as val


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_DIR = os.path.join(_THIS_DIR, "calibration_data")
os.makedirs(CALIB_DIR, exist_ok=True)

INTRINSICS_NPZ = os.path.join(CALIB_DIR, "calibration_intrinsics.npz")
WORKSPACE_NPZ = os.path.join(CALIB_DIR, "calibration_workspace.npz")
EXTRINSICS_NPZ = os.path.join(CALIB_DIR, "calibration_extrinsics.npz")


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


def load_intrinsics(path: str = INTRINSICS_NPZ):
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path, allow_pickle=True)
    except Exception:
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate workspace depth/bounds using an ArUco tag on the user's hand. "
            "This script requires calibration_intrinsics.npz to already exist."
        )
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0).")
    parser.add_argument("--ids", type=str, default="", help="Allowed glove marker ids, e.g. '1,5'.")
    parser.add_argument("--aruco", type=int, default=-1, help="cv2.aruco dictionary id. Default uses values.ARUCO_DICT_NAME.")
    parser.add_argument("--marker-size", type=float, default=-1.0, help="Marker size in meters. Default uses values.ARUCO_MARKER_SIZE_M.")
    parser.add_argument("--hold-frames", type=int, default=20, help="Stable frames to average per capture.")
    args = parser.parse_args()

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