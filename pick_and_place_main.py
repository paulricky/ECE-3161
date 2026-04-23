"""Entry point for YOLO + ArUco pick-and-place on the SO-100 arm.

Mirrors main.py's shape: ensure calibration, open camera, construct detectors,
build the planner, then loop at REAL_ROBOT_HZ reading the camera, updating the
planner with detections and robot feedback, and letting the planner emit
JointCommands which we pass through the rate-limited hardware sender.

Run hand-tracking (main.py) and this script separately — they expect
different cameras and calibration files, but they share the LeRobot driver.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

import values as val
from aruco_marker import TopdownArucoDetector, aruco_dict_id_from_name
from object_detector import ObjectDetector
from pick_place_planner import PickPlaceConfig, PickPlacePlanner, PickPlaceState
from pixel_to_workspace import PixelToWorkspace
from robot_controller import SOArmHardwareController


_THIS_DIR = Path(__file__).resolve().parent


def _resolve_project_path(configured: str, default_subpath: str) -> Path:
    raw = str(configured or default_subpath)
    p = Path(raw)
    if not p.is_absolute():
        p = _THIS_DIR / p
    return p


def _ensure_robot_calibration() -> bool:
    if not getattr(val, "ENABLE_REAL_ROBOT", False):
        return True

    try:
        from robot_calibrate import (
            get_joint_calibration_status,
            get_motor_setup_status,
            run_workflow as run_robot_calibration_workflow,
        )
    except Exception as exc:
        print(f"[pick_and_place_main] failed to import robot_calibrate: {exc}")
        return False

    setup = get_motor_setup_status()
    calib = get_joint_calibration_status()
    if calib.configured:
        print(f"[pick_and_place_main] robot calibration present")
        return True
    if setup.configured:
        print("[pick_and_place_main] motor setup present but joint calibration is missing.")
        reply = input("Run joint calibration now? [Y/n]: ").strip().lower()
        if reply not in ("", "y", "yes"):
            return False
        return run_robot_calibration_workflow("calibration") == 0
    print("[pick_and_place_main] neither motor setup nor joint calibration was found.")
    reply = input("Run motor setup and joint calibration now? [Y/n]: ").strip().lower()
    if reply not in ("", "y", "yes"):
        return False
    return run_robot_calibration_workflow("full") == 0


def _load_topdown_calibration() -> Optional[Tuple[dict, dict]]:
    intr_path = _resolve_project_path(
        getattr(val, "PICKPLACE_TOPDOWN_INTRINSICS_FILE", ""),
        "calibration_data/topdown_intrinsics.npz",
    )
    ext_path = _resolve_project_path(
        getattr(val, "PICKPLACE_TOPDOWN_EXTRINSICS_FILE", ""),
        "calibration_data/topdown_extrinsics.npz",
    )
    if not intr_path.exists() or not ext_path.exists():
        print("[pick_and_place_main] top-down calibration files not found.")
        print(f"  intrinsics: {intr_path} exists={intr_path.exists()}")
        print(f"  extrinsics: {ext_path} exists={ext_path.exists()}")
        print("  Run: python3 topdown_calibrate.py --phase intrinsics")
        print("  Then: python3 topdown_calibrate.py --phase extrinsics")
        return None

    intr_data = np.load(intr_path, allow_pickle=True)
    K = None
    for key in ("camera_matrix", "mtx", "K"):
        if key in intr_data:
            K = np.asarray(intr_data[key], dtype=np.float64).reshape(3, 3)
            break
    dist = None
    for key in ("dist_coeffs", "dist", "distortion_coefficients"):
        if key in intr_data:
            dist = np.asarray(intr_data[key], dtype=np.float64).reshape(-1, 1)
            break
    if K is None or dist is None:
        print(f"[pick_and_place_main] intrinsics file missing required keys")
        return None

    image_size = None
    if "image_size" in intr_data:
        image_size = tuple(int(x) for x in np.asarray(intr_data["image_size"]).reshape(-1)[:2])

    ext_data = np.load(ext_path, allow_pickle=True)
    R = np.asarray(ext_data["R"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(ext_data["t"], dtype=np.float64).reshape(3)
    table_z = float(getattr(val, "PICKPLACE_TABLE_Z_M", 0.02))
    if "table_z_fitted" in ext_data:
        fitted = float(np.asarray(ext_data["table_z_fitted"]).reshape(-1)[0])
        if np.isfinite(fitted):
            table_z = fitted
            print(f"[pick_and_place_main] using calibrated table_z = {table_z:.4f} m")

    return (
        {"K": K, "dist": dist, "image_size": image_size},
        {"R": R, "t": t, "table_z": table_z},
    )


def _draw_hud(frame: np.ndarray, lines):
    for i, line in enumerate(lines):
        y = 30 + 26 * i
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)


def main() -> int:
    if not _ensure_robot_calibration():
        print("[pick_and_place_main] robot calibration not confirmed; exiting.")
        return 1

    calib = _load_topdown_calibration()
    if calib is None:
        return 1
    intr, ext = calib

    pixel_to_ws = PixelToWorkspace(
        K=intr["K"],
        dist=intr["dist"],
        R_cam_from_ws=ext["R"],
        t_cam_from_ws=ext["t"],
        table_z=ext["table_z"],
    )

    detector = ObjectDetector(
        model_path=str(getattr(val, "PICKPLACE_YOLO_MODEL_PATH", "yolov8n.pt")),
        class_whitelist=list(getattr(val, "PICKPLACE_YOLO_CLASS_WHITELIST", []) or []),
        conf_threshold=float(getattr(val, "PICKPLACE_YOLO_CONF_THRESHOLD", 0.45)),
        device=str(getattr(val, "PICKPLACE_YOLO_DEVICE", "cpu")),
        inference_hz=float(getattr(val, "PICKPLACE_YOLO_INFERENCE_HZ", 8.0)),
    )

    drop_dict_id = aruco_dict_id_from_name(
        str(getattr(val, "PICKPLACE_DROP_MARKER_DICT", "DICT_4X4_50"))
    )
    drop_marker_id = int(getattr(val, "PICKPLACE_DROP_MARKER_ID", 42))
    drop_aruco = TopdownArucoDetector(
        aruco_dict_id=drop_dict_id,
        marker_size_m=float(getattr(val, "PICKPLACE_DROP_MARKER_SIZE_M", 0.04)),
        K=intr["K"], dist=intr["dist"],
        R_cam_from_ws=ext["R"], t_cam_from_ws=ext["t"],
        valid_ids=[drop_marker_id],
    )

    config = PickPlaceConfig.from_values(table_z=ext["table_z"])
    planner = PickPlacePlanner(
        pixel_to_ws=pixel_to_ws,
        config=config,
        drop_marker_id=drop_marker_id,
    )

    robot = SOArmHardwareController()
    real_robot_enabled = bool(getattr(val, "ENABLE_REAL_ROBOT", False))
    if real_robot_enabled:
        try:
            robot.connect()
            print("[pick_and_place_main] robot connected")
        except Exception as exc:
            print(f"[pick_and_place_main] failed to connect robot: {exc}")
            return 1
    else:
        print("[pick_and_place_main] ENABLE_REAL_ROBOT=False; running in dry-run mode")

    cam_index = int(getattr(val, "PICKPLACE_CAMERA_INDEX", 1))
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[pick_and_place_main] cannot open camera index {cam_index}")
        if real_robot_enabled:
            robot.disconnect()
        return 1

    hz = float(getattr(val, "REAL_ROBOT_HZ", 20.0))
    period = 1.0 / max(hz, 1e-3)
    window = "Pick and Place"

    try:
        while True:
            loop_start = time.time()
            ok, frame = cap.read()
            if not ok:
                print("[pick_and_place_main] camera read failed")
                break

            now = time.time()
            df = detector.maybe_detect(frame, now)
            markers = drop_aruco.detect(frame)
            fb = robot.read_present_joints_rad() if real_robot_enabled else None

            planner.update_detections(df)
            planner.update_drop_marker(markers)
            planner.update_robot_feedback(fb)

            cmd = planner.tick(now)
            if cmd is not None and real_robot_enabled:
                robot.send_if_due(cmd)

            # Overlay
            if df is not None:
                for det in df.detections:
                    ObjectDetector.draw_overlay(frame, det)

            for mid, pose in markers.items():
                drop_aruco.draw_overlay(frame, pose, color=(255, 128, 0),
                                        label=f"DROP id={mid}")

            chosen = planner.last_chosen_detection
            lines = [
                f"state: {planner.state.value}   continuous={config.continuous_mode}",
                f"detections: {0 if df is None else len(df.detections)}   "
                f"drop_seen: {drop_marker_id in markers}",
            ]
            if chosen is not None:
                lines.append(
                    f"chosen: {chosen.detection.class_name} @ "
                    f"xy=({chosen.xyz_ws[0]:+.3f},{chosen.xyz_ws[1]:+.3f}) "
                    f"theta_deg={np.degrees(chosen.theta_ws):+.1f}"
                )
            lines.append(
                f"aborts: {planner.abort_count}   "
                f"real_robot: {real_robot_enabled}"
            )
            if cmd is not None:
                lines.append(
                    f"cmd: pan={np.degrees(cmd.shoulder_pan):+.1f}  "
                    f"lift={np.degrees(cmd.shoulder_lift):+.1f}  "
                    f"elbow={np.degrees(cmd.elbow_flex):+.1f}  "
                    f"gripper={cmd.gripper_open01:.2f}"
                )
            _draw_hud(frame, lines)

            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            if planner.state == PickPlaceState.ERROR:
                print("[pick_and_place_main] planner in ERROR; exiting loop")
                break
            if planner.state == PickPlaceState.DONE:
                print("[pick_and_place_main] planner reached DONE; exiting loop")
                break

            elapsed = time.time() - loop_start
            sleep = period - elapsed
            if sleep > 0:
                time.sleep(sleep)
    finally:
        try:
            if real_robot_enabled:
                robot.disconnect()
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
