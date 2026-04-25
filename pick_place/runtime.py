from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from config import values as val
from vision.aruco_marker import TopdownArucoDetector, aruco_dict_id_from_name
from vision.object_detector import ObjectDetector
from pick_place.planner import PickPlaceConfig, PickPlacePlanner, PickPlaceState
from vision.pixel_to_workspace import PixelToWorkspace


_THIS_DIR = Path(__file__).resolve().parent


def _resolve_project_path(configured: str, default_subpath: str) -> Path:
    raw = str(configured or default_subpath)
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = _THIS_DIR / p
    return p.resolve()


def load_topdown_calibration() -> Optional[Tuple[dict, dict]]:
    intr_path = _resolve_project_path(
        getattr(val, "PICKPLACE_TOPDOWN_INTRINSICS_FILE", ""),
        "calibration_data/topdown_intrinsics.npz",
    )
    ext_path = _resolve_project_path(
        getattr(val, "PICKPLACE_TOPDOWN_EXTRINSICS_FILE", ""),
        "calibration_data/topdown_extrinsics.npz",
    )
    if not intr_path.exists() or not ext_path.exists():
        print("[pick_place] top-down calibration files not found.")
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
        print("[pick_place] intrinsics file missing camera matrix or distortion keys")
        return None

    image_size = None
    if "image_size" in intr_data:
        image_size = tuple(int(x) for x in np.asarray(intr_data["image_size"]).reshape(-1)[:2])

    ext_data = np.load(ext_path, allow_pickle=True)
    try:
        R = np.asarray(ext_data["R"], dtype=np.float64).reshape(3, 3)
        t = np.asarray(ext_data["t"], dtype=np.float64).reshape(3)
    except Exception as exc:
        print(f"[pick_place] extrinsics file missing R/t: {exc}")
        return None

    table_z = float(getattr(val, "PICKPLACE_TABLE_Z_M", 0.02))
    if "table_z_fitted" in ext_data:
        fitted = float(np.asarray(ext_data["table_z_fitted"]).reshape(-1)[0])
        if np.isfinite(fitted):
            table_z = fitted
            print(f"[pick_place] using calibrated table_z = {table_z:.4f} m")

    return (
        {"K": K, "dist": dist, "image_size": image_size},
        {"R": R, "t": t, "table_z": table_z},
    )


def _draw_hud(frame: np.ndarray, lines) -> None:
    for i, line in enumerate(lines):
        y = 30 + 26 * i
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)


class PickAndPlaceRuntime:
    """Lazy, triggerable pick-and-place runtime used by main.py."""

    def __init__(self, real_robot_enabled: bool):
        self.real_robot_enabled = bool(real_robot_enabled)
        self.initialized = False
        self.available = False
        self.active = False
        self.last_error = "not initialized"
        self.last_trigger = ""

        self.cap = None
        self.pixel_to_ws = None
        self.detector = None
        self.drop_aruco = None
        self.config = None
        self.drop_marker_id = int(getattr(val, "PICKPLACE_DROP_MARKER_ID", 42))
        self.planner = None
        self.window = str(getattr(val, "PICKPLACE_WINDOW_NAME", "Pick and Place"))

    def initialize(self) -> bool:
        if self.initialized:
            return self.available
        self.initialized = True

        calib = load_topdown_calibration()
        if calib is None:
            self.available = False
            self.last_error = "missing top-down calibration"
            return False
        intr, ext = calib

        self.pixel_to_ws = PixelToWorkspace(
            K=intr["K"],
            dist=intr["dist"],
            R_cam_from_ws=ext["R"],
            t_cam_from_ws=ext["t"],
            table_z=ext["table_z"],
        )

        self.detector = ObjectDetector(
            model_path=str(getattr(val, "PICKPLACE_YOLO_MODEL_PATH", "yolov8n.pt")),
            class_whitelist=list(getattr(val, "PICKPLACE_YOLO_CLASS_WHITELIST", []) or []),
            conf_threshold=float(getattr(val, "PICKPLACE_YOLO_CONF_THRESHOLD", 0.45)),
            device=str(getattr(val, "PICKPLACE_YOLO_DEVICE", "cpu")),
            inference_hz=float(getattr(val, "PICKPLACE_YOLO_INFERENCE_HZ", 8.0)),
        )

        drop_dict_id = aruco_dict_id_from_name(
            str(getattr(val, "PICKPLACE_DROP_MARKER_DICT", "DICT_4X4_50"))
        )
        self.drop_marker_id = int(getattr(val, "PICKPLACE_DROP_MARKER_ID", 42))
        self.drop_aruco = TopdownArucoDetector(
            aruco_dict_id=drop_dict_id,
            marker_size_m=float(getattr(val, "PICKPLACE_DROP_MARKER_SIZE_M", 0.04)),
            K=intr["K"],
            dist=intr["dist"],
            R_cam_from_ws=ext["R"],
            t_cam_from_ws=ext["t"],
            valid_ids=[self.drop_marker_id],
        )

        self.config = PickPlaceConfig.from_values(table_z=ext["table_z"])
        self.config.continuous_mode = bool(getattr(val, "PICKPLACE_TRIGGER_CONTINUOUS_MODE", False))

        cam_index = int(getattr(val, "PICKPLACE_CAMERA_INDEX", 1))
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            self.available = False
            self.last_error = f"cannot open pick-place camera index {cam_index}"
            print(f"[pick_place] {self.last_error}")
            return False

        self.available = True
        self.last_error = ""
        return True

    def _new_planner(self) -> PickPlacePlanner:
        return PickPlacePlanner(
            pixel_to_ws=self.pixel_to_ws,
            config=self.config,
            drop_marker_id=self.drop_marker_id,
        )

    def request_start(self, reason: str = "manual") -> bool:
        if self.active:
            return True
        if not self.initialize():
            print(f"[pick_place] cannot start: {self.last_error}")
            return False
        self.planner = self._new_planner()
        self.active = True
        self.last_trigger = str(reason)
        print(f"[pick_place] triggered by {self.last_trigger}")
        return True

    def cancel(self) -> None:
        if self.active:
            print("[pick_place] canceled")
        self.active = False
        self.planner = None

    def tick(self, robot_feedback=None):
        if not self.active or self.planner is None:
            return None
        if self.cap is None or not self.cap.isOpened():
            self.last_error = "pick-place camera is not open"
            self.cancel()
            return None

        ok, frame = self.cap.read()
        if not ok:
            self.last_error = "pick-place camera read failed"
            print(f"[pick_place] {self.last_error}")
            self.cancel()
            return None

        now = time.time()
        df = self.detector.maybe_detect(frame, now)
        markers = self.drop_aruco.detect(frame)

        self.planner.update_detections(df)
        self.planner.update_drop_marker(markers)
        self.planner.update_robot_feedback(robot_feedback)

        cmd = self.planner.tick(now)

        if df is not None:
            for det in df.detections:
                ObjectDetector.draw_overlay(frame, det)
        for mid, pose in markers.items():
            self.drop_aruco.draw_overlay(frame, pose, color=(255, 128, 0), label=f"DROP id={mid}")

        chosen = self.planner.last_chosen_detection
        lines = [
            f"PICK/PLACE ACTIVE - trigger={self.last_trigger} - press c to cancel",
            f"state: {self.planner.state.value}   one_shot={not self.config.continuous_mode}",
            f"detections: {0 if df is None else len(df.detections)}   drop_seen: {self.drop_marker_id in markers}",
        ]
        if chosen is not None:
            lines.append(
                f"chosen: {chosen.detection.class_name} xy=({chosen.xyz_ws[0]:+.3f},{chosen.xyz_ws[1]:+.3f}) "
                f"theta={np.degrees(chosen.theta_ws):+.1f}deg"
            )
        lines.append(f"aborts: {self.planner.abort_count}   real_robot: {self.real_robot_enabled}")
        if cmd is not None:
            lines.append(
                f"cmd pan={np.degrees(cmd.shoulder_pan):+.1f} lift={np.degrees(cmd.shoulder_lift):+.1f} "
                f"elbow={np.degrees(cmd.elbow_flex):+.1f} grip={cmd.gripper_open01:.2f}"
            )
        _draw_hud(frame, lines)
        cv2.imshow(self.window, frame)

        if self.planner.state in (PickPlaceState.ERROR, PickPlaceState.DONE):
            print(f"[pick_place] finished with state={self.planner.state.value}")
            self.active = False
            self.planner = None

        return cmd

    def close(self) -> None:
        self.cancel()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        try:
            cv2.destroyWindow(self.window)
        except Exception:
            pass