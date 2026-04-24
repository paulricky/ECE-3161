from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2

import values as val
from handtracking import HandTracker
from pick_place_runtime import PickAndPlaceRuntime
from robot_calibrate import (
    get_joint_calibration_status,
    get_motor_setup_status,
    run_workflow as run_robot_calibration_workflow,
)
from robot_controller import JointCommand, SOArmHardwareController


DEFAULT_ROBOT_CALIBRATION_FILE = Path(__file__).resolve().parent / "calibration_data" / "robot_joint_calibration.json"


def _robot_calibration_path() -> Path:
    configured = getattr(val, "ROBOT_JOINT_CALIBRATION_FILE", "")
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        return path.resolve()
    return DEFAULT_ROBOT_CALIBRATION_FILE


def _ensure_robot_calibration() -> bool:
    if not getattr(val, "ENABLE_REAL_ROBOT", False):
        return True

    setup_status = get_motor_setup_status()
    calib_status = get_joint_calibration_status()
    calib_path = _robot_calibration_path()

    if calib_status.configured:
        print(f"[main] Using robot calibration: {calib_path}")
        reply = input("Robot calibration already exists. Recalibrate now? [y/N]: ").strip().lower()
        if reply in ("y", "yes"):
            rc = run_robot_calibration_workflow("calibration")
            if rc != 0:
                print("[main] Recalibration did not complete successfully. Exiting.")
                return False
        return True

    if setup_status.configured:
        print("[main] Motor-ID setup already exists, but joint calibration does not.")
        if setup_status.source:
            print(f"[main] Using existing motor setup from: {setup_status.source}")
        reply = input("Run joint calibration now? [Y/n]: ").strip().lower()
        if reply not in ("", "y", "yes"):
            print("[main] Calibration declined. Exiting.")
            return False
        rc = run_robot_calibration_workflow("calibration")
        if rc != 0:
            print("[main] Joint calibration did not complete successfully. Exiting.")
            return False
    else:
        print("[main] No robot motor setup detected and no joint calibration found.")
        reply = input("Run motor setup and joint calibration now? [Y/n]: ").strip().lower()
        if reply not in ("", "y", "yes"):
            print("[main] Calibration declined. Exiting.")
            return False
        rc = run_robot_calibration_workflow("full")
        if rc != 0:
            print("[main] Setup/calibration did not complete successfully. Exiting.")
            return False

    if not calib_path.exists():
        print(f"[main] Calibration finished, but the file was still not found: {calib_path}")
        return False

    print(f"[main] Calibration completed and saved to: {calib_path}")
    return True


def _command_from_hand_data(hand_data: dict) -> JointCommand:
    return JointCommand(
        shoulder_pan=float(hand_data["shoulder_pan"]),
        shoulder_lift=float(hand_data["shoulder_lift"]),
        elbow_flex=float(hand_data["elbow_flex"]),
        wrist_flex=float(hand_data["wrist_flex"]),
        wrist_yaw=float(hand_data["wrist_yaw"]),
        wrist_roll=float(hand_data["wrist_roll"]),
        wrist_pitch=float(hand_data.get("wrist_pitch", 0.0)),
        gripper_open01=float(hand_data["gripper_open01"]),
    )


def _draw_main_hud(frame, pick_runtime: PickAndPlaceRuntime, snap_event: bool) -> None:
    key = str(getattr(val, "PICKPLACE_TRIGGER_KEY", "p"))[:1] or "p"
    lines = [
        f"ESC: quit   {key}: pick/place   c: cancel pick/place",
        f"snap trigger: {'YES' if snap_event else 'no'}   pick/place: {'ACTIVE' if pick_runtime.active else 'idle'}",
    ]
    if pick_runtime.initialized and not pick_runtime.available:
        lines.append(f"pick/place unavailable: {pick_runtime.last_error}")
    for i, line in enumerate(lines):
        y = 24 + i * 24
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def main() -> int:
    if not _ensure_robot_calibration():
        return 1

    hand_cam_index = int(getattr(val, "HANDTRACKING_CAMERA_INDEX", 0))
    cap = cv2.VideoCapture(hand_cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open hand-tracking camera index {hand_cam_index}")

    tracker = HandTracker()
    robot = SOArmHardwareController()
    real_robot_enabled = bool(getattr(val, "ENABLE_REAL_ROBOT", False))

    if real_robot_enabled:
        try:
            robot.connect()
            print("[main] Real robot connected")
        except Exception as exc:
            print(f"[main] Failed to connect robot: {exc}")
            cap.release()
            return 1
    else:
        print("[main] ENABLE_REAL_ROBOT=False; commands will be displayed but not sent")

    pick_runtime = PickAndPlaceRuntime(real_robot_enabled=real_robot_enabled)
    trigger_key = str(getattr(val, "PICKPLACE_TRIGGER_KEY", "p"))[:1].lower() or "p"
    trigger_key_code = ord(trigger_key)

    hz = float(getattr(val, "REAL_ROBOT_HZ", 20.0))
    period = 1.0 / max(hz, 1e-3)
    last_time = time.time()

    try:
        while True:
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                print("[main] hand-tracking camera read failed")
                break

            frame = cv2.flip(frame, 1)

            robot_feedback: Optional[dict] = None
            if real_robot_enabled and bool(getattr(val, "MAIN_READ_ROBOT_FEEDBACK", True)):
                try:
                    robot_feedback = robot.read_present_joints_rad()
                except Exception as exc:
                    print(f"[main] warning: robot feedback read failed: {exc}")
                    robot_feedback = None
            if robot_feedback is not None:
                tracker.update_robot_feedback(robot_feedback)

            hand_data = tracker.process(frame)
            snap_event = tracker.consume_snap_event()
            if snap_event and bool(getattr(val, "PICKPLACE_TRIGGER_ON_SNAP", True)):
                pick_runtime.request_start("snap")

            command_to_send = None
            if pick_runtime.active:
                command_to_send = pick_runtime.tick(robot_feedback=robot_feedback)
            elif hand_data is not None:
                command_to_send = _command_from_hand_data(hand_data)

            if command_to_send is not None and real_robot_enabled:
                robot.send_if_due(command_to_send)

            _draw_main_hud(frame, pick_runtime, snap_event)
            cv2.imshow("Hand Tracking / Main", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == trigger_key_code:
                pick_runtime.request_start(f"key '{trigger_key}'")
            elif key == ord("c"):
                pick_runtime.cancel()

            elapsed = time.time() - loop_start
            sleep = period - elapsed
            if sleep > 0:
                time.sleep(sleep)
            last_time = time.time()
    finally:
        try:
            pick_runtime.close()
        except Exception:
            pass
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