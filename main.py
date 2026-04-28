from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2

import values as val
from camera_utils import (
    CameraOpenError,
    open_handtracking_camera,
    print_camera_failure_help,
    probe_camera_indices,
)
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hand tracking and robot control runtime.")
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Probe camera indices 0..4 and exit without connecting the robot.",
    )
    return parser.parse_args()


def _list_cameras() -> int:
    print("[main] Probing camera indices 0..4. Robot will not be connected.")
    probes = probe_camera_indices(range(5), read_retries=8)
    for p in probes:
        shape = "" if p.frame_shape is None else f" frame_shape={p.frame_shape}"
        print(
            f"  index={p.index} backend={p.backend_name} props={'yes' if p.used_props else 'no'} "
            f"opened={p.opened} read_ok={p.read_ok}{shape}"
        )
    return 0


def main() -> int:
    args = _parse_args()
    if args.list_cameras:
        return _list_cameras()

    if not _ensure_robot_calibration():
        return 1

    try:
        camera = open_handtracking_camera()
    except CameraOpenError as exc:
        print_camera_failure_help(exc)
        return 1 if bool(getattr(val, "MAIN_CAMERA_FAIL_FATAL", True)) else 0

    cap = camera.cap
    pending_first_frame = camera.frame
    tracker = None
    robot = None
    pick_runtime = None
    real_robot_enabled = bool(getattr(val, "ENABLE_REAL_ROBOT", False))
    exit_code = 0

    try:
        from handtracking import HandTracker

        tracker = HandTracker()
    except Exception as exc:
        print(f"[main] Failed to initialize hand tracker after camera startup: {exc}")
        cap.release()
        return 1

    robot = SOArmHardwareController()

    if real_robot_enabled:
        try:
            robot.connect()
            print("[main] Real robot connected")
            if bool(getattr(val, "REAL_ROBOT_ASYNC_COMMAND_SENDER", True)):
                robot.start_async_sender()
                print("[main] Robot command sender running asynchronously")
        except Exception as exc:
            print(f"[main] Failed to connect robot: {exc}")
            try:
                tracker.close()
            except Exception:
                pass
            try:
                cap.release()
            except Exception:
                pass
            return 1
    else:
        print("[main] ENABLE_REAL_ROBOT=False; commands will be displayed but not sent")

    pick_runtime = PickAndPlaceRuntime(real_robot_enabled=real_robot_enabled)
    trigger_key = str(getattr(val, "PICKPLACE_TRIGGER_KEY", "p"))[:1].lower() or "p"
    trigger_key_code = ord(trigger_key)

    hz = float(getattr(val, "MAIN_LOOP_HZ", getattr(val, "REAL_ROBOT_HZ", 20.0)))
    period = 1.0 / max(hz, 1e-3)
    feedback_hz = float(getattr(val, "MAIN_ROBOT_FEEDBACK_HZ", 5.0))
    feedback_period = 1.0 / max(feedback_hz, 1e-3) if feedback_hz > 0.0 else float("inf")
    last_feedback_time = 0.0
    robot_feedback: Optional[dict] = None
    camera_failures = 0
    camera_failure_limit = max(1, int(getattr(val, "MAIN_CAMERA_READ_RETRIES", 30)))
    camera_retry_delay = max(0.0, float(getattr(val, "MAIN_CAMERA_READ_RETRY_DELAY_S", 0.05)))

    try:
        while True:
            loop_start = time.time()
            if pending_first_frame is not None:
                frame = pending_first_frame
                pending_first_frame = None
                ret = True
            else:
                ret, frame = cap.read()

            if not ret or frame is None:
                camera_failures += 1
                if camera_failures == 1:
                    print("[main] hand-tracking camera read failed; retrying")
                if camera_failures >= camera_failure_limit:
                    print("[main] Could not read from hand-tracking camera.")
                    print(f"[main] Consecutive failed reads: {camera_failures}")
                    print("[main] Close other apps using the camera or change HANDTRACKING_CAMERA_INDEX in values.py.")
                    exit_code = 1 if bool(getattr(val, "MAIN_CAMERA_FAIL_FATAL", True)) else 0
                    break
                if camera_retry_delay > 0.0:
                    time.sleep(camera_retry_delay)
                continue
            camera_failures = 0

            frame = cv2.flip(frame, 1)

            now = time.time()
            if (
                real_robot_enabled
                and bool(getattr(val, "MAIN_READ_ROBOT_FEEDBACK", True))
                and (now - last_feedback_time) >= feedback_period
            ):
                try:
                    fresh_feedback = robot.read_present_joints_rad()
                    if fresh_feedback is not None:
                        robot_feedback = fresh_feedback
                        tracker.update_robot_feedback(robot_feedback)
                except Exception as exc:
                    print(f"[main] warning: robot feedback read failed: {exc}")
                last_feedback_time = now

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
                if bool(getattr(val, "REAL_ROBOT_ASYNC_COMMAND_SENDER", True)):
                    robot.submit_latest_command(command_to_send)
                else:
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
            # No work is scheduled here; the sleep above only rate-limits this
            # foreground loop so the camera UI and serial bus stay stable.
    finally:
        if pick_runtime is not None:
            try:
                pick_runtime.close()
            except Exception:
                pass
        try:
            if tracker is not None:
                tracker.close()
        except Exception:
            pass
        try:
            if real_robot_enabled and robot is not None:
                robot.stop_async_sender()
                robot.disconnect()
        except Exception:
            pass
        try:
            cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
