from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2

import values as val
from camera_utils import (
    CameraOpenError,
    LatestFrameCamera,
    open_handtracking_camera,
    open_specific_handtracking_camera,
    print_camera_failure_help,
    read_latest_from_capture,
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


class PerfStats:
    def __init__(self):
        self.last_print = time.time()
        self.camera_frames = 0
        self.processed_frames = 0
        self.robot_commands = 0
        self.frame_age_ms = 0.0
        self.hand_ms = 0.0
        self.ik_ms = 0.0
        self.robot_enqueue_ms = 0.0
        self.display_ms = 0.0
        self.total_ms = 0.0
        self.queue_size = 0
        self.cam_fps = 0.0
        self.proc_fps = 0.0
        self.robot_hz = 0.0
        self._window_start = time.time()

    def update_rates(self, now: float) -> None:
        elapsed = max(now - self._window_start, 1e-6)
        self.cam_fps = self.camera_frames / elapsed
        self.proc_fps = self.processed_frames / elapsed
        self.robot_hz = self.robot_commands / elapsed

    def maybe_print(self) -> None:
        interval = float(getattr(val, "DEBUG_PERF_PRINT_EVERY_S", 2.0))
        if interval <= 0.0:
            return
        now = time.time()
        if (now - self.last_print) < interval:
            return
        self.update_rates(now)
        print(
            "[perf] "
            f"cam_fps={self.cam_fps:.1f} proc_fps={self.proc_fps:.1f} robot_hz={self.robot_hz:.1f} "
            f"frame_age_ms={self.frame_age_ms:.0f} hand_ms={self.hand_ms:.1f} ik_ms={self.ik_ms:.1f} "
            f"enqueue_ms={self.robot_enqueue_ms:.1f} display_ms={self.display_ms:.1f} total_ms={self.total_ms:.1f} "
            f"queue={self.queue_size}"
        )
        self.last_print = now
        self._window_start = now
        self.camera_frames = 0
        self.processed_frames = 0
        self.robot_commands = 0


def _draw_perf_overlay(frame, perf: PerfStats) -> None:
    if not bool(getattr(val, "DEBUG_PERF_OVERLAY", True)):
        return
    lines = [
        f"cam {perf.cam_fps:.1f}fps proc {perf.proc_fps:.1f}fps robot {perf.robot_hz:.1f}Hz q={perf.queue_size}",
        f"age {perf.frame_age_ms:.0f}ms hand {perf.hand_ms:.1f}ms IK {perf.ik_ms:.1f}ms serial {perf.robot_enqueue_ms:.1f}ms total {perf.total_ms:.1f}ms",
    ]
    for i, line in enumerate(lines):
        y = 82 + i * 22
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (40, 255, 40), 1, cv2.LINE_AA)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hand tracking and robot control runtime.")
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Probe camera indices 0..4 and exit without connecting the robot.",
    )
    return parser.parse_args()


def _list_cameras() -> int:
    required = max(2, int(getattr(val, "MAIN_CAMERA_STABILITY_FRAMES", 10)))
    print(f"[main] Probing camera indices 0..4 for {required} stable frames. Robot will not be connected.")
    probes = probe_camera_indices(range(5), read_retries=required)
    for p in probes:
        shape = "" if p.frame_shape is None else f" frame_shape={p.frame_shape}"
        print(
            f"  index={p.index} backend={p.backend_name} props={'yes' if p.used_props else 'no'} "
            f"opened={p.opened} stable_read={'yes' if p.read_ok else 'no'}{shape}"
        )
    return 0


def _initialize_hand_tracker():
    from handtracking import HandTracker

    tracker = HandTracker()
    if bool(getattr(val, "HANDTRACKING_INIT_MEDIAPIPE_BEFORE_CAMERA", True)):
        try:
            ok = bool(tracker.warmup_mediapipe())
            if ok:
                print("[main] MediaPipe hand tracker initialized before camera validation")
            else:
                print("[main] MediaPipe hand tracker warmup unavailable; continuing with existing no-hand fallback")
        except Exception as exc:
            print(f"[main] MediaPipe warmup skipped: {exc}")
    return tracker


def _stop_robot_sender(robot: Optional[SOArmHardwareController], real_robot_enabled: bool) -> None:
    if not real_robot_enabled or robot is None:
        return
    try:
        robot.stop_async_sender()
    except Exception:
        pass


def _start_robot_sender(robot: Optional[SOArmHardwareController], real_robot_enabled: bool) -> None:
    if not real_robot_enabled or robot is None:
        return
    if bool(getattr(val, "ROBOT_COMMAND_ASYNC_ONLY", True)) or bool(getattr(val, "REAL_ROBOT_ASYNC_COMMAND_SENDER", True)):
        robot.start_async_sender()


def main() -> int:
    args = _parse_args()
    if args.list_cameras:
        return _list_cameras()

    if not _ensure_robot_calibration():
        return 1

    tracker = None
    cap = None
    robot = None
    pick_runtime = None
    real_robot_enabled = bool(getattr(val, "ENABLE_REAL_ROBOT", False))
    exit_code = 0

    try:
        tracker = _initialize_hand_tracker()
    except Exception as exc:
        print(f"[main] Failed to initialize hand tracker before camera startup: {exc}")
        return 1

    try:
        camera = open_handtracking_camera()
    except CameraOpenError as exc:
        print_camera_failure_help(exc)
        try:
            tracker.close()
        except Exception:
            pass
        return 1 if bool(getattr(val, "MAIN_CAMERA_FAIL_FATAL", True)) else 0

    cap = camera.cap
    pending_first_frame = camera.frame
    latest_camera = None
    if bool(getattr(val, "CAMERA_THREADED_CAPTURE", True)):
        latest_camera = LatestFrameCamera(cap, initial_frame=pending_first_frame)
        latest_camera.start()
        pending_first_frame = None

    robot = SOArmHardwareController()

    if real_robot_enabled:
        try:
            robot.connect()
            print("[main] Real robot connected")
            _start_robot_sender(robot, real_robot_enabled)
        except Exception as exc:
            print(f"[main] Failed to connect robot: {exc}")
            try:
                tracker.close()
            except Exception:
                pass
            try:
                if latest_camera is not None:
                    latest_camera.release()
                else:
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
    reopen_attempts = 0
    max_reopen_attempts = max(0, int(getattr(val, "MAIN_CAMERA_MAX_REOPEN_ATTEMPTS", 2)))
    reopen_on_live_failure = bool(getattr(val, "MAIN_CAMERA_REOPEN_ON_LIVE_FAILURE", True))
    camera_sender_paused = False
    perf = PerfStats()
    last_frame_sequence = -1
    processed_frame_count = 0
    last_process_time = 0.0
    last_display_frame = None
    last_snap_event = False
    process_every_n = max(1, int(getattr(val, "HANDTRACKING_PROCESS_EVERY_N_FRAMES", 1)))
    process_hz = float(getattr(val, "HANDTRACKING_MAX_PROCESS_HZ", 30.0))
    process_period = 1.0 / max(process_hz, 1e-3) if process_hz > 0.0 else 0.0

    try:
        while True:
            loop_start = time.time()
            frame_timestamp = loop_start
            frame_sequence = last_frame_sequence + 1
            if latest_camera is not None:
                latest = latest_camera.read_latest()
                ret = bool(latest.ok)
                frame = latest.frame
                frame_timestamp = latest.timestamp
                frame_sequence = latest.sequence
                perf.frame_age_ms = latest.age_s * 1000.0
                stats = latest_camera.stats()
                perf.queue_size = int(stats.get("queue_size", 0))
                if latest.failures >= camera_failure_limit:
                    ret = False
                    frame = None
                if frame_sequence == last_frame_sequence:
                    if latest.failures >= camera_failure_limit:
                        ret = False
                        frame = None
                    else:
                        key = cv2.waitKey(1) & 0xFF
                        if key == 27:
                            break
                        time.sleep(0.001)
                        continue
                if frame is None:
                    ret = False
            elif pending_first_frame is not None:
                frame = pending_first_frame
                pending_first_frame = None
                ret = True
            else:
                ret, frame = read_latest_from_capture(cap)

            if not ret or frame is None:
                camera_failures += 1
                if camera_failures == 1:
                    print("[main] hand-tracking camera read failed; retrying")
                    _stop_robot_sender(robot, real_robot_enabled)
                    camera_sender_paused = True
                if camera_failures >= camera_failure_limit:
                    print("[main] Could not read from hand-tracking camera.")
                    print(f"[main] Consecutive failed reads: {camera_failures}")
                    print("[main] Close other apps using the camera or change HANDTRACKING_CAMERA_INDEX in values.py.")
                    if reopen_on_live_failure and reopen_attempts < max_reopen_attempts:
                        reopen_attempts += 1
                        print(f"[main] Attempting to reopen hand-tracking camera ({reopen_attempts}/{max_reopen_attempts})")
                        _stop_robot_sender(robot, real_robot_enabled)
                        try:
                            if latest_camera is not None:
                                latest_camera.release()
                                latest_camera = None
                            elif cap is not None:
                                cap.release()
                        except Exception:
                            pass
                        try:
                            camera = open_specific_handtracking_camera(camera.index, camera.backend_name, camera.used_props)
                            cap = camera.cap
                            pending_first_frame = camera.frame
                            if bool(getattr(val, "CAMERA_THREADED_CAPTURE", True)):
                                latest_camera = LatestFrameCamera(cap, initial_frame=pending_first_frame)
                                latest_camera.start()
                                pending_first_frame = None
                            camera_failures = 0
                            _start_robot_sender(robot, real_robot_enabled)
                            camera_sender_paused = False
                            last_frame_sequence = -1
                            continue
                        except CameraOpenError as exc:
                            print_camera_failure_help(exc)
                    exit_code = 1 if bool(getattr(val, "MAIN_CAMERA_FAIL_FATAL", True)) else 0
                    break
                if camera_retry_delay > 0.0:
                    time.sleep(camera_retry_delay)
                continue
            if camera_sender_paused:
                _start_robot_sender(robot, real_robot_enabled)
                camera_sender_paused = False
            camera_failures = 0
            perf.camera_frames += 1
            last_frame_sequence = frame_sequence

            frame = cv2.flip(frame, 1)
            last_display_frame = frame
            now = time.time()
            should_process = (
                (processed_frame_count % process_every_n) == 0
                and (process_period <= 0.0 or (now - last_process_time) >= process_period)
            )
            snap_event = last_snap_event

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

            command_to_send = None
            if should_process:
                last_process_time = now
                processed_frame_count += 1
                t_hand = time.perf_counter()
                hand_data = tracker.process(frame)
                perf.hand_ms = (time.perf_counter() - t_hand) * 1000.0
                perf.processed_frames += 1
                snap_event = tracker.consume_snap_event()
                last_snap_event = snap_event
                if isinstance(hand_data, dict):
                    diag = hand_data.get("__diagnostics__", {})
                    if isinstance(diag, dict):
                        perf.ik_ms = float(diag.get("perf_ik_ms", diag.get("ik_time_ms", perf.ik_ms)))
                if snap_event and bool(getattr(val, "PICKPLACE_TRIGGER_ON_SNAP", True)):
                    pick_runtime.request_start("snap")

                if pick_runtime.active:
                    command_to_send = pick_runtime.tick(robot_feedback=robot_feedback)
                elif hand_data is not None:
                    command_to_send = _command_from_hand_data(hand_data)
            else:
                processed_frame_count += 1
                if pick_runtime.active:
                    command_to_send = pick_runtime.tick(robot_feedback=robot_feedback)

            if command_to_send is not None and real_robot_enabled:
                t_robot = time.perf_counter()
                if bool(getattr(val, "ROBOT_COMMAND_ASYNC_ONLY", True)) or bool(getattr(val, "REAL_ROBOT_ASYNC_COMMAND_SENDER", True)):
                    robot.submit_latest_command(command_to_send)
                else:
                    robot.send_if_due(command_to_send)
                perf.robot_enqueue_ms = (time.perf_counter() - t_robot) * 1000.0
                if isinstance(getattr(robot, "last_command_diagnostics", None), dict):
                    perf.robot_enqueue_ms = float(robot.last_command_diagnostics.get("serial_write_ms", perf.robot_enqueue_ms))
                perf.robot_commands += 1

            _draw_main_hud(frame, pick_runtime, snap_event)
            perf.update_rates(time.time())
            _draw_perf_overlay(frame, perf)
            t_display = time.perf_counter()
            cv2.imshow("Hand Tracking / Main", frame)

            key = cv2.waitKey(1) & 0xFF
            perf.display_ms = (time.perf_counter() - t_display) * 1000.0
            if key == 27:
                break
            if key == trigger_key_code:
                pick_runtime.request_start(f"key '{trigger_key}'")
            elif key == ord("c"):
                pick_runtime.cancel()

            elapsed = time.time() - loop_start
            perf.total_ms = elapsed * 1000.0
            perf.maybe_print()
            sleep = period - elapsed
            if sleep > 0 and not bool(getattr(val, "LOW_LATENCY_MODE", True)):
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
            if latest_camera is not None:
                latest_camera.release()
            elif cap is not None:
                cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
