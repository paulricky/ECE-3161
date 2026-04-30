"""Minimal hand-tracking → robot teleop.

Pipeline:
  camera → SimpleHandTracker.process(frame) → JointCommand → robot.

ESC always disables motor torque before exit, so the arm goes limp instead of
holding its last commanded position. --dry-run skips the robot connection
entirely and just shows the mapping in the camera window.

Run:
    python3 simple_main.py --dry-run        # see the mapping, arm doesn't move
    python3 simple_main.py                  # send to the robot
    python3 simple_main.py --camera 1       # pick a different camera index
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from typing import Optional

import cv2

import values as val
from robot_controller import JointCommand, SOArmHardwareController
from simple_handtrack import SimpleHandTracker


def _build_joint_command(joints: dict, gripper_open01: float) -> JointCommand:
    return JointCommand(
        shoulder_pan=float(joints["shoulder_pan"]),
        shoulder_lift=float(joints["shoulder_lift"]),
        elbow_flex=float(joints["elbow_flex"]),
        wrist_flex=float(joints["wrist_flex"]),
        wrist_yaw=float(joints.get("wrist_yaw", 0.0)),
        wrist_roll=float(joints.get("wrist_roll", 0.0)),
        wrist_pitch=float(joints.get("wrist_pitch", 0.0)),
        gripper_open01=float(gripper_open01),
    )


def _cleanup(robot: Optional[SOArmHardwareController], cap: Optional[cv2.VideoCapture], tracker: Optional[SimpleHandTracker]) -> None:
    if robot is not None:
        try:
            robot.release_torque()
        except Exception as exc:
            print(f"[simple_main] release_torque failed: {exc}")
        try:
            robot.disconnect()
        except Exception as exc:
            print(f"[simple_main] disconnect failed: {exc}")
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    if tracker is not None:
        try:
            tracker.close()
        except Exception:
            pass
    cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Don't connect to the robot")
    parser.add_argument("--camera", type=int, default=int(getattr(val, "HANDTRACKING_CAMERA_INDEX", 0)))
    parser.add_argument("--no-flip", action="store_true", help="Don't horizontally mirror the camera frame")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[simple_main] could not open camera index {args.camera}")
        return 1

    tracker = SimpleHandTracker()
    robot: Optional[SOArmHardwareController] = None

    if not args.dry_run:
        robot = SOArmHardwareController()
        try:
            robot.connect()
        except Exception as exc:
            print(f"[simple_main] robot connect failed: {exc}")
            _cleanup(None, cap, tracker)
            return 1
        try:
            robot.start_async_sender()
        except Exception as exc:
            print(f"[simple_main] start_async_sender failed: {exc}")

    def sigint_handler(*_):
        print("\n[simple_main] SIGINT — releasing torque and exiting")
        _cleanup(robot, cap, tracker)
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    flip = not args.no_flip
    print(f"[simple_main] running. ESC to quit. dry_run={args.dry_run} flip={flip}")

    last_print = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if flip:
                frame = cv2.flip(frame, 1)

            data = tracker.process(frame)
            if data is not None and robot is not None:
                cmd = _build_joint_command(data["joints"], data["gripper_open01"])
                robot.submit_latest_command(cmd)
            elif data is not None and (time.time() - last_print) > 0.5:
                xyz = data["target_xyz"]
                j = data["joints"]
                print(
                    f"[dry] xyz=({xyz[0]:+.3f},{xyz[1]:+.3f},{xyz[2]:+.3f}) "
                    f"pan={j['shoulder_pan']:+.2f} lift={j['shoulder_lift']:+.2f} "
                    f"elbow={j['elbow_flex']:+.2f} wflex={j['wrist_flex']:+.2f} "
                    f"grip={data['gripper_open01']:.2f}"
                )
                last_print = time.time()

            cv2.imshow("Simple Hand Tracking", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
    finally:
        _cleanup(robot, cap, tracker)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
