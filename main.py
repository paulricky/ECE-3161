import time
from pathlib import Path

import cv2

import values as val
from robot_controller import SOArmHardwareController, JointCommand
from handtracking import HandTracker
from calibrate_robot_joints import main as run_robot_calibration


DEFAULT_ROBOT_CALIBRATION_FILE = Path(__file__).resolve().parent / "calibration_data" / "robot_joint_calibration.json"


def _robot_calibration_path() -> Path:
    configured = getattr(val, "ROBOT_JOINT_CALIBRATION_FILE", "")
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_ROBOT_CALIBRATION_FILE


def _ensure_robot_calibration() -> bool:
    if not getattr(val, "ENABLE_REAL_ROBOT", False):
        return True

    calib_path = _robot_calibration_path()
    if calib_path.exists():
        print(f"[main] Using robot calibration: {calib_path}")
        return True

    print(f"[main] Robot calibration file not found: {calib_path}")
    reply = input("Run robot calibration now? [Y/n]: ").strip().lower()
    if reply not in ("", "y", "yes"):
        print("[main] Calibration declined. Exiting.")
        return False

    rc = run_robot_calibration()
    if rc != 0:
        print("[main] Calibration did not complete successfully. Exiting.")
        return False

    if not calib_path.exists():
        print(f"[main] Calibration finished, but the file was still not found: {calib_path}")
        return False

    print(f"[main] Calibration completed and saved to: {calib_path}")
    return True


def main():
    if not _ensure_robot_calibration():
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    tracker = HandTracker()

    robot = SOArmHardwareController()

    if val.ENABLE_REAL_ROBOT:
        try:
            robot.connect()
            print("[main] Real robot connected")
        except Exception as e:
            print(f"[main] Failed to connect robot: {e}")
            cap.release()
            return
    else:
        print("[main] Real robot disabled")
        cap.release()
        return

    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        hand_data = tracker.process(frame)

        if hand_data is not None:
            cmd = JointCommand(
                shoulder_pan=float(hand_data["shoulder_pan"]),
                shoulder_lift=float(hand_data["shoulder_lift"]),
                elbow_flex=float(hand_data["elbow_flex"]),
                wrist_flex=float(hand_data["wrist_flex"]),
                wrist_yaw=float(hand_data["wrist_yaw"]),
                wrist_roll=float(hand_data["wrist_roll"]),
                gripper_open01=float(hand_data["gripper_open01"]),
            )

            robot.send_if_due(cmd)

        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        now = time.time()
        dt = now - last_time
        if dt < 1.0 / val.REAL_ROBOT_HZ:
            time.sleep((1.0 / val.REAL_ROBOT_HZ) - dt)
        last_time = now

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()