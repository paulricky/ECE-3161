import time
import cv2

import values as val
from robot_controller import SOArmHardwareController, JointCommand
from handtracking import HandTracker


def main():
    # -------------------------
    # Init camera + hand tracker
    # -------------------------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    tracker = HandTracker()

    # -------------------------
    # Init robot
    # -------------------------
    robot = SOArmHardwareController()

    if val.ENABLE_REAL_ROBOT:
        try:
            robot.connect()
            print("[main] Real robot connected")
        except Exception as e:
            print(f"[main] Failed to connect robot: {e}")
            return
    else:
        print("[main] Real robot disabled")
        return

    # -------------------------
    # Main loop
    # -------------------------
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for natural interaction
        frame = cv2.flip(frame, 1)

        # -------------------------
        # Hand tracking
        # -------------------------
        hand_data = tracker.process(frame)

        if hand_data is not None:
            # Expect your tracker to output normalized joint targets
            # Modify this mapping if needed based on your tracker output
            cmd = JointCommand(
                shoulder_pan=hand_data["shoulder_pan"],
                shoulder_lift=hand_data["shoulder_lift"],
                elbow_flex=hand_data["elbow_flex"],
                wrist_flex=hand_data["wrist_flex"],
                wrist_roll=hand_data["wrist_roll"],
                gripper_open01=hand_data["gripper_open01"],
            )

            robot.send_if_due(cmd)

        # -------------------------
        # Display
        # -------------------------
        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        # Optional rate limit
        now = time.time()
        dt = now - last_time
        if dt < 1.0 / val.REAL_ROBOT_HZ:
            time.sleep((1.0 / val.REAL_ROBOT_HZ) - dt)
        last_time = now

    # -------------------------
    # Cleanup
    # -------------------------
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()