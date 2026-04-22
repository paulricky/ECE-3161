import time
import cv2

import values as val
from robot_controller import SOArmHardwareController, JointCommand
from handtracking import HandTracker

def main():

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
            return
    else:
        print("[main] Real robot disabled")
        return

    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        hand_data = tracker.process(frame)

        if hand_data is not None:

            # panned = np.interp(hand_data["shoulder_pan"], [-1, 1], [val.BASE_PAN_MIN, val.BASE_PAN_MAX])
            # lifted = np.interp(hand_data["shoulder_lift"], [-1, 1], [val.SHOULDER_LIFT_MIN, val.SHOULDER_LIFT_MAX])
            # elbowed = np.interp(hand_data["elbow_flex"], [-1, 1], [val.ELBOW_MIN, val.ELBOW_MAX])
            # cmd = JointCommand(
            #     shoulder_pan=panned,  # Use the stretched value
            #     shoulder_lift=lifted,  # Use the stretched value
            #     elbow_flex=elbowed,  # Use the stretched value
            #     wrist_flex=hand_data["wrist_flex"],
            #     wrist_roll=hand_data["wrist_roll"],
            #     gripper_open01=hand_data["gripper_open01"],
            # )

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

        #rate limit
        now = time.time()
        dt = now - last_time
        if dt < 1.0 / val.REAL_ROBOT_HZ:
            time.sleep((1.0 / val.REAL_ROBOT_HZ) - dt)
        last_time = now

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()