import cv2
import numpy as np

CHECKERBOARD = (6, 9)
MIN_FRAMES = 15

def run_calibration():
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    cap = cv2.VideoCapture(1)


    print(f"--- CALIBRATION ---")
    print(f"Press 'c' to capture. Min: {MIN_FRAMES} frames.")
    print(f"q to quit.")

    captured_count = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        if ret == True:
            cv2.drawChessboardCorners(display_frame, CHECKERBOARD, corners, ret)
            cv2.putText(display_frame, "Pattern Found", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(display_frame, f"Captured: {captured_count}/{MIN_FRAMES}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow('Calibration', display_frame)
        key = cv2.waitKey(1)

        if key == ord('c') and ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            captured_count += 1
            print(f"Pic #: ({captured_count})")
            cv2.imshow('Calibration', np.zeros_like(display_frame))
            cv2.waitKey(50)

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured_count < 5:
        print("Not enough frames to calibrate.")
        return

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    np.savez("../robot_arm/vision/calibration_data.npz", mtx=mtx, dist=dist)
    print("Calibration data saved to 'calibration_data.npz'")
    print(f"Camera Matrix:\n{mtx}")

if __name__ == "__main__":
    run_calibration()