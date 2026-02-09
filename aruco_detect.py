import cv2
import math
from ultralytics import YOLO

MARKER_SIZE_MM = 100
ORIGIN_ID = 0
CONFIDENCE_THRESHOLD = 0.5

model = YOLO('yolo26n.pt')

cap = cv2.VideoCapture(1)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

def get_marker_properties(corners):
    pts = corners[0]
    cx = int(pts[:, 0].sum() / 4)
    cy = int(pts[:, 1].sum() / 4)
    dx = pts[1][0] - pts[0][0]
    dy = pts[1][1] - pts[0][1]
    angle_rad = math.atan2(dy, dx)
    px_per_mm = math.sqrt(dx**2 + dy**2) / MARKER_SIZE_MM
    return cx, cy, angle_rad, px_per_mm

def transform_to_robot_frame(obj_x, obj_y, origin_x, origin_y, origin_angle, px_per_mm):
    dx = obj_x - origin_x
    dy = obj_y - origin_y
    rx = dx * math.cos(-origin_angle) - dy * math.sin(-origin_angle)
    ry = dx * math.sin(-origin_angle) + dy * math.cos(-origin_angle)
    return int(rx / px_per_mm), int(ry / px_per_mm)

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    ox, oy, origin_angle, px_per_mm = 0, 0, 0, 0
    origin_found = False

    if ids is not None:
        ids_list = ids.flatten().tolist()
        if ORIGIN_ID in ids_list:
            idx = ids_list.index(ORIGIN_ID)
            ox, oy, origin_angle, px_per_mm = get_marker_properties(corners[idx])
            origin_found = True
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)


    results = model(frame, device='mps', verbose=False) #metal accel

    for r in results:
        for box in r.boxes:
            if float(box.conf[0]) < CONFIDENCE_THRESHOLD: continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            name = model.names[int(box.cls[0])]

            tx, ty = int((x1 + x2) / 2), int(y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)

            if origin_found:
                rx, ry = transform_to_robot_frame(tx, ty, ox, oy, origin_angle, px_per_mm)
                cv2.putText(frame, f"{name}: {rx}mm, {ry}mm", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.line(frame, (ox, oy), (tx, ty), (0, 255, 255), 1)

    cv2.imshow('feed', frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()