import cv2
import numpy as np
import math

from ultralytics import YOLO

import values as val

MARKER_SIZE_MM = 100
ORIGIN_ID = 0
CONFIDENCE_THRESHOLD = 0.5

model = YOLO('yolo26n.pt')

cap = cv2.VideoCapture(1)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)


def _order_quad(pts):
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def detect_aruco(gray, aruco_dict=cv2.aruco.DICT_5X5_250):
    d = cv2.aruco.getPredefinedDictionary(aruco_dict)
    params = cv2.aruco.DetectorParameters()
    det = cv2.aruco.ArucoDetector(d, params)
    corners, ids, _rej = det.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None
    ids = ids.flatten().tolist()
    return corners, ids

def marker_pose_from_corners(corners_4x2):
    pts = np.array(corners_4x2, dtype=np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    v = pts[1] - pts[0]  # top edge direction
    ang = math.atan2(v[1], v[0])
    return (float(c[0]), float(c[1])), float(ang)

def detect_largest_rectangle(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, None, iterations=2)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0.0

    for c in cnts:
        area = cv2.contourArea(c)
        if area < 5000:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue
        if area > best_area:
            best_area = area
            best = approx.reshape(4, 2)

    if best is None:
        return None
    return _order_quad(best)

def rect_homography_to_workspace(rect_img_4x2):
    xmin, xmax = val.WORKSPACE_X_MIN, val.WORKSPACE_X_MAX
    ymin, ymax = val.WORKSPACE_Y_MIN, val.WORKSPACE_Y_MAX

    dst = np.array([
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax],
    ], dtype=np.float32)

    H, _ = cv2.findHomography(rect_img_4x2.astype(np.float32), dst, method=0)
    if H is None:
        return None
    Hinv = np.linalg.inv(H)
    return H, Hinv

def maybe_rotate_rect_using_marker(rect_img_4x2, marker_center, marker_angle):
    # Heuristic: ensure marker "top edge direction" roughly aligns with rectangle top edge.
    # If it aligns better after a 90deg corner rotation, rotate corner ordering.
    rect = rect_img_4x2.copy()
    top_edge = rect[1] - rect[0]
    rect_ang = math.atan2(float(top_edge[1]), float(top_edge[0]))

    def angdiff(a, b):
        d = (a - b + math.pi) % (2 * math.pi) - math.pi
        return abs(d)

    best = rect
    best_score = angdiff(marker_angle, rect_ang)

    # Try rotating the corner list (TL,TR,BR,BL) -> (TR,BR,BL,TL) etc.
    for k in range(1, 4):
        r = np.roll(rect, -k, axis=0)
        ang = math.atan2(float((r[1]-r[0])[1]), float((r[1]-r[0])[0]))
        score = angdiff(marker_angle, ang)
        if score < best_score:
            best_score = score
            best = r

    return best
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