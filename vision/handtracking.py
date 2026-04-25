from __future__ import annotations

import json
import math
import os
import time
import threading
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

from robot import mathmodel as mm
from config import values as val


action_log = deque(maxlen=val.LOG_MAX)

_snap_state = {
    "Left": {"pinched": False, "prev_d": None, "cooldown_until": 0.0},
    "Right": {"pinched": False, "prev_d": None, "cooldown_until": 0.0},
}

_hand_open_state = {"Left": None, "Right": None}
_hand_closed_bool = {"Left": False, "Right": False}

_prev_hands_dist = None
_clap_cooldown_until = 0.0

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=int(getattr(val, "HANDTRACKING_MAX_NUM_HANDS", 2)),
    model_complexity=int(getattr(val, "HANDTRACKING_MODEL_COMPLEXITY", 0)),
    min_detection_confidence=float(getattr(val, "HANDTRACKING_MIN_DETECTION_CONFIDENCE", 0.6)),
    min_tracking_confidence=float(getattr(val, "HANDTRACKING_MIN_TRACKING_CONFIDENCE", 0.6)),
)


def log_event(text: str):
    action_log.append((time.time(), str(text)))


def build_detected_hands(results):
    out = []
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            score = float(handedness.classification[0].score)
            out.append((hand_lms, label, score))
    return out


def choose_driver(detected_hands):
    for h in detected_hands:
        if h[1] == "Right":
            return h
    return detected_hands[0] if detected_hands else None


def fingertips_spread(hand_lms) -> float:
    lm = hand_lms.landmark
    tips = [
        (lm[4].x, lm[4].y),
        (lm[8].x, lm[8].y),
        (lm[12].x, lm[12].y),
        (lm[16].x, lm[16].y),
        (lm[20].x, lm[20].y),
    ]
    cx = sum(p[0] for p in tips) / 5.0
    cy = sum(p[1] for p in tips) / 5.0
    c = (cx, cy)
    return sum(mm.dist(p, c) for p in tips) / 5.0


def _clip(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _get_limit(name: str, default_lo: float, default_hi: float):
    lo = getattr(val, f"{name}_MIN", default_lo)
    hi = getattr(val, f"{name}_MAX", default_hi)
    return float(lo), float(hi)


def _norm_to_range(z: float, lo: float, hi: float) -> float:
    z = _clip(z, 0.0, 1.0)
    return _lerp(lo, hi, z)


def _angle_2d(a, b) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])


def _vec(a, b):
    return np.array([float(b[0]) - float(a[0]), float(b[1]) - float(a[1])], dtype=np.float64)


def _norm(v):
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return v * 0.0
    return v / n


def _load_default_lerobot_calibration_path():
    path = getattr(val, "LEROBOT_CALIBRATION_FILE", "").strip()
    if path:
        return path
    robot_id = getattr(val, "REAL_ROBOT_ID", "my_awesome_follower_arm")
    home = os.path.expanduser("~")
    return os.path.join(
        home,
        ".cache",
        "huggingface",
        "lerobot",
        "calibration",
        "robots",
        "so101_follower",
        f"{robot_id}.json",
    )


def _rvec_tvec_to_T(rvec, tvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    t = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3:] = t
    return T


def _T_inv(T):
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def _T_apply(T, p):
    ph = np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
    q = T @ ph
    return q[:3]


def _rot_to_rpy(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float64)


def _safe_npz_value(d, keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def _count_extended_fingers(hand_lms, label: str):
    lm = hand_lms.landmark

    wrist = (lm[0].x, lm[0].y)
    thumb_mcp = (lm[2].x, lm[2].y)
    thumb_ip = (lm[3].x, lm[3].y)
    thumb_tip = (lm[4].x, lm[4].y)

    index_mcp = (lm[5].x, lm[5].y)
    index_pip = (lm[6].x, lm[6].y)
    index_tip = (lm[8].x, lm[8].y)

    middle_mcp = (lm[9].x, lm[9].y)
    middle_pip = (lm[10].x, lm[10].y)
    middle_tip = (lm[12].x, lm[12].y)

    ring_mcp = (lm[13].x, lm[13].y)
    ring_pip = (lm[14].x, lm[14].y)
    ring_tip = (lm[16].x, lm[16].y)

    pinky_mcp = (lm[17].x, lm[17].y)
    pinky_pip = (lm[18].x, lm[18].y)
    pinky_tip = (lm[20].x, lm[20].y)

    palm_width = mm.dist(index_mcp, pinky_mcp)
    palm_height = mm.dist(wrist, middle_mcp)
    palm_size = max(1e-6, 0.5 * (palm_width + palm_height))

    ext_on = float(getattr(val, "OPEN_FINGER_EXTENDED_ON", 0.62))
    ext_off = float(getattr(val, "OPEN_FINGER_EXTENDED_OFF", 0.52))

    state_closed = _hand_closed_bool[label]
    ext_thr = ext_on if state_closed else ext_off

    def straight_score(mcp, pip, tip):
        a = _norm(_vec(pip, tip))
        b = _norm(_vec(mcp, pip))
        straight = float(np.dot(a, b))
        tip_far = (mm.dist(tip, wrist) - mm.dist(pip, wrist)) / palm_size
        score = 0.55 * straight + 0.45 * tip_far
        return score

    index_score = straight_score(index_mcp, index_pip, index_tip)
    middle_score = straight_score(middle_mcp, middle_pip, middle_tip)
    ring_score = straight_score(ring_mcp, ring_pip, ring_tip)
    pinky_score = straight_score(pinky_mcp, pinky_pip, pinky_tip)

    palm_axis = _norm(_vec(index_mcp, pinky_mcp))
    thumb_dir = _norm(_vec(thumb_mcp, thumb_tip))
    thumb_outward = abs(float(np.cross(palm_axis, thumb_dir)))
    thumb_far = (mm.dist(thumb_tip, wrist) - mm.dist(thumb_ip, wrist)) / palm_size
    thumb_score = 0.55 * thumb_outward + 0.45 * thumb_far

    thumb_extended = thumb_score > ext_thr
    index_extended = index_score > ext_thr
    middle_extended = middle_score > ext_thr
    ring_extended = ring_score > ext_thr
    pinky_extended = pinky_score > ext_thr

    extended_flags = {
        "thumb": thumb_extended,
        "index": index_extended,
        "middle": middle_extended,
        "ring": ring_extended,
        "pinky": pinky_extended,
    }
    extended_scores = {
        "thumb": thumb_score,
        "index": index_score,
        "middle": middle_score,
        "ring": ring_score,
        "pinky": pinky_score,
    }

    extended_count = sum(1 for v in extended_flags.values() if v)

    pinch_dist = mm.dist(thumb_tip, index_tip) / palm_size

    return extended_count, extended_flags, extended_scores, pinch_dist, palm_size


def openness_from_fingertips(hand_lms, label: str):
    global _hand_closed_bool

    extended_count, extended_flags, extended_scores, pinch_dist, _palm_size = _count_extended_fingers(hand_lms, label)

    min_open_fingers = int(getattr(val, "OPEN_HAND_MIN_OPEN_FINGERS", 3))
    max_closed_fingers = int(getattr(val, "OPEN_HAND_MAX_CLOSED_FINGERS", 1))
    pinch_block = float(getattr(val, "OPEN_HAND_PINCH_BLOCK", 0.10))

    st = _hand_closed_bool[label]

    if st:
        if extended_count >= min_open_fingers and pinch_dist > pinch_block:
            st = False
    else:
        if extended_count <= max_closed_fingers:
            st = True

    _hand_closed_bool[label] = st

    open01 = (extended_count - max_closed_fingers) / max(1.0, float(min_open_fingers - max_closed_fingers))
    open01 = max(0.0, min(1.0, open01))

    if pinch_dist < pinch_block:
        open01 *= 0.5

    debug = {
        "extended_count": extended_count,
        "extended_flags": extended_flags,
        "extended_scores": extended_scores,
        "pinch_dist": pinch_dist,
    }

    return st, open01, float(extended_count), debug


def update_snap_and_open_state(hand_lms, label: str, now: float, dt: float):
    lm = hand_lms.landmark
    thumb_tip = (lm[4].x, lm[4].y)
    middle_tip = (lm[12].x, lm[12].y)

    d_tm = mm.dist(thumb_tip, middle_tip)
    st = _snap_state[label]
    snap_event = False

    if now > st["cooldown_until"]:
        pinch_on = d_tm < val.SNAP_PINCH_ON
        pinch_off = d_tm > val.SNAP_PINCH_OFF
        opening_speed = (d_tm - st["prev_d"]) / dt if st["prev_d"] is not None and dt > 1e-9 else 0.0

        if not st["pinched"]:
            if pinch_on:
                st["pinched"] = True
        else:
            if pinch_off and opening_speed > val.SNAP_FAST_RELEASE:
                snap_event = True
                st["pinched"] = False
                st["cooldown_until"] = now + val.SNAP_COOLDOWN_S
                log_event(f"{label} SNAP")
            elif pinch_off:
                st["pinched"] = False

    st["prev_d"] = d_tm

    is_closed, open01, _metric, debug = openness_from_fingertips(hand_lms, label)
    state = "CLOSED" if is_closed else ("OPEN" if open01 > 0.8 else "PARTIAL")

    if _hand_open_state[label] != state:
        _hand_open_state[label] = state
        log_event(f"{label} {state}")

    return state, open01, snap_event, debug


def detect_clap(detected_hands, now: float, dt: float):
    global _prev_hands_dist, _clap_cooldown_until
    clap_event = False

    if len(detected_hands) == 2:
        c0 = mm.hand_center_xy(detected_hands[0][0])
        c1 = mm.hand_center_xy(detected_hands[1][0])
        d = mm.dist(c0, c1)

        if _prev_hands_dist is not None and dt > 1e-9:
            closing_speed = (_prev_hands_dist - d) / dt
            if now > _clap_cooldown_until and d < val.CLAP_CLOSE_ENOUGH and closing_speed > val.CLAP_FAST_CLOSING:
                clap_event = True
                _clap_cooldown_until = now + val.CLAP_COOLDOWN_S
                log_event("CLAP")

        _prev_hands_dist = d
    else:
        _prev_hands_dist = None

    return clap_event


def draw_and_update_gestures(frame, detected_hands, now: float, dt: float):
    clap_event = detect_clap(detected_hands, now, dt)
    per_hand = {}

    for hand_lms, label, score in detected_hands:
        mp_draw.draw_landmarks(
            frame,
            hand_lms,
            mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style(),
        )

        state, open01, snap_event, debug = update_snap_and_open_state(hand_lms, label, now, dt)
        per_hand[label] = {
            "state": state,
            "open01": open01,
            "snap": snap_event,
            "score": score,
            "debug": debug,
        }

        h_img, w_img = frame.shape[:2]
        xw = int(hand_lms.landmark[0].x * w_img)
        yw = int(hand_lms.landmark[0].y * h_img)

        extras = [state]
        if snap_event:
            extras.append("SNAP!")
        if clap_event:
            extras.append("CLAP!")

        finger_count = debug["extended_count"]
        flags = debug["extended_flags"]
        flag_text = (
            f"T{int(flags['thumb'])}"
            f"I{int(flags['index'])}"
            f"M{int(flags['middle'])}"
            f"R{int(flags['ring'])}"
            f"P{int(flags['pinky'])}"
        )

        text1 = f"{label} ({score:.2f}) {extras}"
        text2 = f"fingers={finger_count} open={open01:.2f} {flag_text}"

        cv2.putText(frame, text1, (xw + 10, yw - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, text2, (xw + 10, yw - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2, cv2.LINE_AA)

    return clap_event, per_hand


class ArucoGloveTracker:
    def __init__(self):
        self.enabled = bool(getattr(val, "ARUCO_GLOVE_ENABLED", True))
        self.marker_size_m = float(getattr(val, "ARUCO_MARKER_SIZE_M", 0.03))
        self.front_id = int(getattr(val, "ARUCO_GLOVE_FRONT_ID", 10))
        self.back_id = int(getattr(val, "ARUCO_GLOVE_BACK_ID", 11))
        self.camera_matrix = None
        self.dist_coeffs = None
        self.T_workspace_from_camera = None
        self.workspace_min = np.array(getattr(val, "ARUCO_WORKSPACE_MIN", (-0.18, -0.12, 0.02)), dtype=np.float64)
        self.workspace_max = np.array(getattr(val, "ARUCO_WORKSPACE_MAX", (0.18, 0.18, 0.28)), dtype=np.float64)
        self._stable_id = None
        self._stable_count = 0
        self._required_stable_frames = 3
        self.last_debug = {
            "status": "disabled",
            "all_ids": [],
            "candidate_count": 0,
            "rejected_count": 0,
            "chosen_id": None,
            "pose_ok": False,
        }

        if not self.enabled:
            self.detector = None
            return

        dict_name = getattr(val, "ARUCO_DICT_NAME", "DICT_4X4_50")
        dict_id = getattr(cv2.aruco, dict_name)
        dictionary = cv2.aruco.getPredefinedDictionary(dict_id)

        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 50
        params.cornerRefinementMinAccuracy = 0.01

        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 53
        params.adaptiveThreshWinSizeStep = 4
        params.adaptiveThreshConstant = 7.0

        params.minMarkerPerimeterRate = 0.01
        params.maxMarkerPerimeterRate = 6.0
        params.polygonalApproxAccuracyRate = 0.08
        params.minCornerDistanceRate = 0.02
        params.minDistanceToBorder = 1
        params.minOtsuStdDev = 3.0
        params.perspectiveRemoveIgnoredMarginPerCell = 0.20
        params.maxErroneousBitsInBorderRate = 0.6
        params.errorCorrectionRate = 0.8

        self.detector = cv2.aruco.ArucoDetector(dictionary, params)

        self._load_intrinsics()
        self._load_extrinsics()
        self._load_workspace_bounds()

    def _load_intrinsics(self):
        path = getattr(val, "CALIB_INTRINSICS_FILE", "")
        if not path or not os.path.exists(path):
            return
        d = np.load(path, allow_pickle=True)
        K = _safe_npz_value(d, ["camera_matrix", "K", "mtx"])
        dist = _safe_npz_value(d, ["dist_coeffs", "dist", "distortion_coefficients"])
        if K is None or dist is None:
            return
        self.camera_matrix = np.asarray(K, dtype=np.float64)
        self.dist_coeffs = np.asarray(dist, dtype=np.float64).reshape(-1, 1)

    def _load_extrinsics(self):
        path = getattr(val, "CALIB_EXTRINSICS_FILE", "")
        if not path or not os.path.exists(path):
            return
        d = np.load(path, allow_pickle=True)

        mode = getattr(val, "EXTRINSICS_MODE", "workspace_to_camera")

        R = _safe_npz_value(d, ["R"])
        t = _safe_npz_value(d, ["t", "T"])
        rvec = _safe_npz_value(d, ["rvec"])
        tvec = _safe_npz_value(d, ["tvec"])

        if R is not None and t is not None:
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
            T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
        elif rvec is not None and tvec is not None:
            T = _rvec_tvec_to_T(rvec, tvec)
        else:
            return

        if mode == "workspace_to_camera":
            self.T_workspace_from_camera = _T_inv(T)
        else:
            self.T_workspace_from_camera = T

    def _load_workspace_bounds(self):
        path = getattr(val, "CALIB_WORKSPACE_FILE", "")
        if not path or not os.path.exists(path):
            return
        d = np.load(path, allow_pickle=True)
        mn = _safe_npz_value(d, ["workspace_min", "xyz_min", "min_xyz"])
        mx = _safe_npz_value(d, ["workspace_max", "xyz_max", "max_xyz"])
        if mn is not None and mx is not None:
            self.workspace_min = np.asarray(mn, dtype=np.float64).reshape(3)
            self.workspace_max = np.asarray(mx, dtype=np.float64).reshape(3)

    def _marker_object_points(self):
        s = self.marker_size_m / 2.0
        return np.array(
            [
                [-s, s, 0.0],
                [s, s, 0.0],
                [s, -s, 0.0],
                [-s, -s, 0.0],
            ],
            dtype=np.float64,
        )

    def _detect_candidates(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)

        if ids is None:
            ids_list = []
        else:
            ids_list = [int(x) for x in ids.flatten().tolist()]

        return corners, ids_list, rejected

    def detect(self, frame):
        self.last_debug = {
            "status": "searching",
            "all_ids": [],
            "candidate_count": 0,
            "rejected_count": 0,
            "chosen_id": None,
            "pose_ok": False,
        }

        if self.detector is None:
            self.last_debug["status"] = "disabled"
            return None

        corners, ids_list, rejected = self._detect_candidates(frame)

        self.last_debug["all_ids"] = ids_list
        self.last_debug["candidate_count"] = len(corners)
        self.last_debug["rejected_count"] = 0 if rejected is None else len(rejected)

        if len(corners) == 0:
            self.last_debug["status"] = "no_markers"
            self._stable_id = None
            self._stable_count = 0
            return None

        chosen_idx = None
        chosen_id = None

        for candidate in (self.front_id, self.back_id):
            if candidate in ids_list:
                chosen_idx = ids_list.index(candidate)
                chosen_id = candidate
                break

        if chosen_idx is None:
            self.last_debug["status"] = "wrong_ids"
            self._stable_id = None
            self._stable_count = 0
            return None

        if self._stable_id == chosen_id:
            self._stable_count += 1
        else:
            self._stable_id = chosen_id
            self._stable_count = 1

        self.last_debug["chosen_id"] = int(chosen_id)
        self.last_debug["stable_count"] = self._stable_count

        if self._stable_count < self._required_stable_frames:
            self.last_debug["status"] = "id_unstable"
            return None

        if self.camera_matrix is None or self.dist_coeffs is None or self.T_workspace_from_camera is None:
            self.last_debug["status"] = "missing_calibration"
            return None

        image_points = np.asarray(corners[chosen_idx], dtype=np.float64).reshape(4, 2)

        ok, rvec, tvec = cv2.solvePnP(
            self._marker_object_points(),
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )

        if not ok:
            self.last_debug["status"] = "pose_failed"
            return None

        T_camera_from_marker = _rvec_tvec_to_T(rvec, tvec)
        marker_origin_camera = T_camera_from_marker[:3, 3]
        marker_origin_workspace = _T_apply(self.T_workspace_from_camera, marker_origin_camera)

        R_marker_camera = T_camera_from_marker[:3, :3]
        R_workspace_marker = self.T_workspace_from_camera[:3, :3] @ R_marker_camera
        workspace_rpy = _rot_to_rpy(R_workspace_marker)

        self.last_debug["status"] = "pose_ok"
        self.last_debug["pose_ok"] = True
        self.last_debug["chosen_corners"] = image_points

        return {
            "marker_id": int(chosen_id),
            "workspace_xyz": marker_origin_workspace,
            "workspace_rpy": workspace_rpy,
            "image_corners": image_points,
            "all_corners": corners,
            "all_ids": ids_list,
            "rejected": rejected,
        }

    def normalize_workspace_xyz(self, xyz):
        denom = self.workspace_max - self.workspace_min
        denom = np.where(np.abs(denom) < 1e-9, 1.0, denom)
        z = (np.asarray(xyz, dtype=np.float64) - self.workspace_min) / denom
        return np.clip(z, 0.0, 1.0)

    def draw_debug(self, frame, aruco_pose=None):
        dbg = self.last_debug

        if aruco_pose is not None and "all_corners" in aruco_pose and len(aruco_pose["all_corners"]) > 0:
            ids = None if len(aruco_pose.get("all_ids", [])) == 0 else np.array(aruco_pose["all_ids"], dtype=np.int32)
            cv2.aruco.drawDetectedMarkers(frame, aruco_pose["all_corners"], ids)

        status = dbg.get("status", "unknown")
        all_ids = dbg.get("all_ids", [])
        chosen_id = dbg.get("chosen_id", None)
        rejected_count = dbg.get("rejected_count", 0)

        status = dbg.get("status", "unknown")
        all_ids = dbg.get("all_ids", [])
        chosen_id = dbg.get("chosen_id", None)
        rejected_count = dbg.get("rejected_count", 0)
        stable_count = dbg.get("stable_count", 0)

        if status == "pose_ok":
            color = (0, 255, 0)
            msg = f"ARUCO DETECTED id={chosen_id} ids={all_ids}"
        elif status == "id_unstable":
            color = (0, 255, 255)
            msg = f"ARUCO id={chosen_id} unstable {stable_count}/{self._required_stable_frames}"
        elif status == "wrong_ids":
            color = (0, 0, 255)
            msg = f"ARUCO wrong ids seen={all_ids} expected={[self.front_id, self.back_id]}"
        elif status == "no_markers":
            color = (0, 0, 255)
            msg = f"ARUCO no marker detected rejected={rejected_count}"
        elif status == "pose_failed":
            color = (0, 165, 255)
            msg = f"ARUCO id={chosen_id} found but pose failed"
        elif status == "missing_calibration":
            color = (0, 165, 255)
            msg = "ARUCO marker found but camera/workspace calibration missing"
        elif status == "disabled":
            color = (128, 128, 128)
            msg = "ARUCO disabled"
        else:
            color = (255, 255, 0)
            msg = f"ARUCO status={status} ids={all_ids} rejected={rejected_count}"

        cv2.putText(frame, msg, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


class HandTracker:
    def __init__(self):
        self.prev_time = time.time()
        self._last_cmd = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_yaw": 0.0,
            "wrist_roll": 0.0,
            "wrist_pitch": 0.0,
            "gripper_open01": 1.0,
        }
        self._last_open01 = 1.0
        self._alpha = float(getattr(val, "HAND_CMD_SMOOTHING", 0.25))
        self._open_alpha = float(getattr(val, "HAND_STATE_SMOOTHING", 0.20))
        self.aruco = ArucoGloveTracker()
        self.lerobot_calibration = self._load_lerobot_calibration()
        self._last_ik_solution = None
        self._filtered_target_xyz = None
        self._filtered_target_rpy = None
        self._external_measured_joints = None
        self._feedback_seeded_last_command = False
        self._last_ik_time = 0.0
        self._last_ik_command = None
        self._last_ik_signature = None
        self._ik_async_enabled = bool(getattr(val, "HAND_IK_ASYNC", True))
        self._ik_lock = threading.RLock()
        self._ik_stop = threading.Event()
        self._ik_request = None
        self._ik_worker = None
        self._ik_busy = False
        self._ik_last_request_time = 0.0
        if self._ik_async_enabled:
            self._ik_worker = threading.Thread(target=self._ik_worker_loop, name="hand-ik-worker", daemon=True)
            self._ik_worker.start()
        self.last_gesture_events = {"snap": False, "clap": False, "per_hand": {}}

    def update_robot_feedback(self, joints_rad):
        """Optional hook used by main/pick-place code to seed IK with measured joints.

        The first valid feedback packet is also used to initialize the smoothed
        hand command. Without this, the command smoother starts near all zeros,
        which can create a huge first target and trip the tracking watchdog.
        """
        if isinstance(joints_rad, dict):
            clean = {}
            for k in self._last_cmd:
                if k in joints_rad:
                    try:
                        clean[k] = float(joints_rad[k])
                    except Exception:
                        pass
            if clean:
                self._external_measured_joints = dict(clean)
                if not self._feedback_seeded_last_command:
                    for k, v in clean.items():
                        if k in self._last_cmd and math.isfinite(float(v)):
                            self._last_cmd[k] = float(v)
                    if "gripper_open01" not in clean or not math.isfinite(float(clean.get("gripper_open01", float("nan")))):
                        self._last_cmd["gripper_open01"] = self._last_open01
                    self._last_ik_solution = {
                        k: float(self._last_cmd[k])
                        for k in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "wrist_pitch")
                    }
                    self._feedback_seeded_last_command = True

    def _load_lerobot_calibration(self):
        path = _load_default_lerobot_calibration_path()
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def process(self, frame):
        now = time.time()
        dt = now - self.prev_time
        if dt <= 0.0:
            dt = 1e-3
        self.prev_time = now

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        detected_hands = build_detected_hands(results)

        clap_event, per_hand = draw_and_update_gestures(frame, detected_hands, now, dt)
        snap_event = any(bool(info.get("snap", False)) for info in per_hand.values())
        self.last_gesture_events = {"snap": snap_event, "clap": bool(clap_event), "per_hand": per_hand}

        aruco_pose = self.aruco.detect(frame)
        self.aruco.draw_debug(frame, aruco_pose)

        if aruco_pose is not None:
            open01 = self._estimate_gripper_open01(detected_hands)
            cmd = self._aruco_pose_to_command(aruco_pose, open01)
            if cmd is not None:
                out = self._smooth_command(cmd)
                out["mode"] = "aruco"
                out["snap_event"] = snap_event
                out["clap_event"] = bool(clap_event)
                out["ee_target_xyz"] = aruco_pose["workspace_xyz"].tolist()
                self._draw_command_overlay(frame, out)
                self._draw_aruco_overlay(frame, aruco_pose)
                return out

        driver = choose_driver(detected_hands)
        if driver is None:
            return None

        hand_lms, label, _score = driver
        cmd = self._landmarks_to_command(hand_lms, label)
        out = self._smooth_command(cmd)
        out["mode"] = "mediapipe"
        out["snap_event"] = snap_event
        out["clap_event"] = bool(clap_event)
        self._draw_command_overlay(frame, out)
        return out

    def _smooth_command(self, cmd):
        for k in self._last_cmd:
            self._last_cmd[k] = _lerp(self._last_cmd[k], float(cmd[k]), self._alpha)
        return dict(self._last_cmd)

    def consume_snap_event(self) -> bool:
        snap = bool(self.last_gesture_events.get("snap", False))
        if snap:
            self.last_gesture_events["snap"] = False
            for info in self.last_gesture_events.get("per_hand", {}).values():
                if isinstance(info, dict):
                    info["snap"] = False
        return snap

    def close(self) -> None:
        self._ik_stop.set()
        worker = getattr(self, "_ik_worker", None)
        if worker is not None and worker.is_alive():
            worker.join(timeout=0.5)
        try:
            hands.close()
        except Exception:
            pass

    def _estimate_gripper_open01(self, detected_hands):
        driver = choose_driver(detected_hands)
        if driver is None:
            return self._last_open01

        hand_lms, label, _score = driver
        is_closed, open01, _metric, _debug = openness_from_fingertips(hand_lms, label)
        gripper_open01 = 0.0 if is_closed else float(open01)
        self._last_open01 = _lerp(self._last_open01, gripper_open01, self._open_alpha)
        return self._last_open01

    def _smooth_target_pose(self, xyz, rpy):
        xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
        rpy = np.asarray(rpy, dtype=np.float64).reshape(3)
        alpha = _clip(float(getattr(val, "POSE_SMOOTH_ALPHA", 0.25)), 0.0, 1.0)
        if self._filtered_target_xyz is None:
            self._filtered_target_xyz = xyz.copy()
        else:
            self._filtered_target_xyz = (1.0 - alpha) * self._filtered_target_xyz + alpha * xyz
        if self._filtered_target_rpy is None:
            self._filtered_target_rpy = rpy.copy()
        else:
            prev = self._filtered_target_rpy
            delta = np.array([math.atan2(math.sin(rpy[i] - prev[i]), math.cos(rpy[i] - prev[i])) for i in range(3)], dtype=np.float64)
            self._filtered_target_rpy = prev + alpha * delta
            self._filtered_target_rpy = np.array([math.atan2(math.sin(a), math.cos(a)) for a in self._filtered_target_rpy], dtype=np.float64)
        return self._filtered_target_xyz.copy(), self._filtered_target_rpy.copy()

    def _prev_joints_for_ik(self):
        # Prefer the last solved IK posture once it exists. Measured feedback is
        # best as a cold-start seed, but repeatedly using lagging robot feedback
        # as the seed makes the solver jump around while the arm ramps.
        if isinstance(self._last_ik_solution, dict):
            return self._last_ik_solution
        if isinstance(self._external_measured_joints, dict):
            return self._external_measured_joints
        return None

    def _cached_ik_command_if_fresh(self, xyz_f, rpy_f, open01):
        if self._last_ik_command is None:
            return None
        hz = float(getattr(val, "HAND_IK_HZ", 6.0))
        if hz <= 0.0:
            return None
        now = time.time()
        period = 1.0 / max(hz, 1e-3)
        if (now - self._last_ik_time) >= period:
            return None
        max_delta = float(getattr(val, "HAND_IK_FORCE_SOLVE_TARGET_DELTA_M", 0.025))
        sig = self._last_ik_signature
        if isinstance(sig, dict):
            try:
                old_xyz = np.asarray(sig.get("xyz"), dtype=np.float64).reshape(3)
                delta = float(np.linalg.norm(np.asarray(xyz_f, dtype=np.float64).reshape(3) - old_xyz))
                if delta > max_delta:
                    return None
            except Exception:
                return None
        out = dict(self._last_ik_command)
        out["gripper_open01"] = float(open01)
        out["ee_target_xyz"] = np.asarray(xyz_f, dtype=np.float64).reshape(3).tolist()
        out["ee_target_rpy"] = np.asarray(rpy_f, dtype=np.float64).reshape(3).tolist()
        out["__diagnostics__"] = dict(out.get("__diagnostics__", {}))
        out["__diagnostics__"]["ik_cached"] = True
        return out

    def _ik_base_command(self, open01, xyz_f=None, rpy_f=None, pending=False):
        """Return immediately using the most recent solved IK command.

        The foreground camera loop must not wait for the full 7-DOF numerical
        IK refinement.  A background worker updates _last_ik_command whenever a
        fresh target is available; MediaPipe processing reuses that most recent
        command and only updates the gripper value.
        """
        with self._ik_lock:
            if isinstance(self._last_ik_command, dict):
                out = dict(self._last_ik_command)
            elif isinstance(self._last_ik_solution, dict):
                out = dict(self._last_ik_solution)
            else:
                out = dict(self._last_cmd)
        out["gripper_open01"] = float(open01)
        if xyz_f is not None:
            out["ee_target_xyz"] = np.asarray(xyz_f, dtype=np.float64).reshape(3).tolist()
        if rpy_f is not None:
            out["ee_target_rpy"] = np.asarray(rpy_f, dtype=np.float64).reshape(3).tolist()
        diag = dict(out.get("__diagnostics__", {})) if isinstance(out.get("__diagnostics__", {}), dict) else {}
        diag["ik_async_pending"] = bool(pending)
        diag["ik_cached"] = True
        out["__diagnostics__"] = diag
        return out

    def _queue_async_ik(self, xyz_f, rpy_f, open01) -> None:
        now = time.time()
        with self._ik_lock:
            # Keep only the newest target.  Old hand positions are overwritten
            # instead of forming an IK backlog.
            self._ik_request = {
                "xyz": np.asarray(xyz_f, dtype=np.float64).reshape(3).copy(),
                "rpy": np.asarray(rpy_f, dtype=np.float64).reshape(3).copy(),
                "open01": float(open01),
                "time": now,
            }
            self._ik_last_request_time = now

    def _ik_worker_loop(self) -> None:
        hz = float(getattr(val, "HAND_IK_HZ", 4.0))
        period = 1.0 / max(hz, 1e-3) if hz > 0.0 else 0.25
        last_start = 0.0
        while not self._ik_stop.is_set():
            with self._ik_lock:
                req = self._ik_request
                self._ik_request = None
            if req is None:
                self._ik_stop.wait(0.005)
                continue
            wait = period - (time.time() - last_start)
            if wait > 0.0:
                self._ik_stop.wait(min(wait, 0.05))
                with self._ik_lock:
                    newer = self._ik_request
                    self._ik_request = None
                if newer is not None:
                    req = newer
            if self._ik_stop.is_set():
                break
            last_start = time.time()
            self._ik_busy = True
            try:
                self._solve_ik_now(req["xyz"], req["rpy"], req["open01"])
            except Exception as exc:
                log_event(f"IK worker error: {exc}")
            finally:
                self._ik_busy = False

    def _solve_ik_now(self, xyz_f, rpy_f, open01):
        solved = mm.solve_ik_from_target(
            target_xyz=xyz_f,
            target_rpy=rpy_f,
            gripper_open01=float(open01),
            lerobot_calibration=self.lerobot_calibration,
            previous_joints=self._prev_joints_for_ik(),
            ik_mode="teleop",
            strict_reachability=False,
        )
        if not isinstance(solved, dict):
            return None
        diag = solved.get("__diagnostics__", {})
        max_err = float(getattr(val, "HAND_IK_REJECT_POSITION_ERR_M", 0.06))
        if isinstance(diag, dict) and (not bool(diag.get("reachable", True))) and float(diag.get("position_error_m", 0.0)) > max_err:
            log_event(f"IK target rejected err={float(diag.get('position_error_m', 0.0)):.3f}m")
            return None
        out = {
            "shoulder_pan": float(solved["shoulder_pan"]),
            "shoulder_lift": float(solved["shoulder_lift"]),
            "elbow_flex": float(solved["elbow_flex"]),
            "wrist_flex": float(solved["wrist_flex"]),
            "wrist_yaw": float(solved["wrist_yaw"]),
            "wrist_roll": float(solved["wrist_roll"]),
            "wrist_pitch": float(solved.get("wrist_pitch", 0.0)),
            "gripper_open01": float(solved.get("gripper_open01", open01)),
            "__diagnostics__": diag,
            "ee_target_xyz": np.asarray(xyz_f, dtype=np.float64).reshape(3).tolist(),
            "ee_target_rpy": np.asarray(rpy_f, dtype=np.float64).reshape(3).tolist(),
        }
        with self._ik_lock:
            self._last_ik_solution = {k: float(out[k]) for k in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "wrist_pitch")}
            self._last_ik_time = time.time()
            self._last_ik_command = dict(out)
            self._last_ik_signature = {"xyz": out["ee_target_xyz"], "rpy": out["ee_target_rpy"]}
        return out

    def _solve_cartesian_command(self, xyz, rpy, open01):
        xyz_f, rpy_f = self._smooth_target_pose(xyz, rpy)
        if self._ik_async_enabled:
            self._queue_async_ik(xyz_f, rpy_f, open01)
            return self._ik_base_command(open01, xyz_f, rpy_f, pending=True)

        cached = self._cached_ik_command_if_fresh(xyz_f, rpy_f, open01)
        if cached is not None:
            return cached
        solved = self._solve_ik_now(xyz_f, rpy_f, open01)
        if solved is not None:
            return solved
        return self._ik_base_command(open01, xyz_f, rpy_f, pending=False)

    def _aruco_pose_to_command(self, aruco_pose, open01):
        xyz = np.asarray(aruco_pose["workspace_xyz"], dtype=np.float64)
        rpy = np.asarray(aruco_pose["workspace_rpy"], dtype=np.float64)
        cmd = self._solve_cartesian_command(xyz, rpy, open01)
        if cmd is not None:
            return cmd
        if self._last_ik_solution is not None:
            out = dict(self._last_ik_solution)
            out["gripper_open01"] = float(open01)
            return out
        return None


    def _draw_aruco_overlay(self, frame, aruco_pose):
        c = np.asarray(aruco_pose["image_corners"], dtype=np.int32)
        cv2.polylines(frame, [c.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        xyz = aruco_pose["workspace_xyz"]
        text = f"ARUCO xyz=({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f})"
        cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    def _draw_command_overlay(self, frame, out):
        h_img, _ = frame.shape[:2]
        cv2.putText(
            frame,
            f"{out.get('mode', 'unknown')} pan={out['shoulder_pan']:.2f} lift={out['shoulder_lift']:.2f} elbow={out['elbow_flex']:.2f}",
            (10, h_img - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"wflex={out['wrist_flex']:.2f} wyaw={out['wrist_yaw']:.2f} wroll={out['wrist_roll']:.2f} grip={out['gripper_open01']:.2f}",
            (10, h_img - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _estimate_hand_rpy_from_landmarks(self, hand_lms):
        lm = hand_lms.landmark
        wrist = np.array([lm[0].x, lm[0].y, lm[0].z], dtype=np.float64)
        index_mcp = np.array([lm[5].x, lm[5].y, lm[5].z], dtype=np.float64)
        middle_mcp = np.array([lm[9].x, lm[9].y, lm[9].z], dtype=np.float64)
        pinky_mcp = np.array([lm[17].x, lm[17].y, lm[17].z], dtype=np.float64)

        x_axis = middle_mcp - wrist
        if np.linalg.norm(x_axis) < 1e-8:
            x_axis = index_mcp - wrist
        x_axis = x_axis / max(np.linalg.norm(x_axis), 1e-8)

        across = pinky_mcp - index_mcp
        across = across / max(np.linalg.norm(across), 1e-8)

        z_axis = np.cross(x_axis, across)
        if np.linalg.norm(z_axis) < 1e-8:
            z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        z_axis = z_axis / max(np.linalg.norm(z_axis), 1e-8)

        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / max(np.linalg.norm(y_axis), 1e-8)

        R = np.column_stack([x_axis, y_axis, z_axis])
        return _rot_to_rpy(R)

    def _landmarks_to_command(self, hand_lms, label: str):
        lm = hand_lms.landmark
        wrist = (lm[0].x, lm[0].y)
        thumb_tip = (lm[4].x, lm[4].y)
        index_mcp = (lm[5].x, lm[5].y)
        index_tip = (lm[8].x, lm[8].y)
        middle_mcp = (lm[9].x, lm[9].y)
        pinky_mcp = (lm[17].x, lm[17].y)

        hand_cx = (wrist[0] + middle_mcp[0] + index_mcp[0] + pinky_mcp[0]) / 4.0
        hand_cy = (wrist[1] + middle_mcp[1] + index_mcp[1] + pinky_mcp[1]) / 4.0
        palm_width = mm.dist(index_mcp, pinky_mcp)
        palm_height = mm.dist(wrist, middle_mcp)
        size_metric = 0.5 * (palm_width + palm_height)

        is_closed, open01, _metric, _debug = openness_from_fingertips(hand_lms, label)
        pinch_dist = mm.dist(thumb_tip, index_tip)
        pinch_lo = float(getattr(val, "PINCH_CLOSE_DIST", 0.03))
        pinch_hi = float(getattr(val, "PINCH_OPEN_DIST", 0.12))
        if pinch_hi <= pinch_lo:
            pinch_hi = pinch_lo + 1e-3
        pinch_open01 = _clip((pinch_dist - pinch_lo) / (pinch_hi - pinch_lo), 0.0, 1.0)
        gripper_open01 = min(open01, pinch_open01)
        if is_closed:
            gripper_open01 = 0.0
        self._last_open01 = _lerp(self._last_open01, gripper_open01, self._open_alpha)
        gripper_open01 = self._last_open01

        x = _lerp(float(getattr(val, "WORKSPACE_X_MIN", -0.18)), float(getattr(val, "WORKSPACE_X_MAX", 0.18)), _clip(1.0 - hand_cx, 0.0, 1.0))
        z = _lerp(float(getattr(val, "WORKSPACE_Z_MIN", 0.04)), float(getattr(val, "WORKSPACE_Z_MAX", 0.28)), _clip(1.0 - hand_cy, 0.0, 1.0))
        size_near = float(getattr(val, "HAND_SIZE_NEAR", 0.22))
        size_far = float(getattr(val, "HAND_SIZE_FAR", 0.08))
        if abs(size_near - size_far) < 1e-6:
            size_near = size_far + 1e-3
        reach_norm = _clip((size_metric - size_far) / (size_near - size_far), 0.0, 1.0)
        y = _lerp(float(getattr(val, "WORKSPACE_Y_MAX", 0.38)), float(getattr(val, "WORKSPACE_Y_MIN", 0.12)), reach_norm)

        palm_rpy = self._estimate_hand_rpy_from_landmarks(hand_lms)
        palm_line_angle = _angle_2d(index_mcp, pinky_mcp)
        roll = float(palm_line_angle)
        pitch = _clip(float(palm_rpy[1]), -1.25, 1.25)
        yaw = float(palm_rpy[2])
        if label == "Left":
            yaw = -yaw
            roll = -roll
        rpy = np.array([roll, pitch + float(getattr(val, "HAND_TARGET_PITCH_BIAS_RAD", -0.15)), yaw], dtype=np.float64)
        xyz = np.array([x, y, z], dtype=np.float64)

        cmd = self._solve_cartesian_command(xyz, rpy, gripper_open01)
        if cmd is not None:
            return cmd
        if self._last_ik_solution is not None:
            out = dict(self._last_ik_solution)
            out["gripper_open01"] = gripper_open01
            return out
        return {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_yaw": 0.0,
            "wrist_roll": 0.0,
            "wrist_pitch": 0.0,
            "gripper_open01": gripper_open01,
        }