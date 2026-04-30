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

import mathmodel as mm
import values as val
from depthcalibrator import DepthCalibrator, HandDepthEstimator
from hand_workspace_mapper import HandWorkspaceMapper
from robot_mirror_mapper import RobotMirrorWorkspaceMapper
from robot_workspace_mapper import RobotWorkspaceMapper
from residual_learning import BoundedResidualCorrector


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
hands = None
_hands_init_failed = False


def _get_hands():
    global hands, _hands_init_failed
    if hands is not None:
        return hands
    if _hands_init_failed:
        return None
    try:
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=int(getattr(val, "HANDTRACKING_MAX_NUM_HANDS", 2)),
            model_complexity=int(getattr(
                val,
                "HANDTRACKING_MEDIAPIPE_MODEL_COMPLEXITY" if bool(getattr(val, "LOW_LATENCY_MODE", False)) else "HANDTRACKING_MODEL_COMPLEXITY",
                getattr(val, "HANDTRACKING_MODEL_COMPLEXITY", 0),
            )),
            min_detection_confidence=float(getattr(
                val,
                "HANDTRACKING_MIN_DETECTION_CONFIDENCE",
                getattr(val, "HANDTRACKING_MIN_DETECTION_CONFIDENCE", 0.6),
            )),
            min_tracking_confidence=float(getattr(
                val,
                "HANDTRACKING_MIN_TRACKING_CONFIDENCE",
                getattr(val, "HANDTRACKING_MIN_TRACKING_CONFIDENCE", 0.6),
            )),
        )
        return hands
    except Exception as exc:
        _hands_init_failed = True
        log_event(f"MediaPipe hands unavailable: {exc}")
        return None


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


def _base_joint_limits():
    return {
        "shoulder_pan": _get_limit("BASE_PAN", -math.pi, math.pi),
        "shoulder_lift": _get_limit("SHOULDER_LIFT", -math.pi, math.pi),
        "elbow_flex": _get_limit("ELBOW", -math.pi, math.pi),
        "wrist_flex": _get_limit("WRIST_FLEX", -math.pi, math.pi),
        "wrist_yaw": _get_limit("WRIST_YAW", -math.pi, math.pi),
        "wrist_roll": _get_limit("WRIST_ROLL", -math.pi, math.pi),
        "wrist_pitch": _get_limit("WRIST_PITCH", -math.pi, math.pi),
    }


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


def _finite_float(x, default=None):
    try:
        f = float(x)
    except Exception:
        return default
    return f if math.isfinite(f) else default


def _finite_vec3(x):
    try:
        arr = np.asarray(x, dtype=np.float64).reshape(3)
    except Exception:
        return None
    return arr if np.all(np.isfinite(arr)) else None


def _effective_rate_hz(base_attr: str, current_attr: str, default: float) -> float:
    try:
        base = float(getattr(val, base_attr, getattr(val, current_attr, default)))
    except Exception:
        base = float(default)
    if bool(getattr(val, "REAL_ROBOT_APPLY_SPEED_PERCENT_TO_RATES", False)):
        try:
            pct = float(getattr(val, "REAL_ROBOT_ARM_SPEED_PERCENT", 100.0))
            min_pct = float(getattr(val, "REAL_ROBOT_MIN_ARM_SPEED_PERCENT", 1.0))
            pct = max(min_pct, min(100.0, pct)) / 100.0
        except Exception:
            pct = 1.0
        base *= pct
    return max(1e-6, base)


def _wrap_angle(x: float) -> float:
    return math.atan2(math.sin(float(x)), math.cos(float(x)))


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
            fallback = getattr(val, "CAMERA_CALIBRATION_FILE", "")
            if fallback and os.path.exists(fallback):
                path = fallback
            else:
                return
        try:
            if str(path).lower().endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    jd = json.load(f)
                K = jd.get("camera_matrix", jd.get("K"))
                dist = jd.get("dist_coeffs", jd.get("dist"))
                if K is None or dist is None:
                    return
                self.camera_matrix = np.asarray(K, dtype=np.float64)
                self.dist_coeffs = np.asarray(dist, dtype=np.float64).reshape(-1, 1)
                return
        except Exception as exc:
            self.last_debug["status"] = f"intrinsics_json_load_failed:{exc}"
            return
        try:
            d = np.load(path, allow_pickle=True)
        except Exception as exc:
            self.last_debug["status"] = f"intrinsics_load_failed:{exc}"
            return
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
        try:
            d = np.load(path, allow_pickle=True)
        except Exception as exc:
            self.last_debug["status"] = f"extrinsics_load_failed:{exc}"
            return

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
        try:
            d = np.load(path, allow_pickle=True)
        except Exception as exc:
            self.last_debug["status"] = f"workspace_load_failed:{exc}"
            return
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
            "camera_xyz": marker_origin_camera,
            "camera_depth_m": float(np.linalg.norm(marker_origin_camera)),
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
        if bool(getattr(val, "LOW_LATENCY_MODE", False)):
            self._alpha = float(getattr(val, "HAND_TARGET_SMOOTHING_ALPHA_LOW_LATENCY", self._alpha))
        self.aruco = ArucoGloveTracker() if bool(getattr(val, "HAND_DEPTH_ENABLE_ARUCO_GLOVE", False)) else None
        self.residual_corrector = BoundedResidualCorrector()
        try:
            self.depth_estimator = HandDepthEstimator()
            self.depth_calibrator = self.depth_estimator
        except Exception:
            self.depth_estimator = None
            self.depth_calibrator = None
        try:
            self.workspace_mapper = HandWorkspaceMapper()
        except Exception as exc:
            self.workspace_mapper = None
            log_event(f"hand workspace mapper disabled: {exc}")
        try:
            self.robot_workspace_mapper = RobotWorkspaceMapper()
        except Exception as exc:
            self.robot_workspace_mapper = None
            log_event(f"robot workspace mapper disabled: {exc}")
        try:
            self.robot_mirror_mapper = RobotMirrorWorkspaceMapper()
        except Exception as exc:
            self.robot_mirror_mapper = None
            log_event(f"robot mirror workspace mapper disabled: {exc}")
        self._robot_workspace_anchor_warning = ""
        try:
            if self.robot_workspace_mapper is not None and bool(getattr(self.robot_workspace_mapper, "loaded", False)):
                warn_err = float(getattr(val, "ROBOT_MIRROR_ANCHOR_WARN_ERR_M", 0.015))
                anchor_errors = self.robot_workspace_mapper.evaluate_anchor_errors()
                bad = [
                    f"{name}:{float(item.get('final_error_m', 0.0)):.3f}m"
                    for name, item in anchor_errors.items()
                    if float(item.get("final_error_m", 0.0)) > warn_err
                ]
                if bad:
                    self._robot_workspace_anchor_warning = "anchor_warning " + ", ".join(bad)
                    log_event("robot workspace calibration anchor warning: " + ", ".join(bad))
        except Exception as exc:
            self._robot_workspace_anchor_warning = f"anchor_check_failed:{exc}"
        self._robot_mirror_anchor_warning = ""
        try:
            if self.robot_mirror_mapper is not None and bool(getattr(self.robot_mirror_mapper, "loaded", False)):
                warn_err = float(getattr(val, "ROBOT_MIRROR_ANCHOR_WARN_ERR_M", 0.015))
                anchor_errors = self.robot_mirror_mapper.evaluate_anchor_errors()
                bad = [
                    f"{name}:{float(item.get('final_error_m', 0.0)):.3f}m"
                    for name, item in anchor_errors.items()
                    if float(item.get("final_error_m", 0.0)) > warn_err
                ]
                if bad:
                    self._robot_mirror_anchor_warning = "anchor_warning " + ", ".join(bad)
                    log_event("robot mirror calibration anchor warning: " + ", ".join(bad))
        except Exception as exc:
            self._robot_mirror_anchor_warning = f"anchor_check_failed:{exc}"
        self.lerobot_calibration = self._load_lerobot_calibration()
        self._command_joint_limits = self._build_command_joint_limits()
        (
            self._near_camera_motor2_extension_delta_rad,
            self._near_camera_motor3_extension_delta_rad,
            self._near_camera_extension_delta_source,
        ) = self._init_near_camera_extension_deltas()
        (
            self._near_camera_upward_motor2_delta_rad,
            self._near_camera_upward_motor3_delta_rad,
            self._near_camera_upward_delta_source,
        ) = self._init_near_camera_upward_deltas()
        self._last_ik_solution = None
        self._filtered_target_xyz = None
        self._filtered_target_rpy = None
        self._external_measured_joints = None
        self._feedback_seeded_last_command = False
        self._last_ik_time = 0.0
        self._last_ik_command = None
        self._last_ik_signature = None
        self._last_cartesian_target_xyz = None
        self._last_cartesian_target_rpy = None
        self._filtered_virtual_wrist_rpy = None
        self._simple_palm_roll_smoothed = None
        self._warned_no_hand_calibration = False
        self._warned_no_workspace_calibration = False
        self._warned_no_robot_workspace_calibration = False
        self._warned_no_robot_mirror_calibration = False
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

    def warmup_mediapipe(self) -> bool:
        try:
            return _get_hands() is not None
        except Exception as exc:
            log_event(f"MediaPipe warmup skipped: {exc}")
            return False

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
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
        except Exception as exc:
            log_event(f"calibration load skipped: {exc}")
            return None

    def _build_command_joint_limits(self):
        limits = _base_joint_limits()
        if isinstance(self.lerobot_calibration, dict):
            try:
                merged, _joint_cal = mm._merge_limits(limits, self.lerobot_calibration)
                clean = {}
                for name, pair in merged.items():
                    lo, hi = float(pair[0]), float(pair[1])
                    if math.isfinite(lo) and math.isfinite(hi):
                        clean[name] = (min(lo, hi), max(lo, hi))
                if clean:
                    limits.update(clean)
            except Exception as exc:
                log_event(f"joint limit calibration skipped: {exc}")
        return limits

    def _clamp_command_joint(self, joint_name: str, value: float) -> float:
        x = float(value)
        if not math.isfinite(x):
            raise ValueError(f"non-finite {joint_name} command")
        limits = getattr(self, "_command_joint_limits", None)
        if not isinstance(limits, dict) or joint_name not in limits:
            limits = _base_joint_limits()
        lo, hi = limits.get(joint_name, (-math.pi, math.pi))
        lo, hi = float(lo), float(hi)
        if hi < lo:
            lo, hi = hi, lo
        return _clip(x, lo, hi)

    def _init_near_camera_extension_deltas(self):
        base2 = float(getattr(val, "HAND_NEAR_CAMERA_MOTOR2_EXTENSION_DELTA_RAD", 0.35))
        base3 = float(getattr(val, "HAND_NEAR_CAMERA_MOTOR3_EXTENSION_DELTA_RAD", -0.35))
        if not math.isfinite(base2):
            base2 = 0.0
        if not math.isfinite(base3):
            base3 = 0.0
        if not bool(getattr(val, "HAND_NEAR_CAMERA_EXTENSION_INFER_SIGNS_FROM_WORKSPACE", True)):
            return base2, base3, "config"

        min_delta = abs(float(getattr(val, "HAND_NEAR_CAMERA_EXTENSION_MIN_INFER_DELTA_RAD", 0.08)))
        candidates = (
            (self.robot_workspace_mapper, "near", "robot_workspace_near_pose"),
            (self.robot_mirror_mapper, "mirror_near", "robot_mirror_near_pose"),
        )
        for mapper, near_name, source in candidates:
            pose_joints = getattr(mapper, "pose_joints", None)
            if not isinstance(pose_joints, dict):
                continue
            center = pose_joints.get("center")
            near = pose_joints.get(near_name) or pose_joints.get("near") or pose_joints.get("mirror_near")
            if not isinstance(center, dict) or not isinstance(near, dict):
                continue
            try:
                raw2 = float(near["shoulder_lift"]) - float(center["shoulder_lift"])
                raw3 = float(near["elbow_flex"]) - float(center["elbow_flex"])
            except Exception:
                continue
            if not (math.isfinite(raw2) and math.isfinite(raw3)):
                continue
            if abs(raw2) < min_delta or abs(raw3) < min_delta:
                continue
            sign2 = 1.0 if raw2 > 0.0 else -1.0
            sign3 = 1.0 if raw3 > 0.0 else -1.0
            if sign2 == sign3:
                continue
            return sign2 * abs(base2), sign3 * abs(base3), source
        return base2, base3, "config"

    def _init_near_camera_upward_deltas(self):
        if not bool(getattr(val, "HAND_NEAR_CAMERA_UPWARD_COMPENSATION_ENABLED", True)):
            return 0.0, 0.0, "disabled"
        comp_m = abs(float(getattr(val, "HAND_NEAR_CAMERA_UPWARD_COMPENSATION_M", 0.025)))
        if not math.isfinite(comp_m) or comp_m <= 0.0:
            return 0.0, 0.0, "disabled"
        arm_m = (
            abs(float(getattr(val, "IK_LINK1_M", 0.115)))
            + abs(float(getattr(val, "IK_LINK2_M", 0.115)))
        )
        comp_rad = comp_m / max(arm_m, 1e-6)
        min_delta = abs(float(getattr(val, "HAND_NEAR_CAMERA_EXTENSION_MIN_INFER_DELTA_RAD", 0.08)))
        candidates = (
            (self.robot_workspace_mapper, "up", "robot_workspace_up_pose"),
            (self.robot_mirror_mapper, "mirror_up", "robot_mirror_up_pose"),
        )
        for mapper, up_name, source in candidates:
            pose_joints = getattr(mapper, "pose_joints", None)
            if not isinstance(pose_joints, dict):
                continue
            center = pose_joints.get("center")
            up = pose_joints.get(up_name) or pose_joints.get("up") or pose_joints.get("mirror_up")
            if not isinstance(center, dict) or not isinstance(up, dict):
                continue
            try:
                raw2 = float(up["shoulder_lift"]) - float(center["shoulder_lift"])
                raw3 = float(up["elbow_flex"]) - float(center["elbow_flex"])
            except Exception:
                continue
            if not (math.isfinite(raw2) and math.isfinite(raw3)):
                continue
            use2 = abs(raw2) >= min_delta
            use3 = abs(raw3) >= min_delta
            if not (use2 or use3):
                continue
            delta2 = (1.0 if raw2 > 0.0 else -1.0) * comp_rad if use2 else 0.0
            delta3 = (1.0 if raw3 > 0.0 else -1.0) * comp_rad if use3 else 0.0
            return float(delta2), float(delta3), source
        return 0.0, 0.0, "unavailable"

    def process(self, frame):
        process_t0 = time.perf_counter()
        hand_ms = 0.0
        aruco_ms = 0.0
        command_ms = 0.0
        now = time.time()
        dt = now - self.prev_time
        if dt <= 0.0:
            dt = 1e-3
        self.prev_time = now

        hand_t0 = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            rgb.flags.writeable = False
        except Exception:
            pass
        hands_processor = _get_hands()
        if hands_processor is None:
            detected_hands = []
        else:
            try:
                results = hands_processor.process(rgb)
                detected_hands = build_detected_hands(results)
            except Exception as exc:
                log_event(f"MediaPipe process skipped: {exc}")
                detected_hands = []
        try:
            rgb.flags.writeable = True
        except Exception:
            pass
        hand_ms = (time.perf_counter() - hand_t0) * 1000.0

        clap_event, per_hand = draw_and_update_gestures(frame, detected_hands, now, dt)
        snap_event = any(bool(info.get("snap", False)) for info in per_hand.values())
        self.last_gesture_events = {"snap": snap_event, "clap": bool(clap_event), "per_hand": per_hand}

        aruco_t0 = time.perf_counter()
        aruco_pose = None
        if self.aruco is not None and bool(getattr(val, "HAND_DEPTH_ENABLE_ARUCO_GLOVE", False)):
            aruco_pose = self.aruco.detect(frame)
            if not bool(getattr(val, "DEBUG_DISABLE_EXPENSIVE_OVERLAYS", False)):
                self.aruco.draw_debug(frame, aruco_pose)
        aruco_ms = (time.perf_counter() - aruco_t0) * 1000.0

        driver = choose_driver(detected_hands)

        if aruco_pose is not None and driver is None:
            command_t0 = time.perf_counter()
            open01 = self._estimate_gripper_open01(detected_hands)
            cmd = self._aruco_pose_to_command(aruco_pose, open01)
            if cmd is not None:
                out = self._smooth_command(cmd)
                command_ms = (time.perf_counter() - command_t0) * 1000.0
                out["mode"] = "aruco"
                out["snap_event"] = snap_event
                out["clap_event"] = bool(clap_event)
                out["ee_target_xyz"] = aruco_pose["workspace_xyz"].tolist()
                diag = dict(out.get("__diagnostics__", {})) if isinstance(out.get("__diagnostics__", {}), dict) else {}
                diag.update({
                    "perf_hand_ms": hand_ms,
                    "perf_aruco_ms": aruco_ms,
                    "perf_command_ms": command_ms,
                    "perf_process_ms": (time.perf_counter() - process_t0) * 1000.0,
                })
                out["__diagnostics__"] = diag
                if not bool(getattr(val, "DEBUG_DISABLE_EXPENSIVE_OVERLAYS", False)):
                    self._draw_command_overlay(frame, out)
                    self._draw_aruco_overlay(frame, aruco_pose)
                return out

        if driver is None:
            return None

        hand_lms, label, _score = driver
        command_t0 = time.perf_counter()
        cmd = self._landmarks_to_command(hand_lms, label, frame.shape[1], frame.shape[0], aruco_pose=aruco_pose)
        out = self._smooth_command(cmd)
        command_ms = (time.perf_counter() - command_t0) * 1000.0
        diag = out.get("__diagnostics__", {}) if isinstance(out.get("__diagnostics__", {}), dict) else {}
        diag = dict(diag)
        diag.update({
            "perf_hand_ms": hand_ms,
            "perf_aruco_ms": aruco_ms,
            "perf_command_ms": command_ms,
            "perf_process_ms": (time.perf_counter() - process_t0) * 1000.0,
        })
        out["__diagnostics__"] = diag
        out["mode"] = "cartesian" if diag.get("hand_cartesian_ik_active") else "mediapipe"
        out["snap_event"] = snap_event
        out["clap_event"] = bool(clap_event)
        if not bool(getattr(val, "DEBUG_DISABLE_EXPENSIVE_OVERLAYS", False)):
            self._draw_command_overlay(frame, out)
        return out

    def _smooth_command(self, cmd):
        for k in self._last_cmd:
            self._last_cmd[k] = _lerp(self._last_cmd[k], float(cmd[k]), self._alpha)
        out = dict(self._last_cmd)
        for k in ("__diagnostics__", "ee_target_xyz", "ee_target_rpy"):
            if k in cmd:
                out[k] = cmd[k]
        return out

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
            if hands is not None:
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

    def estimate_simple_palm_roll(self, hand_lms):
        lm = hand_lms.landmark
        index_mcp = lm[5]
        pinky_mcp = lm[17]
        dx = float(pinky_mcp.x) - float(index_mcp.x)
        dy = float(pinky_mcp.y) - float(index_mcp.y)
        if not math.isfinite(dx) or not math.isfinite(dy):
            raise ValueError("non-finite palm roll landmarks")
        roll = math.atan2(dy, dx)
        alpha = _clip(float(getattr(val, "HAND_SIMPLE_PALM_ROLL_SMOOTHING", 0.60)), 0.0, 1.0)
        if self._simple_palm_roll_smoothed is None:
            smoothed = float(roll)
        else:
            delta = _wrap_angle(float(roll) - float(self._simple_palm_roll_smoothed))
            smoothed = _wrap_angle(float(self._simple_palm_roll_smoothed) + alpha * delta)
        self._simple_palm_roll_smoothed = float(smoothed)
        return float(smoothed)

    def _simple_palm_roll_target_joint(self) -> str:
        target = str(getattr(val, "HAND_SIMPLE_PALM_ROLL_TARGET_JOINT", "wrist_pitch")).strip().lower()
        if target not in {"wrist_yaw", "wrist_roll", "wrist_pitch"}:
            target = "wrist_pitch"
        return target

    def map_simple_palm_roll_to_joint(self, roll_rad):
        roll = float(roll_rad)
        if not math.isfinite(roll):
            raise ValueError("non-finite palm roll")
        left = float(getattr(val, "HAND_SIMPLE_PALM_ROLL_LEFT_RAD", -0.6))
        right = float(getattr(val, "HAND_SIMPLE_PALM_ROLL_RIGHT_RAD", 0.6))
        out_left = float(getattr(val, "HAND_SIMPLE_PALM_ROLL_OUTPUT_LEFT_RAD", -1.5))
        out_right = float(getattr(val, "HAND_SIMPLE_PALM_ROLL_OUTPUT_RIGHT_RAD", 1.5))
        if not all(math.isfinite(x) for x in (left, right, out_left, out_right)):
            raise ValueError("non-finite palm roll config")
        denom = right - left
        t = 0.5 if abs(denom) < 1e-9 else (roll - left) / denom
        if bool(getattr(val, "HAND_SIMPLE_PALM_ROLL_CLAMP", True)):
            t = _clip(t, 0.0, 1.0)
        command = _lerp(out_left, out_right, t)
        if bool(getattr(val, "HAND_SIMPLE_PALM_ROLL_CLAMP", True)):
            lo, hi = min(out_left, out_right), max(out_left, out_right)
            command = _clip(command, lo, hi)
        limit_name = {
            "wrist_yaw": "WRIST_YAW",
            "wrist_roll": "WRIST_ROLL",
            "wrist_pitch": "WRIST_PITCH",
        }[self._simple_palm_roll_target_joint()]
        lo, hi = _get_limit(limit_name, -math.pi, math.pi)
        command = _clip(command, lo, hi)
        if not math.isfinite(float(command)):
            raise ValueError("non-finite palm roll command")
        return float(command)

    def _map_direct_wrist_roll_from_roll(self, roll_rad, current_wrist_roll=0.0):
        roll = float(roll_rad)
        if not math.isfinite(roll):
            raise ValueError("non-finite direct wrist-roll input")
        left = float(getattr(val, "HAND_SIMPLE_PALM_ROLL_LEFT_RAD", -0.6))
        right = float(getattr(val, "HAND_SIMPLE_PALM_ROLL_RIGHT_RAD", 0.6))
        out_left = float(getattr(val, "HAND_DIRECT_WRIST_ROLL_OUTPUT_LEFT_RAD", -0.45))
        out_right = float(getattr(val, "HAND_DIRECT_WRIST_ROLL_OUTPUT_RIGHT_RAD", 0.45))
        if not all(math.isfinite(x) for x in (left, right, out_left, out_right)):
            raise ValueError("non-finite direct wrist-roll config")
        denom = right - left
        t = 0.5 if abs(denom) < 1e-9 else (roll - left) / denom
        if bool(getattr(val, "HAND_DIRECT_WRIST_ROLL_CLAMP", True)):
            t = _clip(t, 0.0, 1.0)
        mapped = _lerp(out_left, out_right, t)
        deadband = abs(float(getattr(val, "HAND_DIRECT_WRIST_ROLL_DEADBAND_RAD", 0.03)))
        if abs(mapped) < deadband:
            mapped = 0.0
        blend = _clip(float(getattr(val, "HAND_DIRECT_WRIST_ROLL_BLEND", 1.0)), 0.0, 1.0)
        current = float(current_wrist_roll)
        if not math.isfinite(current):
            current = 0.0
        command = (1.0 - blend) * current + blend * float(mapped)
        if bool(getattr(val, "HAND_DIRECT_WRIST_ROLL_CLAMP", True)):
            lo, hi = _get_limit("WRIST_ROLL", -math.pi, math.pi)
            if hi < lo:
                lo, hi = hi, lo
            command = _clip(command, lo, hi)
        if not math.isfinite(float(command)):
            raise ValueError("non-finite direct wrist-roll command")
        return float(command), float(mapped), float(blend)

    def estimate_palm_pitch_rad(self, hand_lms):
        lm = hand_lms.landmark
        wrist = lm[0]
        index_mcp = lm[5]
        middle_mcp = lm[9]
        pinky_mcp = lm[17]
        fx = float(middle_mcp.x) - float(wrist.x)
        fy = float(middle_mcp.y) - float(wrist.y)
        fz = float(middle_mcp.z) - float(wrist.z)
        lx = float(pinky_mcp.x) - float(index_mcp.x)
        ly = float(pinky_mcp.y) - float(index_mcp.y)
        lz = float(pinky_mcp.z) - float(index_mcp.z)
        if not all(math.isfinite(x) for x in (fx, fy, fz, lx, ly, lz)):
            raise ValueError("non-finite palm pitch landmarks")
        forward_n = math.sqrt(fx * fx + fy * fy + fz * fz)
        lateral_n = math.sqrt(lx * lx + ly * ly + lz * lz)
        if forward_n < 1e-7 or lateral_n < 1e-7:
            raise ValueError("degenerate palm pitch landmarks")
        image_forward_n = math.sqrt(fx * fx + fy * fy)
        pitch = math.atan2(-fz, max(image_forward_n, 1e-7))
        if not math.isfinite(pitch):
            raise ValueError("non-finite palm pitch")
        return float(_clip(pitch, -0.5 * math.pi, 0.5 * math.pi))

    def _map_direct_wrist_roll_from_pitch(self, pitch_rad, current_wrist_roll=0.0):
        pitch = float(pitch_rad)
        if not math.isfinite(pitch):
            raise ValueError("non-finite palm pitch")
        neutral = float(getattr(val, "HAND_DIRECT_WRIST_ROLL_PITCH_NEUTRAL_RAD", 0.0))
        in_min = float(getattr(val, "HAND_DIRECT_WRIST_ROLL_PITCH_INPUT_MIN_RAD", -0.75))
        in_max = float(getattr(val, "HAND_DIRECT_WRIST_ROLL_PITCH_INPUT_MAX_RAD", 0.75))
        out_min = float(getattr(val, "HAND_DIRECT_WRIST_ROLL_PITCH_OUTPUT_MIN_RAD", -0.65))
        out_max = float(getattr(val, "HAND_DIRECT_WRIST_ROLL_PITCH_OUTPUT_MAX_RAD", 0.65))
        if not all(math.isfinite(x) for x in (neutral, in_min, in_max, out_min, out_max)):
            raise ValueError("non-finite palm pitch wrist-roll config")
        if in_max < in_min:
            in_min, in_max = in_max, in_min
            out_min, out_max = out_max, out_min
        pitch_centered = pitch - neutral
        denom = in_max - in_min
        t = 0.5 if abs(denom) < 1e-9 else (pitch_centered - in_min) / denom
        if bool(getattr(val, "HAND_DIRECT_WRIST_ROLL_FROM_PITCH_CLAMP", True)):
            t = _clip(t, 0.0, 1.0)
        mapped = _lerp(out_min, out_max, t)
        deadband = abs(float(getattr(val, "HAND_DIRECT_WRIST_ROLL_PITCH_DEADBAND_RAD", 0.04)))
        if abs(pitch_centered) < deadband:
            mapped = 0.0
        blend = _clip(float(getattr(val, "HAND_DIRECT_WRIST_ROLL_FROM_PITCH_BLEND", 1.0)), 0.0, 1.0)
        current = float(current_wrist_roll)
        if not math.isfinite(current):
            current = 0.0
        command = (1.0 - blend) * current + blend * float(mapped)
        if bool(getattr(val, "HAND_DIRECT_WRIST_ROLL_FROM_PITCH_CLAMP", True)):
            command = self._clamp_command_joint("wrist_roll", command)
        if not math.isfinite(float(command)):
            raise ValueError("non-finite palm pitch wrist-roll command")
        return float(command), float(mapped), float(blend), float(pitch_centered)

    def apply_direct_wrist_roll_from_pitch(self, command_dict, hand_lms, diagnostics):
        if not isinstance(command_dict, dict):
            return command_dict
        diag = diagnostics if isinstance(diagnostics, dict) else {}
        enabled = bool(getattr(val, "HAND_DIRECT_WRIST_ROLL_FROM_PITCH_ENABLED", True))
        diag["direct_wrist_roll_from_pitch_enabled"] = bool(enabled)
        diag["direct_wrist_roll_from_pitch_applied"] = False
        diag["direct_wrist_roll_from_pitch_command_rad"] = None
        diag["direct_wrist_roll_from_pitch_skip_reason"] = ""
        if not enabled:
            diag["direct_wrist_roll_from_pitch_skip_reason"] = "disabled"
            command_dict["__diagnostics__"] = diag
            return command_dict
        if "wrist_roll" not in command_dict:
            diag["direct_wrist_roll_from_pitch_skip_reason"] = "missing_wrist_roll"
            command_dict["__diagnostics__"] = diag
            return command_dict
        try:
            existing_pitch = _finite_float(diag.get("palm_pitch_rad"), None)
            pitch = existing_pitch if existing_pitch is not None else self.estimate_palm_pitch_rad(hand_lms)
            previous = float(command_dict.get("wrist_roll", 0.0))
            command, mapped, blend, pitch_centered = self._map_direct_wrist_roll_from_pitch(pitch, previous)
        except Exception as exc:
            diag["palm_pitch_rad"] = diag.get("palm_pitch_rad", None)
            diag["direct_wrist_roll_from_pitch_skip_reason"] = str(exc)
            command_dict["__diagnostics__"] = diag
            return command_dict
        command_dict["wrist_roll"] = float(command)
        diag["palm_pitch_rad"] = float(pitch)
        diag["direct_wrist_roll_from_pitch_applied"] = True
        diag["direct_wrist_roll_from_pitch_previous_rad"] = float(previous)
        diag["direct_wrist_roll_from_pitch_mapped_rad"] = float(mapped)
        diag["direct_wrist_roll_from_pitch_command_rad"] = float(command)
        diag["direct_wrist_roll_from_pitch_delta_rad"] = float(command - previous)
        diag["direct_wrist_roll_from_pitch_blend"] = float(blend)
        diag["direct_wrist_roll_from_pitch_centered_rad"] = float(pitch_centered)
        diag["wrist_roll_command_rad"] = float(command)
        command_dict["__diagnostics__"] = diag
        return command_dict

    def _near_camera_extension_weight(self, depth_norm: float) -> float:
        depth = float(depth_norm)
        start = float(getattr(val, "HAND_NEAR_CAMERA_EXTENSION_START", 0.60))
        full = float(getattr(val, "HAND_NEAR_CAMERA_EXTENSION_FULL", 0.95))
        gamma = float(getattr(val, "HAND_NEAR_CAMERA_EXTENSION_CURVE_GAMMA", 0.75))
        if not all(math.isfinite(x) for x in (depth, start, full, gamma)):
            raise ValueError("non-finite near-camera extension config")
        if full <= start:
            raise ValueError("invalid near-camera extension range")
        if gamma <= 0.0:
            gamma = 1.0
        near_weight = _clip((depth - start) / (full - start), 0.0, 1.0)
        return float(near_weight ** gamma)

    def _depth_norm_for_near_camera_extension(self, diagnostics):
        if not isinstance(diagnostics, dict):
            return None
        for key in ("depth_norm", "camera_depth_norm", "hand_depth_norm"):
            depth = _finite_float(diagnostics.get(key), None)
            if depth is not None:
                return _clip(depth, 0.0, 1.0)
        return None

    def apply_near_camera_extension_assist(self, command_dict, diagnostics):
        if not isinstance(command_dict, dict):
            return command_dict
        diag = diagnostics if isinstance(diagnostics, dict) else {}
        enabled = bool(getattr(val, "HAND_NEAR_CAMERA_EXTENSION_ASSIST_ENABLED", True))
        diag["near_camera_extension_enabled"] = bool(enabled)
        diag["near_camera_extension_applied"] = False
        diag["near_camera_extension_weight"] = 0.0
        diag["near_camera_motor2_delta_rad"] = 0.0
        diag["near_camera_motor3_delta_rad"] = 0.0
        diag["near_camera_extension_delta_source"] = getattr(self, "_near_camera_extension_delta_source", "config")
        diag["near_camera_extension_preserve_horizontal"] = bool(getattr(val, "HAND_NEAR_CAMERA_EXTENSION_PRESERVE_HORIZONTAL", True))
        diag["near_camera_extension_skip_reason"] = ""
        for key, diag_key in (
            ("shoulder_lift", "motor2_command_rad"),
            ("elbow_flex", "motor3_command_rad"),
            ("wrist_roll", "wrist_roll_command_rad"),
        ):
            current = _finite_float(command_dict.get(key), None)
            if current is not None:
                diag[diag_key] = float(current)
        depth_norm = self._depth_norm_for_near_camera_extension(diag)
        if depth_norm is not None:
            diag["depth_norm"] = float(depth_norm)
        if not enabled:
            diag["near_camera_extension_skip_reason"] = "disabled"
            command_dict["__diagnostics__"] = diag
            return command_dict
        if depth_norm is None:
            diag["near_camera_extension_skip_reason"] = "missing_depth_norm"
            command_dict["__diagnostics__"] = diag
            return command_dict
        try:
            near_weight = self._near_camera_extension_weight(depth_norm)
        except Exception as exc:
            diag["near_camera_extension_skip_reason"] = str(exc)
            command_dict["__diagnostics__"] = diag
            return command_dict
        diag["near_camera_extension_weight"] = float(near_weight)
        if near_weight <= 0.0:
            diag["near_camera_extension_skip_reason"] = "not_near_camera"
            command_dict["__diagnostics__"] = diag
            return command_dict
        if "shoulder_lift" not in command_dict or "elbow_flex" not in command_dict:
            diag["near_camera_extension_skip_reason"] = "missing_motor2_or_motor3"
            command_dict["__diagnostics__"] = diag
            return command_dict
        try:
            shoulder_lift = float(command_dict["shoulder_lift"])
            elbow_flex = float(command_dict["elbow_flex"])
        except Exception:
            diag["near_camera_extension_skip_reason"] = "non_finite_motor2_or_motor3"
            command_dict["__diagnostics__"] = diag
            return command_dict
        if not (math.isfinite(shoulder_lift) and math.isfinite(elbow_flex)):
            diag["near_camera_extension_skip_reason"] = "non_finite_motor2_or_motor3"
            command_dict["__diagnostics__"] = diag
            return command_dict

        blend = _clip(float(getattr(val, "HAND_NEAR_CAMERA_EXTENSION_BLEND", 1.0)), 0.0, 1.0)
        ext2 = near_weight * blend * float(getattr(self, "_near_camera_motor2_extension_delta_rad", 0.0))
        ext3 = near_weight * blend * float(getattr(self, "_near_camera_motor3_extension_delta_rad", 0.0))
        up2 = near_weight * blend * float(getattr(self, "_near_camera_upward_motor2_delta_rad", 0.0))
        up3 = near_weight * blend * float(getattr(self, "_near_camera_upward_motor3_delta_rad", 0.0))
        target2 = shoulder_lift + ext2 + up2
        target3 = elbow_flex + ext3 + up3
        if bool(getattr(val, "HAND_NEAR_CAMERA_EXTENSION_CLAMP_TO_LIMITS", True)):
            target2 = self._clamp_command_joint("shoulder_lift", target2)
            target3 = self._clamp_command_joint("elbow_flex", target3)
        if not (math.isfinite(target2) and math.isfinite(target3)):
            diag["near_camera_extension_skip_reason"] = "non_finite_extension_command"
            command_dict["__diagnostics__"] = diag
            return command_dict

        command_dict["shoulder_lift"] = float(target2)
        command_dict["elbow_flex"] = float(target3)
        diag["near_camera_extension_applied"] = True
        diag["near_camera_motor2_delta_rad"] = float(target2 - shoulder_lift)
        diag["near_camera_motor3_delta_rad"] = float(target3 - elbow_flex)
        diag["near_camera_motor2_extension_delta_rad"] = float(ext2)
        diag["near_camera_motor3_extension_delta_rad"] = float(ext3)
        diag["near_camera_upward_motor2_delta_rad"] = float(up2)
        diag["near_camera_upward_motor3_delta_rad"] = float(up3)
        diag["near_camera_upward_delta_source"] = getattr(self, "_near_camera_upward_delta_source", "unavailable")
        diag["motor2_command_rad"] = float(target2)
        diag["motor3_command_rad"] = float(target3)
        if _finite_float(command_dict.get("wrist_roll"), None) is not None:
            diag["wrist_roll_command_rad"] = float(command_dict["wrist_roll"])
        command_dict["__diagnostics__"] = diag
        return command_dict

    def apply_direct_wrist_roll_override(self, command_dict, hand_lms, diagnostics, palm_roll_rad=None):
        if not isinstance(command_dict, dict):
            return command_dict
        diag = diagnostics if isinstance(diagnostics, dict) else {}
        enabled = bool(getattr(val, "HAND_DIRECT_WRIST_ROLL_ENABLED", False))
        diag["direct_wrist_roll_enabled"] = bool(enabled)
        diag["direct_wrist_roll_source"] = str(getattr(val, "HAND_DIRECT_WRIST_ROLL_SOURCE", "palm_roll"))
        diag["direct_wrist_roll_applied"] = False
        if not enabled:
            command_dict["__diagnostics__"] = diag
            return command_dict
        if "wrist_roll" not in command_dict:
            diag["direct_wrist_roll_skip_reason"] = "missing_wrist_roll"
            command_dict["__diagnostics__"] = diag
            return command_dict
        source = str(getattr(val, "HAND_DIRECT_WRIST_ROLL_SOURCE", "palm_roll")).strip().lower()
        if source != "palm_roll":
            diag["direct_wrist_roll_skip_reason"] = f"unsupported_source:{source}"
            command_dict["__diagnostics__"] = diag
            return command_dict
        try:
            roll = float(palm_roll_rad) if palm_roll_rad is not None else self.estimate_simple_palm_roll(hand_lms)
            previous = float(command_dict.get("wrist_roll", 0.0))
            command, mapped, blend = self._map_direct_wrist_roll_from_roll(roll, previous)
        except Exception as exc:
            diag["direct_wrist_roll_skip_reason"] = str(exc)
            command_dict["__diagnostics__"] = diag
            return command_dict
        command_dict["wrist_roll"] = float(command)
        diag["direct_wrist_roll_applied"] = True
        diag["direct_wrist_roll_palm_roll_rad"] = float(roll)
        diag["direct_wrist_roll_previous_rad"] = float(previous)
        diag["direct_wrist_roll_mapped_rad"] = float(mapped)
        diag["direct_wrist_roll_command_rad"] = float(command)
        diag["direct_wrist_roll_delta_rad"] = float(command - previous)
        diag["direct_wrist_roll_blend"] = float(blend)
        command_dict["__diagnostics__"] = diag
        return command_dict

    def apply_simple_palm_roll_override(self, command_dict, hand_lms, diagnostics):
        if not isinstance(command_dict, dict):
            return command_dict
        diag = diagnostics if isinstance(diagnostics, dict) else {}
        enabled = bool(getattr(val, "HAND_SIMPLE_PALM_ROLL_ENABLED", True))
        target_joint = self._simple_palm_roll_target_joint()
        diag["simple_palm_roll_enabled"] = bool(enabled)
        diag["simple_palm_roll_joint"] = target_joint
        diag["simple_palm_roll_target_joint"] = target_joint
        palm_roll = None
        if enabled:
            if target_joint not in command_dict:
                diag["simple_palm_roll_enabled"] = False
                diag["simple_palm_roll_skip_reason"] = "missing_target_joint"
            else:
                try:
                    palm_roll = self.estimate_simple_palm_roll(hand_lms)
                    joint_command = self.map_simple_palm_roll_to_joint(palm_roll)
                    command_dict[target_joint] = float(joint_command)
                    diag["simple_palm_roll_rad"] = float(palm_roll)
                    diag["simple_palm_roll_command_rad"] = float(joint_command)
                except Exception as exc:
                    diag["simple_palm_roll_enabled"] = False
                    diag["simple_palm_roll_skip_reason"] = str(exc)
        command_dict["__diagnostics__"] = diag
        if not bool(getattr(val, "HAND_DIRECT_WRIST_ROLL_FROM_PITCH_ENABLED", True)):
            command_dict = self.apply_direct_wrist_roll_override(command_dict, hand_lms, diag, palm_roll_rad=palm_roll)
            diag = command_dict.get("__diagnostics__", diag) if isinstance(command_dict, dict) else diag
        command_dict = self.apply_direct_wrist_roll_from_pitch(command_dict, hand_lms, diag)
        diag = command_dict.get("__diagnostics__", diag) if isinstance(command_dict, dict) else diag
        return self.apply_near_camera_extension_assist(command_dict, diag)

    def _shape_centered_extension_axis(self, centered: float, gamma: float) -> float:
        if not math.isfinite(float(gamma)) or float(gamma) <= 0.0:
            gamma = 1.0
        c = _clip(float(centered), -1.0, 1.0)
        shaped = math.copysign(abs(c) ** float(gamma), c)
        if bool(getattr(val, "ROBOT_WORKSPACE_EXTENSION_SHAPING_CLAMP", True)):
            shaped = _clip(shaped, -1.0, 1.0)
        return float(shaped)

    def _shape_norm_for_extension(self, norm01: float, axis: str):
        raw_norm = _clip(float(norm01), 0.0, 1.0)
        centered_raw = _clip(2.0 * (raw_norm - 0.5), -1.0, 1.0)
        axis_l = str(axis).strip().lower()
        if axis_l == "vertical" and bool(getattr(val, "ROBOT_WORKSPACE_VERTICAL_ENDPOINT_BOOST_ENABLED", True)):
            centered_shaped = self._shape_centered_extension_axis(
                centered_raw,
                float(getattr(val, "ROBOT_WORKSPACE_VERTICAL_RESPONSE_GAMMA", 1.0)),
            )
        elif axis_l == "depth" and bool(getattr(val, "ROBOT_WORKSPACE_DEPTH_ENDPOINT_BOOST_ENABLED", True)):
            centered_shaped = self._shape_centered_extension_axis(
                centered_raw,
                float(getattr(val, "ROBOT_WORKSPACE_DEPTH_RESPONSE_GAMMA", 1.0)),
            )
        else:
            centered_shaped = centered_raw
        shaped_norm = _clip(0.5 + 0.5 * centered_shaped, 0.0, 1.0)
        return float(shaped_norm), float(centered_raw), float(centered_shaped)

    def _smooth_target_pose(self, xyz, rpy):
        xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
        rpy = np.asarray(rpy, dtype=np.float64).reshape(3)
        alpha = _clip(float(getattr(val, "POSE_SMOOTH_ALPHA", 0.25)), 0.0, 1.0)
        if bool(getattr(val, "LOW_LATENCY_MODE", False)):
            alpha = _clip(float(getattr(val, "HAND_TARGET_SMOOTHING_ALPHA_LOW_LATENCY", alpha)), 0.0, 1.0)
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
        hz = _effective_rate_hz("HAND_IK_BASE_HZ", "HAND_IK_HZ", 6.0)
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

    def _queue_async_ik(self, xyz_f, rpy_f, open01, q_seed=None, projection_center=None) -> None:
        now = time.time()
        with self._ik_lock:
            # Keep only the newest target.  Old hand positions are overwritten
            # instead of forming an IK backlog.
            self._ik_request = {
                "xyz": np.asarray(xyz_f, dtype=np.float64).reshape(3).copy(),
                "rpy": np.asarray(rpy_f, dtype=np.float64).reshape(3).copy(),
                "open01": float(open01),
                "q_seed": dict(q_seed) if isinstance(q_seed, dict) else None,
                "projection_center": None if projection_center is None else np.asarray(projection_center, dtype=np.float64).reshape(3).copy(),
                "time": now,
            }
            self._ik_last_request_time = now

    def _ik_worker_loop(self) -> None:
        hz = _effective_rate_hz("IK_MAX_SOLVE_BASE_HZ", "IK_MAX_SOLVE_HZ", getattr(val, "HAND_IK_HZ", 4.0))
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
                self._solve_ik_now(
                    req["xyz"],
                    req["rpy"],
                    req["open01"],
                    q_seed=req.get("q_seed"),
                    projection_center=req.get("projection_center"),
                )
            except Exception as exc:
                log_event(f"IK worker error: {exc}")
            finally:
                self._ik_busy = False

    def _solve_ik_now(self, xyz_f, rpy_f, open01, q_seed=None, projection_center=None):
        ik_t0 = time.perf_counter()
        raw_xyz = np.asarray(xyz_f, dtype=np.float64).reshape(3)
        raw_rpy = np.asarray(rpy_f, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(raw_xyz)) or not np.all(np.isfinite(raw_rpy)):
            return None
        projection_diag: dict = {"target_projected": False}
        corrected_xyz, corrected_rpy, residual = self.residual_corrector.apply(raw_xyz, raw_rpy)
        if not np.all(np.isfinite(corrected_xyz)) or corrected_rpy is None or not np.all(np.isfinite(corrected_rpy)):
            return None

        previous_for_ik = q_seed if q_seed is not None else self._prev_joints_for_ik()

        def solve_candidate(candidate_xyz):
            return mm.solve_ik_from_target(
                target_xyz=candidate_xyz,
                target_rpy=corrected_rpy,
                gripper_open01=float(open01),
                lerobot_calibration=self.lerobot_calibration,
                previous_joints=previous_for_ik,
                ik_mode="teleop",
                strict_reachability=False,
            )

        solved = solve_candidate(corrected_xyz)
        if not isinstance(solved, dict):
            return None
        if bool(getattr(val, "ROBOT_WORKSPACE_PROJECT_TO_CENTER_ON_IK_FAIL", True)) and projection_center is not None:
            diag0 = dict(solved.get("__diagnostics__", {})) if isinstance(solved.get("__diagnostics__", {}), dict) else {}
            max_err = float(getattr(val, "HAND_IK_REJECT_POSITION_ERR_M", 0.06))
            needs_projection = (not bool(diag0.get("reachable", True))) and float(diag0.get("position_error_m", 0.0)) > max_err
            if needs_projection:
                try:
                    center = np.asarray(projection_center, dtype=np.float64).reshape(3)
                except Exception:
                    center = None
                if center is not None and np.all(np.isfinite(center)):
                    steps = max(1, int(getattr(val, "ROBOT_WORKSPACE_PROJECTION_STEPS", 8)))
                    original = corrected_xyz.copy()
                    best_projected = None
                    best_projected_diag = None
                    best_projected_xyz = None
                    best_err = float(diag0.get("position_error_m", float("inf")))
                    for i in range(1, steps + 1):
                        alpha = i / float(steps)
                        candidate_xyz = (1.0 - alpha) * original + alpha * center
                        candidate = solve_candidate(candidate_xyz)
                        if not isinstance(candidate, dict):
                            continue
                        cdiag = dict(candidate.get("__diagnostics__", {})) if isinstance(candidate.get("__diagnostics__", {}), dict) else {}
                        cerr = float(cdiag.get("position_error_m", float("inf")))
                        if cerr < best_err:
                            best_projected = candidate
                            best_projected_diag = cdiag
                            best_projected_xyz = candidate_xyz
                            best_err = cerr
                        if bool(cdiag.get("reachable", True)) or cerr <= max_err:
                            solved = candidate
                            corrected_xyz = candidate_xyz
                            projection_diag = {
                                "target_projected": True,
                                "projection_alpha": float(alpha),
                                "original_target_xyz_m": original.tolist(),
                                "final_target_xyz_m": corrected_xyz.tolist(),
                            }
                            break
                    else:
                        if best_projected is not None and best_projected_diag is not None and best_err < float(diag0.get("position_error_m", float("inf"))):
                            solved = best_projected
                            if best_projected_xyz is not None:
                                corrected_xyz = best_projected_xyz
                            projection_diag = {
                                "target_projected": True,
                                "projection_alpha": None,
                                "original_target_xyz_m": original.tolist(),
                                "final_target_xyz_m": corrected_xyz.tolist(),
                            }
        diag = dict(solved.get("__diagnostics__", {})) if isinstance(solved.get("__diagnostics__", {}), dict) else {}
        perf_ik_ms = (time.perf_counter() - ik_t0) * 1000.0
        diag.update({
            "perf_ik_ms": perf_ik_ms,
            "residual_raw_xyz_m": raw_xyz.tolist(),
            "residual_raw_rpy_rad": raw_rpy.tolist(),
            "residual_corrected_xyz_m": corrected_xyz.tolist(),
            "residual_corrected_rpy_rad": None if corrected_rpy is None else corrected_rpy.tolist(),
            "residual_delta_xyz_m": residual.delta_xyz.tolist(),
            "residual_delta_rpy_rad": residual.delta_rpy.tolist(),
            "residual_enabled": bool(residual.enabled),
            "residual_source": str(residual.source),
        })
        diag.update(projection_diag)
        max_err = float(getattr(val, "HAND_IK_REJECT_POSITION_ERR_M", 0.06))
        if isinstance(diag, dict) and (not bool(diag.get("reachable", True))) and float(diag.get("position_error_m", 0.0)) > max_err:
            diag["ik_success"] = False
            diag["failure_reason"] = "position_error_rejected"
            log_event(f"IK target rejected err={float(diag.get('position_error_m', 0.0)):.3f}m")
            return None
        diag["ik_success"] = bool(diag.get("reachable", True))
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
            "ee_target_xyz": corrected_xyz.tolist(),
            "ee_target_rpy": corrected_rpy.tolist(),
        }
        with self._ik_lock:
            self._last_ik_solution = {k: float(out[k]) for k in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "wrist_pitch")}
            self._last_ik_time = time.time()
            self._last_ik_command = dict(out)
            self._last_ik_signature = {"xyz": out["ee_target_xyz"], "rpy": out["ee_target_rpy"]}
        return out

    def _solve_cartesian_command(self, xyz, rpy, open01, q_seed=None, projection_center=None):
        xyz_f, rpy_f = self._smooth_target_pose(xyz, rpy)
        if self._ik_async_enabled:
            self._queue_async_ik(xyz_f, rpy_f, open01, q_seed=q_seed, projection_center=projection_center)
            return self._ik_base_command(open01, xyz_f, rpy_f, pending=True)

        cached = self._cached_ik_command_if_fresh(xyz_f, rpy_f, open01)
        if cached is not None:
            return cached
        solved = self._solve_ik_now(xyz_f, rpy_f, open01, q_seed=q_seed, projection_center=projection_center)
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
        diag = out.get("__diagnostics__", {}) if isinstance(out.get("__diagnostics__", {}), dict) else {}
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
            f"wflex={out['wrist_flex']:.2f} wyaw={out['wrist_yaw']:.2f} wroll={out['wrist_roll']:.2f} wpitch={out['wrist_pitch']:.2f} grip={out['gripper_open01']:.2f}",
            (10, h_img - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if bool(getattr(val, "HAND_CAMERA_AXIS_DEBUG", True)) and diag:
            xyz = diag.get("target_xyz_final_m", out.get("ee_target_xyz", [0.0, 0.0, 0.0]))
            rpy = diag.get("target_rpy_rad", out.get("ee_target_rpy", [0.0, 0.0, 0.0]))
            try:
                xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
                rpy_deg = np.degrees(np.asarray(rpy, dtype=np.float64).reshape(3))
                line1 = (
                    f"map={diag.get('mapping_source', '?')} depth={diag.get('depth_source', '?')} "
                    f"n=({float(diag.get('camera_x_norm', 0.0)):.2f},"
                    f"{float(diag.get('camera_y_norm', 0.0)):.2f},"
                    f"{float(diag.get('camera_depth_norm', 0.5)):.2f}) "
                    f"z={float(diag.get('depth_m', 0.0)):.2f}m c={float(diag.get('depth_confidence', 0.0)):.2f} "
                    f"size={float(diag.get('hand_size_norm', 0.0)):.2f}"
                )
                line2 = (
                    f"xyz=({xyz[0]:+.3f},{xyz[1]:+.3f},{xyz[2]:+.3f}) "
                    f"rpy=({rpy_deg[0]:+.0f},{rpy_deg[1]:+.0f},{rpy_deg[2]:+.0f}) "
                    f"IK={'OK' if bool(diag.get('ik_success', False)) else 'pending/fallback'} "
                    f"map={diag.get('workspace_learning_method', '?')} "
                    f"speed={float(getattr(val, 'REAL_ROBOT_ARM_SPEED_PERCENT', 100.0)):.0f}% "
                    "grip=100%"
                )
                overlay_y = max(22, h_img - 138)
                if (
                    bool(getattr(val, "HAND_SIMPLE_PALM_ROLL_DEBUG", True))
                    and bool(diag.get("simple_palm_roll_enabled", False))
                    and "simple_palm_roll_command_rad" in diag
                ):
                    palm_line = (
                        f"simple roll={float(diag.get('simple_palm_roll_rad', 0.0)):+.2f}rad -> "
                        f"{diag.get('simple_palm_roll_target_joint', diag.get('simple_palm_roll_joint', '?'))}="
                        f"{float(diag.get('simple_palm_roll_command_rad', 0.0)):+.2f}rad"
                    )
                    cv2.putText(frame, palm_line, (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 2, cv2.LINE_AA)
                    overlay_y = max(22, overlay_y - 24)
                if (
                    bool(getattr(val, "HAND_DIRECT_WRIST_ROLL_DEBUG", True))
                    and bool(diag.get("direct_wrist_roll_from_pitch_applied", False))
                    and "direct_wrist_roll_from_pitch_command_rad" in diag
                ):
                    roll6_pitch_line = (
                        f"M6 pitch={float(diag.get('palm_pitch_rad', 0.0)):+.2f}rad -> "
                        f"wroll={float(diag.get('direct_wrist_roll_from_pitch_command_rad', 0.0)):+.2f}rad"
                    )
                    cv2.putText(frame, roll6_pitch_line, (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 2, cv2.LINE_AA)
                    overlay_y = max(22, overlay_y - 24)
                if (
                    bool(getattr(val, "HAND_DIRECT_WRIST_ROLL_DEBUG", True))
                    and bool(diag.get("direct_wrist_roll_applied", False))
                    and "direct_wrist_roll_command_rad" in diag
                ):
                    roll6_line = (
                        f"M6 direct roll={float(diag.get('direct_wrist_roll_command_rad', 0.0)):+.2f}rad "
                        f"delta={float(diag.get('direct_wrist_roll_delta_rad', 0.0)):+.2f}"
                    )
                    cv2.putText(frame, roll6_line, (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 2, cv2.LINE_AA)
                    overlay_y = max(22, overlay_y - 24)
                if bool(diag.get("near_camera_extension_applied", False)):
                    near_line = (
                        f"near w={float(diag.get('near_camera_extension_weight', 0.0)):.2f} "
                        f"M2d={float(diag.get('near_camera_motor2_delta_rad', 0.0)):+.2f} "
                        f"M3d={float(diag.get('near_camera_motor3_delta_rad', 0.0)):+.2f}"
                    )
                    cv2.putText(frame, near_line, (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 2, cv2.LINE_AA)
                if "workspace_mapping_source" in diag:
                    line0 = (
                        f"workspace={diag.get('workspace_method', '?')} legacy={bool(diag.get('robot_workspace_legacy_loaded', False))} "
                        f"clamp={bool(diag.get('target_clamped', False))} proj={bool(diag.get('target_projected', False))} "
                        f"near={diag.get('nearest_pose', '?')}"
                    )
                    cv2.putText(frame, line0, (10, h_img - 114), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 2, cv2.LINE_AA)
                elif "mirror_mapping_source" in diag:
                    line0 = (
                        f"mirror={diag.get('mirror_method', '?')} paired={bool(diag.get('paired_hand_calibration_loaded', False))} "
                        f"depthPair={diag.get('hand_depth_pairing_source', '?')} clamp={bool(diag.get('mirror_target_clamped', False))} "
                        f"near={diag.get('mirror_nearest_pose', '?')}"
                    )
                    cv2.putText(frame, line0, (10, h_img - 114), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, line1, (10, h_img - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, line2, (10, h_img - 66), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 2, cv2.LINE_AA)
            except Exception:
                pass

    def estimate_hand_orientation_rpy(self, hand_lms, world_landmarks=None):
        """Estimate a virtual spherical wrist orientation from MediaPipe hand geometry."""
        try:
            src = world_landmarks.landmark if (
                world_landmarks is not None
                and bool(getattr(val, "HAND_VIRTUAL_WRIST_USE_WORLD_LANDMARKS", True))
            ) else hand_lms.landmark
            pts = []
            for idx in (0, 5, 9, 17):
                p = src[idx]
                pts.append(np.array([float(p.x), float(p.y), float(p.z)], dtype=np.float64))
            wrist, index_mcp, middle_mcp, pinky_mcp = pts
            lateral = index_mcp - pinky_mcp
            forward = middle_mcp - wrist
            lateral_n = float(np.linalg.norm(lateral))
            forward_n = float(np.linalg.norm(forward))
            if lateral_n < 1e-7 or forward_n < 1e-7:
                raise ValueError("degenerate hand frame")
            x_axis = lateral / lateral_n
            y_axis = forward / forward_n
            z_axis = np.cross(x_axis, y_axis)
            z_n = float(np.linalg.norm(z_axis))
            if z_n < 1e-7:
                raise ValueError("degenerate palm normal")
            z_axis /= z_n
            y_axis = np.cross(z_axis, x_axis)
            y_axis /= max(float(np.linalg.norm(y_axis)), 1e-9)
            R = np.column_stack([x_axis, y_axis, z_axis])
            rpy = self._configured_wrist_rpy(_rot_to_rpy(R))
            confidence = _clip(min(lateral_n, forward_n) / 0.05, 0.0, 1.0)
            min_conf = float(getattr(val, "HAND_VIRTUAL_WRIST_CONFIDENCE_MIN", 0.35))
            neutral = np.array([0.0, float(getattr(val, "HAND_TARGET_PITCH_BIAS_RAD", -0.15)), 0.0], dtype=np.float64)
            if confidence < min_conf and bool(getattr(val, "HAND_VIRTUAL_WRIST_BLEND_TO_NEUTRAL_ON_LOW_CONF", True)):
                blend = _clip(confidence / max(min_conf, 1e-6), 0.0, 1.0)
                rpy = neutral + blend * np.array([_wrap_angle(rpy[i] - neutral[i]) for i in range(3)], dtype=np.float64)
                source = "virtual_wrist_low_conf_blend"
            else:
                source = "virtual_wrist"
            alpha = _clip(float(getattr(val, "HAND_VIRTUAL_WRIST_ORIENTATION_SMOOTHING", 0.45)), 0.0, 1.0)
            if bool(getattr(val, "LOW_LATENCY_MODE", False)):
                alpha = _clip(float(getattr(val, "HAND_WRIST_SMOOTHING_ALPHA_LOW_LATENCY", alpha)), 0.0, 1.0)
            if self._filtered_virtual_wrist_rpy is None:
                self._filtered_virtual_wrist_rpy = rpy.copy()
            else:
                prev = self._filtered_virtual_wrist_rpy
                delta = np.array([_wrap_angle(rpy[i] - prev[i]) for i in range(3)], dtype=np.float64)
                self._filtered_virtual_wrist_rpy = prev + alpha * delta
                self._filtered_virtual_wrist_rpy = np.array([_wrap_angle(x) for x in self._filtered_virtual_wrist_rpy], dtype=np.float64)
            return self._filtered_virtual_wrist_rpy.copy(), source, float(confidence)
        except Exception:
            neutral = np.array([0.0, float(getattr(val, "HAND_TARGET_PITCH_BIAS_RAD", -0.15)), 0.0], dtype=np.float64)
            if bool(getattr(val, "HAND_VIRTUAL_WRIST_BLEND_TO_NEUTRAL_ON_LOW_CONF", True)) and self._filtered_virtual_wrist_rpy is not None:
                alpha = _clip(float(getattr(val, "HAND_VIRTUAL_WRIST_ORIENTATION_SMOOTHING", 0.45)), 0.0, 1.0)
                self._filtered_virtual_wrist_rpy = (1.0 - alpha) * self._filtered_virtual_wrist_rpy + alpha * neutral
                return self._filtered_virtual_wrist_rpy.copy(), "virtual_wrist_neutral_blend", 0.0
            return neutral, "neutral_fallback", 0.0

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

    def _hand_size_metrics(self, hand_lms):
        lm = hand_lms.landmark
        pts = np.array([[float(p.x), float(p.y)] for p in lm], dtype=np.float64)
        wrist = (lm[0].x, lm[0].y)
        index_mcp = (lm[5].x, lm[5].y)
        middle_mcp = (lm[9].x, lm[9].y)
        middle_tip = (lm[12].x, lm[12].y)
        pinky_mcp = (lm[17].x, lm[17].y)
        palm_width = mm.dist(index_mcp, pinky_mcp)
        palm_height_mcp = mm.dist(wrist, middle_mcp)
        palm_height_tip = mm.dist(wrist, middle_tip)
        palm_height = palm_height_mcp if palm_height_mcp > 1e-5 else palm_height_tip
        bbox_w = float(np.max(pts[:, 0]) - np.min(pts[:, 0]))
        bbox_h = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))
        bbox_size = math.sqrt(max(0.0, bbox_w * bbox_h))
        candidates = [x for x in (palm_width, palm_height, bbox_size) if math.isfinite(float(x)) and float(x) > 0.0]
        hand_size = float(np.median(candidates)) if candidates else 0.0
        center_pts = pts[[0, 5, 9, 17], :]
        center = np.mean(center_pts, axis=0)
        return {
            "camera_x_norm": _clip(float(center[0]), 0.0, 1.0),
            "camera_y_norm": _clip(float(center[1]), 0.0, 1.0),
            "palm_width_norm": float(palm_width),
            "palm_height_norm": float(palm_height),
            "bbox_size_norm": float(bbox_size),
            "hand_size_norm": float(hand_size),
        }

    def _target_bounds(self):
        return {
            "robot_x": (
                float(getattr(val, "HAND_TARGET_X_MIN_M", getattr(val, "WORKSPACE_X_MIN", -0.12))),
                float(getattr(val, "HAND_TARGET_X_MAX_M", getattr(val, "WORKSPACE_X_MAX", 0.12))),
            ),
            "robot_y": (
                float(getattr(val, "HAND_TARGET_Y_MIN_M", getattr(val, "WORKSPACE_Y_MIN", 0.10))),
                float(getattr(val, "HAND_TARGET_Y_MAX_M", getattr(val, "WORKSPACE_Y_MAX", 0.22))),
            ),
            "robot_z": (
                float(getattr(val, "HAND_TARGET_Z_MIN_M", getattr(val, "WORKSPACE_Z_MIN", 0.00))),
                float(getattr(val, "HAND_TARGET_Z_MAX_M", getattr(val, "WORKSPACE_Z_MAX", 0.22))),
            ),
        }

    def _axis_value_from_norm(self, axis: str, norm: float) -> float:
        bounds = self._target_bounds()
        lo, hi = bounds.get(axis, (0.0, 1.0))
        if hi < lo:
            lo, hi = hi, lo
        return _lerp(lo, hi, _clip(norm, 0.0, 1.0))

    def _axis_midpoint(self, axis: str) -> float:
        lo, hi = self._target_bounds().get(axis, (0.0, 0.0))
        return 0.5 * (float(lo) + float(hi))

    def _clamp_target_xyz(self, xyz):
        arr = np.asarray(xyz, dtype=np.float64).reshape(3)
        before = arr.copy()
        bounds = self._target_bounds()
        for i, axis in enumerate(("robot_x", "robot_y", "robot_z")):
            lo, hi = bounds[axis]
            if hi < lo:
                lo, hi = hi, lo
            arr[i] = _clip(arr[i], lo, hi)
        return arr, bool(np.linalg.norm(arr - before) > 1e-12)

    def _depth_targets_for_axis(self, axis: str):
        near = _finite_float(getattr(val, "HAND_DEPTH_TARGET_NEAR_M", None), None)
        far = _finite_float(getattr(val, "HAND_DEPTH_TARGET_FAR_M", None), None)
        lo, hi = self._target_bounds().get(axis, (0.0, 1.0))
        if near is None:
            near = float(lo)
        if far is None:
            far = float(hi)
        return float(near), float(far)

    def _depth_coord(self, axis: str, depth_norm: float) -> float:
        near, far = self._depth_targets_for_axis(axis)
        return _lerp(far, near, _clip(depth_norm, 0.0, 1.0))

    def _estimate_target_rpy_from_hand(self, hand_lms, aruco_pose=None):
        if not bool(getattr(val, "HAND_USE_WRIST_ORIENTATION", True)):
            return np.array([0.0, float(getattr(val, "HAND_TARGET_PITCH_BIAS_RAD", -0.15)), 0.0], dtype=np.float64), "disabled"

        if aruco_pose is not None and bool(getattr(val, "HAND_DEPTH_ENABLE_ARUCO_GLOVE", False)):
            rpy = _finite_vec3(aruco_pose.get("workspace_rpy"))
            if rpy is not None:
                return self._configured_wrist_rpy(rpy), "aruco"

        if bool(getattr(val, "HAND_VIRTUAL_WRIST_ENABLED", True)):
            rpy, source, _confidence = self.estimate_hand_orientation_rpy(hand_lms)
            if rpy is not None and np.all(np.isfinite(rpy)):
                return rpy, source

        try:
            lm = hand_lms.landmark
            index = np.array([lm[5].x, lm[5].y], dtype=np.float64)
            pinky = np.array([lm[17].x, lm[17].y], dtype=np.float64)
            wrist = np.array([lm[0].x, lm[0].y], dtype=np.float64)
            middle = np.array([lm[9].x, lm[9].y], dtype=np.float64)
            lateral = pinky - index
            forward = middle - wrist
            if np.linalg.norm(lateral) < 1e-6 or np.linalg.norm(forward) < 1e-6:
                raise ValueError("hand axes degenerate")
            roll = math.atan2(float(lateral[1]), float(lateral[0]))
            yaw = math.atan2(float(forward[0]), max(abs(float(forward[1])), 1e-6))
            pitch = -math.atan2(float(forward[1]), max(abs(float(forward[0])), 1e-6))
            rpy3 = self._estimate_hand_rpy_from_landmarks(hand_lms)
            if np.all(np.isfinite(rpy3)):
                pitch = 0.5 * pitch + 0.5 * float(rpy3[1])
                yaw = 0.5 * yaw + 0.5 * float(rpy3[2])
            rpy = np.array([roll, pitch + float(getattr(val, "HAND_TARGET_PITCH_BIAS_RAD", -0.15)), yaw], dtype=np.float64)
            return self._configured_wrist_rpy(rpy), "landmarks"
        except Exception:
            if bool(getattr(val, "HAND_WRIST_ORIENTATION_FALLBACK_TO_NEUTRAL", True)):
                return np.array([0.0, float(getattr(val, "HAND_TARGET_PITCH_BIAS_RAD", -0.15)), 0.0], dtype=np.float64), "neutral_fallback"
            return None, "unavailable"

    def _configured_wrist_rpy(self, rpy):
        raw = np.asarray(rpy, dtype=np.float64).reshape(3)
        roll = _wrap_angle(raw[0] * float(getattr(val, "HAND_WRIST_ROLL_SCALE", 1.0)) + float(getattr(val, "HAND_WRIST_ROLL_OFFSET_RAD", 0.0)))
        pitch = _wrap_angle(raw[1] * float(getattr(val, "HAND_WRIST_PITCH_SCALE", 1.0)) + float(getattr(val, "HAND_WRIST_PITCH_OFFSET_RAD", 0.0)))
        yaw = _wrap_angle(raw[2] * float(getattr(val, "HAND_WRIST_YAW_SCALE", 1.0)) + float(getattr(val, "HAND_WRIST_YAW_OFFSET_RAD", 0.0)))
        roll_max_name = "HAND_VIRTUAL_WRIST_MAX_ROLL_RAD" if bool(getattr(val, "HAND_VIRTUAL_WRIST_ENABLED", True)) else "HAND_WRIST_MAX_ROLL_RAD"
        pitch_max_name = "HAND_VIRTUAL_WRIST_MAX_PITCH_RAD" if bool(getattr(val, "HAND_VIRTUAL_WRIST_ENABLED", True)) else "HAND_WRIST_MAX_PITCH_RAD"
        yaw_max_name = "HAND_VIRTUAL_WRIST_MAX_YAW_RAD" if bool(getattr(val, "HAND_VIRTUAL_WRIST_ENABLED", True)) else "HAND_WRIST_MAX_YAW_RAD"
        roll = _clip(roll, -abs(float(getattr(val, roll_max_name, getattr(val, "HAND_WRIST_MAX_ROLL_RAD", math.pi)))), abs(float(getattr(val, roll_max_name, getattr(val, "HAND_WRIST_MAX_ROLL_RAD", math.pi)))))
        pitch = _clip(pitch, -abs(float(getattr(val, pitch_max_name, getattr(val, "HAND_WRIST_MAX_PITCH_RAD", 2.5)))), abs(float(getattr(val, pitch_max_name, getattr(val, "HAND_WRIST_MAX_PITCH_RAD", 2.5)))))
        yaw = _clip(yaw, -abs(float(getattr(val, yaw_max_name, getattr(val, "HAND_WRIST_MAX_YAW_RAD", math.pi)))), abs(float(getattr(val, yaw_max_name, getattr(val, "HAND_WRIST_MAX_YAW_RAD", math.pi)))))
        return np.array([roll, pitch, yaw], dtype=np.float64)

    def build_hand_cartesian_target(self, hand_lms, frame_w: int, frame_h: int, aruco_pose=None):
        metrics = self._hand_size_metrics(hand_lms)
        x_norm_raw = metrics["camera_x_norm"]
        y_norm_raw = metrics["camera_y_norm"]
        x_norm = 1.0 - x_norm_raw if bool(getattr(val, "HAND_IMAGE_X_FLIP", False)) else x_norm_raw
        y_norm = 1.0 - y_norm_raw if bool(getattr(val, "HAND_IMAGE_Y_FLIP", True)) else y_norm_raw

        depth_debug = {}
        if (
            bool(getattr(val, "HAND_MONOCULAR_DEPTH_ENABLED", True))
            and str(getattr(val, "HAND_DEPTH_MODE", "monocular_size")).strip().lower() == "monocular_size"
            and self.depth_estimator is not None
        ):
            depth_debug = self.depth_estimator.estimate_depth(hand_lms, frame_w=frame_w, frame_h=frame_h)
            depth_source = str(depth_debug.get("source", "monocular_size"))
            depth_raw = float(depth_debug.get("depth_norm", 0.5))
            depth_smooth = depth_raw
        else:
            depth_source = "fixed"
            depth_m = float(getattr(val, "HAND_DEPTH_DEFAULT_M", 0.45))
            if self.depth_estimator is not None:
                depth_raw = self.depth_estimator.depth_m_to_norm(depth_m)
            else:
                near = float(getattr(val, "HAND_MONOCULAR_NEAR_M", 0.20))
                far = float(getattr(val, "HAND_MONOCULAR_FAR_M", 0.70))
                depth_raw = _clip(1.0 - ((depth_m - near) / max(far - near, 1e-6)), 0.0, 1.0)
            depth_smooth = depth_raw
            depth_debug = {
                "depth_m": depth_m,
                "depth_norm": depth_smooth,
                "source": depth_source,
                "confidence": 0.2,
                "raw_candidates": {},
                "hand_size_norm": metrics["hand_size_norm"],
                "palm_width_norm": metrics["palm_width_norm"],
                "wrist_to_middle_mcp_norm": metrics["palm_height_norm"],
                "palm_height_norm": metrics["palm_height_norm"],
                "bbox_size_norm": metrics["bbox_size_norm"],
                "thumb_index_span_norm": 0.0,
                "valid": True,
            }

        mirror_depth_input = float(depth_smooth)
        if bool(getattr(val, "HAND_DEPTH_FLIP", False)):
            depth_raw = 1.0 - depth_raw
            depth_smooth = 1.0 - depth_smooth
        depth_norm = _clip(depth_smooth, float(getattr(val, "HAND_DEPTH_MIN_NORM", 0.0)), float(getattr(val, "HAND_DEPTH_MAX_NORM", 1.0)))

        axis_map = getattr(val, "HAND_CAMERA_TO_ROBOT_AXIS_MAP", {"image_x": "robot_x", "image_y": "robot_z", "depth": "robot_y"})
        if not isinstance(axis_map, dict):
            axis_map = {"image_x": "robot_x", "image_y": "robot_z", "depth": "robot_y"}
        depth_axis = str(getattr(val, "HAND_DEPTH_AXIS", axis_map.get("depth", "robot_y")))
        xyz_by_axis = {
            "robot_x": self._axis_midpoint("robot_x"),
            "robot_y": self._axis_midpoint("robot_y"),
            "robot_z": self._axis_midpoint("robot_z"),
        }
        target_before_depth = dict(xyz_by_axis)
        ix_axis = str(axis_map.get("image_x", "robot_x"))
        iy_axis = str(axis_map.get("image_y", "robot_z"))
        if ix_axis in xyz_by_axis:
            xyz_by_axis[ix_axis] = self._axis_value_from_norm(ix_axis, x_norm)
        if iy_axis in xyz_by_axis:
            xyz_by_axis[iy_axis] = self._axis_value_from_norm(iy_axis, y_norm)
        target_before_depth = dict(xyz_by_axis)
        if depth_axis in xyz_by_axis:
            xyz_by_axis[depth_axis] = self._depth_coord(depth_axis, depth_norm)

        xyz_raw = np.array([xyz_by_axis["robot_x"], xyz_by_axis["robot_y"], xyz_by_axis["robot_z"]], dtype=np.float64)
        mapping_source = "mediapipe_monocular_depth"
        mapper_debug = {}
        q_seed = None
        mirror_mapper_used = False
        workspace_mapper_used = False
        if aruco_pose is not None and bool(getattr(val, "HAND_DEPTH_ENABLE_ARUCO_GLOVE", False)):
            aruco_xyz = _finite_vec3(aruco_pose.get("workspace_xyz"))
            if aruco_xyz is not None:
                xyz_raw = aruco_xyz
                mapping_source = "aruco_calibrated"
            elif not self._warned_no_hand_calibration:
                log_event("Aruco pose unavailable; using normalized hand mapping")
                self._warned_no_hand_calibration = True
        elif depth_source == "midpoint":
            mapping_source = "midpoint"
        elif (
            self.robot_workspace_mapper is not None
            and bool(getattr(val, "ROBOT_WORKSPACE_ENABLED", True))
            and bool(getattr(self.robot_workspace_mapper, "loaded", False))
        ):
            mapped_xyz, mapper_debug = self.robot_workspace_mapper.map_hand_to_workspace(
                x_norm_raw,
                y_norm_raw,
                mirror_depth_input,
            )
            if mapped_xyz is not None:
                xyz_raw = mapped_xyz
                workspace_mapper_used = True
                mapping_source = str(mapper_debug.get("workspace_mapping_source", "robot_workspace_extrema_calibration"))
                if self._robot_workspace_anchor_warning:
                    mapper_debug["workspace_anchor_warning"] = self._robot_workspace_anchor_warning
                q_seed, seed_debug = self.robot_workspace_mapper.choose_ik_seed(
                    x_norm_raw,
                    y_norm_raw,
                    mirror_depth_input,
                    previous_q=self._prev_joints_for_ik(),
                )
                mapper_debug.update(seed_debug)
            elif not self._warned_no_robot_workspace_calibration:
                log_event("robot workspace extrema mapping unavailable; using legacy mirror/workspace mapping")
                self._warned_no_robot_workspace_calibration = True
        elif (
            self.robot_mirror_mapper is not None
            and bool(getattr(val, "ROBOT_MIRROR_WORKSPACE_ENABLED", True))
            and bool(getattr(self.robot_mirror_mapper, "loaded", False))
        ):
            mapped_xyz, mapper_debug = self.robot_mirror_mapper.map_hand_to_robot_target(
                x_norm_raw,
                y_norm_raw,
                mirror_depth_input,
            )
            if mapped_xyz is not None:
                xyz_raw = mapped_xyz
                mirror_mapper_used = True
                mapping_source = str(mapper_debug.get("mirror_mapping_source", "robot_mirror_workspace_calibration"))
                if self._robot_mirror_anchor_warning:
                    mapper_debug["mirror_anchor_warning"] = self._robot_mirror_anchor_warning
                q_seed, seed_debug = self.robot_mirror_mapper.choose_ik_seed(
                    x_norm_raw,
                    y_norm_raw,
                    mirror_depth_input,
                    previous_q=self._prev_joints_for_ik(),
                )
                mapper_debug.update(seed_debug)
            elif not self._warned_no_robot_mirror_calibration:
                log_event("robot mirror workspace mapping unavailable; using legacy hand workspace/values mapping")
                self._warned_no_robot_mirror_calibration = True
        elif self.workspace_mapper is not None and bool(getattr(val, "HAND_WORKSPACE_LEARNING_ENABLED", True)):
            if (
                self.robot_mirror_mapper is not None
                and bool(getattr(val, "ROBOT_MIRROR_WORKSPACE_ENABLED", True))
                and not bool(getattr(self.robot_mirror_mapper, "loaded", False))
                and not self._warned_no_robot_mirror_calibration
            ):
                log_event("robot mirror workspace calibration missing; using legacy hand workspace/values mapping")
                self._warned_no_robot_mirror_calibration = True
            if not bool(getattr(self.workspace_mapper, "loaded", False)) and not self._warned_no_workspace_calibration:
                log_event("hand workspace calibration missing; using values.py mapping")
                self._warned_no_workspace_calibration = True
            mapped_xyz, mapper_debug = self.workspace_mapper.map_hand_to_workspace(
                x_norm,
                y_norm,
                depth_norm,
                hand_size_norm=depth_debug.get("hand_size_norm", metrics["hand_size_norm"]),
            )
            xyz_raw = mapped_xyz
            mapping_source = str(mapper_debug.get("workspace_mapping_source", mapping_source))
            q_seed, seed_debug = self.workspace_mapper.choose_ik_seed(
                x_norm,
                y_norm,
                depth_norm,
                previous_q=self._prev_joints_for_ik(),
            )
            mapper_debug.update(seed_debug)

        if workspace_mapper_used or mirror_mapper_used:
            xyz_final = np.asarray(xyz_raw, dtype=np.float64).reshape(3)
            clamped = bool(mapper_debug.get("target_clamped", mapper_debug.get("mirror_target_clamped", False)))
        else:
            xyz_final, clamped = self._clamp_target_xyz(xyz_raw)
        rpy, rpy_source = self._estimate_target_rpy_from_hand(hand_lms, aruco_pose)
        if rpy is None:
            rpy = np.array([0.0, float(getattr(val, "HAND_TARGET_PITCH_BIAS_RAD", -0.15)), 0.0], dtype=np.float64)
            rpy_source = "neutral_fallback"

        if not np.all(np.isfinite(xyz_final)) or not np.all(np.isfinite(rpy)):
            raise ValueError("non-finite Cartesian target")

        before_depth_arr = np.array([
            target_before_depth["robot_x"],
            target_before_depth["robot_y"],
            target_before_depth["robot_z"],
        ], dtype=np.float64)
        after_depth_arr = np.array([
            xyz_by_axis["robot_x"],
            xyz_by_axis["robot_y"],
            xyz_by_axis["robot_z"],
        ], dtype=np.float64)
        debug = {
            "camera_x_norm": float(x_norm_raw),
            "camera_y_norm": float(y_norm_raw),
            "camera_depth_norm": float(depth_norm),
            "depth_norm": float(depth_norm),
            "depth_m": float(depth_debug.get("depth_m", 0.0)),
            "depth_confidence": float(depth_debug.get("confidence", 0.0)),
            "depth_candidates": dict(depth_debug.get("raw_candidates", {})) if isinstance(depth_debug.get("raw_candidates", {}), dict) else {},
            "hand_size_norm": float(depth_debug.get("hand_size_norm", metrics["hand_size_norm"])),
            "palm_width_norm": float(depth_debug.get("palm_width_norm", metrics["palm_width_norm"])),
            "wrist_to_middle_mcp_norm": float(depth_debug.get("wrist_to_middle_mcp_norm", metrics["palm_height_norm"])),
            "palm_height_norm": float(depth_debug.get("palm_height_norm", metrics["palm_height_norm"])),
            "bbox_size_norm": float(depth_debug.get("bbox_size_norm", metrics["bbox_size_norm"])),
            "thumb_index_span_norm": float(depth_debug.get("thumb_index_span_norm", 0.0)),
            "depth_source": depth_source,
            "depth_norm_raw": float(depth_raw),
            "depth_norm_smoothed": float(depth_smooth),
            "mapped_robot_x_m": float(xyz_final[0]),
            "mapped_robot_y_m": float(xyz_final[1]),
            "mapped_robot_z_m": float(xyz_final[2]),
            "target_depth_axis": depth_axis,
            "target_depth_m": float(xyz_by_axis.get(depth_axis, xyz_final[1])),
            "target_xyz_before_depth_m": before_depth_arr.tolist(),
            "target_xyz_after_depth_m": after_depth_arr.tolist(),
            "target_xyz_raw_m": xyz_raw.tolist(),
            "target_xyz_final_m": xyz_final.tolist(),
            "mapping_source": mapping_source,
            "workspace_mapper_used": bool(workspace_mapper_used),
            "legacy_mirror_mapper_used": bool(mirror_mapper_used),
            "hand_x_norm": float(x_norm),
            "hand_y_norm": float(y_norm),
            "hand_depth_norm": float(depth_norm),
            "axis_map_used": dict(axis_map),
            "axis_flips_used": {
                "image_x": bool(getattr(val, "HAND_IMAGE_X_FLIP", False)),
                "image_y": bool(getattr(val, "HAND_IMAGE_Y_FLIP", True)),
                "depth": bool(getattr(val, "HAND_DEPTH_FLIP", False)),
            },
            "target_clamped": bool(clamped),
            "target_rpy_rad": rpy.tolist(),
            "target_rpy_source": rpy_source,
            "using_monocular_depth": bool(getattr(val, "HAND_MONOCULAR_DEPTH_ENABLED", True)),
            "using_camera_intrinsics": bool(depth_debug.get("using_camera_intrinsics", False)),
            "using_hand_depth_calibration": bool(depth_debug.get("using_hand_depth_calibration", False)),
            "using_depth_camera": False,
            "using_cnn_depth": False,
            "aruco_glove_disabled": not bool(getattr(val, "HAND_DEPTH_ENABLE_ARUCO_GLOVE", False)),
            "aruco_detected": bool(aruco_pose is not None and bool(getattr(val, "HAND_DEPTH_ENABLE_ARUCO_GLOVE", False))),
        }
        debug.update(mapper_debug)
        return xyz_final, rpy, debug, q_seed

    def _landmarks_to_command(self, hand_lms, label: str, frame_w: int = 1, frame_h: int = 1, aruco_pose=None):
        if not (bool(getattr(val, "HAND_USE_CARTESIAN_IK", True)) and bool(getattr(val, "HAND_CARTESIAN_MAPPING_ENABLED", True))):
            return self._old_landmarks_to_command(hand_lms, label)

        is_closed, open01, _metric, _debug = openness_from_fingertips(hand_lms, label)
        gripper_open01 = 0.0 if is_closed else float(open01)
        self._last_open01 = _lerp(self._last_open01, gripper_open01, self._open_alpha)
        gripper_open01 = self._last_open01

        try:
            xyz, rpy, target_debug, q_seed = self.build_hand_cartesian_target(hand_lms, frame_w, frame_h, aruco_pose=aruco_pose)
            projection_center = target_debug.get("workspace_center_xyz_m") if isinstance(target_debug, dict) else None
            cmd = self._solve_cartesian_command(xyz, rpy, gripper_open01, q_seed=q_seed, projection_center=projection_center)
            if cmd is None:
                raise RuntimeError("IK returned no command")
            diag = dict(cmd.get("__diagnostics__", {})) if isinstance(cmd.get("__diagnostics__", {}), dict) else {}
            diag.update(target_debug)
            diag["hand_cartesian_ik_active"] = True
            diag["hand_cartesian_fallback_active"] = False
            diag["ik_success"] = not bool(diag.get("ik_async_pending", False)) and bool(diag.get("reachable", True))
            cmd["__diagnostics__"] = diag
            cmd = self.apply_simple_palm_roll_override(cmd, hand_lms, diag)
            return cmd
        except Exception as exc:
            log_event(f"Cartesian hand mapping fallback: {exc}")
            if bool(getattr(val, "HAND_CARTESIAN_MAPPING_FALLBACK_TO_OLD", True)):
                out = self._old_landmarks_to_command(hand_lms, label)
                diag = dict(out.get("__diagnostics__", {})) if isinstance(out.get("__diagnostics__", {}), dict) else {}
                diag["hand_cartesian_ik_active"] = False
                diag["hand_cartesian_fallback_active"] = True
                diag["mapping_source"] = "old_fallback"
                diag["failure_reason"] = str(exc)
                out["__diagnostics__"] = diag
                return out
            out = self._ik_base_command(gripper_open01, pending=False)
            diag = dict(out.get("__diagnostics__", {})) if isinstance(out.get("__diagnostics__", {}), dict) else {}
            return self.apply_simple_palm_roll_override(out, hand_lms, diag)

    def _old_landmarks_to_command(self, hand_lms, label: str):
        lm = hand_lms.landmark
        wrist = (lm[0].x, lm[0].y)
        index_mcp = (lm[5].x, lm[5].y)
        middle_mcp = (lm[9].x, lm[9].y)
        pinky_mcp = (lm[17].x, lm[17].y)

        hand_cx = (wrist[0] + middle_mcp[0] + index_mcp[0] + pinky_mcp[0]) / 4.0
        hand_cy = (wrist[1] + middle_mcp[1] + index_mcp[1] + pinky_mcp[1]) / 4.0
        palm_width = mm.dist(index_mcp, pinky_mcp)
        palm_height = mm.dist(wrist, middle_mcp)
        size_metric = 0.5 * (palm_width + palm_height)

        is_closed, open01, _metric, _debug = openness_from_fingertips(hand_lms, label)
        gripper_open01 = 0.0 if is_closed else float(open01)
        self._last_open01 = _lerp(self._last_open01, gripper_open01, self._open_alpha)
        gripper_open01 = self._last_open01

        # Hand position to (forward, height) target in the arm's vertical plane.
        # palm_size large (hand close to camera) -> arm pulls back.
        # hand high in frame                     -> end-effector high.
        size_near = float(getattr(val, "HAND_SIZE_NEAR", 0.22))
        size_far = float(getattr(val, "HAND_SIZE_FAR", 0.08))
        if abs(size_near - size_far) < 1e-6:
            size_near = size_far + 1e-3
        depth_autocal = bool(getattr(val, "HAND_DEPTH_AUTOCALIBRATE", False))
        if depth_autocal and self.depth_calibrator is not None:
            try:
                reach_norm = float(self.depth_calibrator.normalize01(size_metric))
            except Exception:
                reach_norm = _clip((size_metric - size_far) / (size_near - size_far), 0.0, 1.0)
                depth_autocal = False
        else:
            reach_norm = _clip((size_metric - size_far) / (size_near - size_far), 0.0, 1.0)

        # Keep horizontal/left-right mapping untouched.  Apply only O(1)
        # monotonic shaping to depth and vertical controls so hand positions
        # near their endpoints push closer to the calibrated full extension.
        reach_norm_raw = float(reach_norm)
        reach_norm, depth_centered_raw, depth_centered_shaped = self._shape_norm_for_extension(reach_norm_raw, "depth")
        vertical_norm_raw = _clip(1.0 - hand_cy, 0.0, 1.0)
        vertical_norm, vertical_centered_raw, vertical_centered_shaped = self._shape_norm_for_extension(vertical_norm_raw, "vertical")

        forward = _lerp(float(getattr(val, "WORKSPACE_Y_MAX", 0.22)),
                        float(getattr(val, "WORKSPACE_Y_MIN", 0.10)),
                        reach_norm)
        z_world = _lerp(float(getattr(val, "WORKSPACE_Z_MIN", 0.00)),
                        float(getattr(val, "WORKSPACE_Z_MAX", 0.22)),
                        vertical_norm)

        # Closed-form 3-DOF planar IK with a gripper-horizontal constraint, so
        # wrist_flex (motor 4) actively participates instead of sitting at zero.
        # Three links in the vertical plane: upper_arm L1, forearm L2, tool L3.
        # We require theta1 + theta2 + theta3 = 0 (last segment parallel to
        # ground), which means the wrist center sits L3 in front of the EE.
        L1 = float(getattr(val, "IK_LINK1_M", 0.115))
        L2 = float(getattr(val, "IK_LINK2_M", 0.115))
        L3 = (float(getattr(val, "IK_TOOL_A_M", 0.025))
              + float(getattr(val, "IK_TOOL_B_M", 0.025)))
        shoulder_z = float(getattr(val, "IK_SHOULDER_Z_M", 0.06))

        forward_wc = forward - L3
        height_wc = z_world - shoulder_z

        target_dist = math.sqrt(forward_wc * forward_wc + height_wc * height_wc)

        # Clamp the wrist-center target to the reachable annulus so cos(elbow)
        # stays in [-1, 1] and the arm never has to extend past its limits.
        max_reach = (L1 + L2) * 0.98
        min_reach = max(abs(L1 - L2) + 0.01, 1e-3)
        if target_dist > max_reach:
            scale = max_reach / max(target_dist, 1e-6)
            forward_wc *= scale
            height_wc *= scale
            target_dist = max_reach
        elif target_dist < min_reach:
            scale = min_reach / max(target_dist, 1e-6)
            forward_wc *= scale
            height_wc *= scale
            target_dist = min_reach

        cos_t2 = (target_dist * target_dist - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
        cos_t2 = max(-1.0, min(1.0, cos_t2))
        theta2 = math.acos(cos_t2)

        alpha = math.atan2(height_wc, max(forward_wc, 1e-6))
        beta = math.atan2(L2 * math.sin(theta2), L1 + L2 * math.cos(theta2))
        theta1 = alpha - beta

        # shoulder_pan: direct mapping (mirror-flipped frame, so hand_cx=0 is
        # the user's left side of the image -> pan_lo).
        pan_lo, pan_hi = _get_limit("BASE_PAN", -3.0, 3.0)
        shoulder_pan = _lerp(pan_lo, pan_hi, _clip(hand_cx, 0.0, 1.0))

        # The calibration's neutral is now the user's chosen pose (claw forward,
        # arm extended), so "math zero" already coincides with motor zero for
        # shoulder_lift. No -pi/2 offset is needed. wrist_flex carries the
        # keep-gripper-horizontal correction theta2-theta1.
        out = {
            "shoulder_pan": float(shoulder_pan),
            "shoulder_lift": float(-theta1),
            "elbow_flex": float(theta2),
            "wrist_flex": float(theta2 - theta1),
            "wrist_yaw": 0.0,
            "wrist_roll": 0.0,
            "wrist_pitch": 0.0,
            "gripper_open01": float(gripper_open01),
            "__diagnostics__": {
                "hand_size_metric": float(size_metric),
                "hand_depth_norm": float(reach_norm),
                "depth_norm": float(reach_norm),
                "hand_depth_norm_raw": float(reach_norm_raw),
                "hand_depth_centered_raw": float(depth_centered_raw),
                "hand_depth_centered_shaped": float(depth_centered_shaped),
                "hand_vertical_norm_raw": float(vertical_norm_raw),
                "hand_vertical_norm": float(vertical_norm),
                "hand_vertical_centered_raw": float(vertical_centered_raw),
                "hand_vertical_centered_shaped": float(vertical_centered_shaped),
                "workspace_h_centered_raw": float(_clip(2.0 * (hand_cx - 0.5), -1.0, 1.0)),
                "workspace_h_centered_shaped": float(_clip(2.0 * (hand_cx - 0.5), -1.0, 1.0)),
                "workspace_extension_shaping_enabled": bool(
                    bool(getattr(val, "ROBOT_WORKSPACE_VERTICAL_ENDPOINT_BOOST_ENABLED", True))
                    or bool(getattr(val, "ROBOT_WORKSPACE_DEPTH_ENDPOINT_BOOST_ENABLED", True))
                ),
                "hand_depth_autocalibrated": bool(depth_autocal),
            },
        }
        diag = dict(out.get("__diagnostics__", {}))
        return self.apply_simple_palm_roll_override(out, hand_lms, diag)
