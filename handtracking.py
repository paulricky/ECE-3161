from __future__ import annotations

import json
import math
import os
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

import mathmodel as mm
import values as val


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
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
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


def openness_from_fingertips(hand_lms, label: str):
    global _hand_closed_bool

    spread = fingertips_spread(hand_lms)

    st = _hand_closed_bool[label]
    if st:
        if spread > val.CLOSED_TIPS_OFF:
            st = False
    else:
        if spread < val.CLOSED_TIPS_ON:
            st = True
    _hand_closed_bool[label] = st

    denom = val.CLOSED_TIPS_OFF - val.CLOSED_TIPS_ON
    if denom <= 1e-9:
        open01 = 0.0 if st else 1.0
    else:
        open01 = (spread - val.CLOSED_TIPS_ON) / denom
        open01 = max(0.0, min(1.0, open01))

    return st, open01, spread


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

    is_closed, open01, _spread = openness_from_fingertips(hand_lms, label)
    state = "CLOSED" if is_closed else ("OPEN" if open01 > 0.8 else "PARTIAL")

    if _hand_open_state[label] != state:
        _hand_open_state[label] = state
        log_event(f"{label} {state}")

    return state, open01, snap_event


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

        state, open01, snap_event = update_snap_and_open_state(hand_lms, label, now, dt)
        per_hand[label] = {"state": state, "open01": open01, "snap": snap_event, "score": score}

        h_img, w_img = frame.shape[:2]
        xw = int(hand_lms.landmark[0].x * w_img)
        yw = int(hand_lms.landmark[0].y * h_img)

        extras = [state]
        if snap_event:
            extras.append("SNAP!")
        if clap_event:
            extras.append("CLAP!")

        text = f"{label} ({score:.2f}) {extras}"
        cv2.putText(frame, text, (xw + 10, yw - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return clap_event, per_hand


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

        if not self.enabled:
            self.detector = None
            return

        dict_name = getattr(val, "ARUCO_DICT_NAME", "DICT_4X4_50")
        dict_id = getattr(cv2.aruco, dict_name)
        dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        params = cv2.aruco.DetectorParameters()
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

    def detect(self, frame):
        if self.detector is None or self.camera_matrix is None or self.dist_coeffs is None or self.T_workspace_from_camera is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return None

        ids = ids.flatten().tolist()
        chosen_idx = None
        chosen_id = None

        for candidate in (self.front_id, self.back_id):
            if candidate in ids:
                chosen_idx = ids.index(candidate)
                chosen_id = candidate
                break

        if chosen_idx is None:
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
            return None

        T_camera_from_marker = _rvec_tvec_to_T(rvec, tvec)
        marker_origin_camera = T_camera_from_marker[:3, 3]
        marker_origin_workspace = _T_apply(self.T_workspace_from_camera, marker_origin_camera)

        R_marker_camera = T_camera_from_marker[:3, :3]
        R_workspace_marker = self.T_workspace_from_camera[:3, :3] @ R_marker_camera
        workspace_rpy = _rot_to_rpy(R_workspace_marker)

        return {
            "marker_id": int(chosen_id),
            "workspace_xyz": marker_origin_workspace,
            "workspace_rpy": workspace_rpy,
            "image_corners": image_points,
        }

    def normalize_workspace_xyz(self, xyz):
        denom = self.workspace_max - self.workspace_min
        denom = np.where(np.abs(denom) < 1e-9, 1.0, denom)
        z = (np.asarray(xyz, dtype=np.float64) - self.workspace_min) / denom
        return np.clip(z, 0.0, 1.0)


class HandTracker:
    def __init__(self):
        self.prev_time = time.time()
        self._last_cmd = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper_open01": 1.0,
        }
        self._last_open01 = 1.0
        self._alpha = float(getattr(val, "HAND_CMD_SMOOTHING", 0.25))
        self.aruco = ArucoGloveTracker()
        self.lerobot_calibration = self._load_lerobot_calibration()

    def _load_lerobot_calibration(self):
        path = getattr(val, "LEROBOT_CALIBRATION_FILE", "").strip()
        if not path:
            robot_id = getattr(val, "REAL_ROBOT_ID", "my_awesome_follower_arm")
            home = os.path.expanduser("~")
            path = os.path.join(
                home,
                ".cache",
                "huggingface",
                "lerobot",
                "calibration",
                "robots",
                "so101_follower",
                f"{robot_id}.json",
            )

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

        draw_and_update_gestures(frame, detected_hands, now, dt)

        aruco_pose = self.aruco.detect(frame)
        if aruco_pose is not None:
            open01 = self._estimate_gripper_open01(detected_hands)
            cmd = self._aruco_pose_to_command(aruco_pose, open01)
            if cmd is not None:
                out = self._smooth_command(cmd)
                out["mode"] = "aruco"
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
        self._draw_command_overlay(frame, out)
        return out

    def _smooth_command(self, cmd):
        for k in self._last_cmd:
            self._last_cmd[k] = _lerp(self._last_cmd[k], float(cmd[k]), self._alpha)
        return dict(self._last_cmd)

    def _estimate_gripper_open01(self, detected_hands):
        driver = choose_driver(detected_hands)
        if driver is None:
            return self._last_open01
        hand_lms, label, _score = driver
        is_closed, open01, _spread = openness_from_fingertips(hand_lms, label)
        if is_closed:
            open01 = 0.0
        self._last_open01 = open01
        return open01

    def _aruco_pose_to_command(self, aruco_pose, open01):
        xyz = np.asarray(aruco_pose["workspace_xyz"], dtype=np.float64)
        rpy = np.asarray(aruco_pose["workspace_rpy"], dtype=np.float64)
        xyz_norm = self.aruco.normalize_workspace_xyz(xyz)

        if hasattr(mm, "solve_ik_from_target"):
            try:
                solved = mm.solve_ik_from_target(
                    target_xyz=xyz,
                    target_rpy=rpy,
                    gripper_open01=float(open01),
                    lerobot_calibration=self.lerobot_calibration,
                )
                if isinstance(solved, dict):
                    return {
                        "shoulder_pan": float(solved["shoulder_pan"]),
                        "shoulder_lift": float(solved["shoulder_lift"]),
                        "elbow_flex": float(solved["elbow_flex"]),
                        "wrist_flex": float(solved["wrist_flex"]),
                        "wrist_roll": float(solved["wrist_roll"]),
                        "gripper_open01": float(solved["gripper_open01"]),
                    }
                if isinstance(solved, (list, tuple)) and len(solved) >= 5:
                    return {
                        "shoulder_pan": float(solved[0]),
                        "shoulder_lift": float(solved[1]),
                        "elbow_flex": float(solved[2]),
                        "wrist_flex": float(solved[3]),
                        "wrist_roll": float(solved[4]),
                        "gripper_open01": float(open01),
                    }
            except Exception:
                pass

        pan_lo, pan_hi = _get_limit("BASE_PAN", -1.2, 1.2)
        lift_lo, lift_hi = _get_limit("SHOULDER_LIFT", -0.8, 1.0)
        elbow_lo, elbow_hi = _get_limit("ELBOW", -0.9, 1.2)
        wflex_lo, wflex_hi = _get_limit("WRIST_FLEX", -0.9, 0.9)
        wroll_lo, wroll_hi = _get_limit("WRIST_ROLL", -1.5, 1.5)

        shoulder_pan = _norm_to_range(1.0 - xyz_norm[0], pan_lo, pan_hi)
        shoulder_lift = _norm_to_range(1.0 - xyz_norm[1], lift_lo, lift_hi)
        elbow_flex = _norm_to_range(xyz_norm[2], elbow_lo, elbow_hi)

        pitch_norm = _clip((rpy[1] + math.pi / 2.0) / math.pi, 0.0, 1.0)
        roll_norm = _clip((rpy[0] + math.pi) / (2.0 * math.pi), 0.0, 1.0)

        wrist_flex = _norm_to_range(pitch_norm, wflex_lo, wflex_hi)
        wrist_roll = _norm_to_range(roll_norm, wroll_lo, wroll_hi)

        return {
            "shoulder_pan": float(shoulder_pan),
            "shoulder_lift": float(shoulder_lift),
            "elbow_flex": float(elbow_flex),
            "wrist_flex": float(wrist_flex),
            "wrist_roll": float(wrist_roll),
            "gripper_open01": float(open01),
        }

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
            f"wflex={out['wrist_flex']:.2f} wroll={out['wrist_roll']:.2f} grip={out['gripper_open01']:.2f}",
            (10, h_img - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

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

        is_closed, open01, _spread = openness_from_fingertips(hand_lms, label)

        pan_lo, pan_hi = _get_limit("BASE_PAN", -1.2, 1.2)
        lift_lo, lift_hi = _get_limit("SHOULDER_LIFT", -0.8, 1.0)
        elbow_lo, elbow_hi = _get_limit("ELBOW", -0.9, 1.2)
        wflex_lo, wflex_hi = _get_limit("WRIST_FLEX", -0.9, 0.9)
        wroll_lo, wroll_hi = _get_limit("WRIST_ROLL", -1.5, 1.5)

        pan_norm = 1.0 - hand_cx
        lift_norm = 1.0 - hand_cy

        size_lo = float(getattr(val, "HAND_SIZE_NEAR", 0.08))
        size_hi = float(getattr(val, "HAND_SIZE_FAR", 0.22))
        if size_hi <= size_lo:
            size_hi = size_lo + 1e-3
        depth_norm = (size_metric - size_lo) / (size_hi - size_lo)
        depth_norm = _clip(depth_norm, 0.0, 1.0)

        palm_tilt = _clip((wrist[1] - middle_mcp[1]) / 0.25, -1.0, 1.0)

        palm_line_angle = _angle_2d(index_mcp, pinky_mcp)
        wrist_roll_norm = _clip((palm_line_angle + math.pi / 2.0) / math.pi, 0.0, 1.0)

        shoulder_pan = _norm_to_range(pan_norm, pan_lo, pan_hi)
        shoulder_lift = _norm_to_range(lift_norm, lift_lo, lift_hi)
        elbow_flex = _norm_to_range(depth_norm, elbow_lo, elbow_hi)
        wrist_flex = _norm_to_range(0.5 * (1.0 + palm_tilt), wflex_lo, wflex_hi)
        wrist_roll = _norm_to_range(wrist_roll_norm, wroll_lo, wroll_hi)

        if label == "Left":
            shoulder_pan = -shoulder_pan
            wrist_roll = -wrist_roll

        pinch_dist = mm.dist(thumb_tip, index_tip)
        pinch_lo = float(getattr(val, "PINCH_CLOSE_DIST", 0.03))
        pinch_hi = float(getattr(val, "PINCH_OPEN_DIST", 0.12))
        if pinch_hi <= pinch_lo:
            pinch_hi = pinch_lo + 1e-3
        pinch_open01 = _clip((pinch_dist - pinch_lo) / (pinch_hi - pinch_lo), 0.0, 1.0)

        gripper_open01 = min(open01, pinch_open01)
        if is_closed:
            gripper_open01 = 0.0

        self._last_open01 = gripper_open01

        return {
            "shoulder_pan": shoulder_pan,
            "shoulder_lift": shoulder_lift,
            "elbow_flex": elbow_flex,
            "wrist_flex": wrist_flex,
            "wrist_roll": wrist_roll,
            "gripper_open01": gripper_open01,
        }