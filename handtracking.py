from __future__ import annotations

import math
import time
from collections import deque

import cv2
import mediapipe as mp

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
        if open01 < 0.0:
            open01 = 0.0
        if open01 > 1.0:
            open01 = 1.0

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
        self._alpha = float(getattr(val, "HAND_CMD_SMOOTHING", 0.25))

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

        driver = choose_driver(detected_hands)
        if driver is None:
            return None

        hand_lms, label, _score = driver
        cmd = self._landmarks_to_command(hand_lms, label)

        for k in self._last_cmd:
            self._last_cmd[k] = _lerp(self._last_cmd[k], cmd[k], self._alpha)

        out = dict(self._last_cmd)

        h_img, w_img = frame.shape[:2]
        cv2.putText(
            frame,
            f"CMD pan={out['shoulder_pan']:.2f} lift={out['shoulder_lift']:.2f} elbow={out['elbow_flex']:.2f}",
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

        return out

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

        return {
            "shoulder_pan": shoulder_pan,
            "shoulder_lift": shoulder_lift,
            "elbow_flex": elbow_flex,
            "wrist_flex": wrist_flex,
            "wrist_roll": wrist_roll,
            "gripper_open01": gripper_open01,
        }