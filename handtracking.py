from __future__ import annotations

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
    tips = [(lm[4].x, lm[4].y), (lm[8].x, lm[8].y), (lm[12].x, lm[12].y), (lm[16].x, lm[16].y), (lm[20].x, lm[20].y)]
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

    denom = (val.CLOSED_TIPS_OFF - val.CLOSED_TIPS_ON)
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
