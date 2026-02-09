import cv2
import collections
import time
import mediapipe as mp

import mathmodel as mm
import values as val

action_log = collections.deque()

def log_event(text):
    t = time.time()
    action_log.append((t, text))
    while len(action_log) > val.LOG_MAX:
        action_log.popleft()

def fingers_up(hand_lms, handedness_label):
    lm = hand_lms.landmark
    pts = [(lm[i].x, lm[i].y) for i in range(21)]
    fingers = [0, 0, 0, 0, 0]

    thumb_tip = pts[4][0]
    thumb_ip = pts[3][0]
    if handedness_label.lower() == "right":
        fingers[0] = 1 if thumb_tip < thumb_ip else 0
    else:
        fingers[0] = 1 if thumb_tip > thumb_ip else 0

    fingers[1] = 1 if pts[8][1]  < pts[6][1]  else 0
    fingers[2] = 1 if pts[12][1] < pts[10][1] else 0
    fingers[3] = 1 if pts[16][1] < pts[14][1] else 0
    fingers[4] = 1 if pts[20][1] < pts[18][1] else 0
    return fingers

def hand_center(hand_lms):
    lm = hand_lms.landmark
    idxs = [0, 5, 9, 13, 17]
    x = sum(lm[i].x for i in idxs) / len(idxs)
    y = sum(lm[i].y for i in idxs) / len(idxs)
    return (x, y)

mp_hands = mp.solutions.hands
draw = mp.solutions.drawing_utils
styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

_prev_hands_dist = None
_clap_cooldown_until = 0.0

_snap_state = {
    "Left": {"pinched": False, "prev_d": None, "cooldown_until": 0.0},
    "Right": {"pinched": False, "prev_d": None, "cooldown_until": 0.0},
}

_hand_open_state = {"Left": None, "Right": None}
_wrist_angle_deg_sm = {"Left": None, "Right": None}


def reset_gestures():
    global _prev_hands_dist, _clap_cooldown_until, _snap_state, _hand_open_state, _wrist_angle_deg_sm
    _prev_hands_dist = None
    _clap_cooldown_until = 0.0
    _snap_state = {
        "Left": {"pinched": False, "prev_d": None, "cooldown_until": 0.0},
        "Right": {"pinched": False, "prev_d": None, "cooldown_until": 0.0},
    }
    _hand_open_state = {"Left": None, "Right": None}
    _wrist_angle_deg_sm = {"Left": None, "Right": None}


def build_detected_hands(results):
    detected = []
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            score = handedness.classification[0].score
            detected.append((hand_lms, label, score))
    return detected


def choose_driver(detected_hands):
    # prefer Right hand
    for h in detected_hands:
        if h[1] == "Right":
            return h
    return detected_hands[0] if detected_hands else None


def update_snap_and_open_state(hand_lms, label, now, dt):
    """
    Updates snap/open/closed and wrist angle smoothing, logs transitions.
    Returns: (open_state, up_count, ang_show, snap_event)
    """
    global _hand_open_state, _wrist_angle_deg_sm, _snap_state

    lm = hand_lms.landmark

    # wrist angle (for overlay/debug)
    raw_ang = mm.angle_deg((lm[0].x, lm[0].y), (lm[9].x, lm[9].y))
    _wrist_angle_deg_sm[label] = mm.ema(_wrist_angle_deg_sm[label], raw_ang, alpha=val.ANGLE_SMOOTH_ALPHA)
    ang_show = _wrist_angle_deg_sm[label]

    # open/closed
    f = fingers_up(hand_lms, label)
    up_count = sum(f)
    if up_count <= 1:
        open_state = "CLOSED"
    elif up_count >= 4:
        open_state = "OPEN"
    else:
        open_state = "PARTIAL"

    if _hand_open_state[label] != open_state:
        _hand_open_state[label] = open_state
        log_event(f"{label} {open_state}")

    # SNAP (thumb -> middle tip)
    thumb_tip = (lm[4].x, lm[4].y)
    middle_tip = (lm[12].x, lm[12].y)
    d_tm = mm.dist(thumb_tip, middle_tip)

    st = _snap_state[label]
    snap_event = False

    if now > st["cooldown_until"]:
        pinch_on = d_tm < val.SNAP_PINCH_ON
        pinch_off = d_tm > val.SNAP_PINCH_OFF
        opening_speed = (d_tm - st["prev_d"]) / dt if st["prev_d"] is not None else 0.0

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

    return open_state, up_count, ang_show, snap_event


def update_clap(detected_hands, now, dt):
    """
    Clap = two hands moving together fast enough and close enough.
    """
    global _prev_hands_dist, _clap_cooldown_until

    clap_event = False
    if len(detected_hands) == 2:
        c0 = hand_center(detected_hands[0][0])
        c1 = hand_center(detected_hands[1][0])
        d = mm.dist(c0, c1)

        if _prev_hands_dist is not None:
            closing_speed = (_prev_hands_dist - d) / dt
            if now > _clap_cooldown_until and d < val.CLAP_CLOSE_ENOUGH and closing_speed > val.CLAP_FAST_CLOSING:
                clap_event = True
                _clap_cooldown_until = now + val.CLAP_COOLDOWN_S
                log_event("CLAP")

        _prev_hands_dist = d
    else:
        _prev_hands_dist = None

    return clap_event


def draw_and_update_gestures(frame, detected_hands, now, dt):
    """
    Draw landmarks + per-hand overlay + update snap/open/wrist + compute clap.
    Returns: clap_event, per_hand_info dict keyed by label.
    """
    clap_event = update_clap(detected_hands, now, dt)

    per_hand = {}
    for hand_lms, label, score in detected_hands:
        # draw landmarks
        draw.draw_landmarks(
            frame, hand_lms, mp_hands.HAND_CONNECTIONS,
            styles.get_default_hand_landmarks_style(),
            styles.get_default_hand_connections_style()
        )

        open_state, up_count, ang_show, snap_event = update_snap_and_open_state(hand_lms, label, now, dt)

        # overlay near wrist
        lm = hand_lms.landmark
        h_img, w_img = frame.shape[:2]
        xw, yw = int(lm[0].x * w_img), int(lm[0].y * h_img)

        extras = [open_state]
        if snap_event:
            extras.append("SNAP!")
        if clap_event:
            extras.append("CLAP!")

        text = f"{label} ({score:.2f}) up:{up_count} ang:{ang_show:.2f} {extras}"
        cv2.putText(frame, text, (xw + 10, yw - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        per_hand[label] = {
            "score": score,
            "open_state": open_state,
            "up_count": up_count,
            "wrist_angle_deg": ang_show,
            "snap": snap_event,
        }

    return clap_event, per_hand

