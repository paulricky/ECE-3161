"""Simple hand → end-effector mapping.

Pipeline (no calibration files, no async, no fallbacks):

    camera frame
      → MediaPipe gives palm center (cx, cy ∈ [0,1]) and palm size
      → linear lerp into a fixed XYZ workspace box
      → simple closed-form 4-DOF IK (pan + lift + elbow + wrist_flex)
      → JointCommand

The four wrist orientation joints (wrist_yaw / wrist_roll / wrist_pitch) are
held at zero. wrist_flex is set to keep the gripper horizontal (tool axis
aligned with the radial direction). Sign conventions match mathmodel.py:
positive shoulder_lift rotates the arm downward (Ry(+) takes +x to -z).

Tune the workspace box and palm-size depth gain at the top of this file.
If a single axis tracks the wrong way, flip JOINT_SIGN_* for that joint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:
    raise ImportError("mediapipe is required for simple_handtrack") from exc


# ---------------------------------------------------------------------------
# Workspace box. The end effector tip will move within this XYZ region (in the
# robot base frame, meters). Verified all 27 corners+center reach with FK
# round-trip < 5 mm using the chain lengths below.
# ---------------------------------------------------------------------------
EE_X_MIN, EE_X_MAX = -0.10, 0.10   # left ↔ right
EE_Y_MIN, EE_Y_MAX = 0.13, 0.20    # forward depth (mapped from palm size)
EE_Z_MIN, EE_Z_MAX = 0.04, 0.18    # down ↔ up

# Kinematic chain (must match mathmodel.py / values.py defaults).
L1 = 0.115        # upper arm
L2 = 0.115        # forearm
L3 = 0.05         # tool (tool_a + tool_b)
BASE_Z = 0.06     # shoulder height above base

# Palm-size → depth gain. The wrist-to-middle-MCP distance in normalized image
# coords ranges roughly 0.10 (hand far / small) to 0.30 (hand near / large).
# Tune by reading the on-screen palm value at your two extremes.
PALM_NEAR = 0.30  # hand all the way in: maps to EE_Y_MIN (closest to base)
PALM_FAR = 0.10   # hand all the way out: maps to EE_Y_MAX (farthest)

# Smoothing on the EE target (0..1, where 1.0 = no smoothing, replace).
SMOOTHING_ALPHA = 0.35

# Per-joint sign overrides for if the arm runs an axis the wrong way.
# Multiply the IK output by these before sending to the robot. Default +1 each.
JOINT_SIGN_PAN = 1.0
JOINT_SIGN_LIFT = 1.0
JOINT_SIGN_ELBOW = 1.0
JOINT_SIGN_WRIST_FLEX = 1.0

# Math-frame angles at the calibrated NEUTRAL pose.
# Calibration treats the arm's folded/stowed pose as motor command (0,0,0,0).
# But the IK math is naturally written in "horizontal extended = 0" frame, so
# we subtract these offsets before sending to the motor.
#
# Defaults below correspond to: upper arm vertical UP, forearm folded back
# 180 deg so it points DOWN parallel to the upper arm, gripper inline with
# forearm (also pointing down). If your folded pose is different, change
# these constants and re-run.
NEUTRAL_PAN_RAD = 0.0
NEUTRAL_LIFT_RAD = -math.pi / 2.0    # upper arm pointing up (math: -lift = up)
NEUTRAL_ELBOW_RAD = math.pi          # forearm folded back fully
NEUTRAL_WRIST_FLEX_RAD = 0.0         # gripper continues forearm direction


def _lerp(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * t)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _wrap_to_pi(angle: float) -> float:
    """Wrap an angle into the range [-pi, pi] so the motor takes the shortest
    angular path. After subtracting NEUTRAL_*_RAD, raw values can land outside
    one revolution; wrapping picks the equivalent angle within the motor's
    natural travel range.
    """
    a = (float(angle) + math.pi) % (2.0 * math.pi)
    return a - math.pi


def _ik_math(target_x: float, target_y: float, target_z: float) -> Optional[dict]:
    """Closed-form 4-DOF IK in MATH frame (math 0 = arm horizontal extended).

    Convention (matches mathmodel.py FK):
      shoulder_pan: rotation around +z, atan2(y, x).
      shoulder_lift: rotation around +y, positive = arm tilts down.
      elbow_flex:   rotation around +y, negative = elbow up branch here.
      wrist_flex:   rotation around +y, set so tool axis stays horizontal.
    """
    pan = math.atan2(target_y, target_x) if (target_x or target_y) else 0.0
    radial = math.hypot(target_x, target_y)
    height = target_z - BASE_Z

    xc = radial - L3
    zc = height
    d2 = xc * xc + zc * zc
    d = math.sqrt(d2)
    if d > L1 + L2 - 0.005:
        return None
    if d < abs(L1 - L2) + 0.005:
        return None

    cos_t2 = (d2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    cos_t2 = max(-1.0, min(1.0, cos_t2))
    t2 = math.acos(cos_t2)
    alpha = math.atan2(zc, xc)
    beta = math.atan2(L2 * math.sin(t2), L1 + L2 * math.cos(t2))
    t1 = alpha - beta

    return {
        "shoulder_pan": pan,
        "shoulder_lift": -t1,
        "elbow_flex": -t2,
        "wrist_flex": -(-t1 + -t2),  # so total Ry rotation = 0 (tool horizontal)
    }


def simple_ik(target_x: float, target_y: float, target_z: float) -> Optional[dict]:
    """Closed-form IK that returns MOTOR commands.

    Motor 0 = calibrated neutral (folded pose). The IK math runs in the
    math frame (math 0 = horizontal extended), so we subtract the math-frame
    angles of the folded neutral before applying per-joint sign flips.
    """
    m = _ik_math(target_x, target_y, target_z)
    if m is None:
        return None
    return {
        "shoulder_pan": _wrap_to_pi((m["shoulder_pan"] - NEUTRAL_PAN_RAD) * JOINT_SIGN_PAN),
        "shoulder_lift": _wrap_to_pi((m["shoulder_lift"] - NEUTRAL_LIFT_RAD) * JOINT_SIGN_LIFT),
        "elbow_flex": _wrap_to_pi((m["elbow_flex"] - NEUTRAL_ELBOW_RAD) * JOINT_SIGN_ELBOW),
        "wrist_flex": _wrap_to_pi((m["wrist_flex"] - NEUTRAL_WRIST_FLEX_RAD) * JOINT_SIGN_WRIST_FLEX),
        "wrist_yaw": 0.0,
        "wrist_roll": 0.0,
        "wrist_pitch": 0.0,
    }


def fk_position(pan: float, lift: float, elbow: float, wrist_flex: float) -> Tuple[float, float, float]:
    """FK in the same convention as simple_ik. Returns world-frame XYZ."""
    a1 = lift
    a2 = lift + elbow
    a3 = lift + elbow + wrist_flex
    radial = L1 * math.cos(a1) + L2 * math.cos(a2) + L3 * math.cos(a3)
    height = -(L1 * math.sin(a1) + L2 * math.sin(a2) + L3 * math.sin(a3))
    return (radial * math.cos(pan), radial * math.sin(pan), height + BASE_Z)


# ---------------------------------------------------------------------------
# MediaPipe hand detection
# ---------------------------------------------------------------------------

@dataclass
class HandReading:
    cx: float           # palm center X in normalized image coords [0,1]
    cy: float           # palm center Y in normalized image coords [0,1]
    palm: float         # palm size (wrist→middle MCP distance, normalized)
    open01: float       # gripper openness [0,1] from finger extension
    landmarks: object   # raw MediaPipe landmarks list (for debug overlay)


class SimpleHandTracker:
    def __init__(self):
        self._mp = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self._smoothed_xyz: Optional[np.ndarray] = None
        self._smoothed_open: float = 1.0

    def close(self) -> None:
        try:
            self._mp.close()
        except Exception:
            pass

    def _read_hand(self, frame: np.ndarray) -> Optional[HandReading]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._mp.process(rgb)
        if not results.multi_hand_landmarks:
            return None
        lms = results.multi_hand_landmarks[0].landmark
        wrist = lms[0]
        middle_mcp = lms[9]
        middle_tip = lms[12]
        cx = 0.5 * (wrist.x + middle_mcp.x)
        cy = 0.5 * (wrist.y + middle_mcp.y)
        palm = math.hypot(middle_mcp.x - wrist.x, middle_mcp.y - wrist.y)
        finger = math.hypot(middle_tip.x - middle_mcp.x, middle_tip.y - middle_mcp.y)
        ratio = finger / max(palm, 1e-6)
        open01 = _clip01((ratio - 0.3) / 0.7)
        return HandReading(cx=cx, cy=cy, palm=palm, open01=open01, landmarks=lms)

    def hand_to_xyz(self, hand: HandReading) -> np.ndarray:
        x = _lerp(EE_X_MIN, EE_X_MAX, _clip01(hand.cx))
        z = _lerp(EE_Z_MIN, EE_Z_MAX, _clip01(1.0 - hand.cy))
        depth_norm = _clip01((hand.palm - PALM_FAR) / max(PALM_NEAR - PALM_FAR, 1e-6))
        y = _lerp(EE_Y_MAX, EE_Y_MIN, depth_norm)
        return np.array([x, y, z], dtype=np.float64)

    def process(self, frame: np.ndarray) -> Optional[dict]:
        """Returns dict with target_xyz, joints, gripper_open01, or None."""
        hand = self._read_hand(frame)
        if hand is None:
            self._draw_idle(frame)
            return None

        target = self.hand_to_xyz(hand)
        if self._smoothed_xyz is None:
            self._smoothed_xyz = target.copy()
        else:
            a = SMOOTHING_ALPHA
            self._smoothed_xyz = (1.0 - a) * self._smoothed_xyz + a * target
        self._smoothed_open = (1.0 - SMOOTHING_ALPHA) * self._smoothed_open + SMOOTHING_ALPHA * hand.open01

        xyz = self._smoothed_xyz
        joints = simple_ik(float(xyz[0]), float(xyz[1]), float(xyz[2]))
        self._draw_overlay(frame, hand, xyz, joints, self._smoothed_open)
        if joints is None:
            return None
        return {
            "target_xyz": xyz.copy(),
            "joints": joints,
            "gripper_open01": float(self._smoothed_open),
        }

    # ---------------- visual debug ----------------

    def _draw_landmarks(self, frame, lms) -> None:
        h, w = frame.shape[:2]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17),
        ]
        pts = [(int(p.x * w), int(p.y * h)) for p in lms]
        for i, j in edges:
            cv2.line(frame, pts[i], pts[j], (180, 180, 180), 2)
        for p in pts:
            cv2.circle(frame, p, 3, (0, 255, 0), -1)

    def _draw_idle(self, frame) -> None:
        cv2.putText(frame, "no hand", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def _draw_overlay(self, frame, hand: HandReading, xyz: np.ndarray, joints, open01: float) -> None:
        self._draw_landmarks(frame, hand.landmarks)
        h, w = frame.shape[:2]
        lines = [
            f"hand n=({hand.cx:.2f},{hand.cy:.2f}) palm={hand.palm:.3f} open={open01:.2f}",
            f"target xyz=({xyz[0]:+.3f},{xyz[1]:+.3f},{xyz[2]:+.3f}) m",
        ]
        if joints is None:
            lines.append("IK: UNREACHABLE")
        else:
            lines.append(
                f"joints pan={joints['shoulder_pan']:+.2f} lift={joints['shoulder_lift']:+.2f} "
                f"elbow={joints['elbow_flex']:+.2f} wflex={joints['wrist_flex']:+.2f}"
            )
        for i, line in enumerate(lines):
            y = 30 + i * 24
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
