"""Minimal hand-tracking → robot teleop.

Pipeline:
  camera → SimpleHandTracker.process(frame) → JointCommand → robot.

ESC always disables motor torque before exit, so the arm goes limp instead of
holding its last commanded position. --dry-run skips the robot connection
entirely and just shows the mapping in the camera window.

Run:
    python3 simple_main.py --dry-run        # see the mapping, arm doesn't move
    python3 simple_main.py                  # send to the robot
    python3 simple_main.py --camera 1       # pick a different camera index
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from typing import Optional

import cv2

import values as val
from robot_controller import JointCommand, SOArmHardwareController
import simple_handtrack as _sh
from simple_handtrack import SimpleHandTracker


# Debug mode: only command shoulder_lift, hold every other joint at motor 0.
# In direct mode we bypass the IK entirely and just lerp hand_cy into a known
# command range, so the motor command is unambiguous and not subject to IK
# wrap-around or workspace-shape constraints. Adjust LIFT_DIRECT_AT_TOP /
# LIFT_DIRECT_AT_BOTTOM (radians) until lift moves the way you want; if it
# moves the wrong direction, swap their values.
# Debug modes. Set TEST_JOINT to:
#   "lift"    - drive only motor 2 from hand_cy
#   "elbow"   - drive only motor 3 from hand.palm
#   "3dof"    - drive pan/lift/elbow from simple_handtrack's 4-DOF analytic
#               IK (only 3 of the 4 outputs used; wrist_flex held).
#   "4dof"    - 3dof + wrist_flex (motor 4) auto-compensated to keep tool
#               axis horizontal forward. Elbow is palm-direct override.
#   "5dof"    - 4dof + wrist_pitch (motor 7, physically a roll axis in this
#               build) driven from hand_roll (index_mcp -> pinky_mcp tilt).
#   None      - drive all 4 IK outputs + gripper.
# In every "test" mode the non-driven joints hold their captured startup
# position so the arm doesn't drift.
TEST_JOINT = "5dof"

# --- pan (motor 1) direct, driven by hand_cx ---
PAN_DIRECT_AT_LEFT = -0.7
PAN_DIRECT_AT_RIGHT = +0.7
PAN_SMOOTH_ALPHA = 0.60

# --- lift-only direct (used when TEST_JOINT == "lift") ---
LIFT_DIRECT_AT_TOP = +1.5
LIFT_DIRECT_AT_BOTTOM = -0.4
LIFT_SMOOTH_ALPHA = 0.60

# --- elbow-only direct (used when TEST_JOINT == "elbow") ---
ELBOW_DIRECT_AT_NEAR = -0.4
ELBOW_DIRECT_AT_FAR  = +1.0
ELBOW_SMOOTH_ALPHA = 0.60
PALM_NEAR_NORM = 0.35   # palm >= this -> saturated to "arm forward"
PALM_FAR_NORM = 0.05    # palm <= this -> saturated to "arm back"

# --- 2-link planar IK for lift + elbow (used when TEST_JOINT == "3dof") ---
# Treats the arm as a 2-link planar manipulator in the (radial, height) plane.
# Hand depth (palm) controls radial, hand_cy controls height. Both lift and
# elbow are computed so the wrist center reaches that target point, which is
# what makes them "share the work".
LINK1 = 0.115
LINK2 = 0.115
# Workspace box for the planar IK (meters). Tighten if commands run past
# motor 2's available ROM; widen for bigger swings.
PLANAR_RADIAL_AT_NEAR = 0.20  # palm large (hand near camera) -> arm forward toward user
PLANAR_RADIAL_AT_FAR  = 0.06  # palm small (hand far)         -> arm pulled back
PLANAR_HEIGHT_AT_BOTTOM = -0.13
PLANAR_HEIGHT_AT_TOP    = +0.13
# Elbow branch: +1 = forearm reaches "up" over shoulder-wrist line,
# -1 = forearm reaches "down" under it. Pick whichever uses the half of
# motor 2's ROM that has clearance.
ELBOW_BRANCH_SIGN = +1.0
# Offsets and signs applied to the IK math output before it becomes a motor
# command. OFFSET shifts the math zero so motor 0 lines up with the user's
# neutral pose; SIGN flips the direction if a joint runs the wrong way.
# Defaults chosen so the lift command uses most of the asymmetric ROM
# (motor 2 has ~ -39 deg .. +133 deg) and the elbow stays centered.
PLANAR_LIFT_OFFSET = -1.40
PLANAR_LIFT_SIGN = +1.0
PLANAR_ELBOW_OFFSET = +1.00
PLANAR_ELBOW_SIGN = +1.0

# --- 3-DOF direct mapping (used when TEST_JOINT == "3dof") ---
# Per-axis lerp into a known motor command range. No IK, no coupling between
# joints. Tune endpoints if any axis runs the wrong way (just swap the two
# values for that joint) or if you want a different range. Pan reuses the
# PAN_DIRECT_* constants above; lift reuses PALM_NEAR_NORM/PALM_FAR_NORM.
# Both lift and elbow are driven by palm depth so motor 2 and motor 3 move
# together. Palm near (hand close to camera) extends the arm forward (lift
# up, elbow open). Palm far folds the arm back (lift down, elbow folded).
# cy is currently unused in 3DOF mode; reserve for wrist_flex when needed.
LIFT_3DOF_AT_PALM_NEAR  = +1.5  # palm large -> motor 2 fully extended forward
LIFT_3DOF_AT_PALM_FAR   = -0.4  # palm small -> motor 2 fully pulled back
ELBOW_3DOF_AT_PALM_NEAR = -0.4  # palm large -> motor 3 elbow open (extended)
ELBOW_3DOF_AT_PALM_FAR  = +1.0  # palm small -> motor 3 elbow folded

# --- 4-DOF elbow override (used when TEST_JOINT == "4dof") ---
# IK can't span the full motor 3 ROM given the workspace + tool-length geometry,
# so we drive elbow directly from palm depth using the tested ROM endpoints.
# wrist_flex is then recomputed from (lift_math, override elbow_math) so the
# tool axis stays horizontal even though we ignored the IK's elbow.
# Separate palm thresholds from simple_handtrack's so the IK target x mapping
# isn't affected. Tune these based on the palm sizes at YOUR hand-rest pose.
ELBOW_4DOF_PALM_FAR     = 0.14  # palm <= this -> motor 3 saturated to ELBOW_4DOF_AT_PALM_FAR
ELBOW_4DOF_PALM_NEAR    = 0.40  # palm >= this -> motor 3 saturated to ELBOW_4DOF_AT_PALM_NEAR
ELBOW_4DOF_AT_PALM_NEAR = -0.4  # palm large -> motor 3 lower extreme (extended)
ELBOW_4DOF_AT_PALM_FAR  = +1.0  # palm small -> motor 3 upper extreme (folded)

# --- 5-DOF wrist-roll override (used when TEST_JOINT == "5dof") ---
# Motor 7 is `wrist_pitch` in the JointCommand field per the kinematic chain
# in CLAUDE.md, but in this physical build it acts as a roll axis. Drive it
# from the user's hand roll (angle of the index_mcp -> pinky_mcp line in the
# image). hand.roll is in radians; ~0 when palm faces camera with hand
# upright, swings ~±π/2 as the hand twists. Tune endpoints to your comfort.
HAND_ROLL_LEFT_NORM   = -0.6    # rad: hand rolled to one extreme (CCW in image)
HAND_ROLL_RIGHT_NORM  = +0.6    # rad: hand rolled to other extreme (CW in image)
M7_AT_HAND_ROLL_LEFT  = -1.5    # motor 7 cmd at hand rolled left
M7_AT_HAND_ROLL_RIGHT = +1.5    # motor 7 cmd at hand rolled right


# Per-joint hold values captured at startup. In LIFT_ONLY_MODE we feed these
# back to every non-lift joint each frame so the rest of the arm doesn't drift
# toward motor 0.
_HOLD_JOINTS: Optional[dict] = None
_PAN_SMOOTHED: Optional[float] = None
_LIFT_SMOOTHED: Optional[float] = None
_ELBOW_SMOOTHED: Optional[float] = None


def _ema(prev: Optional[float], target: float, alpha: float) -> float:
    if prev is None:
        return float(target)
    return float((1.0 - alpha) * prev + alpha * target)


def _capture_hold_joints(robot: Optional[SOArmHardwareController]) -> None:
    global _HOLD_JOINTS
    if robot is None:
        _HOLD_JOINTS = None
        return
    try:
        present = robot.read_present_joints_rad()
    except Exception:
        present = None
    if isinstance(present, dict):
        _HOLD_JOINTS = {
            "shoulder_pan": float(present.get("shoulder_pan", 0.0)),
            "elbow_flex": float(present.get("elbow_flex", 0.0)),
            "wrist_flex": float(present.get("wrist_flex", 0.0)),
            "wrist_yaw": float(present.get("wrist_yaw", 0.0)),
            "wrist_roll": float(present.get("wrist_roll", 0.0)),
            "wrist_pitch": float(present.get("wrist_pitch", 0.0)),
            "gripper_open01": float(present.get("gripper_open01", 1.0)),
        }
        print(f"[simple_main] captured hold joints: {{k: round(v, 2) for k, v in _HOLD_JOINTS.items()}}")
    else:
        _HOLD_JOINTS = None
        print("[simple_main] could not read joint feedback; non-lift joints will hold at 0")


def _direct_pan(hand_cx: float) -> float:
    global _PAN_SMOOTHED
    t = max(0.0, min(1.0, float(hand_cx)))
    tgt = PAN_DIRECT_AT_LEFT + (PAN_DIRECT_AT_RIGHT - PAN_DIRECT_AT_LEFT) * t
    _PAN_SMOOTHED = _ema(_PAN_SMOOTHED, tgt, PAN_SMOOTH_ALPHA)
    return _PAN_SMOOTHED


def _direct_lift(hand_cy: float) -> float:
    global _LIFT_SMOOTHED
    t = max(0.0, min(1.0, float(hand_cy)))
    tgt = LIFT_DIRECT_AT_TOP + (LIFT_DIRECT_AT_BOTTOM - LIFT_DIRECT_AT_TOP) * t
    _LIFT_SMOOTHED = _ema(_LIFT_SMOOTHED, tgt, LIFT_SMOOTH_ALPHA)
    return _LIFT_SMOOTHED


def _direct_elbow(hand_palm: float) -> float:
    global _ELBOW_SMOOTHED
    depth_norm = (float(hand_palm) - PALM_FAR_NORM) / max(PALM_NEAR_NORM - PALM_FAR_NORM, 1e-6)
    depth_norm = max(0.0, min(1.0, depth_norm))
    tgt = ELBOW_DIRECT_AT_FAR + (ELBOW_DIRECT_AT_NEAR - ELBOW_DIRECT_AT_FAR) * depth_norm
    _ELBOW_SMOOTHED = _ema(_ELBOW_SMOOTHED, tgt, ELBOW_SMOOTH_ALPHA)
    return _ELBOW_SMOOTHED


def _planar2_ik(hand_cy: float, hand_palm: float) -> tuple[float, float, float, float]:
    """2-link planar IK in (radial, height). Returns (lift_motor, elbow_motor,
    radial, height) where lift_motor/elbow_motor are EMA-smoothed motor
    commands ready to send.
    """
    global _LIFT_SMOOTHED, _ELBOW_SMOOTHED
    depth_norm = (float(hand_palm) - PALM_FAR_NORM) / max(PALM_NEAR_NORM - PALM_FAR_NORM, 1e-6)
    depth_norm = max(0.0, min(1.0, depth_norm))
    # palm large (hand near) -> radial large (forward toward user);
    # palm small (hand far)  -> radial small (pulled back toward base).
    radial = PLANAR_RADIAL_AT_FAR + (PLANAR_RADIAL_AT_NEAR - PLANAR_RADIAL_AT_FAR) * depth_norm
    # cy=0 (top) -> height_at_top; cy=1 (bottom) -> height_at_bottom
    cy = max(0.0, min(1.0, float(hand_cy)))
    height = PLANAR_HEIGHT_AT_TOP + (PLANAR_HEIGHT_AT_BOTTOM - PLANAR_HEIGHT_AT_TOP) * cy

    # Clamp target onto reachable annulus.
    d2 = radial * radial + height * height
    d = math.sqrt(d2)
    max_d = LINK1 + LINK2 - 0.005
    min_d = abs(LINK1 - LINK2) + 0.005
    if d > max_d:
        s = max_d / max(d, 1e-6); radial *= s; height *= s; d = max_d; d2 = d * d
    if d < min_d:
        s = min_d / max(d, 1e-6); radial *= s; height *= s; d = min_d; d2 = d * d

    cos_e = (d2 - LINK1 * LINK1 - LINK2 * LINK2) / (2.0 * LINK1 * LINK2)
    cos_e = max(-1.0, min(1.0, cos_e))
    elbow = ELBOW_BRANCH_SIGN * math.acos(cos_e)
    alpha = math.atan2(height, radial)
    beta = math.atan2(LINK2 * math.sin(elbow), LINK1 + LINK2 * math.cos(elbow))
    lift = alpha - beta

    motor_lift = (lift - PLANAR_LIFT_OFFSET) * PLANAR_LIFT_SIGN
    motor_elbow = (elbow - PLANAR_ELBOW_OFFSET) * PLANAR_ELBOW_SIGN

    _LIFT_SMOOTHED = _ema(_LIFT_SMOOTHED, motor_lift, LIFT_SMOOTH_ALPHA)
    _ELBOW_SMOOTHED = _ema(_ELBOW_SMOOTHED, motor_elbow, ELBOW_SMOOTH_ALPHA)
    return _LIFT_SMOOTHED, _ELBOW_SMOOTHED, radial, height


def _direct_3dof(hand_cx: float, hand_cy: float, hand_palm: float) -> tuple[float, float, float]:
    """Per-axis direct mapping: pan from cx, lift from palm, elbow from cy.

    No IK; each joint is independent. Returns EMA-smoothed motor commands
    (pan, lift, elbow) ready to send.
    """
    global _PAN_SMOOTHED, _LIFT_SMOOTHED, _ELBOW_SMOOTHED

    pan_t = max(0.0, min(1.0, float(hand_cx)))
    pan_tgt = PAN_DIRECT_AT_LEFT + (PAN_DIRECT_AT_RIGHT - PAN_DIRECT_AT_LEFT) * pan_t
    _PAN_SMOOTHED = _ema(_PAN_SMOOTHED, pan_tgt, PAN_SMOOTH_ALPHA)

    depth_norm = (float(hand_palm) - PALM_FAR_NORM) / max(PALM_NEAR_NORM - PALM_FAR_NORM, 1e-6)
    depth_norm = max(0.0, min(1.0, depth_norm))
    lift_tgt = LIFT_3DOF_AT_PALM_FAR + (LIFT_3DOF_AT_PALM_NEAR - LIFT_3DOF_AT_PALM_FAR) * depth_norm
    _LIFT_SMOOTHED = _ema(_LIFT_SMOOTHED, lift_tgt, LIFT_SMOOTH_ALPHA)

    elbow_tgt = ELBOW_3DOF_AT_PALM_FAR + (ELBOW_3DOF_AT_PALM_NEAR - ELBOW_3DOF_AT_PALM_FAR) * depth_norm
    _ELBOW_SMOOTHED = _ema(_ELBOW_SMOOTHED, elbow_tgt, ELBOW_SMOOTH_ALPHA)

    return _PAN_SMOOTHED, _LIFT_SMOOTHED, _ELBOW_SMOOTHED


def _build_joint_command(
    joints: dict,
    gripper_open01: float,
    hand_cx: float = 0.5,
    hand_cy: float = 0.5,
    hand_palm: float = 0.20,
    hand_roll: float = 0.0,
) -> JointCommand:
    if TEST_JOINT in ("lift", "elbow", "3dof", "4dof", "5dof"):
        hold = _HOLD_JOINTS or {}
        cmd_pan = float(hold.get("shoulder_pan", 0.0))
        cmd_lift = float(hold.get("shoulder_lift", 0.0))
        cmd_elbow = float(hold.get("elbow_flex", 0.0))
        cmd_wrist_flex = float(hold.get("wrist_flex", 0.0))
        cmd_wrist_pitch = float(hold.get("wrist_pitch", 0.0))

        if TEST_JOINT == "lift":
            cmd_lift = _direct_lift(hand_cy)
        elif TEST_JOINT == "elbow":
            cmd_elbow = _direct_elbow(hand_palm)
        elif TEST_JOINT == "3dof":
            cmd_pan = float(joints["shoulder_pan"])
            cmd_lift = float(joints["shoulder_lift"])
            cmd_elbow = float(joints["elbow_flex"])
        elif TEST_JOINT in ("4dof", "5dof"):
            cmd_pan = float(joints["shoulder_pan"])
            cmd_lift = float(joints["shoulder_lift"])
            # Override elbow with palm-direct mapping for full motor 3 ROM use.
            palm_t = (float(hand_palm) - ELBOW_4DOF_PALM_FAR) / max(ELBOW_4DOF_PALM_NEAR - ELBOW_4DOF_PALM_FAR, 1e-6)
            palm_t = max(0.0, min(1.0, palm_t))
            cmd_elbow = ELBOW_4DOF_AT_PALM_FAR + (ELBOW_4DOF_AT_PALM_NEAR - ELBOW_4DOF_AT_PALM_FAR) * palm_t
            # Recompute wrist_flex against the overridden elbow so tool stays
            # horizontal (math = motor / sign + neutral; wflex = -(lift+elbow)).
            lift_math = cmd_lift / _sh.JOINT_SIGN_LIFT + _sh.NEUTRAL_LIFT_RAD
            elbow_math = cmd_elbow / _sh.JOINT_SIGN_ELBOW + _sh.NEUTRAL_ELBOW_RAD
            wrist_flex_math = -(lift_math + elbow_math)
            cmd_wrist_flex = (wrist_flex_math - _sh.NEUTRAL_WRIST_FLEX_RAD) * _sh.JOINT_SIGN_WRIST_FLEX
            if TEST_JOINT == "5dof":
                # Motor 7 (wrist_pitch field, physically a roll axis) from hand
                # roll. Direct lerp from the index_mcp -> pinky_mcp tilt angle.
                roll_t = (float(hand_roll) - HAND_ROLL_LEFT_NORM) / max(HAND_ROLL_RIGHT_NORM - HAND_ROLL_LEFT_NORM, 1e-6)
                roll_t = max(0.0, min(1.0, roll_t))
                cmd_wrist_pitch = M7_AT_HAND_ROLL_LEFT + (M7_AT_HAND_ROLL_RIGHT - M7_AT_HAND_ROLL_LEFT) * roll_t

        return JointCommand(
            shoulder_pan=cmd_pan,
            shoulder_lift=cmd_lift,
            elbow_flex=cmd_elbow,
            wrist_flex=cmd_wrist_flex,
            wrist_yaw=float(hold.get("wrist_yaw", 0.0)),
            wrist_roll=float(hold.get("wrist_roll", 0.0)),
            wrist_pitch=cmd_wrist_pitch,
            gripper_open01=float(hold.get("gripper_open01", 1.0)),
        )
    return JointCommand(
        shoulder_pan=float(joints["shoulder_pan"]),
        shoulder_lift=float(joints["shoulder_lift"]),
        elbow_flex=float(joints["elbow_flex"]),
        wrist_flex=float(joints["wrist_flex"]),
        wrist_yaw=float(joints.get("wrist_yaw", 0.0)),
        wrist_roll=float(joints.get("wrist_roll", 0.0)),
        wrist_pitch=float(joints.get("wrist_pitch", 0.0)),
        gripper_open01=float(gripper_open01),
    )


def _cleanup(robot: Optional[SOArmHardwareController], cap: Optional[cv2.VideoCapture], tracker: Optional[SimpleHandTracker]) -> None:
    if robot is not None:
        try:
            robot.release_torque()
        except Exception as exc:
            print(f"[simple_main] release_torque failed: {exc}")
        try:
            robot.disconnect()
        except Exception as exc:
            print(f"[simple_main] disconnect failed: {exc}")
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    if tracker is not None:
        try:
            tracker.close()
        except Exception:
            pass
    cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Don't connect to the robot")
    parser.add_argument("--camera", type=int, default=int(getattr(val, "HANDTRACKING_CAMERA_INDEX", 0)))
    parser.add_argument("--no-flip", action="store_true", help="Don't horizontally mirror the camera frame")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[simple_main] could not open camera index {args.camera}")
        return 1

    tracker = SimpleHandTracker()
    robot: Optional[SOArmHardwareController] = None

    if not args.dry_run:
        robot = SOArmHardwareController()
        try:
            robot.connect()
        except Exception as exc:
            print(f"[simple_main] robot connect failed: {exc}")
            _cleanup(None, cap, tracker)
            return 1
        # Capture present joint angles before starting the async sender so the
        # hold-still values reflect the live arm pose, not 0.
        _capture_hold_joints(robot)
        try:
            robot.start_async_sender()
        except Exception as exc:
            print(f"[simple_main] start_async_sender failed: {exc}")

    def sigint_handler(*_):
        print("\n[simple_main] SIGINT — releasing torque and exiting")
        _cleanup(robot, cap, tracker)
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    flip = not args.no_flip
    print(f"[simple_main] running. ESC to quit. dry_run={args.dry_run} flip={flip}")

    last_print = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if flip:
                frame = cv2.flip(frame, 1)

            data = tracker.process(frame)
            if data is not None:
                cmd = _build_joint_command(
                    data["joints"], data["gripper_open01"],
                    hand_cx=data["hand_cx"], hand_cy=data["hand_cy"],
                    hand_palm=data["hand_palm"],
                    hand_roll=data.get("hand_roll", 0.0),
                )
                if robot is not None:
                    robot.submit_latest_command(cmd)
                if (time.time() - last_print) > 0.3:
                    tag = "live" if robot is not None else "dry "
                    fb = None
                    if robot is not None:
                        try:
                            fb = robot.read_present_joints_rad()
                        except Exception:
                            pass
                    if TEST_JOINT in ("lift", "elbow"):
                        joint_name = "shoulder_lift" if TEST_JOINT == "lift" else "elbow_flex"
                        cmd_val = cmd.shoulder_lift if TEST_JOINT == "lift" else cmd.elbow_flex
                        present_v = fb.get(joint_name) if isinstance(fb, dict) else None
                        present_str = (
                            f"  present={present_v:+.3f} ({math.degrees(present_v):+.0f} deg)"
                            if present_v is not None else "  present=?"
                        )
                        err_str = (
                            f"  err={cmd_val - present_v:+.3f}"
                            if present_v is not None else ""
                        )
                        input_str = (
                            f"hand cy={data['hand_cy']:.2f}" if TEST_JOINT == "lift"
                            else f"hand palm={data['hand_palm']:.3f}"
                        )
                        print(
                            f"[{tag}] {input_str}  "
                            f"{TEST_JOINT}_cmd={cmd_val:+.3f} ({math.degrees(cmd_val):+.0f} deg)"
                            f"{present_str}{err_str}"
                        )
                    elif TEST_JOINT in ("3dof", "4dof", "5dof"):
                        def _pres(name):
                            return fb.get(name) if isinstance(fb, dict) else None
                        p_pan, p_lift, p_elbow, p_wflex, p_wpitch = (
                            _pres("shoulder_pan"), _pres("shoulder_lift"),
                            _pres("elbow_flex"), _pres("wrist_flex"),
                            _pres("wrist_pitch"),
                        )
                        def _fmt(v):
                            return f"{v:+.2f}" if v is not None else "  ?  "
                        xyz = data["target_xyz"]
                        roll_str = (
                            f"  roll={data.get('hand_roll', 0.0):+.2f}" if TEST_JOINT == "5dof" else ""
                        )
                        wpitch_str = (
                            f"  m7={cmd.wrist_pitch:+.2f}/{_fmt(p_wpitch)}" if TEST_JOINT == "5dof" else ""
                        )
                        print(
                            f"[{tag}] hand cx={data['hand_cx']:.2f} cy={data['hand_cy']:.2f} palm={data['hand_palm']:.3f}{roll_str}  "
                            f"xyz=({xyz[0]:+.3f},{xyz[1]:+.3f},{xyz[2]:+.3f})  "
                            f"pan={cmd.shoulder_pan:+.2f}/{_fmt(p_pan)}  "
                            f"lift={cmd.shoulder_lift:+.2f}/{_fmt(p_lift)}  "
                            f"elbow={cmd.elbow_flex:+.2f}/{_fmt(p_elbow)}  "
                            f"wflex={cmd.wrist_flex:+.2f}/{_fmt(p_wflex)}"
                            f"{wpitch_str}"
                        )
                    else:
                        xyz = data["target_xyz"]
                        j = data["joints"]
                        print(
                            f"[{tag}] hand cx={data['hand_cx']:.2f} cy={data['hand_cy']:.2f} palm={data['hand_palm']:.3f}  "
                            f"xyz=({xyz[0]:+.3f},{xyz[1]:+.3f},{xyz[2]:+.3f})  "
                            f"pan={j['shoulder_pan']:+.2f} lift={j['shoulder_lift']:+.2f} "
                            f"elbow={j['elbow_flex']:+.2f} wflex={j['wrist_flex']:+.2f} "
                            f"grip={data['gripper_open01']:.2f}"
                        )
                    last_print = time.time()

            cv2.imshow("Simple Hand Tracking", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
    finally:
        _cleanup(robot, cap, tracker)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
