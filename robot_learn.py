"""Pose-capture learning script for the modified 7-DOF SO-100 arm.

Walks the user through a sequence of named poses while torque is disabled,
recording raw motor positions at each pose. Output is written to
calibration_data/robot_learn_results.json with per-motor neutral / extreme
positions plus combined-pose validation captures.

This script does NOT modify robot_joint_calibration.json. Use the output to
manually update calibration values, derive INVERT_* flags, or sanity-check the
existing calibration.

Run:
    python3 robot_learn.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import robot_calibrate as rc


MOTORS = [
    {
        "id": 1,
        "config_name": "shoulder_pan",
        "human_name": "shoulder pan (M1)",
        "ext_a": "rotate the WHOLE arm fully to the LEFT (counterclockwise viewed from above)",
        "ext_b": "rotate the WHOLE arm fully to the RIGHT (clockwise viewed from above)",
    },
    {
        "id": 2,
        "config_name": "shoulder_lift",
        "human_name": "shoulder lift (M2)",
        "ext_a": "lower the upper arm fully (drop it DOWN toward the table)",
        "ext_b": "raise the upper arm fully (UP toward the ceiling)",
    },
    {
        "id": 3,
        "config_name": "elbow_flex",
        "human_name": "shoulder bend / M3 (the extra joint between shoulder lift and elbow)",
        "ext_a": "bend M3 fully in one direction (pick whichever is easier)",
        "ext_b": "bend M3 fully in the OPPOSITE direction",
    },
    {
        "id": 4,
        "config_name": "wrist_flex",
        "human_name": "elbow / M4",
        "ext_a": "FOLD the elbow fully (pull forearm toward upper arm)",
        "ext_b": "EXTEND the elbow fully (forearm in line with upper arm)",
    },
    {
        "id": 5,
        "config_name": "wrist_yaw",
        "human_name": "forearm twist / M5",
        "ext_a": "twist the forearm fully one way (around its long axis)",
        "ext_b": "twist the forearm fully the OPPOSITE way",
    },
    {
        "id": 6,
        "config_name": "wrist_roll",
        "human_name": "wrist bend / M6",
        "ext_a": "tilt the claw DOWN as far as it goes (gripper aimed at the table)",
        "ext_b": "tilt the claw UP as far as it goes (gripper aimed at the ceiling)",
    },
    {
        "id": 7,
        "config_name": "wrist_pitch",
        "human_name": "claw twist / M7",
        "ext_a": "rotate the claw fully one way (around its own long axis)",
        "ext_b": "rotate the claw fully the OPPOSITE way",
    },
]


OUTPUT_PATH = Path(__file__).resolve().parent / "calibration_data" / "robot_learn_results.json"


def _print_header(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}")


def _input_capture_or_retry(prompt: str) -> str:
    while True:
        s = input(prompt).strip().lower()
        if s == "":
            return "capture"
        if s == "r":
            return "retry"
        if s == "q":
            return "quit"
        print("Press Enter to capture, 'r' to redo this pose's instructions, or 'q' to quit.")


def _capture_pose(session, pose_id: str, title: str, instructions: str, captures: dict) -> bool:
    while True:
        _print_header(title)
        print(instructions)
        print()
        action = _input_capture_or_retry("[Enter] capture | [r] redo | [q] quit > ")
        if action == "quit":
            return False
        if action == "retry":
            continue
        try:
            positions = rc.read_positions(session)
        except RuntimeError as exc:
            print(f"\nRead failed: {exc}")
            print("Make sure the arm is powered and the bus is connected, then try again.")
            continue
        captures[pose_id] = positions
        print("\nCaptured:")
        for name in session.motor_names:
            print(f"  {name:16s} {positions[name]}")
        print()
        return True


def _build_pose_sequence() -> list[tuple[str, str, str]]:
    seq: list[tuple[str, str, str]] = []
    seq.append((
        "neutral",
        "POSE: Neutral (anchor pose)",
        (
            "Hold the arm by hand in this neutral pose:\n"
            "  - Shoulder pan (M1): facing straight forward (away from the base mount).\n"
            "  - Shoulder lift (M2): upper arm horizontal.\n"
            "  - Shoulder bend (M3): zero bend - upper arm continues straight.\n"
            "  - Elbow (M4): forearm horizontal, in line with upper arm.\n"
            "  - Forearm twist (M5): no rotation.\n"
            "  - Wrist bend (M6): wrist straight, claw level.\n"
            "  - Claw twist (M7): no rotation.\n"
            "  - Gripper: half open.\n"
            "Hold steady. This pose is the reference 'zero' for every joint."
        ),
    ))
    seq.append((
        "gravity_rest",
        "POSE: Gravity rest (let the arm droop)",
        (
            "Stop holding the arm. Let it droop naturally under gravity until it stops moving.\n"
            "Don't help it - just steady any wobbling joints once they settle.\n"
            "If a joint bottoms out hard, support it lightly so it isn't loaded.\n"
            "This anchors gravity-down for the arm."
        ),
    ))
    for m in MOTORS:
        seq.append((
            f"M{m['id']}_extA",
            f"POSE: {m['human_name']} -- extreme A",
            (
                f"Return all OTHER joints to the neutral pose. Then move ONLY this joint:\n"
                f"  Joint:     {m['human_name']}\n"
                f"  Direction: {m['ext_a']}\n"
                f"Move slowly until you reach the soft limit, then back off about 2-5 degrees.\n"
                f"Hold steady."
            ),
        ))
        seq.append((
            f"M{m['id']}_extB",
            f"POSE: {m['human_name']} -- extreme B",
            (
                f"Same joint, opposite direction. Other joints back at neutral.\n"
                f"  Joint:     {m['human_name']}\n"
                f"  Direction: {m['ext_b']}\n"
                f"Hold at the soft limit (don't force the hard stop)."
            ),
        ))
    seq.append((
        "gripper_closed",
        "POSE: Gripper fully CLOSED",
        (
            "Squeeze the gripper jaws together by hand. Other joints at neutral.\n"
            "Close gently - if the gripper is geared and stiff to back-drive, that's expected."
        ),
    ))
    seq.append((
        "gripper_open",
        "POSE: Gripper fully OPEN",
        (
            "Spread the gripper jaws to maximum opening. Other joints at neutral.\n"
            "Stop at the mechanical limit - don't bend the linkage."
        ),
    ))
    seq.append((
        "extended_horizontal_forward",
        "POSE: Arm fully extended HORIZONTALLY, pointing forward",
        (
            "Set the arm so the entire chain is horizontal, pointing forward:\n"
            "  - Shoulder pan: forward (zero rotation from neutral).\n"
            "  - Shoulder lift, shoulder bend, elbow, wrist all aligned in a straight line.\n"
            "  - Wrist bend (M6): claw level (parallel to the table).\n"
            "  - Forearm/claw twists: any (we're checking position, not orientation).\n"
            "Imagine the arm sticking straight out forward like a pointer. Hold steady."
        ),
    ))
    seq.append((
        "extended_vertical_up",
        "POSE: Arm fully extended VERTICALLY, EE pointing up",
        (
            "Same chain alignment as the previous pose, but rotated up 90 degrees at the\n"
            "shoulder so the arm points straight UP toward the ceiling.\n"
            "  - Whole arm vertical.\n"
            "  - Claw pointing up.\n"
            "Don't lean forward or backward - vertical only."
        ),
    ))
    seq.append((
        "claw_forward_arm_horizontal",
        "POSE: Claw pointing exactly FORWARD, arm horizontal",
        (
            "From the neutral pose, fine-tune so the claw is pointing forward (same\n"
            "direction as shoulder pan):\n"
            "  - Whole arm horizontal at shoulder height.\n"
            "  - Claw pointing forward (NOT up, NOT down, NOT left, NOT right).\n"
            "  - Forearm twist + claw twist set so the gripper jaws would close on an\n"
            "    object approaching from above (vertical pinch).\n"
            "This locks the wrist sub-chain offsets."
        ),
    ))
    return seq


def _build_output_payload(session, captures: dict) -> dict:
    payload: dict = {
        "created_at_unix": time.time(),
        "motor_names": list(session.motor_names),
        "motor_ids": list(map(int, session.motor_ids)),
        "captures": {
            pid: {k: int(v) for k, v in pos.items()}
            for pid, pos in captures.items()
        },
    }
    derived: dict[str, dict] = {}
    neutral = captures.get("neutral", {})
    for m in MOTORS:
        name = m["config_name"]
        entry: dict = {"motor_id": m["id"], "human_name": m["human_name"]}
        if name in neutral:
            entry["neutral"] = int(neutral[name])
        cap_a = captures.get(f"M{m['id']}_extA", {})
        cap_b = captures.get(f"M{m['id']}_extB", {})
        if name in cap_a:
            entry["pos_at_extreme_a"] = int(cap_a[name])
            entry["extreme_a_label"] = m["ext_a"]
        if name in cap_b:
            entry["pos_at_extreme_b"] = int(cap_b[name])
            entry["extreme_b_label"] = m["ext_b"]
        if "pos_at_extreme_a" in entry and "pos_at_extreme_b" in entry:
            a = entry["pos_at_extreme_a"]
            b = entry["pos_at_extreme_b"]
            entry["min_pos"] = min(a, b)
            entry["max_pos"] = max(a, b)
            entry["span"] = entry["max_pos"] - entry["min_pos"]
            entry["a_minus_b_sign"] = 1 if a > b else (-1 if a < b else 0)
        derived[name] = entry
    if "gripper_closed" in captures or "gripper_open" in captures:
        gentry: dict = {"motor_id": 8, "human_name": "gripper (M8)"}
        if "gripper" in neutral:
            gentry["neutral"] = int(neutral["gripper"])
        if "gripper_closed" in captures and "gripper" in captures["gripper_closed"]:
            gentry["pos_closed"] = int(captures["gripper_closed"]["gripper"])
        if "gripper_open" in captures and "gripper" in captures["gripper_open"]:
            gentry["pos_open"] = int(captures["gripper_open"]["gripper"])
        if "pos_closed" in gentry and "pos_open" in gentry:
            gentry["min_pos"] = min(gentry["pos_closed"], gentry["pos_open"])
            gentry["max_pos"] = max(gentry["pos_closed"], gentry["pos_open"])
            gentry["span"] = gentry["max_pos"] - gentry["min_pos"]
        derived["gripper"] = gentry
    payload["derived_per_motor"] = derived
    return payload


def _write_output(payload: dict) -> Path:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return OUTPUT_PATH


def _print_derived_summary(payload: dict) -> None:
    derived = payload.get("derived_per_motor", {})
    if not derived:
        print("  (no derived data - neutral pose was not captured)")
        return
    fmt = "  {n:18s}  neutral={ne:>6}  extA={a:>6}  extB={b:>6}  min={mn:>6}  max={mx:>6}  span={sp:>6}"
    for name, entry in derived.items():
        a_key = "pos_at_extreme_a" if "pos_at_extreme_a" in entry else "pos_closed"
        b_key = "pos_at_extreme_b" if "pos_at_extreme_b" in entry else "pos_open"
        print(fmt.format(
            n=name,
            ne=str(entry.get("neutral", "-")),
            a=str(entry.get(a_key, "-")),
            b=str(entry.get(b_key, "-")),
            mn=str(entry.get("min_pos", "-")),
            mx=str(entry.get("max_pos", "-")),
            sp=str(entry.get("span", "-")),
        ))


def main() -> int:
    print("==== Modified 7-DOF SO-100 arm: pose-learning script ====")
    print()
    print("This walks you through ~21 poses. At each pose:")
    print("  1. Read the on-screen instructions for how to position the arm by hand.")
    print("  2. Hold the arm steady at the requested position.")
    print("  3. Press Enter. The script captures all 8 motor positions.")
    print()
    print("Controls at each pose:")
    print("  Enter -- capture and continue")
    print("  r     -- redo this pose's instructions (don't capture, restart pose)")
    print("  q     -- quit (partial captures will still be saved)")
    print()
    print("Safety:")
    print("  - Torque is DISABLED at startup. The arm will not hold itself up.")
    print("  - Keep one hand under the arm at all times so it doesn't fall.")
    print("  - Never force a joint past its mechanical hard stop. Back off slightly.")
    print("  - Move slowly. Steady the arm before pressing Enter.")
    print()
    confirm = input("Type 'go' and press Enter to begin (anything else aborts): ").strip().lower()
    if confirm != "go":
        print("Aborted.")
        return 1

    print("\nConnecting to arm...")
    try:
        session = rc.connect_session()
    except Exception as exc:
        print(f"Failed to connect: {exc}")
        return 1

    captures: dict[str, dict[str, int]] = {}
    try:
        print("Disabling torque on all motors...")
        rc.set_torque(session, enabled=False)
        print()
        seq = _build_pose_sequence()
        total = len(seq)
        for idx, (pose_id, title, instructions) in enumerate(seq, start=1):
            full_title = f"[{idx}/{total}] {title}"
            ok = _capture_pose(session, pose_id, full_title, instructions, captures)
            if not ok:
                print("Quit requested. Saving partial captures...")
                break
        payload = _build_output_payload(session, captures)
        out_path = _write_output(payload)
        print(f"\nSaved {len(captures)} pose capture(s) to: {out_path}")
        print("\nDerived per-motor summary (extA/extB are the two extremes you swept):")
        _print_derived_summary(payload)
    finally:
        # Leave torque OFF after running so the arm can be safely parked by hand.
        try:
            session.disconnect()
        except Exception:
            pass

    print("\nNext steps:")
    print("  - Compare 'min_pos' / 'max_pos' to robot_joint_calibration.json values.")
    print("  - Compare 'neutral' to the calibrated neutral_pos for each motor.")
    print("  - For each motor, the sign of (extA - extB) plus the physical direction you")
    print("    pushed tells you whether the motor's positive direction matches the")
    print("    convention used in IK (informs INVERT_* flag decisions).")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
