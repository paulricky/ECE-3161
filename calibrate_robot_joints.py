from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

import values as val
from robot_controller import SOArmHardwareController, RealRobotUnavailableError


CALIB_DIR = Path(__file__).resolve().parent / "calibration_data"
JSON_PATH = CALIB_DIR / "robot_joint_calibration.json"
TXT_PATH = CALIB_DIR / "robot_joint_calibration_summary.txt"

@dataclass
class CalibrationSession:
    controller: SOArmHardwareController
    robot: Any
    bus: Any
    motor_names: list[str]


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _is_gripper(name: str) -> bool:
    return name.lower() == "gripper"


def _get_configured_motor_names() -> list[str]:
    names = list(getattr(val, "REAL_ROBOT_MOTOR_NAMES", []))
    if names:
        return names
    return [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]


def _to_list_of_numbers(value: Any) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        flat = value.reshape(-1).tolist()
        return [int(round(float(v))) for v in flat]
    if isinstance(value, (list, tuple)):
        out: list[int] = []
        for item in value:
            if isinstance(item, (list, tuple, np.ndarray)):
                out.extend(_to_list_of_numbers(item))
            else:
                out.append(int(round(float(item))))
        return out
    if hasattr(value, "item"):
        return [int(round(float(value.item())))]
    return [int(round(float(value)))]


def _get_bus(robot: Any) -> Any:
    candidates = [
        getattr(robot, "bus", None),
        getattr(robot, "motors_bus", None),
        getattr(robot, "arm", None),
    ]
    follower_arms = getattr(robot, "follower_arms", None)
    if isinstance(follower_arms, dict) and follower_arms:
        candidates.extend(follower_arms.values())
    leader_arms = getattr(robot, "leader_arms", None)
    if isinstance(leader_arms, dict) and leader_arms:
        candidates.extend(leader_arms.values())

    for candidate in candidates:
        if candidate is not None:
            return candidate
    raise RuntimeError(
        "Could not find a readable motor bus on the connected robot object. "
        "The calibration script needs either robot.bus or a follower arm bus."
    )


def _read_positions_from_bus(bus: Any, motor_names: Sequence[str]) -> list[int] | None:
    read = getattr(bus, "read", None)
    if read is None:
        return None

    call_patterns = [
        ("Present_Position", list(motor_names)),
        ("Present_Position", tuple(motor_names)),
        ("Present_Position",),
    ]

    for args in call_patterns:
        try:
            value = read(*args)
            values = _to_list_of_numbers(value)
            if len(values) == len(motor_names):
                return values
        except Exception:
            pass

    values: list[int] = []
    for name in motor_names:
        try:
            value = read("Present_Position", name)
            one = _to_list_of_numbers(value)
            if len(one) != 1:
                return None
            values.append(one[0])
        except Exception:
            return None
    return values


def _read_positions_from_robot(robot: Any, motor_names: Sequence[str]) -> list[int] | None:
    capture_names = ["capture_observation", "get_observation", "read_observation"]
    for fn_name in capture_names:
        fn = getattr(robot, fn_name, None)
        if fn is None:
            continue
        try:
            obs = fn()
        except Exception:
            continue
        if not isinstance(obs, dict):
            continue
        positions: list[int] = []
        ok = True
        for name in motor_names:
            found = None
            for key in (f"{name}.pos", name, f"observation.{name}.pos"):
                if key in obs:
                    found = obs[key]
                    break
            if found is None:
                ok = False
                break
            positions.extend(_to_list_of_numbers(found))
        if ok and len(positions) == len(motor_names):
            return positions
    return None


def read_positions(session: CalibrationSession) -> dict[str, int]:
    values = _read_positions_from_bus(session.bus, session.motor_names)
    if values is None:
        values = _read_positions_from_robot(session.robot, session.motor_names)
    if values is None:
        raise RuntimeError(
            "Could not read present positions from the robot. "
            "The connected driver did not expose a supported read interface."
        )
    return {name: int(v) for name, v in zip(session.motor_names, values, strict=True)}


def _write_torque_with_bus(bus: Any, motor_names: Sequence[str], enabled: bool) -> bool:
    write = getattr(bus, "write", None)
    sync_write = getattr(bus, "sync_write", None)
    methods = [m for m in (write, sync_write) if m is not None]
    if not methods:
        return False

    register = "Torque_Enable"
    value = 1 if enabled else 0

    payloads = [
        (register, value, list(motor_names)),
        (register, value, tuple(motor_names)),
        (register, {name: value for name in motor_names}),
        (register, np.full(len(motor_names), value, dtype=np.int32), list(motor_names)),
        (register, np.full(len(motor_names), value, dtype=np.int32)),
        (register, [value] * len(motor_names), list(motor_names)),
    ]

    for method in methods:
        for args in payloads:
            try:
                method(*args)
                return True
            except Exception:
                pass

        ok = True
        for name in motor_names:
            try:
                method(register, value, name)
            except Exception:
                ok = False
                break
        if ok:
            return True

    return False


def set_torque(session: CalibrationSession, enabled: bool) -> None:
    ok = _write_torque_with_bus(session.bus, session.motor_names, enabled)
    if not ok:
        print(
            f"[calibrate] Warning: could not {'enable' if enabled else 'disable'} torque through the detected bus. "
            "If the arm feels stiff, manually power-cycle or release torque before moving it by hand."
        )
    else:
        print(f"[calibrate] Torque {'ENABLED' if enabled else 'DISABLED'} for {len(session.motor_names)} motors.")


def connect_session() -> CalibrationSession:
    original_auto_calibrate = getattr(val, "REAL_ROBOT_AUTO_CALIBRATE", False)
    val.REAL_ROBOT_AUTO_CALIBRATE = False

    controller = SOArmHardwareController()
    try:
        controller.connect()
    except RealRobotUnavailableError:
        raise
    except Exception:
        raise
    finally:
        val.REAL_ROBOT_AUTO_CALIBRATE = original_auto_calibrate

    if controller.robot is None:
        raise RuntimeError("Robot connected path returned no robot object.")

    bus = _get_bus(controller.robot)

    configured_names = _get_configured_motor_names()
    bus_names = list(getattr(bus, "motor_names", []))
    motor_names = bus_names if bus_names else configured_names

    if not motor_names:
        raise RuntimeError("Could not determine motor names for calibration.")

    return CalibrationSession(
        controller=controller,
        robot=controller.robot,
        bus=bus,
        motor_names=motor_names,
    )


def prompt_capture(session: CalibrationSession, title: str, instructions: str) -> dict[str, int]:
    _print_header(title)
    print(instructions)
    input("Press Enter to capture the current motor positions... ")
    positions = read_positions(session)
    print("Captured positions:")
    for name in session.motor_names:
        print(f"  {name:16s} {positions[name]:6d}")
    return positions


def infer_drive_mode(neutral: dict[str, int], max_pos: dict[str, int]) -> dict[str, int]:
    drive_mode: dict[str, int] = {}
    for name in neutral:
        if _is_gripper(name):
            drive_mode[name] = 0
        else:
            drive_mode[name] = int(max_pos[name] < neutral[name])
    return drive_mode


def infer_homing_offset(neutral: dict[str, int], drive_mode: dict[str, int]) -> dict[str, int]:
    offsets: dict[str, int] = {}
    for name, neutral_pos in neutral.items():
        offsets[name] = int(neutral_pos if drive_mode[name] else -neutral_pos)
    return offsets


def capture_joint_limits(session: CalibrationSession) -> tuple[dict[str, int], dict[str, int]]:
    min_pos: dict[str, int] = {}
    max_pos: dict[str, int] = {}

    for name in session.motor_names:
        if _is_gripper(name):
            closed = prompt_capture(
                session,
                f"Capture gripper CLOSED position ({name})",
                "Move the gripper to its fully closed position. Keep the rest of the arm stable.",
            )
            opened = prompt_capture(
                session,
                f"Capture gripper OPEN position ({name})",
                "Move the gripper to its fully open position. Keep the rest of the arm stable.",
            )
            min_pos[name] = int(closed[name])
            max_pos[name] = int(opened[name])
            continue

        minimum = prompt_capture(
            session,
            f"Capture MIN position for {name}",
            f"Move only '{name}' to its minimum safe mechanical position.\n"
            "Keep the other joints as close to the neutral pose as practical.\n"
            "Do not force the joint hard into a stop.",
        )
        maximum = prompt_capture(
            session,
            f"Capture MAX position for {name}",
            f"Move only '{name}' to its maximum safe mechanical position.\n"
            "Keep the other joints as close to the neutral pose as practical.\n"
            "Do not force the joint hard into a stop.",
        )
        min_pos[name] = int(minimum[name])
        max_pos[name] = int(maximum[name])

    return min_pos, max_pos


def build_calibration_payload(session: CalibrationSession, neutral: dict[str, int], min_pos: dict[str, int], max_pos: dict[str, int]) -> dict[str, Any]:
    drive_mode = infer_drive_mode(neutral, max_pos)
    homing_offset = infer_homing_offset(neutral, drive_mode)
    calib_mode = ["LINEAR" if _is_gripper(name) else "DEGREE" for name in session.motor_names]

    payload = {
        "created_at_unix": time.time(),
        "motor_names": list(session.motor_names),
        "neutral_pos": [neutral[name] for name in session.motor_names],
        "min_pos": [min_pos[name] for name in session.motor_names],
        "max_pos": [max_pos[name] for name in session.motor_names],
        "homing_offset": [homing_offset[name] for name in session.motor_names],
        "drive_mode": [drive_mode[name] for name in session.motor_names],
        "start_pos": [min_pos[name] for name in session.motor_names],
        "end_pos": [max_pos[name] for name in session.motor_names],
        "calib_mode": calib_mode,
        "notes": {
            "neutral_pose": "User-defined zero/neutral pose captured interactively.",
            "min_max_pose": "Per-joint minimum/maximum safe positions captured interactively.",
            "format": "Project-local calibration file with LeRobot-style fields plus explicit neutral/min/max arrays.",
        },
    }
    return payload


def write_outputs(payload: dict[str, Any]) -> None:
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n")

    lines: list[str] = []
    lines.append("Robot joint calibration summary")
    lines.append("=" * 40)
    lines.append(f"JSON file: {JSON_PATH}")
    lines.append("")

    motor_names: list[str] = payload["motor_names"]
    neutral_pos: list[int] = payload["neutral_pos"]
    min_pos: list[int] = payload["min_pos"]
    max_pos: list[int] = payload["max_pos"]
    homing_offset: list[int] = payload["homing_offset"]
    drive_mode: list[int] = payload["drive_mode"]

    for i, name in enumerate(motor_names):
        lines.append(f"{name}:")
        lines.append(f"  neutral_pos   = {neutral_pos[i]}")
        lines.append(f"  min_pos       = {min_pos[i]}")
        lines.append(f"  max_pos       = {max_pos[i]}")
        lines.append(f"  homing_offset = {homing_offset[i]}")
        lines.append(f"  drive_mode    = {drive_mode[i]}")
        lines.append("")

    lines.append("JSON one-line paths for easy copy/paste:")
    lines.append(f"robot_joint_calibration = {JSON_PATH.as_posix()}")
    TXT_PATH.write_text("\n".join(lines) + "\n")


def main() -> int:
    _print_header("Interactive robot calibration")
    print(
        "This script mimics the interactive style of the LeRobot manual calibration flow,\n"
        "but records a neutral pose first and then captures per-joint minimum and maximum\n"
        "positions for your current hardware setup.\n"
    )

    try:
        session = connect_session()
    except Exception as exc:
        print(f"[calibrate] Failed to connect to robot: {exc}")
        return 1

    print(f"[calibrate] Connected. Motor names: {session.motor_names}")

    try:
        set_torque(session, enabled=False)

        neutral = prompt_capture(
            session,
            "Capture NEUTRAL pose",
            "Move the arm into your desired neutral/zero pose.\n"
            "Suggested pose: shoulder_pan centered, shoulder_lift neutral, elbow neutral,\n"
            "wrist joints centered, and gripper in the pose you want to treat as its neutral reference.",
        )

        min_pos, max_pos = capture_joint_limits(session)
        payload = build_calibration_payload(session, neutral, min_pos, max_pos)
        write_outputs(payload)

        _print_header("Calibration complete")
        print(f"Saved JSON calibration to: {JSON_PATH}")
        print(f"Saved text summary to:     {TXT_PATH}")
        print("\nSummary:")
        for name, neutral_pos, minp, maxp in zip(
            payload["motor_names"],
            payload["neutral_pos"],
            payload["min_pos"],
            payload["max_pos"],
            strict=True,
        ):
            print(f"  {name:16s} neutral={neutral_pos:6d}  min={minp:6d}  max={maxp:6d}")

    finally:
        try:
            set_torque(session, enabled=True)
        except Exception:
            pass
        try:
            session.controller.disconnect()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
