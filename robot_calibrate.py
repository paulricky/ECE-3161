from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import values as val
from robot_controller import SOArmHardwareController, RealRobotUnavailableError


CALIB_DIR = Path(__file__).resolve().parent / "calibration_data"
PROJECT_JSON_PATH = CALIB_DIR / "robot_joint_calibration.json"
SETUP_JSON_PATH = CALIB_DIR / "robot_motor_setup.json"
TXT_PATH = CALIB_DIR / "robot_joint_calibration_summary.txt"


def _resolve_optional_path(raw: str) -> Path | None:
    raw = str(raw).strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path


def _default_driver_calibration_path() -> Path:
    robot_id = getattr(val, "REAL_ROBOT_ID", "my_awesome_follower_arm")
    return (
        Path.home()
        / ".cache"
        / "huggingface"
        / "lerobot"
        / "calibration"
        / "robots"
        / "so101_follower"
        / f"{robot_id}.json"
    )


def _get_driver_calibration_path() -> Path:
    driver_path = _resolve_optional_path(getattr(val, "LEROBOT_DRIVER_CALIBRATION_FILE", ""))
    if driver_path is None:
        driver_path = _default_driver_calibration_path()
    return driver_path


def _get_output_json_paths() -> list[Path]:
    paths: list[Path] = [PROJECT_JSON_PATH]
    driver_path = _get_driver_calibration_path()
    if driver_path not in paths:
        paths.append(driver_path)
    return paths


@dataclass
class CalibrationSession:
    controller: SOArmHardwareController
    robot: Any
    bus: Any
    motor_names: list[str]


@dataclass
class SetupStatus:
    configured: bool
    source: str | None
    motor_names: list[str]
    motor_ids: list[int] | None


@dataclass
class CalibrationStatus:
    configured: bool
    source: str | None
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


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text())
    except Exception:
        return None


def _extract_motor_ids(payload: dict[str, Any], motor_names: Sequence[str]) -> list[int] | None:
    if not isinstance(payload, dict):
        return None

    ids = payload.get("motor_ids")
    if isinstance(ids, list) and len(ids) == len(motor_names):
        try:
            return [int(x) for x in ids]
        except Exception:
            pass

    out: list[int] = []
    for name in motor_names:
        entry = payload.get(name)
        if not isinstance(entry, dict) or entry.get("id") is None:
            return None
        try:
            out.append(int(entry["id"]))
        except Exception:
            return None
    return out if len(out) == len(motor_names) else None


def get_motor_setup_status() -> SetupStatus:
    configured_names = _get_configured_motor_names()

    for path in (SETUP_JSON_PATH, _get_driver_calibration_path()):
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        motor_names = list(payload.get("motor_names", [])) or configured_names
        motor_ids = _extract_motor_ids(payload, motor_names)
        if motor_ids is not None:
            return SetupStatus(True, str(path), motor_names, motor_ids)

    return SetupStatus(False, None, configured_names, None)


def get_joint_calibration_status() -> CalibrationStatus:
    payload = _load_json(PROJECT_JSON_PATH)
    configured_names = _get_configured_motor_names()
    if not isinstance(payload, dict):
        return CalibrationStatus(False, None, configured_names)

    motor_names = list(payload.get("motor_names", [])) or configured_names
    neutral = payload.get("neutral_pos")
    min_pos = payload.get("min_pos")
    max_pos = payload.get("max_pos")
    if isinstance(neutral, list) and isinstance(min_pos, list) and isinstance(max_pos, list):
        if len(neutral) == len(min_pos) == len(max_pos) == len(motor_names):
            return CalibrationStatus(True, str(PROJECT_JSON_PATH), motor_names)

    return CalibrationStatus(False, None, motor_names)


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


def _read_motor_ids_from_bus(bus: Any, motor_names: Sequence[str]) -> list[int] | None:
    read = getattr(bus, "read", None)
    if read is None:
        return None

    call_patterns = [
        ("ID", list(motor_names)),
        ("ID", tuple(motor_names)),
        ("Id", list(motor_names)),
        ("Id", tuple(motor_names)),
    ]
    for args in call_patterns:
        try:
            values = _to_list_of_numbers(read(*args))
            if len(values) == len(motor_names):
                return values
        except Exception:
            pass

    out: list[int] = []
    for name in motor_names:
        value = None
        for reg in ("ID", "Id"):
            try:
                one = _to_list_of_numbers(read(reg, name))
                if len(one) == 1:
                    value = one[0]
                    break
            except Exception:
                pass
        if value is None:
            return None
        out.append(value)
    return out


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


def build_calibration_payload(
    session: CalibrationSession,
    neutral: dict[str, int],
    min_pos: dict[str, int],
    max_pos: dict[str, int],
    motor_ids: list[int] | None,
) -> dict[str, Any]:
    drive_mode = infer_drive_mode(neutral, max_pos)
    homing_offset = infer_homing_offset(neutral, drive_mode)
    calib_mode = ["LINEAR" if _is_gripper(name) else "DEGREE" for name in session.motor_names]

    payload: dict[str, Any] = {
        "created_at_unix": time.time(),
        "motor_names": list(session.motor_names),
        "motor_ids": list(motor_ids) if motor_ids is not None else None,
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
            "format": "Project-local calibration file with array fields and per-joint LeRobot-style entries.",
        },
    }

    for idx, name in enumerate(session.motor_names):
        raw_min = int(min_pos[name])
        raw_max = int(max_pos[name])
        range_min = min(raw_min, raw_max)
        range_max = max(raw_min, raw_max)
        payload[name] = {
            "name": name,
            "id": None if motor_ids is None else int(motor_ids[idx]),
            "neutral": int(neutral[name]),
            "recorded_min": raw_min,
            "recorded_max": raw_max,
            "range_min": range_min,
            "range_max": range_max,
            "range_center": 0.5 * (range_min + range_max),
            "range_span": range_max - range_min,
            "homing_offset": int(homing_offset[name]),
            "drive_mode": int(drive_mode[name]),
            "start_pos": raw_min,
            "end_pos": raw_max,
            "calib_mode": "LINEAR" if _is_gripper(name) else "DEGREE",
        }

    return payload


def _build_setup_payload(session: CalibrationSession, motor_ids: list[int] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "created_at_unix": time.time(),
        "motor_names": list(session.motor_names),
        "motor_ids": list(motor_ids) if motor_ids is not None else None,
        "setup_only": True,
        "notes": {
            "purpose": "Records that motor IDs were configured and verified separately from joint calibration.",
        },
    }
    for idx, name in enumerate(session.motor_names):
        payload[name] = {
            "name": name,
            "id": None if motor_ids is None else int(motor_ids[idx]),
        }
    return payload


def write_setup_output(payload: dict[str, Any]) -> Path:
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    SETUP_JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    return SETUP_JSON_PATH


def write_outputs(payload: dict[str, Any]) -> list[Path]:
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    output_paths = _get_output_json_paths()
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")

    lines: list[str] = []
    lines.append("Robot joint calibration summary")
    lines.append("=" * 40)
    lines.append("JSON files:")
    for path in output_paths:
        lines.append(f"  {path}")
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
    for path in output_paths:
        lines.append(f"robot_joint_calibration = {path.as_posix()}")
    TXT_PATH.write_text("\n".join(lines) + "\n")
    return output_paths


def _best_effort_identify(session: CalibrationSession) -> None:
    _print_header("Identify motors")
    print("Motor order currently visible to the bus:")
    ids = _read_motor_ids_from_bus(session.bus, session.motor_names)
    for idx, name in enumerate(session.motor_names):
        suffix = ""
        if ids is not None and idx < len(ids):
            suffix = f"  id={ids[idx]}"
        print(f"  {idx + 1}. {name}{suffix}")
    print(
        "\nIf your driver exposes a native LED/blink identification method, you can add it here later.\n"
        "For now this step verifies the discovered motor order and IDs after setup."
    )


def run_motor_setup_only() -> int:
    _print_header("Robot motor setup")
    print(
        "This stage is separate from joint calibration. It is for assigning/verifying the servo IDs\n"
        "and any driver-side configuration needed before min/max calibration."
    )

    try:
        session = connect_session()
    except Exception as exc:
        print(f"[calibrate] Failed to connect to robot: {exc}")
        return 1

    try:
        print(f"[calibrate] Connected. Motor names: {session.motor_names}")
        setup_motors = getattr(session.robot, "setup_motors", None)
        if callable(setup_motors):
            print("[calibrate] Running native LeRobot setup_motors() workflow...")
            setup_motors()
            time.sleep(0.5)
        else:
            print(
                "[calibrate] The connected driver does not expose setup_motors().\n"
                "Skipping native ID assignment and only recording the currently visible IDs."
            )

        try:
            session.motor_names = list(getattr(session.bus, "motor_names", [])) or session.motor_names
        except Exception:
            pass
        motor_ids = _read_motor_ids_from_bus(session.bus, session.motor_names)
        _best_effort_identify(session)

        payload = _build_setup_payload(session, motor_ids)
        path = write_setup_output(payload)
        print(f"[calibrate] Saved motor-setup metadata to: {path}")

        if motor_ids is None:
            print(
                "[calibrate] Warning: the bus did not expose readable motor IDs after setup.\n"
                "Calibration can still proceed, but automatic detection of 'setup already done' may not be reliable."
            )
        return 0
    finally:
        try:
            session.controller.disconnect()
        except Exception:
            pass


def run_calibration_only() -> int:
    setup_status = get_motor_setup_status()
    if not setup_status.configured:
        print(
            "[calibrate] Motor-ID setup was not detected. Run setup first so calibration behaves like LeRobot:\n"
            "first setup motor IDs, then perform neutral/min/max calibration."
        )
        return 1

    _print_header("Interactive robot calibration")
    print(
        "This stage assumes motor IDs are already configured. It records a neutral pose and then\n"
        "captures per-joint minimum and maximum positions for the current hardware setup."
    )

    try:
        session = connect_session()
    except Exception as exc:
        print(f"[calibrate] Failed to connect to robot: {exc}")
        return 1

    print(f"[calibrate] Connected. Motor names: {session.motor_names}")
    if setup_status.motor_ids is not None:
        print(f"[calibrate] Using previously detected motor IDs: {setup_status.motor_ids}")

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
        motor_ids = _read_motor_ids_from_bus(session.bus, session.motor_names)
        if motor_ids is None:
            motor_ids = setup_status.motor_ids
        payload = build_calibration_payload(session, neutral, min_pos, max_pos, motor_ids)
        output_paths = write_outputs(payload)

        _print_header("Calibration complete")
        print("Saved JSON calibration to:")
        for path in output_paths:
            print(f"  {path}")
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
        return 0
    finally:
        try:
            set_torque(session, enabled=True)
        except Exception:
            pass
        try:
            session.controller.disconnect()
        except Exception:
            pass


def run_setup_and_calibration() -> int:
    rc = run_motor_setup_only()
    if rc != 0:
        return rc
    return run_calibration_only()


def run_workflow(mode: str) -> int:
    mode = str(mode).strip().lower()
    if mode in {"full", "setup+calibration", "setup_and_calibration"}:
        return run_setup_and_calibration()
    if mode in {"setup", "setup_only", "motor_setup"}:
        return run_motor_setup_only()
    if mode in {"calibration", "calibrate", "calibration_only"}:
        return run_calibration_only()
    if mode in {"identify", "identify_only"}:
        try:
            session = connect_session()
        except Exception as exc:
            print(f"[calibrate] Failed to connect to robot: {exc}")
            return 1
        try:
            _best_effort_identify(session)
            return 0
        finally:
            try:
                session.controller.disconnect()
            except Exception:
                pass
    print(f"Unknown workflow mode: {mode}")
    return 1


def _interactive_menu_choice() -> str:
    setup_status = get_motor_setup_status()
    calib_status = get_joint_calibration_status()

    _print_header("Robot setup / calibration")
    print(f"Motor-ID setup detected: {'yes' if setup_status.configured else 'no'}")
    if setup_status.source:
        print(f"  source: {setup_status.source}")
    print(f"Joint calibration detected: {'yes' if calib_status.configured else 'no'}")
    if calib_status.source:
        print(f"  source: {calib_status.source}")
    print("\nChoose an action:")
    print("  1) Full workflow (setup motors, then calibrate joints)")
    print("  2) Setup motors only")
    print("  3) Calibrate joints only")
    print("  4) Identify/verify motors only")
    reply = input("Selection [1/2/3/4]: ").strip()
    return {"1": "full", "2": "setup", "3": "calibration", "4": "identify"}.get(reply, "full")


def main(mode: str | None = None) -> int:
    if mode is None and len(sys.argv) > 1:
        mode = sys.argv[1]
    if mode is None:
        mode = _interactive_menu_choice()
    return run_workflow(mode)


if __name__ == "__main__":
    raise SystemExit(main())