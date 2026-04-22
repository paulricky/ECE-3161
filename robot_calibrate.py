from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

import values as val
from robot_controller import SOArmHardwareController, RealRobotUnavailableError


CALIB_DIR = Path(__file__).resolve().parent / "calibration_data"
PROJECT_JSON_PATH = CALIB_DIR / "robot_joint_calibration.json"
TXT_PATH = CALIB_DIR / "robot_joint_calibration_summary.txt"


LEROBOT_FIELDS = (
    "homing_offset",
    "drive_mode",
    "start_pos",
    "end_pos",
    "calib_mode",
    "motor_names",
)


@dataclass
class CalibrationSession:
    controller: SOArmHardwareController
    robot: Any
    bus: Any
    motor_names: list[str]


@dataclass
class DriverHandle:
    device: Any
    config: Any
    follower_cls: Any
    config_cls: Any


class CalibrationError(RuntimeError):
    pass


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _resolve_optional_path(raw: str) -> Path | None:
    raw = str(raw).strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
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


def _project_calibration_path() -> Path:
    configured = _resolve_optional_path(getattr(val, "ROBOT_JOINT_CALIBRATION_FILE", ""))
    return configured or PROJECT_JSON_PATH


def _driver_calibration_path() -> Path:
    configured = _resolve_optional_path(getattr(val, "LEROBOT_DRIVER_CALIBRATION_FILE", ""))
    return configured or _default_driver_calibration_path()


def _get_output_json_paths() -> tuple[Path, Path]:
    return _project_calibration_path(), _driver_calibration_path()


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


def _expected_motor_ids(motor_names: Sequence[str]) -> dict[str, int]:
    return {name: i + 1 for i, name in enumerate(motor_names)}


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


def _import_symbol(module_path: str, symbol_name: str):
    module = __import__(module_path, fromlist=[symbol_name])
    return getattr(module, symbol_name)


def _import_lerobot_so101() -> tuple[Any, Any]:
    attempts = [
        (
            "lerobot.robots.so101_follower.so101_follower",
            "SO101Follower",
            "lerobot.robots.so101_follower.config_so101_follower",
            "SO101FollowerConfig",
        ),
    ]

    import_errors: list[str] = []
    for follower_module, follower_symbol, config_module, config_symbol in attempts:
        try:
            follower_cls = _import_symbol(follower_module, follower_symbol)
            config_cls = _import_symbol(config_module, config_symbol)
            return follower_cls, config_cls
        except Exception as exc:
            import_errors.append(f"{follower_module} failed: {exc}")

    raise RealRobotUnavailableError(
        "Could not import the SO101 follower driver from LeRobot. " + " | ".join(import_errors)
    )


def _find_candidate_ports(controller: SOArmHardwareController) -> list[str]:
    try:
        ports = controller._find_candidate_ports()
    except Exception:
        ports = []
    return [p for p in ports if p]


def _select_port() -> str:
    configured = str(getattr(val, "REAL_ROBOT_PORT", "")).strip()
    if configured:
        return configured

    controller = SOArmHardwareController()
    ports = _find_candidate_ports(controller)
    if ports:
        print("[calibrate] Auto-detected candidate serial ports:")
        for i, port in enumerate(ports, start=1):
            print(f"  {i}. {port}")
        print(f"[calibrate] Using {ports[0]}")
        return ports[0]

    raise CalibrationError(
        "Could not determine the serial port automatically. Set values.REAL_ROBOT_PORT first."
    )


def _make_driver_handle() -> DriverHandle:
    follower_cls, config_cls = _import_lerobot_so101()
    port = _select_port()
    robot_id = getattr(val, "REAL_ROBOT_ID", "my_awesome_follower_arm")
    cfg = config_cls(
        port=port,
        id=robot_id,
        use_degrees=True,
        max_relative_target=float(getattr(val, "REAL_ROBOT_MAX_RELATIVE_TARGET_DEG", 2.0)),
    )
    device = follower_cls(cfg)
    return DriverHandle(device=device, config=cfg, follower_cls=follower_cls, config_cls=config_cls)


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
    raise CalibrationError(
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
        raise CalibrationError(
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
        raise CalibrationError("Robot connected path returned no robot object.")

    bus = _get_bus(controller.robot)

    configured_names = _get_configured_motor_names()
    bus_names = list(getattr(bus, "motor_names", []))
    motor_names = bus_names if bus_names else configured_names

    if not motor_names:
        raise CalibrationError("Could not determine motor names for calibration.")

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


def _blink_led(bus: Any, target: Any, cycles: int = 2, on_s: float = 0.20, off_s: float = 0.15) -> bool:
    write = getattr(bus, "write", None)
    sync_write = getattr(bus, "sync_write", None)
    methods = [m for m in (write, sync_write) if m is not None]
    if not methods:
        return False

    register_names = ["LED", "LED_Status"]
    for register in register_names:
        for method in methods:
            try:
                for _ in range(cycles):
                    try:
                        method(register, 1, target)
                    except Exception:
                        method(register, {target: 1})
                    time.sleep(on_s)
                    try:
                        method(register, 0, target)
                    except Exception:
                        method(register, {target: 0})
                    time.sleep(off_s)
                return True
            except Exception:
                continue
    return False


def identify_motors(session: CalibrationSession) -> None:
    _print_header("Motor identification / LED blink test")
    expected_ids = _expected_motor_ids(session.motor_names)
    print("The script will try to blink each motor one at a time in LeRobot order.")
    print("Expected ID order:")
    for name in session.motor_names:
        print(f"  id={expected_ids[name]:2d} -> {name}")

    for name in session.motor_names:
        target_candidates: list[Any] = []
        target_candidates.append(name)
        target_candidates.append(expected_ids[name])
        success = False
        for target in target_candidates:
            if _blink_led(session.bus, target):
                success = True
                break
        print(f"[calibrate] {'Blink OK' if success else 'Blink unavailable'} for {name} (expected id={expected_ids[name]})")
        time.sleep(0.10)


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


def _validate_captured_ranges(motor_names: Sequence[str], min_pos: dict[str, int], max_pos: dict[str, int]) -> None:
    errors: list[str] = []
    for name in motor_names:
        span = abs(int(max_pos[name]) - int(min_pos[name]))
        if span <= 0:
            errors.append(f"{name}: min and max are identical ({min_pos[name]}).")
        elif _is_gripper(name) and span < 5:
            errors.append(f"{name}: gripper span is too small ({span}); reopen and reclose more fully.")
        elif (not _is_gripper(name)) and span < 20:
            errors.append(f"{name}: joint span is suspiciously small ({span}); re-capture min/max.")
    if errors:
        msg = "\n".join(errors)
        raise CalibrationError("Calibration capture failed validation:\n" + msg)


def build_calibration_payload(
    session: CalibrationSession,
    neutral: dict[str, int],
    min_pos: dict[str, int],
    max_pos: dict[str, int],
) -> dict[str, Any]:
    drive_mode = infer_drive_mode(neutral, max_pos)
    homing_offset = infer_homing_offset(neutral, drive_mode)
    calib_mode = ["LINEAR" if _is_gripper(name) else "DEGREE" for name in session.motor_names]
    motor_ids = _expected_motor_ids(session.motor_names)

    payload: dict[str, Any] = {
        "created_at_unix": time.time(),
        "format_version": 2,
        "format_kind": "lerobot-compatible-manual-calibration",
        "motor_names": list(session.motor_names),
        "motor_ids": [motor_ids[name] for name in session.motor_names],
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
            "setup_motors": "Use --setup-motors or Full setup+calibrate mode to run LeRobot-style motor ID setup first.",
        },
    }

    for name in session.motor_names:
        raw_min = int(min_pos[name])
        raw_max = int(max_pos[name])
        range_min = min(raw_min, raw_max)
        range_max = max(raw_min, raw_max)
        payload[name] = {
            "name": name,
            "id": motor_ids[name],
            "neutral": int(neutral[name]),
            "recorded_min": raw_min,
            "recorded_max": raw_max,
            "range_min": range_min,
            "range_max": range_max,
            "range_center": 0.5 * (range_min + range_max),
            "range_span": range_max - range_min,
            "homing_offset": int(homing_offset[name]),
            "drive_mode": int(drive_mode[name]),
            "calib_mode": "LINEAR" if _is_gripper(name) else "DEGREE",
        }

    return payload


def _lerobot_driver_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: payload[key] for key in LEROBOT_FIELDS}


def write_outputs(payload: dict[str, Any]) -> tuple[Path, Path]:
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    project_path, driver_path = _get_output_json_paths()

    project_path.parent.mkdir(parents=True, exist_ok=True)
    project_path.write_text(json.dumps(payload, indent=2) + "\n")

    driver_path.parent.mkdir(parents=True, exist_ok=True)
    driver_path.write_text(json.dumps(_lerobot_driver_payload(payload), indent=2) + "\n")

    lines: list[str] = []
    lines.append("Robot joint calibration summary")
    lines.append("=" * 40)
    lines.append("Project JSON:")
    lines.append(f"  {project_path}")
    lines.append("LeRobot driver JSON:")
    lines.append(f"  {driver_path}")
    lines.append("")

    motor_names: list[str] = payload["motor_names"]
    motor_ids: list[int] = payload["motor_ids"]
    neutral_pos: list[int] = payload["neutral_pos"]
    min_pos: list[int] = payload["min_pos"]
    max_pos: list[int] = payload["max_pos"]
    homing_offset: list[int] = payload["homing_offset"]
    drive_mode: list[int] = payload["drive_mode"]

    for i, name in enumerate(motor_names):
        lines.append(f"{name}:")
        lines.append(f"  motor_id      = {motor_ids[i]}")
        lines.append(f"  neutral_pos   = {neutral_pos[i]}")
        lines.append(f"  min_pos       = {min_pos[i]}")
        lines.append(f"  max_pos       = {max_pos[i]}")
        lines.append(f"  homing_offset = {homing_offset[i]}")
        lines.append(f"  drive_mode    = {drive_mode[i]}")
        lines.append("")

    TXT_PATH.write_text("\n".join(lines) + "\n")
    return project_path, driver_path


def setup_motors_native() -> int:
    _print_header("LeRobot-style motor setup (ID / baudrate)")
    print(
        "This step matches the native LeRobot motor setup stage as closely as possible.\n"
        "You will be asked to connect one motor at a time so the driver can assign IDs\n"
        "and baudrate in the expected bus order."
    )

    handle = _make_driver_handle()
    device = handle.device
    try:
        setup_fn = getattr(device, "setup_motors", None)
        if setup_fn is None:
            raise CalibrationError(
                "The installed LeRobot follower class does not expose setup_motors(). "
                "Update LeRobot or perform ID setup with lerobot-setup-motors."
            )
        setup_fn()
        print("[calibrate] Native motor setup completed.")
        return 0
    except Exception as exc:
        print(f"[calibrate] Native motor setup failed: {exc}")
        return 1
    finally:
        try:
            disconnect = getattr(device, "disconnect", None)
            if disconnect is not None:
                disconnect()
        except Exception:
            pass


def run_interactive_calibration(identify_first: bool = True) -> int:
    _print_header("Interactive robot calibration")
    print(
        "This script performs the full calibration capture for the modified follower arm.\n"
        "It records a neutral pose first, then min/max motion for every joint, and writes\n"
        "both a project-local JSON and a LeRobot-compatible driver JSON."
    )

    try:
        session = connect_session()
    except Exception as exc:
        print(f"[calibrate] Failed to connect to robot: {exc}")
        return 1

    print(f"[calibrate] Connected. Motor names: {session.motor_names}")
    print(f"[calibrate] Expected ID map: {_expected_motor_ids(session.motor_names)}")

    try:
        if identify_first:
            identify_motors(session)

        set_torque(session, enabled=False)

        neutral = prompt_capture(
            session,
            "Capture NEUTRAL pose",
            "Move the arm into your desired neutral/zero pose.\n"
            "Suggested pose: shoulder_pan centered, shoulder_lift neutral, elbow neutral,\n"
            "wrist joints centered, and gripper in the pose you want to treat as its neutral reference.",
        )

        min_pos, max_pos = capture_joint_limits(session)
        _validate_captured_ranges(session.motor_names, min_pos, max_pos)
        payload = build_calibration_payload(session, neutral, min_pos, max_pos)
        project_path, driver_path = write_outputs(payload)

        _print_header("Calibration complete")
        print("Saved JSON calibration to:")
        print(f"  Project file: {project_path}")
        print(f"  Driver file:  {driver_path}")
        print(f"Saved text summary to: {TXT_PATH}")
        print("\nSummary:")
        for name, motor_id, neutral_pos, minp, maxp in zip(
            payload["motor_names"],
            payload["motor_ids"],
            payload["neutral_pos"],
            payload["min_pos"],
            payload["max_pos"],
            strict=True,
        ):
            print(
                f"  id={motor_id:2d} {name:16s} neutral={neutral_pos:6d}  min={minp:6d}  max={maxp:6d}"
            )

    except Exception as exc:
        print(f"[calibrate] Calibration failed: {exc}")
        return 1
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


def run_full_workflow() -> int:
    rc = setup_motors_native()
    if rc != 0:
        print("[calibrate] Motor setup did not complete. Stopping before calibration.")
        return rc
    return run_interactive_calibration(identify_first=True)


def _menu_choice() -> str:
    _print_header("Robot motor setup / calibration")
    print("1. Full setup + calibrate")
    print("2. Setup motors only (LeRobot-style ID / baudrate setup)")
    print("3. Calibrate only (IDs already configured)")
    print("4. Identify / blink current motors only")
    print("5. Quit")
    choice = input("Select an option [1-5]: ").strip() or "1"
    return choice


def run_identify_only() -> int:
    try:
        session = connect_session()
    except Exception as exc:
        print(f"[calibrate] Failed to connect for identification: {exc}")
        return 1
    try:
        identify_motors(session)
    finally:
        try:
            session.controller.disconnect()
        except Exception:
            pass
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LeRobot-style motor setup and calibration for the modified SO100/SO101 follower arm.")
    parser.add_argument("--setup-motors", action="store_true", help="Run only the LeRobot-style motor ID/baudrate setup stage.")
    parser.add_argument("--calibrate-only", action="store_true", help="Run only interactive min/max calibration, assuming IDs are already configured.")
    parser.add_argument("--identify-only", action="store_true", help="Connect and blink motors one at a time in expected LeRobot order.")
    parser.add_argument("--full", action="store_true", help="Run setup motors first and then full calibration.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.identify_only:
        return run_identify_only()
    if args.setup_motors:
        return setup_motors_native()
    if args.calibrate_only:
        return run_interactive_calibration(identify_first=True)
    if args.full:
        return run_full_workflow()

    choice = _menu_choice()
    if choice == "1":
        return run_full_workflow()
    if choice == "2":
        return setup_motors_native()
    if choice == "3":
        return run_interactive_calibration(identify_first=True)
    if choice == "4":
        return run_identify_only()
    print("[calibrate] Cancelled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))