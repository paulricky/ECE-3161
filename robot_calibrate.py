from __future__ import annotations

import importlib
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
        / "so_follower"
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
        "wrist_yaw",
        "wrist_roll",
        "wrist_pitch",
        "gripper",
    ]


def _get_configured_motor_ids(motor_names: Sequence[str] | None = None) -> list[int]:
    if motor_names is None:
        motor_names = _get_configured_motor_names()
    ids = list(getattr(val, "REAL_ROBOT_MOTOR_IDS", []))
    if len(ids) == len(motor_names):
        return [int(x) for x in ids]
    return list(range(1, len(motor_names) + 1))


def _get_configured_motor_model_numbers(motor_names: Sequence[str] | None = None) -> list[int]:
    if motor_names is None:
        motor_names = _get_configured_motor_names()
    models = list(getattr(val, "REAL_ROBOT_MOTOR_MODEL_NUMBERS", []))
    if len(models) == len(motor_names):
        return [int(x) for x in models]
    model = int(getattr(val, "REAL_ROBOT_MOTOR_MODEL_NUMBER", 777))
    return [model] * len(motor_names)


def _configured_motor_map(motor_names: Sequence[str] | None = None) -> dict[str, int]:
    if motor_names is None:
        motor_names = _get_configured_motor_names()
    motor_ids = _get_configured_motor_ids(motor_names)
    return {str(name): int(mid) for name, mid in zip(motor_names, motor_ids, strict=True)}


def _find_robot_port() -> str:
    configured = str(getattr(val, "REAL_ROBOT_PORT", "")).strip()
    if configured:
        return configured
    controller = SOArmHardwareController()
    port = controller._auto_detect_port()
    if not port:
        raise RuntimeError("Could not auto-detect the robot serial port. Set values.REAL_ROBOT_PORT manually.")
    return str(port)


def _safe_setattr(obj: Any, name: str, value: Any) -> bool:
    try:
        setattr(obj, name, value)
        return True
    except Exception:
        return False


def _patch_bus_like_object(bus: Any, port: str, motor_names: Sequence[str], motor_ids: Sequence[int]) -> bool:
    changed = False
    motor_map = {str(name): int(mid) for name, mid in zip(motor_names, motor_ids, strict=True)}
    model_numbers = _get_configured_motor_model_numbers(motor_names)
    models_by_name = {str(name): int(model) for name, model in zip(motor_names, model_numbers, strict=True)}

    for attr_name in ("port", "serial_port"):
        if hasattr(bus, attr_name):
            changed = _safe_setattr(bus, attr_name, port) or changed

    candidate_motor_payloads = [
        motor_map,
        {name: (motor_map[name], models_by_name[name]) for name in motor_map},
        {name: {"id": motor_map[name], "model": models_by_name[name]} for name in motor_map},
        {name: {"id": motor_map[name], "model_number": models_by_name[name]} for name in motor_map},
    ]
    for attr_name in ("motors", "motor_names", "motor_ids", "ids"):
        if hasattr(bus, attr_name):
            if attr_name == "motor_names":
                changed = _safe_setattr(bus, attr_name, list(motor_names)) or changed
            elif attr_name in {"motor_ids", "ids"}:
                changed = _safe_setattr(bus, attr_name, list(map(int, motor_ids))) or changed
            else:
                for payload in candidate_motor_payloads:
                    if _safe_setattr(bus, attr_name, payload):
                        changed = True
                        break
    return changed


def _patch_config_for_custom_motors(cfg: Any, port: str, motor_names: Sequence[str], motor_ids: Sequence[int]) -> None:
    _safe_setattr(cfg, "port", port)
    _safe_setattr(cfg, "motor_names", list(motor_names))
    _safe_setattr(cfg, "motor_ids", list(map(int, motor_ids)))

    patched = False
    for attr_name in ("bus", "motors_bus", "arm", "follower_arm", "follower_bus"):
        obj = getattr(cfg, attr_name, None)
        if obj is not None:
            patched = _patch_bus_like_object(obj, port, motor_names, motor_ids) or patched

    for attr_name in ("arms", "follower_arms", "buses", "motor_buses"):
        mapping = getattr(cfg, attr_name, None)
        if isinstance(mapping, dict):
            for obj in mapping.values():
                patched = _patch_bus_like_object(obj, port, motor_names, motor_ids) or patched

    if hasattr(cfg, "motors"):
        motor_map = {str(name): int(mid) for name, mid in zip(motor_names, motor_ids, strict=True)}
        patched = _safe_setattr(cfg, "motors", motor_map) or patched

    if patched:
        print(f"[calibrate] configured setup/calibration motor list: {_configured_motor_map(motor_names)}")


def _make_unconnected_robot_for_setup() -> tuple[SOArmHardwareController, Any, str, list[str], list[int]]:
    controller = SOArmHardwareController()
    follower_cls, config_cls = controller._import_lerobot_so_follower()
    port = _find_robot_port()
    print(f"[calibrate] using port = {port}")
    robot_id = getattr(val, "REAL_ROBOT_ID", "my_awesome_follower_arm")
    cfg = controller._build_lerobot_config(config_cls, port, robot_id)
    motor_names = _get_configured_motor_names()
    motor_ids = _get_configured_motor_ids(motor_names)
    _patch_config_for_custom_motors(cfg, port, motor_names, motor_ids)
    robot = follower_cls(cfg)
    return controller, robot, port, motor_names, motor_ids


def _import_feetech_bus_classes() -> tuple[Any, Any | None]:
    """Return (FeetechMotorsBus, FeetechMotorsBusConfig or None) across LeRobot API versions."""
    candidates = [
        (
            "lerobot.motors.feetech.feetech",
            "FeetechMotorsBus",
            "FeetechMotorsBusConfig",
        ),
        (
            "lerobot.motors.feetech",
            "FeetechMotorsBus",
            "FeetechMotorsBusConfig",
        ),
        (
            "lerobot.common.robot_devices.motors.feetech",
            "FeetechMotorsBus",
            "FeetechMotorsBusConfig",
        ),
        (
            "lerobot.common.robot_devices.motors.feetech",
            "FeetechMotorsBus",
            None,
        ),
    ]
    errors: list[str] = []
    for module_name, bus_name, cfg_name in candidates:
        try:
            module = importlib.import_module(module_name)
            bus_cls = getattr(module, bus_name)
            cfg_cls = getattr(module, cfg_name) if cfg_name else None
            return bus_cls, cfg_cls
        except Exception as exc:
            errors.append(f"{module_name}: {exc}")
    raise RuntimeError(
        "Could not import a Feetech motor bus from the installed LeRobot package. "
        "Tried: " + "; ".join(errors)
    )


def _motor_payload_variants(name: str, motor_id: int, model_number: int) -> list[dict[str, Any]]:
    """Different LeRobot versions expect different motor-map shapes."""
    return [
        {name: int(motor_id)},
        {name: (int(motor_id), int(model_number))},
        {name: [int(motor_id), int(model_number)]},
        {name: {"id": int(motor_id), "model": int(model_number)}},
        {name: {"id": int(motor_id), "model_number": int(model_number)}},
    ]


def _instantiate_feetech_bus(port: str, name: str, motor_id: int, model_number: int) -> Any:
    bus_cls, cfg_cls = _import_feetech_bus_classes()
    last_error: Exception | None = None
    for motors in _motor_payload_variants(name, motor_id, model_number):
        constructor_attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        if cfg_cls is not None:
            cfg_attempts = [
                ((port, motors), {}),
                ((), {"port": port, "motors": motors}),
                ((), {"serial_port": port, "motors": motors}),
            ]
            for cfg_args, cfg_kwargs in cfg_attempts:
                try:
                    cfg = cfg_cls(*cfg_args, **cfg_kwargs)
                    constructor_attempts.extend([
                        ((cfg,), {}),
                        ((), {"config": cfg}),
                    ])
                except Exception as exc:
                    last_error = exc
        constructor_attempts.extend([
            ((port, motors), {}),
            ((), {"port": port, "motors": motors}),
            ((), {"serial_port": port, "motors": motors}),
        ])
        for args, kwargs in constructor_attempts:
            try:
                return bus_cls(*args, **kwargs)
            except Exception as exc:
                last_error = exc
    raise RuntimeError(f"Could not construct a Feetech motor bus: {last_error}")


def _connect_bus_no_raise(bus: Any) -> bool:
    for fn_name in ("connect", "open"):
        fn = getattr(bus, fn_name, None)
        if callable(fn):
            try:
                fn()
                return True
            except Exception:
                return False
    return True


def _disconnect_bus_no_raise(bus: Any) -> None:
    for fn_name in ("disconnect", "close"):
        fn = getattr(bus, fn_name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass


def _read_one_register(bus: Any, register_names: Sequence[str], motor_name: str) -> list[int] | None:
    read = getattr(bus, "read", None)
    if read is None:
        return None
    for register in register_names:
        for args in (
            (register, motor_name),
            (register, [motor_name]),
            (register, (motor_name,)),
            (register,),
        ):
            try:
                values = _to_list_of_numbers(read(*args))
                if values:
                    return values
            except Exception:
                pass
    return None


def _write_one_register(bus: Any, register_names: Sequence[str], value: int, motor_name: str) -> bool:
    methods = [getattr(bus, name, None) for name in ("write", "sync_write")]
    methods = [method for method in methods if callable(method)]
    for method in methods:
        for register in register_names:
            for args in (
                (register, int(value), motor_name),
                (register, int(value), [motor_name]),
                (register, int(value), (motor_name,)),
                (register, {motor_name: int(value)}),
            ):
                try:
                    method(*args)
                    return True
                except Exception:
                    pass
    return False


def _probe_single_motor_at_id(port: str, current_id: int, model_number: int) -> Any | None:
    """Try opening a temporary one-motor bus at a specific ID."""
    temp_name = "motor"
    try:
        bus = _instantiate_feetech_bus(port, temp_name, int(current_id), int(model_number))
    except Exception:
        return None
    if not _connect_bus_no_raise(bus):
        _disconnect_bus_no_raise(bus)
        return None

    model = _read_one_register(bus, ("Model_Number", "Model", "model_number", "ModelNumber"), temp_name)
    if model is not None:
        if not model or int(model[0]) == int(model_number):
            return bus
    # Some LeRobot versions do not expose Model_Number cleanly; Present_Position is enough to
    # prove there is exactly one readable servo on this temporary bus.
    pos = _read_one_register(bus, ("Present_Position", "present_position", "Position"), temp_name)
    if pos is not None:
        return bus

    _disconnect_bus_no_raise(bus)
    return None


def _scan_single_connected_motor(port: str, model_number: int) -> tuple[Any, int] | None:
    """Find the currently connected single motor by trying likely Feetech IDs."""
    preferred_ids = list(range(1, 9)) + [0, 9, 10, 11, 12, 13, 14, 15]
    all_ids = preferred_ids + [i for i in range(16, 254) if i not in preferred_ids]
    print("[calibrate] Scanning for the single connected motor ID...")
    for current_id in all_ids:
        bus = _probe_single_motor_at_id(port, current_id, model_number)
        if bus is not None:
            print(f"[calibrate] Found one motor currently responding as ID {current_id}.")
            return bus, int(current_id)
    return None


def _manual_setup_one_motor(port: str, name: str, target_id: int, model_number: int) -> bool:
    _print_header(f"Set motor ID {target_id}: {name}")
    print(
        f"Disconnect the full daisy chain. Connect ONLY the '{name}' motor to the controller board.\n"
        "Power-cycle the controller/servo bus if the previous motor was just disconnected.\n"
        "Then press Enter.\n\n"
        "This patched setup first uses a direct Feetech packet scan. If that still cannot see\n"
        "the servo, the issue is below the Python calibration logic: wrong port, no data line,\n"
        "wrong baudrate, bad USB/TTL board, or servo stuck at an unknown protocol/baudrate."
    )
    input("Ready? ")

    direct = _direct_scan_single_connected_motor(port)
    if direct is not None:
        current_id, baudrate = direct
        if int(current_id) == int(target_id):
            print(f"[calibrate] '{name}' already has correct ID {target_id}.")
            return True
        print(
            f"[calibrate] Writing ID {target_id} to motor currently at ID {current_id} "
            f"using direct Feetech packets at baudrate {baudrate}..."
        )
        if _direct_write_feetech_id(port, int(current_id), int(target_id), int(baudrate)):
            print(f"[calibrate] Verified '{name}' as ID {target_id}.")
            return True
        print("[calibrate] Direct scan found the motor, but direct ID write/verify failed.")
        return False

    print(
        "[calibrate] Direct packet scan could not get any servo response. Trying the LeRobot bus wrapper fallback..."
    )
    scanned = _scan_single_connected_motor(port, model_number)
    if scanned is None:
        print(
            "[calibrate] Could not get any data response from the connected motor.\n"
            "The red LED only confirms power; it does NOT confirm serial communication. Check:\n"
            "  1) the servo signal plug orientation,\n"
            "  2) that the controller board data line is connected to the servo bus,\n"
            "  3) that the selected port is the Feetech controller,\n"
            "  4) that no other app has the serial port open,\n"
            "  5) that the motor is an STS/SCS Feetech protocol servo,\n"
            "  6) whether the servo is at a nonstandard baudrate not in the scan list."
        )
        return False

    bus, current_id = scanned
    try:
        if current_id == int(target_id):
            print(f"[calibrate] '{name}' already has correct ID {target_id}.")
            return True

        print(f"[calibrate] Writing ID {target_id} to motor currently at ID {current_id}...")
        ok = _write_one_register(bus, ("ID", "Id", "id"), int(target_id), "motor")
        if not ok:
            print(
                "[calibrate] Failed to write the ID register through the detected bus API. "
                "Your installed LeRobot/Feetech bus exposes a different write signature."
            )
            return False
        time.sleep(0.5)
    finally:
        _disconnect_bus_no_raise(bus)

    verify_bus = _probe_single_motor_at_id(port, int(target_id), model_number)
    if verify_bus is None:
        print(
            f"[calibrate] Wrote ID {target_id}, but could not verify it. Power-cycle the bus and rerun setup/identify."
        )
        return False
    _disconnect_bus_no_raise(verify_bus)
    print(f"[calibrate] Verified '{name}' as ID {target_id}.")
    return True


def _run_manual_motor_setup(port: str, motor_names: Sequence[str], motor_ids: Sequence[int]) -> bool:
    models = _get_configured_motor_model_numbers(motor_names)
    print("[calibrate] Manual 8-motor setup order:")
    for name, mid in zip(motor_names, motor_ids, strict=True):
        print(f"  {mid}: {name}")
    print(
        "\nThis avoids LeRobot's built-in 6-motor SO setup order, which assigns gripper as ID 6.\n"
        "Each prompt below assigns exactly the configured ID shown above."
    )
    for name, mid, model in zip(motor_names, motor_ids, models, strict=True):
        if not _manual_setup_one_motor(port, str(name), int(mid), int(model)):
            return False
    return True



def _feetech_checksum(packet_tail: Sequence[int]) -> int:
    return (~sum(int(x) & 0xFF for x in packet_tail)) & 0xFF


def _feetech_packet(servo_id: int, instruction: int, params: Sequence[int] = ()) -> bytes:
    servo_id = int(servo_id) & 0xFF
    params_b = [int(x) & 0xFF for x in params]
    length = len(params_b) + 2
    tail = [servo_id, length, int(instruction) & 0xFF, *params_b]
    return bytes([0xFF, 0xFF, *tail, _feetech_checksum(tail)])


def _feetech_read_status(serial_obj: Any, timeout_s: float = 0.08) -> tuple[int, int, list[int]] | None:
    """Read one Feetech/ST-series status packet: returns (id, error, params)."""
    deadline = time.monotonic() + float(timeout_s)
    window = bytearray()
    while time.monotonic() < deadline:
        b = serial_obj.read(1)
        if not b:
            continue
        window += b
        if len(window) > 2:
            window[:] = window[-2:]
        if len(window) == 2 and window[0] == 0xFF and window[1] == 0xFF:
            header = serial_obj.read(3)
            if len(header) != 3:
                return None
            sid = int(header[0])
            length = int(header[1])
            error = int(header[2])
            rest = serial_obj.read(max(0, length - 1))
            if len(rest) != max(0, length - 1):
                return None
            params = list(rest[:-1]) if rest else []
            checksum = int(rest[-1]) if rest else -1
            tail = [sid, length, error, *params]
            if checksum != _feetech_checksum(tail):
                return None
            return sid, error, params
    return None


def _direct_feetech_ping(port: str, servo_id: int, baudrate: int) -> bool:
    try:
        import serial
    except Exception as exc:
        print(f"[calibrate] pyserial import failed, cannot direct-ping servos: {exc}")
        return False
    try:
        with serial.Serial(str(port), int(baudrate), timeout=0.025, write_timeout=0.1) as ser:
            try:
                ser.reset_input_buffer()
                ser.reset_output_buffer()
            except Exception:
                pass
            ser.write(_feetech_packet(int(servo_id), 0x01, ()))
            ser.flush()
            status = _feetech_read_status(ser, timeout_s=0.08)
            return status is not None and int(status[0]) == int(servo_id)
    except Exception:
        return False


def _direct_scan_single_connected_motor(port: str) -> tuple[int, int] | None:
    """Direct packet-level scan independent of LeRobot's bus wrapper."""
    baudrates = list(dict.fromkeys([
        int(getattr(val, "REAL_ROBOT_BAUDRATE", 1000000)),
        1000000,
        128000,
        500000,
        115200,
        57600,
        38400,
        19200,
        250000,
    ]))
    preferred_ids = list(range(1, 9)) + [0, 9, 10, 11, 12, 13, 14, 15]
    all_ids = preferred_ids + [i for i in range(16, 254) if i not in preferred_ids]
    print("[calibrate] Direct Feetech packet scan for the connected motor...")
    for baud in baudrates:
        print(f"[calibrate]   direct ping baudrate {baud}...")
        found: list[int] = []
        for sid in all_ids:
            if _direct_feetech_ping(port, sid, baud):
                found.append(int(sid))
                print(f"[calibrate]     response from ID {sid}")
                if len(found) > 1:
                    break
        if len(found) == 1:
            print(f"[calibrate] Direct scan found motor ID {found[0]} at baudrate {baud}.")
            return found[0], int(baud)
        if len(found) > 1:
            print(
                f"[calibrate] More than one motor responded at baudrate {baud}: {found}. "
                "Disconnect all except the requested motor."
            )
            return None
    return None


def _direct_write_feetech_id(port: str, current_id: int, target_id: int, baudrate: int) -> bool:
    """Write the Feetech/ST-series ID register directly."""
    try:
        import serial
    except Exception as exc:
        print(f"[calibrate] pyserial import failed, cannot write servo ID directly: {exc}")
        return False
    id_register_addr = int(getattr(val, "FEETECH_ID_REGISTER_ADDR", 5))
    torque_enable_addr = int(getattr(val, "FEETECH_TORQUE_ENABLE_ADDR", 40))
    try:
        with serial.Serial(str(port), int(baudrate), timeout=0.05, write_timeout=0.2) as ser:
            try:
                ser.reset_input_buffer()
                ser.reset_output_buffer()
            except Exception:
                pass
            ser.write(_feetech_packet(int(current_id), 0x03, (torque_enable_addr, 0)))
            ser.flush()
            _feetech_read_status(ser, timeout_s=0.05)
            time.sleep(0.05)
            ser.write(_feetech_packet(int(current_id), 0x03, (id_register_addr, int(target_id))))
            ser.flush()
            status = _feetech_read_status(ser, timeout_s=0.15)
            if status is not None and int(status[1]) != 0:
                print(f"[calibrate] Servo returned error byte {status[1]} while writing ID.")
                return False
            time.sleep(0.5)
    except Exception as exc:
        print(f"[calibrate] Direct ID write failed: {exc}")
        return False

    return _direct_feetech_ping(port, int(target_id), int(baudrate))

def _build_setup_payload_from_config(motor_names: Sequence[str], motor_ids: Sequence[int]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "created_at_unix": time.time(),
        "motor_names": list(motor_names),
        "motor_ids": [int(x) for x in motor_ids],
        "setup_only": True,
        "notes": {
            "purpose": "Records that motor IDs were configured and verified separately from joint calibration.",
            "workflow": "Setup does not require the full chain to be present; connect one motor at a time when prompted.",
        },
    }
    for name, mid in zip(motor_names, motor_ids, strict=True):
        payload[str(name)] = {"name": str(name), "id": int(mid)}
    return payload


def _motor_setup_matches_config(motor_names: Sequence[str], motor_ids: Sequence[int] | None) -> bool:
    configured_names = _get_configured_motor_names()
    if list(motor_names) != configured_names:
        return False
    if motor_ids is None:
        return False
    expected_ids = _get_configured_motor_ids(configured_names)
    return list(map(int, motor_ids)) == expected_ids


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
        if _motor_setup_matches_config(motor_names, motor_ids):
            return SetupStatus(True, str(path), list(motor_names), list(motor_ids))
        print(
            f"[calibrate] Ignoring stale/incompatible motor setup file {path}: "
            f"expected names={configured_names}, ids={_get_configured_motor_ids(configured_names)}; "
            f"file has names={motor_names}, ids={motor_ids}"
        )

    return SetupStatus(False, None, configured_names, None)


def get_joint_calibration_status() -> CalibrationStatus:
    payload = _load_json(PROJECT_JSON_PATH)
    configured_names = _get_configured_motor_names()
    if not isinstance(payload, dict):
        return CalibrationStatus(False, None, configured_names)

    motor_names = list(payload.get("motor_names", [])) or configured_names
    if motor_names != configured_names:
        print(
            f"[calibrate] Ignoring stale/incompatible joint calibration {PROJECT_JSON_PATH}: "
            f"expected motor_names={configured_names}, file has motor_names={motor_names}"
        )
        return CalibrationStatus(False, None, motor_names)

    neutral = payload.get("neutral_pos")
    min_pos = payload.get("min_pos")
    max_pos = payload.get("max_pos")
    if isinstance(neutral, list) and isinstance(min_pos, list) and isinstance(max_pos, list):
        if len(neutral) == len(min_pos) == len(max_pos) == len(configured_names):
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
        "This stage is separate from joint calibration. It assigns/verifies servo IDs before min/max calibration.\n\n"
        "IMPORTANT: this patched setup does NOT use LeRobot's native setup_motors() order, because\n"
        "the native SO follower workflow is still based on the old 6-motor arm and can assign\n"
        "the gripper to ID 6. This workflow assigns IDs manually in your configured 8-motor order."
    )

    motor_names = _get_configured_motor_names()
    motor_ids = _get_configured_motor_ids(motor_names)
    if len(motor_names) != len(motor_ids):
        print(f"[calibrate] Invalid motor configuration: names={motor_names}, ids={motor_ids}")
        return 1

    try:
        port = _find_robot_port()
    except Exception as exc:
        print(f"[calibrate] Failed to locate robot serial port: {exc}")
        return 1

    print(f"[calibrate] using port = {port}")
    ok = _run_manual_motor_setup(port, motor_names, motor_ids)
    if not ok:
        print("[calibrate] Manual motor setup did not complete. No setup file was written.")
        return 1

    payload = _build_setup_payload_from_config(motor_names, motor_ids)
    path = write_setup_output(payload)
    print(f"[calibrate] Saved motor-setup metadata to: {path}")
    print(
        "[calibrate] Motor setup complete. Reconnect the full daisy chain, then run\n"
        "            option 4 to verify or option 3 to calibrate joints."
    )
    return 0

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