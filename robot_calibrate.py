from __future__ import annotations

import glob
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import values as val

try:
    import serial
except Exception:  # pragma: no cover
    serial = None


CALIB_DIR = Path(__file__).resolve().parent / "calibration_data"
PROJECT_JSON_PATH = CALIB_DIR / "robot_joint_calibration.json"
SETUP_JSON_PATH = CALIB_DIR / "robot_motor_setup.json"
TXT_PATH = CALIB_DIR / "robot_joint_calibration_summary.txt"

# Feetech/ST-series defaults. Override in values.py if your servo table differs.
FEETECH_MODEL_NUMBER_ADDR = int(getattr(val, "FEETECH_MODEL_NUMBER_ADDR", 3))
FEETECH_ID_REGISTER_ADDR = int(getattr(val, "FEETECH_ID_REGISTER_ADDR", 5))
FEETECH_TORQUE_ENABLE_ADDR = int(getattr(val, "FEETECH_TORQUE_ENABLE_ADDR", 40))
FEETECH_GOAL_POSITION_ADDR = int(getattr(val, "FEETECH_GOAL_POSITION_ADDR", 42))
FEETECH_PRESENT_POSITION_ADDR = int(getattr(val, "FEETECH_PRESENT_POSITION_ADDR", 56))
FEETECH_PRESENT_POSITION_LEN = int(getattr(val, "FEETECH_PRESENT_POSITION_LEN", 2))

DEFAULT_BAUDRATE = int(getattr(val, "REAL_ROBOT_BAUDRATE", 1000000))
PING_RETRIES = int(getattr(val, "FEETECH_DIRECT_PING_RETRIES", 4))
READ_RETRIES = int(getattr(val, "FEETECH_DIRECT_READ_RETRIES", 5))
PACKET_TIMEOUT_S = float(getattr(val, "FEETECH_DIRECT_PACKET_TIMEOUT_S", 0.18))


@dataclass
class SetupStatus:
    configured: bool
    source: str | None
    motor_names: list[str]
    motor_ids: list[int] | None
    motor_baudrates: list[int] | None = None


@dataclass
class CalibrationStatus:
    configured: bool
    source: str | None
    motor_names: list[str]


@dataclass
class DirectCalibrationSession:
    port: str
    motor_names: list[str]
    motor_ids: list[int]
    model_numbers: list[int]
    motor_baudrates: list[int] = field(default_factory=list)
    controller: Any = None
    robot: Any = None
    bus: Any = None

    def __post_init__(self) -> None:
        if not self.motor_baudrates or len(self.motor_baudrates) != len(self.motor_ids):
            self.motor_baudrates = [DEFAULT_BAUDRATE] * len(self.motor_ids)
        self.bus = self

    @property
    def baudrate(self) -> int:
        return int(self.motor_baudrates[0]) if self.motor_baudrates else DEFAULT_BAUDRATE

    def baud_for_id(self, servo_id: int) -> int:
        for mid, baud in zip(self.motor_ids, self.motor_baudrates, strict=False):
            if int(mid) == int(servo_id):
                return int(baud)
        return DEFAULT_BAUDRATE

    def set_baud_for_id(self, servo_id: int, baudrate: int) -> None:
        for i, mid in enumerate(self.motor_ids):
            if int(mid) == int(servo_id):
                self.motor_baudrates[i] = int(baudrate)
                return

    def disconnect(self) -> None:
        return None

    def read(self, register: str, motor_names: Sequence[str] | str | None = None) -> list[int] | int:
        names = _normalize_motor_selection(motor_names, self.motor_names)
        reg = str(register).lower().replace("_", "")
        values: list[int] = []
        for name in names:
            idx = self.motor_names.index(name)
            sid = self.motor_ids[idx]
            if reg in {"id", "ids"}:
                values.append(int(sid))
            elif reg in {"model", "modelnumber"}:
                model = _direct_read_model_any_baud(self.port, sid, [self.motor_baudrates[idx], *_get_scan_baudrates()])
                values.append(int(model) if model is not None else int(self.model_numbers[idx]))
            elif reg in {"presentposition", "position"}:
                pos, baud = _direct_read_position_any_baud(self.port, sid, [self.motor_baudrates[idx], *_get_scan_baudrates()])
                if pos is None:
                    raise RuntimeError(f"No Present_Position response from {name} / ID {sid}")
                self.set_baud_for_id(sid, baud)
                values.append(int(pos))
            else:
                raise RuntimeError(f"Unsupported direct register read: {register}")
        return values[0] if isinstance(motor_names, str) and len(values) == 1 else values

    def write(self, register: str, value: Any, motor_names: Sequence[str] | str | None = None) -> None:
        names = _normalize_motor_selection(motor_names, self.motor_names)
        reg = str(register).lower().replace("_", "")
        if isinstance(value, dict):
            items = [(name, int(value[name])) for name in names if name in value]
        elif isinstance(value, (list, tuple)):
            items = [(name, int(v)) for name, v in zip(names, value, strict=False)]
        else:
            items = [(name, int(value)) for name in names]

        for name, val_int in items:
            idx = self.motor_names.index(name)
            sid = self.motor_ids[idx]
            baud = self.motor_baudrates[idx]
            if reg in {"torqueenable", "torque"}:
                ok = _direct_write_u8(self.port, sid, FEETECH_TORQUE_ENABLE_ADDR, val_int, baud)
            elif reg in {"goalposition"}:
                ok = _direct_write_u16(self.port, sid, FEETECH_GOAL_POSITION_ADDR, val_int, baud)
            else:
                raise RuntimeError(f"Unsupported direct register write: {register}")
            if not ok:
                raise RuntimeError(f"Write {register} failed for {name} / ID {sid}")

    sync_write = write


def _normalize_motor_selection(selection: Sequence[str] | str | None, all_names: Sequence[str]) -> list[str]:
    if selection is None:
        return list(all_names)
    if isinstance(selection, str):
        return [selection]
    return [str(x) for x in selection]


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _is_gripper(name: str) -> bool:
    return str(name).lower() == "gripper"


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
    return Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration" / "robots" / "so_follower" / f"{robot_id}.json"


def _get_driver_calibration_path() -> Path:
    driver_path = _resolve_optional_path(getattr(val, "LEROBOT_DRIVER_CALIBRATION_FILE", ""))
    return driver_path if driver_path is not None else _default_driver_calibration_path()


def _get_output_json_paths() -> list[Path]:
    paths = [PROJECT_JSON_PATH]
    driver_path = _get_driver_calibration_path()
    if driver_path not in paths:
        paths.append(driver_path)
    return paths


def _get_configured_motor_names() -> list[str]:
    names = list(getattr(val, "REAL_ROBOT_MOTOR_NAMES", []))
    if names:
        return [str(x) for x in names]
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


def _get_scan_baudrates(extra: Sequence[int] | None = None) -> list[int]:
    raw = list(getattr(val, "REAL_ROBOT_SCAN_BAUDRATES", []))
    candidates = list(extra or []) + raw + [DEFAULT_BAUDRATE, 1000000, 128000, 500000, 115200, 57600, 38400, 19200, 250000]
    out: list[int] = []
    for baud in candidates:
        try:
            baud = int(baud)
        except Exception:
            continue
        if baud > 0 and baud not in out:
            out.append(baud)
    return out


def _candidate_robot_ports() -> list[str]:
    configured = str(getattr(val, "REAL_ROBOT_PORT", "")).strip()
    ports: list[str] = []
    if configured:
        ports.append(configured)
        if "/cu." in configured:
            ports.append(configured.replace("/cu.", "/tty."))
        elif "/tty." in configured:
            ports.append(configured.replace("/tty.", "/cu."))
    patterns = ["/dev/cu.usbmodem*", "/dev/cu.usbserial*", "/dev/tty.usbmodem*", "/dev/tty.usbserial*"]
    for pattern in patterns:
        ports.extend(glob.glob(pattern))
    return sorted(dict.fromkeys(ports))


def _find_robot_port() -> str:
    ports = _candidate_robot_ports()
    if not ports:
        raise RuntimeError("Could not auto-detect the robot serial port. Set values.REAL_ROBOT_PORT manually.")
    print(f"[calibrate] candidate ports = {ports}")
    return ports[0]


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text())
    except Exception:
        return None


def _extract_motor_ids(payload: dict[str, Any], motor_names: Sequence[str]) -> list[int] | None:
    ids = payload.get("motor_ids")
    if isinstance(ids, list) and len(ids) == len(motor_names):
        try:
            return [int(x) for x in ids]
        except Exception:
            pass
    out: list[int] = []
    for name in motor_names:
        entry = payload.get(str(name))
        if not isinstance(entry, dict) or entry.get("id") is None:
            return None
        try:
            out.append(int(entry["id"]))
        except Exception:
            return None
    return out if len(out) == len(motor_names) else None


def _extract_motor_baudrates(payload: dict[str, Any], motor_names: Sequence[str]) -> list[int] | None:
    bauds = payload.get("motor_baudrates")
    if isinstance(bauds, list) and len(bauds) == len(motor_names):
        try:
            return [int(x) for x in bauds]
        except Exception:
            pass
    out: list[int] = []
    for name in motor_names:
        entry = payload.get(str(name))
        if isinstance(entry, dict) and entry.get("baudrate") is not None:
            try:
                out.append(int(entry["baudrate"]))
                continue
            except Exception:
                pass
        return None
    return out if len(out) == len(motor_names) else None


def _motor_setup_matches_config(motor_names: Sequence[str], motor_ids: Sequence[int] | None) -> bool:
    configured_names = _get_configured_motor_names()
    if list(motor_names) != configured_names or motor_ids is None:
        return False
    return list(map(int, motor_ids)) == _get_configured_motor_ids(configured_names)


def get_motor_setup_status() -> SetupStatus:
    configured_names = _get_configured_motor_names()
    for path in (SETUP_JSON_PATH, _get_driver_calibration_path()):
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        motor_names = list(payload.get("motor_names", [])) or configured_names
        motor_ids = _extract_motor_ids(payload, motor_names)
        motor_baudrates = _extract_motor_baudrates(payload, motor_names)
        if _motor_setup_matches_config(motor_names, motor_ids):
            return SetupStatus(True, str(path), list(motor_names), list(motor_ids), motor_baudrates)
        print(
            f"[calibrate] Ignoring stale/incompatible motor setup file {path}: "
            f"expected names={configured_names}, ids={_get_configured_motor_ids(configured_names)}; "
            f"file has names={motor_names}, ids={motor_ids}"
        )
    return SetupStatus(False, None, configured_names, None, None)


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


# -------------------------- direct Feetech packet I/O -------------------------

def _feetech_checksum(packet_tail: Sequence[int]) -> int:
    return (~sum(int(x) & 0xFF for x in packet_tail)) & 0xFF


def _feetech_packet(servo_id: int, instruction: int, params: Sequence[int] = ()) -> bytes:
    params_b = [int(x) & 0xFF for x in params]
    tail = [int(servo_id) & 0xFF, len(params_b) + 2, int(instruction) & 0xFF, *params_b]
    return bytes([0xFF, 0xFF, *tail, _feetech_checksum(tail)])


def _read_status(ser: Any, timeout_s: float = PACKET_TIMEOUT_S) -> tuple[int, int, list[int]] | None:
    deadline = time.monotonic() + float(timeout_s)
    window = bytearray()
    while time.monotonic() < deadline:
        b = ser.read(1)
        if not b:
            continue
        window += b
        if len(window) > 2:
            window[:] = window[-2:]
        if len(window) == 2 and window[0] == 0xFF and window[1] == 0xFF:
            header = ser.read(3)
            if len(header) != 3:
                return None
            sid, length, error = int(header[0]), int(header[1]), int(header[2])
            rest = ser.read(max(0, length - 1))
            if len(rest) != max(0, length - 1) or not rest:
                return None
            params = list(rest[:-1])
            checksum = int(rest[-1])
            if checksum != _feetech_checksum([sid, length, error, *params]):
                return None
            return sid, error, params
    return None


def _open_serial(port: str, baudrate: int, timeout: float = 0.04) -> Any:
    if serial is None:
        raise RuntimeError("pyserial is not installed/importable; install pyserial to use direct calibration.")
    ser = serial.Serial(str(port), int(baudrate), timeout=timeout, write_timeout=0.25)
    try:
        ser.reset_input_buffer()
        ser.reset_output_buffer()
    except Exception:
        pass
    time.sleep(0.01)
    return ser


def _direct_ping_once(port: str, servo_id: int, baudrate: int) -> bool:
    try:
        with _open_serial(port, baudrate) as ser:
            ser.write(_feetech_packet(servo_id, 0x01, ()))
            ser.flush()
            status = _read_status(ser, timeout_s=PACKET_TIMEOUT_S)
            return status is not None and int(status[0]) == int(servo_id)
    except Exception:
        return False


def _direct_read_register_once(port: str, servo_id: int, addr: int, length: int, baudrate: int) -> list[int] | None:
    try:
        with _open_serial(port, baudrate, timeout=0.05) as ser:
            ser.write(_feetech_packet(servo_id, 0x02, (int(addr), int(length))))
            ser.flush()
            status = _read_status(ser, timeout_s=PACKET_TIMEOUT_S)
            if status is None or int(status[0]) != int(servo_id) or int(status[1]) != 0:
                return None
            params = list(status[2])
            if len(params) < int(length):
                return None
            return params[: int(length)]
    except Exception:
        return None


def _direct_ping(port: str, servo_id: int, baudrate: int) -> bool:
    for _ in range(max(1, PING_RETRIES)):
        if _direct_ping_once(port, servo_id, baudrate):
            return True
        # Some boards drop ping statuses but allow register reads.
        if _direct_read_register_once(port, servo_id, FEETECH_PRESENT_POSITION_ADDR, FEETECH_PRESENT_POSITION_LEN, baudrate) is not None:
            return True
        time.sleep(0.02)
    return False


def _direct_read_register(port: str, servo_id: int, addr: int, length: int, baudrate: int) -> list[int] | None:
    for _ in range(max(1, READ_RETRIES)):
        data = _direct_read_register_once(port, servo_id, addr, length, baudrate)
        if data is not None:
            return data
        time.sleep(0.02)
    return None


def _direct_write_register(port: str, servo_id: int, addr: int, data: Sequence[int], baudrate: int) -> bool:
    for _ in range(3):
        try:
            with _open_serial(port, baudrate, timeout=0.05) as ser:
                ser.write(_feetech_packet(servo_id, 0x03, (int(addr), *[int(x) & 0xFF for x in data])))
                ser.flush()
                status = _read_status(ser, timeout_s=0.16)
                if status is None or (int(status[0]) == int(servo_id) and int(status[1]) == 0):
                    return True
        except Exception:
            pass
        time.sleep(0.02)
    return False


def _direct_read_u16(port: str, servo_id: int, addr: int, baudrate: int) -> int | None:
    data = _direct_read_register(port, servo_id, addr, 2, baudrate)
    if data is None or len(data) < 2:
        return None
    return int(data[0]) | (int(data[1]) << 8)


def _direct_write_u8(port: str, servo_id: int, addr: int, value: int, baudrate: int) -> bool:
    return _direct_write_register(port, servo_id, addr, [int(value) & 0xFF], baudrate)


def _direct_write_u16(port: str, servo_id: int, addr: int, value: int, baudrate: int) -> bool:
    value = int(value)
    return _direct_write_register(port, servo_id, addr, [value & 0xFF, (value >> 8) & 0xFF], baudrate)


def _direct_read_position(port: str, servo_id: int, baudrate: int) -> int | None:
    return _direct_read_u16(port, servo_id, FEETECH_PRESENT_POSITION_ADDR, baudrate)


def _direct_read_model(port: str, servo_id: int, baudrate: int) -> int | None:
    return _direct_read_u16(port, servo_id, FEETECH_MODEL_NUMBER_ADDR, baudrate)


def _direct_detect_baud_for_id(port: str, servo_id: int, preferred_bauds: Sequence[int] | None = None) -> int | None:
    for baud in _get_scan_baudrates(preferred_bauds):
        if _direct_ping(port, servo_id, baud):
            return int(baud)
    return None


def _direct_read_position_any_baud(port: str, servo_id: int, preferred_bauds: Sequence[int] | None = None) -> tuple[int | None, int]:
    for baud in _get_scan_baudrates(preferred_bauds):
        pos = _direct_read_position(port, servo_id, baud)
        if pos is not None:
            return int(pos), int(baud)
        if _direct_ping(port, servo_id, baud):
            pos = _direct_read_position(port, servo_id, baud)
            if pos is not None:
                return int(pos), int(baud)
    return None, int(_get_scan_baudrates(preferred_bauds)[0])


def _direct_read_model_any_baud(port: str, servo_id: int, preferred_bauds: Sequence[int] | None = None) -> int | None:
    for baud in _get_scan_baudrates(preferred_bauds):
        model = _direct_read_model(port, servo_id, baud)
        if model is not None:
            return int(model)
    return None


def _scan_single_connected_motor(port: str) -> tuple[int, int] | None:
    preferred_ids = list(range(1, 9)) + [0, 9, 10, 11, 12, 13, 14, 15]
    all_ids = preferred_ids + [i for i in range(16, 254) if i not in preferred_ids]
    print("[calibrate] Direct Feetech packet scan for the single connected motor...")
    for baud in _get_scan_baudrates():
        print(f"[calibrate]   ping baudrate {baud}...")
        found: list[int] = []
        for sid in all_ids:
            if _direct_ping(port, sid, baud):
                found.append(sid)
                print(f"[calibrate]     response from ID {sid}")
                if len(found) > 1:
                    print("[calibrate] More than one motor responded. Connect only the requested motor.")
                    return None
        if len(found) == 1:
            print(f"[calibrate] Found motor ID {found[0]} at baudrate {baud}.")
            return found[0], baud
    return None


def _direct_write_feetech_id(port: str, current_id: int, target_id: int, baudrate: int) -> bool:
    _direct_write_u8(port, current_id, FEETECH_TORQUE_ENABLE_ADDR, 0, baudrate)
    time.sleep(0.05)
    if not _direct_write_u8(port, current_id, FEETECH_ID_REGISTER_ADDR, target_id, baudrate):
        return False
    time.sleep(0.5)
    return _direct_ping(port, target_id, baudrate)


def _build_setup_payload_from_config(
    motor_names: Sequence[str], motor_ids: Sequence[int], motor_baudrates: Sequence[int] | None = None
) -> dict[str, Any]:
    if motor_baudrates is None or len(motor_baudrates) != len(motor_names):
        motor_baudrates = [DEFAULT_BAUDRATE] * len(motor_names)
    payload: dict[str, Any] = {
        "created_at_unix": time.time(),
        "motor_names": list(motor_names),
        "motor_ids": [int(x) for x in motor_ids],
        "motor_baudrates": [int(x) for x in motor_baudrates],
        "setup_only": True,
        "notes": {
            "purpose": "Records that motor IDs were configured and verified separately from joint calibration.",
            "workflow": "Setup uses direct Feetech packet-level ID assignment in configured order.",
        },
    }
    for name, mid, baud in zip(motor_names, motor_ids, motor_baudrates, strict=True):
        payload[str(name)] = {"name": str(name), "id": int(mid), "baudrate": int(baud)}
    return payload


def write_setup_output(payload: dict[str, Any]) -> Path:
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    SETUP_JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    return SETUP_JSON_PATH


def _manual_setup_one_motor(port: str, name: str, target_id: int) -> int | None:
    _print_header(f"Set motor ID {target_id}: {name}")
    print(
        f"Disconnect the full daisy chain. Connect ONLY the '{name}' motor to the controller board.\n"
        "Power-cycle the controller/servo bus if the previous motor was just disconnected.\n"
        "Then press Enter."
    )
    input("Ready? ")
    scanned = _scan_single_connected_motor(port)
    if scanned is None:
        print(
            "[calibrate] Could not get any data response from the connected motor.\n"
            "The red LED only confirms power; it does NOT confirm serial communication. Check cable orientation,\n"
            "port, data line, bus power ground, and that no other app has the serial port open."
        )
        return None
    current_id, baudrate = scanned
    if current_id == int(target_id):
        print(f"[calibrate] '{name}' already has correct ID {target_id}.")
        return int(baudrate)
    print(f"[calibrate] Writing ID {target_id} to motor currently at ID {current_id}...")
    if not _direct_write_feetech_id(port, current_id, target_id, baudrate):
        print(f"[calibrate] Failed to write/verify ID {target_id}.")
        return None
    print(f"[calibrate] Verified '{name}' as ID {target_id}.")
    return int(baudrate)


def _run_manual_motor_setup(port: str, motor_names: Sequence[str], motor_ids: Sequence[int]) -> list[int] | None:
    print("[calibrate] Manual 8-motor setup order:")
    for name, mid in zip(motor_names, motor_ids, strict=True):
        print(f"  {mid}: {name}")
    print(
        "\nThis avoids LeRobot's built-in 6-motor SO setup order.\n"
        "Each prompt below assigns exactly the configured ID shown above."
    )
    bauds: list[int] = []
    for name, mid in zip(motor_names, motor_ids, strict=True):
        baud = _manual_setup_one_motor(port, str(name), int(mid))
        if baud is None:
            return None
        bauds.append(int(baud))
    return bauds


def run_motor_setup_only() -> int:
    _print_header("Robot motor setup")
    print(
        "This stage is separate from joint calibration. It assigns/verifies servo IDs before min/max calibration.\n\n"
        "This patched setup uses direct Feetech packet-level communication instead of LeRobot's native\n"
        "6-motor setup order, so the configured 8-motor order is preserved."
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
    motor_baudrates = _run_manual_motor_setup(port, motor_names, motor_ids)
    if motor_baudrates is None:
        print("[calibrate] Manual motor setup did not complete. No setup file was written.")
        return 1
    payload = _build_setup_payload_from_config(motor_names, motor_ids, motor_baudrates)
    path = write_setup_output(payload)
    print(f"[calibrate] Saved motor-setup metadata to: {path}")
    print("[calibrate] Motor setup complete. Reconnect the full daisy chain, then run option 4 or 3.")
    return 0


# ----------------------------- calibration workflow ---------------------------

def _detect_chain_baudrates(
    port: str,
    motor_ids: Sequence[int],
    setup_baudrates: Sequence[int] | None = None,
) -> tuple[dict[int, int], dict[int, int | None]]:
    found_baud: dict[int, int] = {}
    found_model: dict[int, int | None] = {}
    setup_map: dict[int, int] = {}
    if setup_baudrates and len(setup_baudrates) == len(motor_ids):
        setup_map = {int(mid): int(baud) for mid, baud in zip(motor_ids, setup_baudrates, strict=True)}

    # Per-ID detection is more reliable than a one-shot full-chain pass because the bus can drop packets.
    for sid in map(int, motor_ids):
        preferred = [setup_map[sid]] if sid in setup_map else []
        baud = _direct_detect_baud_for_id(port, sid, preferred)
        if baud is not None:
            found_baud[sid] = int(baud)
            found_model[sid] = _direct_read_model(port, sid, int(baud))
    return found_baud, found_model


def connect_session() -> DirectCalibrationSession:
    ports = _candidate_robot_ports()
    if not ports:
        raise RuntimeError("Could not auto-detect the robot serial port. Set values.REAL_ROBOT_PORT manually.")

    setup_status = get_motor_setup_status()
    motor_names = _get_configured_motor_names()
    motor_ids = _get_configured_motor_ids(motor_names)
    model_numbers = _get_configured_motor_model_numbers(motor_names)
    setup_baudrates = setup_status.motor_baudrates if setup_status.configured else None

    best_message = ""
    for port in ports:
        print(f"[calibrate] using direct port = {port}")
        found_baud, found_models = _detect_chain_baudrates(port, motor_ids, setup_baudrates)
        missing = [int(sid) for sid in motor_ids if int(sid) not in found_baud]
        if not missing:
            bauds = [int(found_baud[int(mid)]) for mid in motor_ids]
            print("[calibrate] Direct chain verification succeeded:")
            for name, sid, baud in zip(motor_names, motor_ids, bauds, strict=True):
                print(f"  {name:16s} id={sid:3d} baud={baud}")
            return DirectCalibrationSession(str(port), list(motor_names), list(motor_ids), list(model_numbers), bauds)

        if found_baud:
            print(f"[calibrate] port {port}: found IDs {sorted(found_baud)}, missing IDs {missing}")
        expected = {int(mid): int(model) for mid, model in zip(motor_ids, model_numbers, strict=True)}
        best_message = (
            "\n[calibrate] Direct chain check did not see every configured motor.\n"
            "Full expected motor list (id: model_number):\n"
            f"{expected}\n\n"
            "Full found motor list (id: model_number):\n"
            f"{found_models}\n"
        )

        # Preserve workflow even when the chain check is flaky after successful setup.
        # The actual calibration captures will retry each read by ID and update baudrates dynamically.
        if setup_status.configured:
            print(
                "[calibrate] Warning: setup metadata exists, so calibration will continue despite the incomplete\n"
                "            pre-check. Each capture will retry reads per motor; if a motor truly is unreachable,\n"
                "            the capture step will name the exact motor/ID that failed."
            )
            inferred_bauds = []
            for mid in motor_ids:
                if int(mid) in found_baud:
                    inferred_bauds.append(int(found_baud[int(mid)]))
                elif setup_baudrates and len(setup_baudrates) == len(motor_ids):
                    inferred_bauds.append(int(setup_baudrates[list(motor_ids).index(mid)]))
                else:
                    inferred_bauds.append(DEFAULT_BAUDRATE)
            return DirectCalibrationSession(str(port), list(motor_names), list(motor_ids), list(model_numbers), inferred_bauds)

    print(best_message)
    raise RuntimeError("Not all configured motors responded through direct Feetech packets.")


def read_positions(session: DirectCalibrationSession) -> dict[str, int]:
    positions: dict[str, int] = {}
    failures: list[str] = []
    for name, sid in zip(session.motor_names, session.motor_ids, strict=True):
        preferred = [session.baud_for_id(sid)]
        pos, baud = _direct_read_position_any_baud(session.port, sid, preferred)
        if pos is None:
            failures.append(f"{name}/ID{sid}")
            continue
        session.set_baud_for_id(sid, baud)
        positions[name] = int(pos)
    if failures:
        raise RuntimeError(
            "Could not read Present_Position from: "
            + ", ".join(failures)
            + ". Check that those IDs are actually present on the daisy chain and that their cables pass data."
        )
    return positions


def set_torque(session: DirectCalibrationSession, enabled: bool) -> None:
    value = 1 if enabled else 0
    failed: list[str] = []
    for name, sid in zip(session.motor_names, session.motor_ids, strict=True):
        baud = session.baud_for_id(sid)
        if not _direct_write_u8(session.port, sid, FEETECH_TORQUE_ENABLE_ADDR, value, baud):
            # Do not fail calibration on torque write; some servos may reject this while still allowing position reads.
            failed.append(f"{name}/ID{sid}")
    if failed:
        print(f"[calibrate] Warning: torque write failed for: {', '.join(failed)}")
    else:
        print(f"[calibrate] Torque {'ENABLED' if enabled else 'DISABLED'} for {len(session.motor_names)} motors.")


def prompt_capture(session: DirectCalibrationSession, title: str, instructions: str) -> dict[str, int]:
    _print_header(title)
    print(instructions)
    while True:
        input("Press Enter to capture the current motor positions... ")
        try:
            positions = read_positions(session)
            break
        except Exception as exc:
            print(f"[calibrate] Capture failed: {exc}")
            reply = input("Retry capture? [Y/n]: ").strip().lower()
            if reply in {"n", "no"}:
                raise
    print("Captured positions:")
    for name in session.motor_names:
        print(f"  {name:16s} {positions[name]:6d}")
    return positions


def infer_drive_mode(neutral: dict[str, int], max_pos: dict[str, int]) -> dict[str, int]:
    return {name: 0 if _is_gripper(name) else int(max_pos[name] < neutral[name]) for name in neutral}


def infer_homing_offset(neutral: dict[str, int], drive_mode: dict[str, int]) -> dict[str, int]:
    return {name: int(neutral_pos if drive_mode[name] else -neutral_pos) for name, neutral_pos in neutral.items()}


def capture_joint_limits(session: DirectCalibrationSession) -> tuple[dict[str, int], dict[str, int]]:
    min_pos: dict[str, int] = {}
    max_pos: dict[str, int] = {}
    for name in session.motor_names:
        if _is_gripper(name):
            closed = prompt_capture(session, f"Capture gripper CLOSED position ({name})", "Move the gripper fully closed.")
            opened = prompt_capture(session, f"Capture gripper OPEN position ({name})", "Move the gripper fully open.")
            min_pos[name] = int(closed[name])
            max_pos[name] = int(opened[name])
        else:
            minimum = prompt_capture(
                session,
                f"Capture MIN position for {name}",
                f"Move only '{name}' to its minimum safe mechanical position. Do not force hard stops.",
            )
            maximum = prompt_capture(
                session,
                f"Capture MAX position for {name}",
                f"Move only '{name}' to its maximum safe mechanical position. Do not force hard stops.",
            )
            min_pos[name] = int(minimum[name])
            max_pos[name] = int(maximum[name])
    return min_pos, max_pos


def build_calibration_payload(
    session: DirectCalibrationSession,
    neutral: dict[str, int],
    min_pos: dict[str, int],
    max_pos: dict[str, int],
    motor_ids: list[int] | None,
) -> dict[str, Any]:
    drive_mode = infer_drive_mode(neutral, max_pos)
    homing_offset = infer_homing_offset(neutral, drive_mode)
    calib_mode = ["LINEAR" if _is_gripper(name) else "DEGREE" for name in session.motor_names]
    ids = list(motor_ids) if motor_ids is not None else list(session.motor_ids)
    payload: dict[str, Any] = {
        "created_at_unix": time.time(),
        "motor_names": list(session.motor_names),
        "motor_ids": list(map(int, ids)),
        "motor_baudrates": list(map(int, session.motor_baudrates)),
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
            "io_path": "Captured using direct Feetech packet-level communication.",
        },
    }
    for idx, name in enumerate(session.motor_names):
        raw_min = int(min_pos[name])
        raw_max = int(max_pos[name])
        range_min = min(raw_min, raw_max)
        range_max = max(raw_min, raw_max)
        payload[name] = {
            "name": name,
            "id": int(ids[idx]),
            "baudrate": int(session.motor_baudrates[idx]),
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


def write_outputs(payload: dict[str, Any]) -> list[Path]:
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    output_paths = _get_output_json_paths()
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["Robot joint calibration summary", "=" * 40, "JSON files:"]
    for path in output_paths:
        lines.append(f"  {path}")
    lines.append("")
    for i, name in enumerate(payload["motor_names"]):
        lines += [
            f"{name}:",
            f"  id            = {payload['motor_ids'][i]}",
            f"  baudrate      = {payload.get('motor_baudrates', ['?'] * len(payload['motor_names']))[i]}",
            f"  neutral_pos   = {payload['neutral_pos'][i]}",
            f"  min_pos       = {payload['min_pos'][i]}",
            f"  max_pos       = {payload['max_pos'][i]}",
            f"  homing_offset = {payload['homing_offset'][i]}",
            f"  drive_mode    = {payload['drive_mode'][i]}",
            "",
        ]
    TXT_PATH.write_text("\n".join(lines) + "\n")
    return output_paths


def _best_effort_identify(session: DirectCalibrationSession) -> None:
    _print_header("Identify / verify motors")
    for name, sid in zip(session.motor_names, session.motor_ids, strict=True):
        preferred = [session.baud_for_id(sid)]
        baud = _direct_detect_baud_for_id(session.port, sid, _get_scan_baudrates(preferred))
        if baud is not None:
            session.set_baud_for_id(sid, baud)
        model = _direct_read_model(session.port, sid, session.baud_for_id(sid)) if baud is not None else None
        pos, pos_baud = _direct_read_position_any_baud(session.port, sid, _baud_candidates_for_session(session, sid))
        if pos is not None:
            session.set_baud_for_id(sid, pos_baud)
        print(f"  {name:16s} id={sid:3d} baud={session.baud_for_id(sid):7d} model={str(model):>5s} position={str(pos):>6s}")




# ---------------------------------------------------------------------------
# Robust direct-chain overrides
# ---------------------------------------------------------------------------
# These definitions intentionally replace the earlier connect/read helpers above.
# The motor setup phase already proves the configured IDs were assigned one at a
# time.  Full-chain scans can still miss later motors intermittently on long
# daisy chains, so calibration must not abort simply because the pre-check misses
# ID 7/8 once.  Instead, setup metadata is used as the authoritative motor list,
# and individual motor reads are retried heavily during each capture.

TRUST_SETUP_METADATA_FOR_CHAIN = bool(getattr(val, "REAL_ROBOT_TRUST_SETUP_METADATA", False))
CHAIN_VERIFY_EXTRA_RETRIES = int(getattr(val, "FEETECH_CHAIN_VERIFY_EXTRA_RETRIES", 8))
CHAIN_READ_EXTRA_RETRIES = int(getattr(val, "FEETECH_CHAIN_READ_EXTRA_RETRIES", 12))
CHAIN_INTER_MOTOR_DELAY_S = float(getattr(val, "FEETECH_CHAIN_INTER_MOTOR_DELAY_S", 0.035))


def _direct_ping_reliable(port: str, servo_id: int, baudrate: int, retries: int | None = None) -> bool:
    """Ping/read a servo repeatedly.  Present_Position counts as proof of life.

    Some Feetech adapters drop ping status packets but still answer register reads,
    especially on longer daisy chains.  This helper deliberately treats a valid
    position read as stronger evidence than a ping response.
    """
    attempts = max(1, int(retries if retries is not None else CHAIN_VERIFY_EXTRA_RETRIES))
    for _ in range(attempts):
        if _direct_read_position(port, int(servo_id), int(baudrate)) is not None:
            return True
        if _direct_ping(port, int(servo_id), int(baudrate)):
            return True
        time.sleep(CHAIN_INTER_MOTOR_DELAY_S)
    return False


def _direct_read_position_reliable(
    port: str,
    servo_id: int,
    preferred_bauds: Sequence[int] | None = None,
) -> tuple[int | None, int]:
    """Read Present_Position with many retries and baud fallback."""
    bauds = _get_scan_baudrates(preferred_bauds)
    fallback_baud = int(bauds[0]) if bauds else DEFAULT_BAUDRATE
    for baud in bauds:
        for _ in range(max(1, CHAIN_READ_EXTRA_RETRIES)):
            pos = _direct_read_position(port, int(servo_id), int(baud))
            if pos is not None:
                return int(pos), int(baud)
            # If ping works but the read was dropped, immediately try one more read.
            if _direct_ping_once(port, int(servo_id), int(baud)):
                pos = _direct_read_position(port, int(servo_id), int(baud))
                if pos is not None:
                    return int(pos), int(baud)
            time.sleep(CHAIN_INTER_MOTOR_DELAY_S)
    return None, fallback_baud


def _detect_chain_baudrates(
    port: str,
    motor_ids: Sequence[int],
    setup_baudrates: Sequence[int] | None = None,
) -> tuple[dict[int, int], dict[int, int | None]]:
    """Detect each configured motor independently with robust retries.

    This replaces the earlier detector.  The old detector could mark later motors
    missing because it only accepted a quick ping/read result during a pre-check.
    Here, each motor gets its own preferred baud list and repeated position reads.
    """
    found_baud: dict[int, int] = {}
    found_model: dict[int, int | None] = {}
    setup_map: dict[int, int] = {}
    if setup_baudrates and len(setup_baudrates) == len(motor_ids):
        setup_map = {int(mid): int(baud) for mid, baud in zip(motor_ids, setup_baudrates, strict=True)}

    for sid in map(int, motor_ids):
        preferred = [setup_map[sid]] if sid in setup_map else []
        pos, baud = _direct_read_position_reliable(port, sid, preferred)
        if pos is not None:
            found_baud[sid] = int(baud)
            found_model[sid] = _direct_read_model(port, sid, int(baud))
            continue
        baud = _direct_detect_baud_for_id(port, sid, preferred)
        if baud is not None:
            found_baud[sid] = int(baud)
            found_model[sid] = _direct_read_model(port, sid, int(baud))
    return found_baud, found_model


def connect_session() -> DirectCalibrationSession:
    """Create a direct calibration session for the configured motor chain.

    Motor setup is performed one servo at a time, so setup metadata only proves
    that each ID was assigned individually. Calibration must verify that the
    assembled daisy chain can read every configured ID before continuing. This
    prevents the live table from silently starting with motors 7/8 unreachable.
    """
    ports = _candidate_robot_ports()
    if not ports:
        raise RuntimeError("Could not auto-detect the robot serial port. Set values.REAL_ROBOT_PORT manually.")

    setup_status = get_motor_setup_status()
    motor_names = list(setup_status.motor_names if setup_status.configured else _get_configured_motor_names())
    motor_ids = list(setup_status.motor_ids if setup_status.configured and setup_status.motor_ids else _get_configured_motor_ids(motor_names))
    model_numbers = _get_configured_motor_model_numbers(motor_names)

    if setup_status.configured and setup_status.motor_baudrates and len(setup_status.motor_baudrates) == len(motor_ids):
        motor_baudrates = [int(x) for x in setup_status.motor_baudrates]
    else:
        motor_baudrates = [DEFAULT_BAUDRATE] * len(motor_ids)

    # Prefer the configured / first detected port, then verify the assembled chain
    # unless the user explicitly opts back into trusting setup metadata.
    port = str(ports[0])
    print(f"[calibrate] using direct port = {port}")
    if setup_status.configured and TRUST_SETUP_METADATA_FOR_CHAIN:
        print(
            "[calibrate] WARNING: values.REAL_ROBOT_TRUST_SETUP_METADATA is True, so the saved setup file\n"
            "            is being used without full-chain verification. This is not recommended for\n"
            "            debugging motors 7/8; set it to False to prove the daisy chain is readable."
        )
        for name, sid, baud in zip(motor_names, motor_ids, motor_baudrates, strict=True):
            print(f"  {name:16s} id={int(sid):3d} baud={int(baud)}")
        return DirectCalibrationSession(port, list(motor_names), list(motor_ids), list(model_numbers), list(motor_baudrates))

    # If there is no setup metadata, do a bounded diagnostic scan rather than an
    # unbounded-feeling pre-check. This path should normally only be reached if the
    # user bypassed setup.
    print("[calibrate] No trusted setup metadata found; running bounded direct chain check...")
    found_baud, found_models = _detect_chain_baudrates(port, motor_ids, motor_baudrates)
    missing = [int(sid) for sid in motor_ids if int(sid) not in found_baud]
    if missing:
        expected = {int(mid): int(model) for mid, model in zip(motor_ids, model_numbers, strict=True)}
        print("\n[calibrate] Direct chain check did not see every configured motor.")
        print("Full expected motor list (id: model_number):")
        print(expected)
        print("\nFull found motor list (id: model_number):")
        print(found_models)
        raise RuntimeError("Not all configured motors responded through direct Feetech packets.")

    bauds = [int(found_baud[int(mid)]) for mid in motor_ids]
    print("[calibrate] Direct chain verification succeeded:")
    for name, sid, baud in zip(motor_names, motor_ids, bauds, strict=True):
        print(f"  {name:16s} id={sid:3d} baud={baud}")
    return DirectCalibrationSession(port, list(motor_names), list(motor_ids), list(model_numbers), bauds)


def read_positions(session: DirectCalibrationSession) -> dict[str, int]:
    """Read all motor positions with setup-baud hints and robust retries."""
    positions: dict[str, int] = {}
    failures: list[str] = []
    for name, sid in zip(session.motor_names, session.motor_ids, strict=True):
        preferred = [session.baud_for_id(sid)]
        pos, baud = _direct_read_position_reliable(session.port, int(sid), preferred)
        if pos is None:
            failures.append(f"{name}/ID{sid}")
            continue
        session.set_baud_for_id(sid, baud)
        positions[name] = int(pos)
    if failures:
        raise RuntimeError(
            "Could not read Present_Position from: "
            + ", ".join(failures)
            + ". These IDs were not readable after repeated direct-packet attempts. "
              "Check the data path through the daisy chain after the last readable motor."
        )
    return positions


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
        "This stage uses direct Feetech packet communication. It records a neutral pose and then\n"
        "captures per-joint minimum and maximum positions for the current hardware setup."
    )
    try:
        session = connect_session()
    except Exception as exc:
        print(f"[calibrate] Failed to connect to robot through direct bus: {exc}")
        return 1
    print(f"[calibrate] Connected. Motor names: {session.motor_names}")
    try:
        set_torque(session, enabled=False)
        neutral = prompt_capture(
            session,
            "Capture NEUTRAL pose",
            "Move the arm into your desired neutral/zero pose. Center all wrist joints and set gripper neutral.",
        )
        min_pos, max_pos = capture_joint_limits(session)
        payload = build_calibration_payload(session, neutral, min_pos, max_pos, session.motor_ids)
        output_paths = write_outputs(payload)
        _print_header("Calibration complete")
        print("Saved JSON calibration to:")
        for path in output_paths:
            print(f"  {path}")
        print(f"Saved text summary to:     {TXT_PATH}")
        return 0
    finally:
        try:
            set_torque(session, enabled=True)
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
            print(f"[calibrate] Failed to connect to robot through direct bus: {exc}")
            return 1
        _best_effort_identify(session)
        return 0
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



# ---------------------------------------------------------------------------
# LeRobot-style min/max live table calibration overrides for 8-motor arm
# ---------------------------------------------------------------------------
# These overrides preserve the setup + direct packet workflow above, but change
# joint calibration to match the LeRobot-style calibration behavior: capture a
# neutral pose once, then continuously read all motors while the user moves the
# robot through its safe range. A live table tracks current/min/max for all 8
# motors until Enter is pressed.

FAST_CAPTURE_READ_RETRIES = int(getattr(val, "FEETECH_CAPTURE_READ_RETRIES", 3))
FAST_CAPTURE_INTER_RETRY_S = float(getattr(val, "FEETECH_CAPTURE_INTER_RETRY_S", 0.015))
LIVE_TABLE_PERIOD_S = float(getattr(val, "FEETECH_LIVE_TABLE_PERIOD_S", 0.5))
LIVE_TABLE_MAX_SECONDS = float(getattr(val, "FEETECH_LIVE_TABLE_MAX_SECONDS", 0.0))  # 0 = no limit


def _baud_candidates_for_session(session: DirectCalibrationSession, servo_id: int) -> list[int]:
    """Return the complete baud search order for a live capture read.

    The setup stage may find a motor at a non-default baudrate, and older setup
    files may not contain per-motor baudrates at all. The previous live-table
    code only tried the session baud, DEFAULT_BAUDRATE, and 1 Mbps, which could
    make motors 7/8 appear dead even though the setup scanner could see them.
    Use the same full scan list everywhere, with the current per-ID baud first.
    """
    return _get_scan_baudrates([session.baud_for_id(int(servo_id)), DEFAULT_BAUDRATE, 1000000])


def _read_position_fast(session: DirectCalibrationSession, servo_id: int) -> tuple[int | None, int]:
    """Read one Present_Position quickly with a hard bounded retry count."""
    fallback = session.baud_for_id(int(servo_id))
    for baud in _baud_candidates_for_session(session, int(servo_id)):
        for _ in range(max(1, FAST_CAPTURE_READ_RETRIES)):
            pos = _direct_read_position(session.port, int(servo_id), int(baud))
            if pos is not None:
                return int(pos), int(baud)
            time.sleep(FAST_CAPTURE_INTER_RETRY_S)
    return None, int(fallback)


def read_positions(session: DirectCalibrationSession) -> dict[str, int]:
    """Read all motor positions with progress and bounded per-motor time."""
    positions: dict[str, int] = {}
    failures: list[str] = []
    print("[calibrate] Reading current positions from all configured motors...", flush=True)
    for idx, (name, sid) in enumerate(zip(session.motor_names, session.motor_ids, strict=True), start=1):
        print(f"[calibrate]   ({idx}/{len(session.motor_ids)}) reading {name}/ID{sid} at baud {session.baud_for_id(sid)}...", flush=True)
        pos, baud = _read_position_fast(session, int(sid))
        if pos is None:
            print(f"[calibrate]        FAILED: no Present_Position from {name}/ID{sid}", flush=True)
            failures.append(f"{name}/ID{sid}")
            continue
        session.set_baud_for_id(int(sid), int(baud))
        positions[str(name)] = int(pos)
        print(f"[calibrate]        OK: {pos} (baud {baud})", flush=True)
    if failures:
        raise RuntimeError(
            "Could not read Present_Position from: "
            + ", ".join(failures)
            + ". Calibration cannot continue until these IDs return position data. Power/red LED alone does not prove the serial data line is readable."
        )
    return positions


def _read_positions_partial(session: DirectCalibrationSession) -> tuple[dict[str, int], list[str]]:
    positions: dict[str, int] = {}
    failures: list[str] = []
    for name, sid in zip(session.motor_names, session.motor_ids, strict=True):
        pos, baud = _read_position_fast(session, int(sid))
        if pos is None:
            failures.append(f"{name}/ID{sid}")
            continue
        session.set_baud_for_id(int(sid), int(baud))
        positions[str(name)] = int(pos)
    return positions, failures


def _format_live_table(session: DirectCalibrationSession, current: dict[str, int | None], min_pos: dict[str, int | None], max_pos: dict[str, int | None], failures: Sequence[str], elapsed_s: float) -> str:
    lines: list[str] = []
    lines.append("\nLeRobot-style live range capture")
    lines.append("Move every joint through its full SAFE range. Press Enter when done.")
    lines.append(f"Elapsed: {elapsed_s:6.1f}s")
    lines.append("-" * 82)
    lines.append(f"{'motor':16s} {'id':>3s} {'baud':>8s} {'current':>9s} {'min':>9s} {'max':>9s} {'span':>9s}")
    lines.append("-" * 82)
    for name, sid in zip(session.motor_names, session.motor_ids, strict=True):
        cur = current.get(name)
        mn = min_pos.get(name)
        mx = max_pos.get(name)
        span = None if mn is None or mx is None else int(mx) - int(mn)
        def fmt(v: int | None) -> str:
            return "----" if v is None else str(int(v))
        lines.append(f"{name:16s} {int(sid):3d} {session.baud_for_id(int(sid)):8d} {fmt(cur):>9s} {fmt(mn):>9s} {fmt(mx):>9s} {fmt(span):>9s}")
    if failures:
        lines.append("-" * 82)
        lines.append("Unreadable this cycle: " + ", ".join(failures))
    lines.append("-" * 82)
    return "\n".join(lines)


def _wait_for_enter_event(prompt: str = "Press Enter when finished moving through all safe ranges... ") -> Any:
    import threading
    done = threading.Event()
    def _reader() -> None:
        try:
            input(prompt)
        except Exception:
            pass
        done.set()
    threading.Thread(target=_reader, daemon=True).start()
    return done


def prompt_capture(session: DirectCalibrationSession, title: str, instructions: str) -> dict[str, int]:
    _print_header(title)
    print(instructions)
    print("[calibrate] Waiting for you to physically move the arm, then press Enter.")
    input("READY TO CAPTURE? Press Enter now... ")
    return read_positions(session)


def capture_joint_limits(session: DirectCalibrationSession) -> tuple[dict[str, int], dict[str, int]]:
    """LeRobot-style min/max calibration with a live min/max/current table."""
    _print_header("Capture MIN/MAX ranges for all 8 motors")
    print("Move the arm slowly through the full SAFE motion range for every joint.\nInclude the gripper fully open and fully closed. Do not force hard stops.\nThe table below tracks current, min, and max positions like LeRobot calibration.")
    current: dict[str, int | None] = {name: None for name in session.motor_names}
    min_pos_nullable: dict[str, int | None] = {name: None for name in session.motor_names}
    max_pos_nullable: dict[str, int | None] = {name: None for name in session.motor_names}
    done = _wait_for_enter_event()
    start = time.monotonic()
    last_print = 0.0
    while not done.is_set():
        positions, failures = _read_positions_partial(session)
        for name, value in positions.items():
            current[name] = int(value)
            old_min = min_pos_nullable[name]
            old_max = max_pos_nullable[name]
            min_pos_nullable[name] = int(value) if old_min is None else min(int(old_min), int(value))
            max_pos_nullable[name] = int(value) if old_max is None else max(int(old_max), int(value))
        now = time.monotonic()
        elapsed = now - start
        if now - last_print >= LIVE_TABLE_PERIOD_S:
            print(_format_live_table(session, current, min_pos_nullable, max_pos_nullable, failures, elapsed), flush=True)
            last_print = now
        if LIVE_TABLE_MAX_SECONDS > 0 and elapsed >= LIVE_TABLE_MAX_SECONDS:
            print("[calibrate] Live table time limit reached; finishing range capture.")
            break
        time.sleep(0.02)
    positions, failures = _read_positions_partial(session)
    for name, value in positions.items():
        current[name] = int(value)
        old_min = min_pos_nullable[name]
        old_max = max_pos_nullable[name]
        min_pos_nullable[name] = int(value) if old_min is None else min(int(old_min), int(value))
        max_pos_nullable[name] = int(value) if old_max is None else max(int(old_max), int(value))
    missing = [name for name in session.motor_names if min_pos_nullable[name] is None or max_pos_nullable[name] is None]
    if missing:
        raise RuntimeError("No usable min/max data was captured for: " + ", ".join(missing) + ". Calibration cannot be saved until every configured motor returns Present_Position at least once.")
    min_pos = {name: int(min_pos_nullable[name]) for name in session.motor_names}
    max_pos = {name: int(max_pos_nullable[name]) for name in session.motor_names}
    print(_format_live_table(session, current, min_pos, max_pos, failures, time.monotonic() - start), flush=True)
    print("[calibrate] Min/max range capture complete.")
    return min_pos, max_pos


def _best_effort_identify(session: DirectCalibrationSession) -> None:
    _print_header("Identify / verify motors")
    print("This reads each motor independently with bounded timeout so identify does not hang forever.")
    for name, sid in zip(session.motor_names, session.motor_ids, strict=True):
        pos, baud = _read_position_fast(session, int(sid))
        if pos is not None:
            session.set_baud_for_id(int(sid), int(baud))
        model = _direct_read_model(session.port, int(sid), session.baud_for_id(int(sid)))
        model_display = model if model is not None else session.model_numbers[session.motor_ids.index(sid)]
        print(f"  {name:16s} id={int(sid):3d} baud={session.baud_for_id(int(sid)):7d} model={str(model_display):>5s} position={str(pos):>6s}", flush=True)



# ---------------------------------------------------------------------------
# Final robust live-table calibration overrides
# ---------------------------------------------------------------------------
# These overrides fix the remaining failure mode where some motors can accept
# writes/torque but intermittent direct reads cause calibration to abort.  The
# read path now keeps one serial connection open during each capture cycle, tries
# multiple Present_Position register addresses, and records which address works
# per motor.  This better matches the behavior needed for an 8-motor daisy chain.

POSITION_ADDR_CANDIDATES = [
    int(x) for x in getattr(
        val,
        "FEETECH_PRESENT_POSITION_ADDR_CANDIDATES",
        [FEETECH_PRESENT_POSITION_ADDR, 56, 48, 36],
    )
]
POSITION_ADDR_CANDIDATES = list(dict.fromkeys(POSITION_ADDR_CANDIDATES))
LIVE_REQUIRED_FIRST_CAPTURE = bool(getattr(val, "FEETECH_REQUIRE_ALL_MOTORS_FOR_FIRST_CAPTURE", True))
LIVE_CAPTURE_MAX_MISSED_CYCLES = int(getattr(val, "FEETECH_LIVE_CAPTURE_MAX_MISSED_CYCLES", 30))


def _read_status_from_open_serial(ser: Any, timeout_s: float = PACKET_TIMEOUT_S) -> tuple[int, int, list[int]] | None:
    """Read a Feetech status packet from an already-open serial object."""
    return _read_status(ser, timeout_s=timeout_s)


def _direct_read_register_open(
    ser: Any,
    servo_id: int,
    addr: int,
    length: int,
    timeout_s: float = PACKET_TIMEOUT_S,
) -> list[int] | None:
    """Read a register using an already-open serial object."""
    try:
        try:
            ser.reset_input_buffer()
        except Exception:
            pass
        ser.write(_feetech_packet(int(servo_id), 0x02, (int(addr), int(length))))
        ser.flush()
        status = _read_status_from_open_serial(ser, timeout_s=timeout_s)
        if status is None or int(status[0]) != int(servo_id) or int(status[1]) != 0:
            return None
        params = list(status[2])
        if len(params) < int(length):
            return None
        return params[: int(length)]
    except Exception:
        return None


def _direct_read_u16_open(ser: Any, servo_id: int, addr: int) -> int | None:
    data = _direct_read_register_open(ser, int(servo_id), int(addr), 2)
    if data is None or len(data) < 2:
        return None
    return int(data[0]) | (int(data[1]) << 8)


def _read_position_with_addr_candidates_open(
    ser: Any,
    servo_id: int,
    addr_candidates: Sequence[int] | None = None,
) -> tuple[int | None, int | None]:
    for addr in list(addr_candidates or POSITION_ADDR_CANDIDATES):
        value = _direct_read_u16_open(ser, int(servo_id), int(addr))
        if value is not None:
            return int(value), int(addr)
    return None, None


def _read_position_with_addr_candidates(
    port: str,
    servo_id: int,
    baudrate: int,
    addr_candidates: Sequence[int] | None = None,
) -> tuple[int | None, int | None]:
    try:
        with _open_serial(str(port), int(baudrate), timeout=0.06) as ser:
            return _read_position_with_addr_candidates_open(ser, int(servo_id), addr_candidates)
    except Exception:
        return None, None


def _read_position_fast(session: DirectCalibrationSession, servo_id: int) -> tuple[int | None, int]:
    """Read one motor with a bounded retry count, using known and fallback position addresses."""
    fallback_baud = session.baud_for_id(int(servo_id))
    known_addr = getattr(session, "_position_addr_by_id", {}).get(int(servo_id)) if hasattr(session, "_position_addr_by_id") else None
    addr_candidates = ([known_addr] if known_addr is not None else []) + POSITION_ADDR_CANDIDATES
    addr_candidates = list(dict.fromkeys(int(a) for a in addr_candidates if a is not None))

    for baud in _baud_candidates_for_session(session, int(servo_id)):
        for _ in range(max(1, FAST_CAPTURE_READ_RETRIES)):
            pos, addr = _read_position_with_addr_candidates(session.port, int(servo_id), int(baud), addr_candidates)
            if pos is not None:
                if not hasattr(session, "_position_addr_by_id"):
                    setattr(session, "_position_addr_by_id", {})
                session._position_addr_by_id[int(servo_id)] = int(addr)
                return int(pos), int(baud)
            time.sleep(FAST_CAPTURE_INTER_RETRY_S)
    return None, int(fallback_baud)


def _read_all_positions_one_baud_open(
    session: DirectCalibrationSession,
    baudrate: int,
    include_names: Sequence[str] | None = None,
) -> tuple[dict[str, int], dict[int, int], list[str]]:
    """Read selected motors using one persistent serial connection at one baudrate."""
    selected = set(include_names) if include_names is not None else set(session.motor_names)
    positions: dict[str, int] = {}
    addr_hits: dict[int, int] = {}
    failures: list[str] = []

    try:
        with _open_serial(session.port, int(baudrate), timeout=0.06) as ser:
            for name, sid in zip(session.motor_names, session.motor_ids, strict=True):
                if name not in selected:
                    continue
                known_addr = getattr(session, "_position_addr_by_id", {}).get(int(sid)) if hasattr(session, "_position_addr_by_id") else None
                addr_candidates = ([known_addr] if known_addr is not None else []) + POSITION_ADDR_CANDIDATES
                addr_candidates = list(dict.fromkeys(int(a) for a in addr_candidates if a is not None))
                pos = None
                addr = None
                for _ in range(max(1, FAST_CAPTURE_READ_RETRIES)):
                    pos, addr = _read_position_with_addr_candidates_open(ser, int(sid), addr_candidates)
                    if pos is not None:
                        break
                    time.sleep(FAST_CAPTURE_INTER_RETRY_S)
                if pos is None:
                    failures.append(f"{name}/ID{sid}")
                    continue
                positions[name] = int(pos)
                addr_hits[int(sid)] = int(addr)
    except Exception:
        failures = [f"{name}/ID{sid}" for name, sid in zip(session.motor_names, session.motor_ids, strict=True) if name in selected]
    return positions, addr_hits, failures


def _read_positions_partial(session: DirectCalibrationSession) -> tuple[dict[str, int], list[str]]:
    """Read all motors for live table without aborting on partial failures."""
    positions: dict[str, int] = {}
    failures: list[str] = []

    # First try the common/session baud in one persistent serial connection. This
    # avoids repeatedly opening/closing the port and is much more reliable on long chains.
    primary_baud = session.baudrate
    primary_positions, addr_hits, primary_failures = _read_all_positions_one_baud_open(session, primary_baud)
    for name, pos in primary_positions.items():
        idx = session.motor_names.index(name)
        sid = int(session.motor_ids[idx])
        session.set_baud_for_id(sid, int(primary_baud))
        if not hasattr(session, "_position_addr_by_id"):
            setattr(session, "_position_addr_by_id", {})
        if sid in addr_hits:
            session._position_addr_by_id[sid] = int(addr_hits[sid])
        positions[name] = int(pos)

    # Individually retry only the motors that failed the persistent-baud pass.
    failed_names = []
    for item in primary_failures:
        failed_names.append(item.split("/ID", 1)[0])

    for name in failed_names:
        idx = session.motor_names.index(name)
        sid = int(session.motor_ids[idx])
        pos, baud = _read_position_fast(session, sid)
        if pos is None:
            failures.append(f"{name}/ID{sid}")
            continue
        session.set_baud_for_id(sid, int(baud))
        positions[name] = int(pos)

    return positions, failures


def read_positions(session: DirectCalibrationSession) -> dict[str, int]:
    """Read all motor positions, printing progress, but using the robust persistent read path."""
    print("[calibrate] Reading current positions from all configured motors...", flush=True)
    positions, failures = _read_positions_partial(session)
    for idx, (name, sid) in enumerate(zip(session.motor_names, session.motor_ids, strict=True), start=1):
        if name in positions:
            addr = getattr(session, "_position_addr_by_id", {}).get(int(sid), FEETECH_PRESENT_POSITION_ADDR)
            print(
                f"[calibrate]   ({idx}/{len(session.motor_ids)}) {name}/ID{sid}: "
                f"{positions[name]}  baud={session.baud_for_id(int(sid))}  pos_addr={addr}",
                flush=True,
            )
        else:
            print(f"[calibrate]   ({idx}/{len(session.motor_ids)}) {name}/ID{sid}: FAILED", flush=True)

    if failures:
        raise RuntimeError(
            "Could not read Present_Position from: "
            + ", ".join(failures)
            + ". Calibration cannot continue until every configured motor returns position data. "
              "If these motors lock but do not report position, verify their data-line continuity and/or set "
              "FEETECH_PRESENT_POSITION_ADDR_CANDIDATES in values.py."
        )
    return positions


def _format_live_table(
    session: DirectCalibrationSession,
    current: dict[str, int | None],
    min_pos: dict[str, int | None],
    max_pos: dict[str, int | None],
    failures: Sequence[str],
    elapsed_s: float,
) -> str:
    lines: list[str] = []
    lines.append("\nLeRobot-style range table")
    lines.append("Move every joint through its full SAFE range. Press Enter when done.")
    lines.append(f"Elapsed: {elapsed_s:6.1f}s")
    lines.append("+" + "-" * 18 + "+" + "-" * 6 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 10 + "+")
    lines.append(f"| {'motor':16s} | {'id':>4s} | {'current':>8s} | {'min':>8s} | {'max':>8s} | {'span':>8s} |")
    lines.append("+" + "-" * 18 + "+" + "-" * 6 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 10 + "+")
    for name, sid in zip(session.motor_names, session.motor_ids, strict=True):
        cur = current.get(name)
        mn = min_pos.get(name)
        mx = max_pos.get(name)
        span = None if mn is None or mx is None else int(mx) - int(mn)

        def fmt(v: int | None) -> str:
            return "----" if v is None else str(int(v))

        lines.append(
            f"| {name:16s} | {int(sid):4d} | {fmt(cur):>8s} | {fmt(mn):>8s} | {fmt(mx):>8s} | {fmt(span):>8s} |"
        )
    lines.append("+" + "-" * 18 + "+" + "-" * 6 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 10 + "+")
    if failures:
        lines.append("Unreadable this cycle: " + ", ".join(failures))
    return "\n".join(lines)


def prompt_capture(session: DirectCalibrationSession, title: str, instructions: str) -> dict[str, int]:
    _print_header(title)
    print(instructions)
    print("[calibrate] Waiting for you to physically move the arm, then press Enter.")
    input("READY TO CAPTURE? Press Enter now... ")
    return read_positions(session)


def capture_joint_limits(session: DirectCalibrationSession) -> tuple[dict[str, int], dict[str, int]]:
    """LeRobot-style min/max calibration with a vertical current/min/max table."""
    _print_header("Capture MIN/MAX ranges for all 8 motors")
    print(
        "Move the arm slowly through the full SAFE motion range for every joint.\n"
        "Include the gripper fully open and fully closed. Do not force hard stops.\n"
        "The table below tracks current, min, and max positions like LeRobot calibration."
    )
    current: dict[str, int | None] = {name: None for name in session.motor_names}
    min_pos_nullable: dict[str, int | None] = {name: None for name in session.motor_names}
    max_pos_nullable: dict[str, int | None] = {name: None for name in session.motor_names}
    missed_cycles: dict[str, int] = {name: 0 for name in session.motor_names}

    done = _wait_for_enter_event()
    start = time.monotonic()
    last_print = 0.0
    while not done.is_set():
        positions, failures = _read_positions_partial(session)
        failed_names = {item.split("/ID", 1)[0] for item in failures}
        for name in session.motor_names:
            if name in positions:
                value = int(positions[name])
                current[name] = value
                min_pos_nullable[name] = value if min_pos_nullable[name] is None else min(int(min_pos_nullable[name]), value)
                max_pos_nullable[name] = value if max_pos_nullable[name] is None else max(int(max_pos_nullable[name]), value)
                missed_cycles[name] = 0
            elif name in failed_names:
                missed_cycles[name] += 1

        now = time.monotonic()
        elapsed = now - start
        if now - last_print >= LIVE_TABLE_PERIOD_S:
            print(_format_live_table(session, current, min_pos_nullable, max_pos_nullable, failures, elapsed), flush=True)
            last_print = now

        if any(count >= LIVE_CAPTURE_MAX_MISSED_CYCLES for count in missed_cycles.values() if LIVE_CAPTURE_MAX_MISSED_CYCLES > 0):
            bad = [name for name, count in missed_cycles.items() if count >= LIVE_CAPTURE_MAX_MISSED_CYCLES]
            raise RuntimeError("Repeatedly failed to read during live range capture: " + ", ".join(bad))

        if LIVE_TABLE_MAX_SECONDS > 0 and elapsed >= LIVE_TABLE_MAX_SECONDS:
            print("[calibrate] Live table time limit reached; finishing range capture.")
            break
        time.sleep(0.02)

    # Final update pass.
    positions, failures = _read_positions_partial(session)
    for name, value in positions.items():
        value = int(value)
        current[name] = value
        min_pos_nullable[name] = value if min_pos_nullable[name] is None else min(int(min_pos_nullable[name]), value)
        max_pos_nullable[name] = value if max_pos_nullable[name] is None else max(int(max_pos_nullable[name]), value)

    missing = [name for name in session.motor_names if min_pos_nullable[name] is None or max_pos_nullable[name] is None]
    if missing:
        raise RuntimeError(
            "No usable min/max data was captured for: "
            + ", ".join(missing)
            + ". Each motor must return Present_Position at least once before calibration can be saved."
        )

    min_pos = {name: int(min_pos_nullable[name]) for name in session.motor_names}
    max_pos = {name: int(max_pos_nullable[name]) for name in session.motor_names}
    print(_format_live_table(session, current, min_pos, max_pos, failures, time.monotonic() - start), flush=True)
    print("[calibrate] Min/max range capture complete.")
    return min_pos, max_pos


def _best_effort_identify(session: DirectCalibrationSession) -> None:
    _print_header("Identify / verify motors")
    print("Vertical verification table. A position of ---- means that ID did not return Present_Position.")
    positions, failures = _read_positions_partial(session)
    print("+" + "-" * 18 + "+" + "-" * 6 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 12 + "+")
    print(f"| {'motor':16s} | {'id':>4s} | {'baud':>8s} | {'position':>8s} | {'model':>10s} |")
    print("+" + "-" * 18 + "+" + "-" * 6 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 12 + "+")
    for name, sid in zip(session.motor_names, session.motor_ids, strict=True):
        model = _direct_read_model(session.port, int(sid), session.baud_for_id(int(sid)))
        if model is None:
            model = session.model_numbers[session.motor_ids.index(sid)]
        pos_txt = "----" if name not in positions else str(int(positions[name]))
        print(f"| {name:16s} | {int(sid):4d} | {session.baud_for_id(int(sid)):8d} | {pos_txt:>8s} | {str(model):>10s} |")
    print("+" + "-" * 18 + "+" + "-" * 6 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 12 + "+")
    if failures:
        print("Unreadable: " + ", ".join(failures))

# ---------------------------------------------------------------------------
# Extra-motor diagnostics + strict readable-position calibration patch
# ---------------------------------------------------------------------------
# The code below is intentionally placed after the earlier helpers so these
# definitions override the previous read/identify functions.  It does not treat
# motors 7 and 8 differently; it proves whether they are actually readable on the
# full daisy chain.  If they were set one-at-a-time but disappear only when the
# full chain is connected, that points to chain data continuity, duplicate IDs,
# baud mismatch, or a different position-register address rather than the old
# SO100/SO101 6-motor configuration.

POSITION_ADDR_CANDIDATES = list(dict.fromkeys([
    int(x) for x in getattr(
        val,
        "FEETECH_PRESENT_POSITION_ADDR_CANDIDATES",
        [FEETECH_PRESENT_POSITION_ADDR, 56, 48, 36, 60, 58, 52, 50, 46, 44, 42, 40, 38, 34, 32, 30, 28],
    )
]))
RAW_CHAIN_SCAN_MAX_ID = int(getattr(val, "FEETECH_RAW_CHAIN_SCAN_MAX_ID", 20))
EXTRA_DIAGNOSTIC_BAUDS = list(dict.fromkeys(_get_scan_baudrates([DEFAULT_BAUDRATE, 1000000])))


def _read_model_or_none(session: DirectCalibrationSession, sid: int) -> int | None:
    for baud in _baud_candidates_for_session(session, int(sid)):
        model = _direct_read_model(session.port, int(sid), int(baud))
        if model is not None:
            session.set_baud_for_id(int(sid), int(baud))
            return int(model)
    return None


def _raw_scan_chain_ids(port: str, baudrates: Sequence[int] | None = None, max_id: int | None = None) -> dict[int, int]:
    """Return {id: baudrate} for IDs that respond on the full connected chain."""
    found: dict[int, int] = {}
    max_id = int(max_id if max_id is not None else RAW_CHAIN_SCAN_MAX_ID)
    for baud in list(baudrates or EXTRA_DIAGNOSTIC_BAUDS):
        for sid in range(1, max_id + 1):
            if sid in found:
                continue
            if _direct_ping_once(port, sid, int(baud)) or _direct_read_model(port, sid, int(baud)) is not None:
                found[int(sid)] = int(baud)
    return found


def _diagnose_unreadable_ids(session: DirectCalibrationSession, failures: Sequence[str]) -> str:
    failed_ids: list[int] = []
    failed_names: list[str] = []
    for item in failures:
        if "/ID" in item:
            name, sid_s = item.rsplit("/ID", 1)
            try:
                failed_ids.append(int(sid_s))
                failed_names.append(name)
            except Exception:
                pass

    lines: list[str] = []
    lines.append("")
    lines.append("Extra motor diagnostic:")
    lines.append("  The code is configured for all 8 motors; IDs 7 and 8 are not being ignored as old SO100/SO101 extra motors.")
    lines.append("  A red light or a motor feeling locked only proves power/torque state. Calibration requires each servo to return Present_Position data.")
    lines.append("  Running a short raw full-chain scan to separate code/config issues from data-line/ID issues...")

    raw_found = _raw_scan_chain_ids(session.port, max_id=max(max(session.motor_ids), RAW_CHAIN_SCAN_MAX_ID))
    lines.append(f"  Raw responsive IDs on full chain: {sorted(raw_found.keys()) if raw_found else []}")

    for name, sid in zip(failed_names, failed_ids, strict=False):
        lines.append(f"  {name}/ID{sid}:")
        if sid in raw_found:
            lines.append(f"    - ID responds to raw ping/model scan at baud {raw_found[sid]}, so the ID is visible on the data bus.")
        else:
            lines.append("    - ID did NOT respond to raw ping/model scan on the full chain.")
            lines.append("      This is not a min/max calibration bug. Check for a duplicate/mis-set ID, a reversed/damaged cable,")
            lines.append("      or data-line continuity between the last readable motor and this motor. Power/red LED is not enough.")
        model = _read_model_or_none(session, sid)
        if model is None:
            lines.append("    - Model number read: failed")
        else:
            lines.append(f"    - Model number read: {model}")
        address_hits: list[tuple[int, int]] = []
        for baud in _baud_candidates_for_session(session, sid):
            for addr in POSITION_ADDR_CANDIDATES:
                value = _direct_read_u16(session.port, sid, int(addr), int(baud))
                if value is not None:
                    address_hits.append((int(addr), int(value)))
                    break
            if address_hits:
                session.set_baud_for_id(sid, int(baud))
                break
        if address_hits:
            addr, value = address_hits[0]
            lines.append(f"    - Position-like register read succeeded at addr {addr}: {value}")
            lines.append("      Add this address to FEETECH_PRESENT_POSITION_ADDR_CANDIDATES in values.py if it is the correct moving position register.")
        else:
            lines.append("    - No configured Present_Position candidate address returned data.")
    return "\n".join(lines)


def _read_positions_partial(session: DirectCalibrationSession) -> tuple[dict[str, int], list[str]]:
    """Read all motors for live table without aborting; keeps one serial connection open when possible."""
    positions: dict[str, int] = {}
    failures: list[str] = []

    primary_baud = session.baudrate
    primary_positions, addr_hits, primary_failures = _read_all_positions_one_baud_open(session, primary_baud)
    for name, pos in primary_positions.items():
        idx = session.motor_names.index(name)
        sid = int(session.motor_ids[idx])
        session.set_baud_for_id(sid, int(primary_baud))
        if not hasattr(session, "_position_addr_by_id"):
            setattr(session, "_position_addr_by_id", {})
        if sid in addr_hits:
            session._position_addr_by_id[sid] = int(addr_hits[sid])
        positions[name] = int(pos)

    failed_names = [item.split("/ID", 1)[0] for item in primary_failures]

    for name in failed_names:
        idx = session.motor_names.index(name)
        sid = int(session.motor_ids[idx])
        pos, baud = _read_position_fast(session, sid)
        if pos is None:
            failures.append(f"{name}/ID{sid}")
            continue
        session.set_baud_for_id(sid, int(baud))
        positions[name] = int(pos)

    return positions, failures


def read_positions(session: DirectCalibrationSession) -> dict[str, int]:
    """Read all motor positions and run diagnostics if any configured motor is unreadable."""
    print("[calibrate] Reading current positions from all configured motors...", flush=True)
    positions, failures = _read_positions_partial(session)
    for idx, (name, sid) in enumerate(zip(session.motor_names, session.motor_ids, strict=True), start=1):
        if name in positions:
            addr = getattr(session, "_position_addr_by_id", {}).get(int(sid), FEETECH_PRESENT_POSITION_ADDR)
            print(
                f"[calibrate]   ({idx}/{len(session.motor_ids)}) {name}/ID{sid}: "
                f"{positions[name]}  baud={session.baud_for_id(int(sid))}  pos_addr={addr}",
                flush=True,
            )
        else:
            print(f"[calibrate]   ({idx}/{len(session.motor_ids)}) {name}/ID{sid}: FAILED", flush=True)

    if failures:
        diag = _diagnose_unreadable_ids(session, failures)
        raise RuntimeError(
            "Could not read Present_Position from: "
            + ", ".join(failures)
            + ". Calibration cannot continue until every configured motor returns position data."
            + diag
        )
    return positions


def _best_effort_identify(session: DirectCalibrationSession) -> None:
    _print_header("Identify / verify motors")
    print("Vertical verification table. A position of ---- means that ID did not return Present_Position.")
    positions, failures = _read_positions_partial(session)
    raw_found = _raw_scan_chain_ids(session.port, max_id=max(max(session.motor_ids), RAW_CHAIN_SCAN_MAX_ID))
    print(f"Raw responsive IDs on full chain: {sorted(raw_found.keys()) if raw_found else []}")
    print("+" + "-" * 18 + "+" + "-" * 6 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 12 + "+" + "-" * 10 + "+")
    print(f"| {'motor':16s} | {'id':>4s} | {'baud':>8s} | {'position':>8s} | {'model':>10s} | {'rawseen':>8s} |")
    print("+" + "-" * 18 + "+" + "-" * 6 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 12 + "+" + "-" * 10 + "+")
    for name, sid in zip(session.motor_names, session.motor_ids, strict=True):
        model = _direct_read_model(session.port, int(sid), session.baud_for_id(int(sid)))
        if model is None:
            model = session.model_numbers[session.motor_ids.index(sid)]
        pos_txt = "----" if name not in positions else str(int(positions[name]))
        raw_txt = "yes" if int(sid) in raw_found else "no"
        print(f"| {name:16s} | {int(sid):4d} | {session.baud_for_id(int(sid)):8d} | {pos_txt:>8s} | {str(model):>10s} | {raw_txt:>8s} |")
    print("+" + "-" * 18 + "+" + "-" * 6 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 12 + "+" + "-" * 10 + "+")
    if failures:
        print("Unreadable: " + ", ".join(failures))




# ---------------------------------------------------------------------------
# No-freeze connection/capture overrides
# ---------------------------------------------------------------------------
# The previous patched file still performed a long full-chain pre-check when
# REAL_ROBOT_TRUST_SETUP_METADATA was False. With 8 motors, several baudrates,
# many position-register candidates, and repeated serial-open timeouts, that can
# look like the program is frozen at:
#   "No trusted setup metadata found; running bounded direct chain check..."
# These final overrides intentionally make startup non-blocking: setup metadata
# defines the expected 8-motor chain, and actual readability is tested at the
# neutral/range capture steps with bounded, visible per-motor reads.

STARTUP_CHAIN_VERIFY = bool(getattr(val, "FEETECH_STARTUP_CHAIN_VERIFY", False))
CAPTURE_FULL_BAUD_SCAN = bool(getattr(val, "FEETECH_CAPTURE_FULL_BAUD_SCAN", False))

# Capture read address list. Prefer the new explicit capture constant, but also
# honor the older FEETECH_PRESENT_POSITION_ADDR_CANDIDATES name if it exists.
_DEFAULT_CAPTURE_POSITION_ADDR_CANDIDATES = [
    FEETECH_PRESENT_POSITION_ADDR, 56, 58, 60, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28
]
_CAPTURE_ADDR_RAW = getattr(
    val,
    "FEETECH_CAPTURE_POSITION_ADDR_CANDIDATES",
    getattr(val, "FEETECH_PRESENT_POSITION_ADDR_CANDIDATES", _DEFAULT_CAPTURE_POSITION_ADDR_CANDIDATES),
)
CAPTURE_POSITION_ADDR_CANDIDATES = list(dict.fromkeys(int(x) for x in _CAPTURE_ADDR_RAW))
CAPTURE_READ_RETRIES = int(getattr(val, "FEETECH_CAPTURE_READ_RETRIES", 6))
CAPTURE_INTER_RETRY_S = float(getattr(val, "FEETECH_CAPTURE_INTER_RETRY_S", 0.03))


def _setup_paths_message() -> str:
    return f"checked setup paths: {SETUP_JSON_PATH} and {_get_driver_calibration_path()}"


def _baud_candidates_for_session(session: DirectCalibrationSession, servo_id: int) -> list[int]:
    """Short, non-freezing baud search order for capture reads."""
    base = [session.baud_for_id(int(servo_id)), DEFAULT_BAUDRATE, 1000000]
    return _get_scan_baudrates(base) if CAPTURE_FULL_BAUD_SCAN else list(dict.fromkeys(int(b) for b in base if int(b) > 0))


def connect_session() -> DirectCalibrationSession:
    """Create a session without a long blocking startup chain scan."""
    ports = _candidate_robot_ports()
    if not ports:
        raise RuntimeError("Could not auto-detect the robot serial port. Set values.REAL_ROBOT_PORT manually.")

    setup_status = get_motor_setup_status()
    if not setup_status.configured:
        raise RuntimeError(
            "Motor setup metadata was not found or did not match values.REAL_ROBOT_MOTOR_NAMES / "
            "values.REAL_ROBOT_MOTOR_IDS. Run option 2 first, or fix calibration_data/robot_motor_setup.json; "
            + _setup_paths_message()
        )

    motor_names = list(setup_status.motor_names)
    motor_ids = list(setup_status.motor_ids or _get_configured_motor_ids(motor_names))
    model_numbers = _get_configured_motor_model_numbers(motor_names)
    if setup_status.motor_baudrates and len(setup_status.motor_baudrates) == len(motor_ids):
        motor_baudrates = [int(x) for x in setup_status.motor_baudrates]
    else:
        motor_baudrates = [DEFAULT_BAUDRATE] * len(motor_ids)

    port = str(ports[0])
    print(f"[calibrate] using direct port = {port}", flush=True)
    print(f"[calibrate] using setup metadata = {setup_status.source}", flush=True)
    print("[calibrate] startup full-chain scan is disabled to avoid long serial timeouts.", flush=True)
    print("[calibrate] Each motor will be read with bounded per-motor time during capture.", flush=True)
    for name, sid, baud in zip(motor_names, motor_ids, motor_baudrates, strict=True):
        print(f"  {name:16s} id={int(sid):3d} baud={int(baud)}", flush=True)

    session = DirectCalibrationSession(port, list(motor_names), list(motor_ids), list(model_numbers), list(motor_baudrates))

    if STARTUP_CHAIN_VERIFY:
        print("[calibrate] FEETECH_STARTUP_CHAIN_VERIFY=True; running one quick read pass...", flush=True)
        positions, failures = _read_positions_partial(session)
        print(f"[calibrate] quick readable motors: {sorted(positions.keys())}", flush=True)
        if failures:
            print(f"[calibrate] quick unreadable motors: {', '.join(failures)}", flush=True)
    return session


def _read_position_fast(session: DirectCalibrationSession, servo_id: int) -> tuple[int | None, int]:
    """Fast bounded Present_Position read for one motor."""
    fallback_baud = session.baud_for_id(int(servo_id))
    known_addr = getattr(session, "_position_addr_by_id", {}).get(int(servo_id)) if hasattr(session, "_position_addr_by_id") else None
    addr_candidates = ([known_addr] if known_addr is not None else []) + CAPTURE_POSITION_ADDR_CANDIDATES
    addr_candidates = list(dict.fromkeys(int(a) for a in addr_candidates if a is not None))

    for baud in _baud_candidates_for_session(session, int(servo_id)):
        for _ in range(max(1, CAPTURE_READ_RETRIES)):
            pos, addr = _read_position_with_addr_candidates(session.port, int(servo_id), int(baud), addr_candidates)
            if pos is not None:
                if not hasattr(session, "_position_addr_by_id"):
                    setattr(session, "_position_addr_by_id", {})
                session._position_addr_by_id[int(servo_id)] = int(addr)
                return int(pos), int(baud)
            time.sleep(CAPTURE_INTER_RETRY_S)
    return None, int(fallback_baud)


def _read_all_positions_one_baud_open(
    session: DirectCalibrationSession,
    baudrate: int,
    include_names: Sequence[str] | None = None,
) -> tuple[dict[str, int], dict[int, int], list[str]]:
    """Read selected motors using one persistent serial connection at one baudrate."""
    selected = set(include_names) if include_names is not None else set(session.motor_names)
    positions: dict[str, int] = {}
    addr_hits: dict[int, int] = {}
    failures: list[str] = []

    try:
        with _open_serial(session.port, int(baudrate), timeout=0.06) as ser:
            for name, sid in zip(session.motor_names, session.motor_ids, strict=True):
                if name not in selected:
                    continue
                known_addr = getattr(session, "_position_addr_by_id", {}).get(int(sid)) if hasattr(session, "_position_addr_by_id") else None
                addr_candidates = ([known_addr] if known_addr is not None else []) + CAPTURE_POSITION_ADDR_CANDIDATES
                addr_candidates = list(dict.fromkeys(int(a) for a in addr_candidates if a is not None))
                pos = None
                addr = None
                for _ in range(max(1, CAPTURE_READ_RETRIES)):
                    pos, addr = _read_position_with_addr_candidates_open(ser, int(sid), addr_candidates)
                    if pos is not None:
                        break
                    time.sleep(CAPTURE_INTER_RETRY_S)
                if pos is None:
                    failures.append(f"{name}/ID{sid}")
                    continue
                positions[name] = int(pos)
                addr_hits[int(sid)] = int(addr)
    except Exception:
        failures = [f"{name}/ID{sid}" for name, sid in zip(session.motor_names, session.motor_ids, strict=True) if name in selected]
    return positions, addr_hits, failures


def _diagnose_unreadable_ids(session: DirectCalibrationSession, failures: Sequence[str]) -> str:
    """Return no extra diagnostic text; keep errors/output concise."""
    return ""


def read_positions(session: DirectCalibrationSession) -> dict[str, int]:
    """Read all motor positions with visible progress and bounded time."""
    print("[calibrate] Reading current positions from all configured motors...", flush=True)
    positions, failures = _read_positions_partial(session)
    for idx, (name, sid) in enumerate(zip(session.motor_names, session.motor_ids, strict=True), start=1):
        if name in positions:
            addr = getattr(session, "_position_addr_by_id", {}).get(int(sid), FEETECH_PRESENT_POSITION_ADDR)
            print(
                f"[calibrate]   ({idx}/{len(session.motor_ids)}) {name}/ID{sid}: "
                f"{positions[name]}  baud={session.baud_for_id(int(sid))}  pos_addr={addr}",
                flush=True,
            )
        else:
            print(f"[calibrate]   ({idx}/{len(session.motor_ids)}) {name}/ID{sid}: FAILED", flush=True)

    if failures:
        raise RuntimeError(
            "Could not read Present_Position from: "
            + ", ".join(failures)
            + ". Calibration cannot continue until every configured motor returns position data."
            + _diagnose_unreadable_ids(session, failures)
        )
    return positions


# ---------------------------------------------------------------------------
# Final no-freeze identify/read overrides
# ---------------------------------------------------------------------------
IDENTIFY_FULL_BAUD_SCAN = bool(getattr(val, "FEETECH_IDENTIFY_FULL_BAUD_SCAN", False))
IDENTIFY_READ_RETRIES = int(getattr(val, "FEETECH_IDENTIFY_READ_RETRIES", 2))
IDENTIFY_INTER_RETRY_S = float(getattr(val, "FEETECH_IDENTIFY_INTER_RETRY_S", 0.015))
IDENTIFY_POSITION_ADDR_CANDIDATES = list(dict.fromkeys(
    int(x) for x in getattr(val, "FEETECH_IDENTIFY_POSITION_ADDR_CANDIDATES", [FEETECH_PRESENT_POSITION_ADDR, 56])
))


def _quick_baud_candidates_for_session(session: DirectCalibrationSession, servo_id: int) -> list[int]:
    base = [session.baud_for_id(int(servo_id)), DEFAULT_BAUDRATE, 1000000]
    base = [int(b) for b in base if int(b) > 0]
    if IDENTIFY_FULL_BAUD_SCAN:
        return _get_scan_baudrates(base)
    return list(dict.fromkeys(base))


def _read_position_quick_identify(session: DirectCalibrationSession, servo_id: int) -> tuple[int | None, int, int | None]:
    fallback_baud = session.baud_for_id(int(servo_id))
    known_addr = getattr(session, "_position_addr_by_id", {}).get(int(servo_id)) if hasattr(session, "_position_addr_by_id") else None
    addr_candidates = ([known_addr] if known_addr is not None else []) + IDENTIFY_POSITION_ADDR_CANDIDATES
    addr_candidates = list(dict.fromkeys(int(a) for a in addr_candidates if a is not None))
    for baud in _quick_baud_candidates_for_session(session, int(servo_id)):
        for addr in addr_candidates:
            for _ in range(max(1, IDENTIFY_READ_RETRIES)):
                pos, hit_addr = _read_position_with_addr_candidates(session.port, int(servo_id), int(baud), [int(addr)])
                if pos is not None:
                    if not hasattr(session, "_position_addr_by_id"):
                        setattr(session, "_position_addr_by_id", {})
                    session._position_addr_by_id[int(servo_id)] = int(hit_addr)
                    return int(pos), int(baud), int(hit_addr)
                time.sleep(IDENTIFY_INTER_RETRY_S)
    return None, int(fallback_baud), None


def _best_effort_identify(session: DirectCalibrationSession) -> None:
    _print_header("Identify / verify motors")
    print("Fast verification table. A position of ---- means that ID did not return Present_Position.", flush=True)
    print("This mode does not run a long raw full-chain scan by default.", flush=True)
    print("Set FEETECH_IDENTIFY_FULL_BAUD_SCAN = True in values.py only for a slow exhaustive scan.\n", flush=True)

    border = "+" + "-" * 18 + "+" + "-" * 6 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 10 + "+"
    print(border, flush=True)
    print(f"| {'motor':16s} | {'id':>4s} | {'baud':>8s} | {'position':>8s} | {'pos_addr':>8s} |", flush=True)
    print(border, flush=True)

    failures: list[str] = []
    for name, sid in zip(session.motor_names, session.motor_ids, strict=True):
        print(f"[calibrate] testing {name}/ID{int(sid)}...", flush=True)
        pos, baud, addr = _read_position_quick_identify(session, int(sid))
        if pos is not None:
            session.set_baud_for_id(int(sid), int(baud))
            pos_txt = str(int(pos))
            addr_txt = str(int(addr)) if addr is not None else "----"
        else:
            failures.append(f"{name}/ID{int(sid)}")
            pos_txt = "----"
            addr_txt = "----"
        print(f"| {name:16s} | {int(sid):4d} | {session.baud_for_id(int(sid)):8d} | {pos_txt:>8s} | {addr_txt:>8s} |", flush=True)

    print(border, flush=True)
    if failures:
        print("Unreadable: " + ", ".join(failures), flush=True)
        print(_diagnose_unreadable_ids(session, failures), flush=True)


def _read_positions_partial(session: DirectCalibrationSession) -> tuple[dict[str, int], list[str]]:
    positions: dict[str, int] = {}
    failures: list[str] = []
    for name, sid in zip(session.motor_names, session.motor_ids, strict=True):
        pos, baud = _read_position_fast(session, int(sid))
        if pos is None:
            failures.append(f"{name}/ID{int(sid)}")
            continue
        session.set_baud_for_id(int(sid), int(baud))
        positions[str(name)] = int(pos)
    return positions, failures


# ---------------------------------------------------------------------------
# No-freeze motor setup + quiet diagnostics overrides
# ---------------------------------------------------------------------------
# Motor setup previously scanned every ID up to 253 at every baudrate using slow
# multi-retry helpers. If the only connected motor did not answer quickly, option
# 2 looked frozen. These final overrides keep setup bounded and visible: scan the
# expected target ID first, then a small configurable ID set, and only perform a
# full slow scan if explicitly enabled in values.py.

SETUP_FULL_BAUD_SCAN = bool(getattr(val, "FEETECH_SETUP_FULL_BAUD_SCAN", False))
SETUP_FULL_ID_SCAN = bool(getattr(val, "FEETECH_SETUP_FULL_ID_SCAN", False))
SETUP_MAX_ID = int(getattr(val, "FEETECH_SETUP_SCAN_MAX_ID", 20))
SETUP_PROBE_TIMEOUT_S = float(getattr(val, "FEETECH_SETUP_PROBE_TIMEOUT_S", 0.07))
SETUP_INTER_PROBE_DELAY_S = float(getattr(val, "FEETECH_SETUP_INTER_PROBE_DELAY_S", 0.005))
SETUP_EXTRA_SCAN_IDS = [int(x) for x in getattr(val, "FEETECH_SETUP_EXTRA_SCAN_IDS", [0, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])]


def _diagnose_unreadable_ids(session: DirectCalibrationSession, failures: Sequence[str]) -> str:
    return ""


def _setup_baud_candidates() -> list[int]:
    base = [DEFAULT_BAUDRATE, 1000000]
    if SETUP_FULL_BAUD_SCAN:
        return _get_scan_baudrates(base)
    return list(dict.fromkeys(int(b) for b in base if int(b) > 0))


def _setup_id_candidates(target_id: int | None = None) -> list[int]:
    ids: list[int] = []
    if target_id is not None:
        ids.append(int(target_id))
    ids.extend(_get_configured_motor_ids(_get_configured_motor_names()))
    ids.extend(SETUP_EXTRA_SCAN_IDS)
    if SETUP_FULL_ID_SCAN:
        ids.extend(range(0, 254))
    elif SETUP_MAX_ID > 0:
        ids.extend(range(0, SETUP_MAX_ID + 1))
    out: list[int] = []
    for sid in ids:
        sid = int(sid)
        if 0 <= sid <= 253 and sid not in out:
            out.append(sid)
    return out


def _quick_probe_open(ser: Any, servo_id: int) -> bool:
    sid = int(servo_id)
    try:
        try:
            ser.reset_input_buffer()
            ser.reset_output_buffer()
        except Exception:
            pass
        ser.write(_feetech_packet(sid, 0x01, ()))
        ser.flush()
        status = _read_status(ser, timeout_s=SETUP_PROBE_TIMEOUT_S)
        if status is not None and int(status[0]) == sid:
            return True
    except Exception:
        pass
    for addr in (FEETECH_MODEL_NUMBER_ADDR, FEETECH_PRESENT_POSITION_ADDR):
        try:
            try:
                ser.reset_input_buffer()
            except Exception:
                pass
            ser.write(_feetech_packet(sid, 0x02, (int(addr), 2)))
            ser.flush()
            status = _read_status(ser, timeout_s=SETUP_PROBE_TIMEOUT_S)
            if status is not None and int(status[0]) == sid and int(status[1]) == 0:
                if len(list(status[2])) >= 2:
                    return True
        except Exception:
            pass
    return False


def _quick_probe_id_at_baud(port: str, servo_id: int, baudrate: int) -> bool:
    try:
        with _open_serial(str(port), int(baudrate), timeout=min(0.04, SETUP_PROBE_TIMEOUT_S)) as ser:
            return _quick_probe_open(ser, int(servo_id))
    except Exception:
        return False


def _scan_single_connected_motor(port: str, target_id: int | None = None) -> tuple[int, int] | None:
    ids = _setup_id_candidates(target_id)
    bauds = _setup_baud_candidates()
    print("[calibrate] Fast direct Feetech scan for the single connected motor...", flush=True)
    print(f"[calibrate]   baud candidates: {bauds}", flush=True)
    print(f"[calibrate]   id candidates: {ids}", flush=True)
    if not SETUP_FULL_ID_SCAN:
        print("[calibrate]   full 0..253 ID scan disabled to avoid setup freezes.", flush=True)
        print("[calibrate]   Set FEETECH_SETUP_FULL_ID_SCAN = True only for a slow exhaustive scan.", flush=True)
    for baud in bauds:
        print(f"[calibrate]   scanning baudrate {baud}...", flush=True)
        found: list[int] = []
        try:
            with _open_serial(str(port), int(baud), timeout=min(0.04, SETUP_PROBE_TIMEOUT_S)) as ser:
                for sid in ids:
                    if _quick_probe_open(ser, int(sid)):
                        found.append(int(sid))
                        print(f"[calibrate]     response from ID {sid}", flush=True)
                        if len(found) > 1:
                            print("[calibrate] More than one motor responded. Connect ONLY the requested motor.", flush=True)
                            return None
                    time.sleep(SETUP_INTER_PROBE_DELAY_S)
        except Exception as exc:
            print(f"[calibrate]   baudrate {baud} scan failed to open/read: {exc}", flush=True)
            continue
        if len(found) == 1:
            print(f"[calibrate] Found motor ID {found[0]} at baudrate {baud}.", flush=True)
            return int(found[0]), int(baud)
    print("[calibrate] No motor responded in the fast setup scan.", flush=True)
    print("[calibrate] If this motor has an unknown ID outside the displayed candidates, enable FEETECH_SETUP_FULL_ID_SCAN.", flush=True)
    return None


def _manual_setup_one_motor(port: str, name: str, target_id: int) -> int | None:
    _print_header(f"Set motor ID {target_id}: {name}")
    print(
        f"Disconnect the full daisy chain. Connect ONLY the '{name}' motor to the controller board.\n"
        "Power-cycle the controller/servo bus if the previous motor was just disconnected.\n"
        "Then press Enter."
    )
    input("Ready? ")
    scanned = _scan_single_connected_motor(port, int(target_id))
    if scanned is None:
        print(
            "[calibrate] Could not get any data response from the connected motor.\n"
            "The red LED only confirms power; it does NOT confirm serial communication. Check cable orientation,\n"
            "port, data line, bus power ground, and that no other app has the serial port open."
        )
        return None
    current_id, baudrate = scanned
    if current_id == int(target_id):
        print(f"[calibrate] '{name}' already has correct ID {target_id}.", flush=True)
        return int(baudrate)
    print(f"[calibrate] Writing ID {target_id} to motor currently at ID {current_id}...", flush=True)
    if not _direct_write_feetech_id(port, int(current_id), int(target_id), int(baudrate)):
        print(f"[calibrate] Failed to write/verify ID {target_id}.", flush=True)
        return None
    if not _quick_probe_id_at_baud(port, int(target_id), int(baudrate)):
        print(f"[calibrate] ID write appeared to finish, but ID {target_id} did not answer the fast verification read.", flush=True)
        return None
    print(f"[calibrate] Verified '{name}' as ID {target_id} at baudrate {baudrate}.", flush=True)
    return int(baudrate)


def _best_effort_identify(session: DirectCalibrationSession) -> None:
    _print_header("Identify / verify motors")
    print("Fast verification table. A position of ---- means that ID did not return Present_Position.", flush=True)
    print("This mode does not run a long raw full-chain scan by default.\n", flush=True)
    border = "+" + "-" * 18 + "+" + "-" * 6 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 10 + "+"
    print(border, flush=True)
    print(f"| {'motor':16s} | {'id':>4s} | {'baud':>8s} | {'position':>8s} | {'pos_addr':>8s} |", flush=True)
    print(border, flush=True)
    failures: list[str] = []
    for name, sid in zip(session.motor_names, session.motor_ids, strict=True):
        print(f"[calibrate] testing {name}/ID{int(sid)}...", flush=True)
        pos, baud, addr = _read_position_quick_identify(session, int(sid))
        if pos is not None:
            session.set_baud_for_id(int(sid), int(baud))
            pos_txt = str(int(pos))
            addr_txt = str(int(addr)) if addr is not None else "----"
        else:
            failures.append(f"{name}/ID{int(sid)}")
            pos_txt = "----"
            addr_txt = "----"
        print(f"| {name:16s} | {int(sid):4d} | {session.baud_for_id(int(sid)):8d} | {pos_txt:>8s} | {addr_txt:>8s} |", flush=True)
    print(border, flush=True)
    if failures:
        print("Unreadable: " + ", ".join(failures), flush=True)



# ---------------------------------------------------------------------------
# Stable persistent motor-bus communication overrides
# ---------------------------------------------------------------------------
# These final overrides make the calibration/identify path behave much more like
# a real motor bus session: keep one serial connection open, avoid repeated
# open/close/reset cycles for every motor, retry inside that persistent session,
# and report a success count during identify. This is intentionally placed at the
# end of the file so it overrides the earlier direct one-shot helpers.

import threading

STABLE_BUS_ENABLED = bool(getattr(val, "FEETECH_STABLE_BUS_ENABLED", True))
STABLE_BUS_PACKET_TIMEOUT_S = float(getattr(val, "FEETECH_STABLE_BUS_PACKET_TIMEOUT_S", 0.10))
STABLE_BUS_READ_RETRIES = int(getattr(val, "FEETECH_STABLE_BUS_READ_RETRIES", 5))
STABLE_BUS_WRITE_RETRIES = int(getattr(val, "FEETECH_STABLE_BUS_WRITE_RETRIES", 3))
STABLE_BUS_INTER_PACKET_DELAY_S = float(getattr(val, "FEETECH_STABLE_BUS_INTER_PACKET_DELAY_S", 0.006))
STABLE_BUS_DRAIN_BEFORE_TX = bool(getattr(val, "FEETECH_STABLE_BUS_DRAIN_BEFORE_TX", True))
STABLE_IDENTIFY_ATTEMPTS = int(getattr(val, "FEETECH_STABLE_IDENTIFY_ATTEMPTS", 3))
STABLE_BAUD_SWITCH_DELAY_S = float(getattr(val, "FEETECH_STABLE_BAUD_SWITCH_DELAY_S", 0.08))


def _stable_position_addr_candidates() -> list[int]:
    candidates = getattr(val, "FEETECH_PRESENT_POSITION_ADDR_CANDIDATES", None)
    if candidates is None:
        candidates = getattr(val, "FEETECH_CAPTURE_POSITION_ADDR_CANDIDATES", [FEETECH_PRESENT_POSITION_ADDR, 56])
    out: list[int] = []
    for addr in candidates:
        try:
            addr_i = int(addr)
        except Exception:
            continue
        if addr_i not in out:
            out.append(addr_i)
    return out or [FEETECH_PRESENT_POSITION_ADDR]


class StableFeetechBus:
    """Persistent Feetech serial bus wrapper for a long daisy chain."""

    def __init__(self, port: str, baudrate: int) -> None:
        self.port = str(port)
        self.baudrate = int(baudrate)
        self.ser: Any = None
        self._lock = threading.RLock()
        self.open()

    def open(self) -> None:
        if self.ser is not None and getattr(self.ser, "is_open", True):
            return
        if serial is None:
            raise RuntimeError("pyserial is not installed/importable; install pyserial to use direct calibration.")
        self.ser = serial.Serial(
            self.port,
            int(self.baudrate),
            timeout=min(0.05, max(0.01, STABLE_BUS_PACKET_TIMEOUT_S / 2.0)),
            write_timeout=0.25,
        )
        time.sleep(0.02)
        self.drain()

    def close(self) -> None:
        with self._lock:
            try:
                if self.ser is not None:
                    self.ser.close()
            except Exception:
                pass
            self.ser = None

    def set_baudrate(self, baudrate: int) -> None:
        baudrate = int(baudrate)
        with self._lock:
            if baudrate == self.baudrate and self.ser is not None:
                return
            self.close()
            self.baudrate = baudrate
            time.sleep(STABLE_BAUD_SWITCH_DELAY_S)
            self.open()

    def drain(self) -> None:
        if self.ser is None:
            return
        try:
            waiting = int(getattr(self.ser, "in_waiting", 0) or 0)
            if waiting > 0:
                self.ser.read(waiting)
        except Exception:
            try:
                self.ser.reset_input_buffer()
            except Exception:
                pass

    def transact(self, servo_id: int, instruction: int, params: Sequence[int] = (), timeout_s: float | None = None) -> tuple[int, int, list[int]] | None:
        timeout = STABLE_BUS_PACKET_TIMEOUT_S if timeout_s is None else float(timeout_s)
        packet = _feetech_packet(int(servo_id), int(instruction), params)
        with self._lock:
            self.open()
            try:
                if STABLE_BUS_DRAIN_BEFORE_TX:
                    self.drain()
                self.ser.write(packet)
                self.ser.flush()
                status = _read_status(self.ser, timeout_s=timeout)
                time.sleep(STABLE_BUS_INTER_PACKET_DELAY_S)
                return status
            except Exception:
                try:
                    self.close()
                    self.open()
                except Exception:
                    pass
                time.sleep(STABLE_BUS_INTER_PACKET_DELAY_S)
                return None

    def read_register(self, servo_id: int, addr: int, length: int, retries: int | None = None) -> list[int] | None:
        sid = int(servo_id)
        attempts = max(1, int(retries if retries is not None else STABLE_BUS_READ_RETRIES))
        for _ in range(attempts):
            status = self.transact(sid, 0x02, (int(addr), int(length)))
            if status is None:
                continue
            rx_id, err, params = status
            if int(rx_id) == sid and int(err) == 0 and len(params) >= int(length):
                return list(params[: int(length)])
        return None

    def read_u16(self, servo_id: int, addr: int, retries: int | None = None) -> int | None:
        data = self.read_register(int(servo_id), int(addr), 2, retries=retries)
        if data is None or len(data) < 2:
            return None
        return int(data[0]) | (int(data[1]) << 8)

    def write_register(self, servo_id: int, addr: int, data: Sequence[int], retries: int | None = None, require_status: bool = False) -> bool:
        sid = int(servo_id)
        attempts = max(1, int(retries if retries is not None else STABLE_BUS_WRITE_RETRIES))
        payload = (int(addr), *[int(x) & 0xFF for x in data])
        for _ in range(attempts):
            status = self.transact(sid, 0x03, payload)
            if status is None:
                if not require_status:
                    return True
                continue
            if int(status[0]) == sid and int(status[1]) == 0:
                return True
        return False

    def write_u8(self, servo_id: int, addr: int, value: int) -> bool:
        return self.write_register(int(servo_id), int(addr), [int(value) & 0xFF])

    def write_u16(self, servo_id: int, addr: int, value: int) -> bool:
        value = int(value)
        return self.write_register(int(servo_id), int(addr), [value & 0xFF, (value >> 8) & 0xFF])


def _get_stable_bus(session: DirectCalibrationSession, baudrate: int | None = None) -> StableFeetechBus:
    baud = int(baudrate if baudrate is not None else session.baudrate)
    bus = getattr(session, "_stable_bus", None)
    if bus is None or not isinstance(bus, StableFeetechBus):
        bus = StableFeetechBus(session.port, baud)
        setattr(session, "_stable_bus", bus)
    else:
        bus.set_baudrate(baud)
    return bus


def _close_stable_bus(session: DirectCalibrationSession) -> None:
    bus = getattr(session, "_stable_bus", None)
    if isinstance(bus, StableFeetechBus):
        bus.close()


def _stable_read_position_at_baud(session: DirectCalibrationSession, servo_id: int, baudrate: int) -> tuple[int | None, int | None]:
    bus = _get_stable_bus(session, int(baudrate))
    known_addr = getattr(session, "_position_addr_by_id", {}).get(int(servo_id)) if hasattr(session, "_position_addr_by_id") else None
    addr_candidates = ([known_addr] if known_addr is not None else []) + _stable_position_addr_candidates()
    addr_candidates = list(dict.fromkeys(int(a) for a in addr_candidates if a is not None))
    for addr in addr_candidates:
        value = bus.read_u16(int(servo_id), int(addr), retries=STABLE_BUS_READ_RETRIES)
        if value is not None:
            if not hasattr(session, "_position_addr_by_id"):
                setattr(session, "_position_addr_by_id", {})
            session._position_addr_by_id[int(servo_id)] = int(addr)
            return int(value), int(addr)
    return None, None


def _stable_baud_candidates_for_session(session: DirectCalibrationSession, servo_id: int) -> list[int]:
    base = [session.baud_for_id(int(servo_id)), DEFAULT_BAUDRATE, 1000000]
    if bool(getattr(val, "FEETECH_CAPTURE_FULL_BAUD_SCAN", False)):
        return _get_scan_baudrates(base)
    return list(dict.fromkeys(int(b) for b in base if int(b) > 0))


def _read_position_fast(session: DirectCalibrationSession, servo_id: int) -> tuple[int | None, int]:
    fallback_baud = session.baud_for_id(int(servo_id))
    if not STABLE_BUS_ENABLED:
        for baud in _stable_baud_candidates_for_session(session, int(servo_id)):
            pos, _addr = _read_position_with_addr_candidates(session.port, int(servo_id), int(baud), _stable_position_addr_candidates())
            if pos is not None:
                return int(pos), int(baud)
        return None, int(fallback_baud)
    for baud in _stable_baud_candidates_for_session(session, int(servo_id)):
        pos, _addr = _stable_read_position_at_baud(session, int(servo_id), int(baud))
        if pos is not None:
            return int(pos), int(baud)
    return None, int(fallback_baud)


def _read_positions_partial(session: DirectCalibrationSession) -> tuple[dict[str, int], list[str]]:
    positions: dict[str, int] = {}
    failures: list[str] = []
    for name, sid in zip(session.motor_names, session.motor_ids, strict=True):
        pos, baud = _read_position_fast(session, int(sid))
        if pos is None:
            failures.append(f"{name}/ID{int(sid)}")
            continue
        session.set_baud_for_id(int(sid), int(baud))
        positions[str(name)] = int(pos)
    return positions, failures


def read_positions(session: DirectCalibrationSession) -> dict[str, int]:
    print("[calibrate] Reading current positions from all configured motors using stable persistent bus...", flush=True)
    positions, failures = _read_positions_partial(session)
    for idx, (name, sid) in enumerate(zip(session.motor_names, session.motor_ids, strict=True), start=1):
        if name in positions:
            addr = getattr(session, "_position_addr_by_id", {}).get(int(sid), FEETECH_PRESENT_POSITION_ADDR)
            print(f"[calibrate]   ({idx}/{len(session.motor_ids)}) {name}/ID{int(sid)}: {positions[name]}  baud={session.baud_for_id(int(sid))}  pos_addr={addr}", flush=True)
        else:
            print(f"[calibrate]   ({idx}/{len(session.motor_ids)}) {name}/ID{int(sid)}: FAILED", flush=True)
    if failures:
        raise RuntimeError("Could not read Present_Position from: " + ", ".join(failures) + ". Calibration cannot continue until every configured motor returns position data.")
    return positions


def set_torque(session: DirectCalibrationSession, enabled: bool) -> None:
    value = 1 if enabled else 0
    failed: list[str] = []
    for name, sid in zip(session.motor_names, session.motor_ids, strict=True):
        baud = session.baud_for_id(int(sid))
        try:
            bus = _get_stable_bus(session, baud)
            ok = bus.write_u8(int(sid), FEETECH_TORQUE_ENABLE_ADDR, value)
        except Exception:
            ok = _direct_write_u8(session.port, int(sid), FEETECH_TORQUE_ENABLE_ADDR, value, baud)
        if not ok:
            failed.append(f"{name}/ID{int(sid)}")
    if failed:
        print(f"[calibrate] Warning: torque write failed for: {', '.join(failed)}", flush=True)
    else:
        print(f"[calibrate] Torque {'ENABLED' if enabled else 'DISABLED'} for {len(session.motor_names)} motors.", flush=True)


def connect_session() -> DirectCalibrationSession:
    ports = _candidate_robot_ports()
    if not ports:
        raise RuntimeError("Could not auto-detect the robot serial port. Set values.REAL_ROBOT_PORT manually.")
    setup_status = get_motor_setup_status()
    if not setup_status.configured:
        raise RuntimeError("Motor setup metadata was not found or did not match the configured motor names/IDs. Run option 2 first, or fix calibration_data/robot_motor_setup.json.")
    motor_names = list(setup_status.motor_names)
    motor_ids = list(setup_status.motor_ids or _get_configured_motor_ids(motor_names))
    model_numbers = _get_configured_motor_model_numbers(motor_names)
    if setup_status.motor_baudrates and len(setup_status.motor_baudrates) == len(motor_ids):
        motor_baudrates = [int(x) for x in setup_status.motor_baudrates]
    else:
        motor_baudrates = [DEFAULT_BAUDRATE] * len(motor_ids)
    port = str(ports[0])
    print(f"[calibrate] using direct port = {port}", flush=True)
    print(f"[calibrate] using setup metadata = {setup_status.source}", flush=True)
    print("[calibrate] using stable persistent serial motor-bus session." if STABLE_BUS_ENABLED else "[calibrate] stable bus disabled; using direct one-shot serial reads.", flush=True)
    for name, sid, baud in zip(motor_names, motor_ids, motor_baudrates, strict=True):
        print(f"  {name:16s} id={int(sid):3d} baud={int(baud)}", flush=True)
    session = DirectCalibrationSession(port, list(motor_names), list(motor_ids), list(model_numbers), list(motor_baudrates))
    if STABLE_BUS_ENABLED:
        _get_stable_bus(session, session.baudrate)
    return session


def _read_position_quick_identify(session: DirectCalibrationSession, servo_id: int) -> tuple[int | None, int, int | None, int, int]:
    attempts = max(1, STABLE_IDENTIFY_ATTEMPTS)
    successes = 0
    first_pos: int | None = None
    first_baud = session.baud_for_id(int(servo_id))
    first_addr: int | None = None
    for _ in range(attempts):
        pos, baud = _read_position_fast(session, int(servo_id))
        if pos is not None:
            successes += 1
            session.set_baud_for_id(int(servo_id), int(baud))
            if first_pos is None:
                first_pos = int(pos)
                first_baud = int(baud)
                first_addr = getattr(session, "_position_addr_by_id", {}).get(int(servo_id), FEETECH_PRESENT_POSITION_ADDR)
    return first_pos, int(first_baud), first_addr, int(successes), int(attempts)


def _best_effort_identify(session: DirectCalibrationSession) -> None:
    _print_header("Identify / verify motors")
    print("Stable verification table. Each motor is read several times through one persistent bus session.", flush=True)
    print("A position of ---- means that ID did not return Present_Position in this pass.\n", flush=True)
    border = "+" + "-" * 18 + "+" + "-" * 6 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 10 + "+"
    print(border, flush=True)
    print(f"| {'motor':16s} | {'id':>4s} | {'baud':>8s} | {'position':>8s} | {'pos_addr':>8s} | {'ok/try':>8s} |", flush=True)
    print(border, flush=True)
    failures: list[str] = []
    for name, sid in zip(session.motor_names, session.motor_ids, strict=True):
        print(f"[calibrate] testing {name}/ID{int(sid)}...", flush=True)
        pos, baud, addr, ok_count, tries = _read_position_quick_identify(session, int(sid))
        if pos is not None:
            session.set_baud_for_id(int(sid), int(baud))
            pos_txt = str(int(pos))
            addr_txt = str(int(addr)) if addr is not None else "----"
        else:
            failures.append(f"{name}/ID{int(sid)}")
            pos_txt = "----"
            addr_txt = "----"
        print(f"| {name:16s} | {int(sid):4d} | {session.baud_for_id(int(sid)):8d} | {pos_txt:>8s} | {addr_txt:>8s} | {f'{ok_count}/{tries}':>8s} |", flush=True)
    print(border, flush=True)
    if failures:
        print("Unreadable: " + ", ".join(failures), flush=True)
    _close_stable_bus(session)


def run_calibration_only() -> int:
    setup_status = get_motor_setup_status()
    if not setup_status.configured:
        print("[calibrate] Motor-ID setup was not detected. Run setup first so calibration behaves like LeRobot:\nfirst setup motor IDs, then perform neutral/min/max calibration.")
        return 1
    _print_header("Interactive robot calibration")
    print("This stage uses a stable persistent Feetech motor-bus session. It records a neutral pose and then\ncaptures per-joint minimum and maximum positions for the current hardware setup.")
    try:
        session = connect_session()
    except Exception as exc:
        print(f"[calibrate] Failed to connect to robot through direct bus: {exc}")
        return 1
    print(f"[calibrate] Connected. Motor names: {session.motor_names}")
    try:
        set_torque(session, enabled=False)
        neutral = prompt_capture(session, "Capture NEUTRAL pose", "Move the arm into your desired neutral/zero pose. Center all wrist joints and set gripper neutral.")
        min_pos, max_pos = capture_joint_limits(session)
        payload = build_calibration_payload(session, neutral, min_pos, max_pos, session.motor_ids)
        output_paths = write_outputs(payload)
        _print_header("Calibration complete")
        print("Saved JSON calibration to:")
        for path in output_paths:
            print(f"  {path}")
        print(f"Saved text summary to:     {TXT_PATH}")
        return 0
    finally:
        try:
            set_torque(session, enabled=True)
        except Exception:
            pass
        _close_stable_bus(session)



# ---------------------------------------------------------------------------
# Read-only ID scan mode
# ---------------------------------------------------------------------------
# This mode never writes to a servo. It only reads/pings possible IDs so a single
# connected motor can be identified without running the flashing/setup workflow.

READONLY_SCAN_MAX_ID = int(getattr(val, "FEETECH_READONLY_SCAN_MAX_ID", 20))
READONLY_SCAN_EXTRA_IDS = list(getattr(val, "FEETECH_READONLY_SCAN_EXTRA_IDS", [0]))
READONLY_SCAN_FULL_BAUD_SCAN = bool(getattr(val, "FEETECH_READONLY_SCAN_FULL_BAUD_SCAN", False))
READONLY_SCAN_READ_RETRIES = int(getattr(val, "FEETECH_READONLY_SCAN_READ_RETRIES", 2))
READONLY_SCAN_SHOW_MISSES = bool(getattr(val, "FEETECH_READONLY_SCAN_SHOW_MISSES", False))


def _readonly_scan_id_candidates() -> list[int]:
    ids = list(range(1, READONLY_SCAN_MAX_ID + 1)) + [int(x) for x in READONLY_SCAN_EXTRA_IDS]
    out: list[int] = []
    for sid in ids:
        try:
            sid = int(sid)
        except Exception:
            continue
        if 0 <= sid <= 253 and sid not in out:
            out.append(sid)
    return out


def _readonly_scan_baud_candidates() -> list[int]:
    base = [DEFAULT_BAUDRATE, 1000000]
    if READONLY_SCAN_FULL_BAUD_SCAN:
        return _get_scan_baudrates(base)
    out: list[int] = []
    for baud in base:
        baud = int(baud)
        if baud > 0 and baud not in out:
            out.append(baud)
    return out


def _readonly_scan_read_position(bus: StableFeetechBus, servo_id: int) -> tuple[int | None, int | None]:
    for addr in _stable_position_addr_candidates():
        pos = bus.read_u16(int(servo_id), int(addr), retries=max(1, READONLY_SCAN_READ_RETRIES))
        if pos is not None:
            return int(pos), int(addr)
    return None, None


def run_readonly_scan() -> int:
    _print_header("Read-only motor ID scan")
    print("This mode does NOT write, flash, or change servo IDs.", flush=True)
    print("Connect one motor, or a small motor segment, then use this to see which IDs actually respond.", flush=True)
    ports = _candidate_robot_ports()
    if not ports:
        print("[scan] Could not auto-detect the robot serial port. Set values.REAL_ROBOT_PORT manually.", flush=True)
        return 1
    port = str(ports[0])
    ids = _readonly_scan_id_candidates()
    bauds = _readonly_scan_baud_candidates()
    print(f"[scan] port = {port}", flush=True)
    print(f"[scan] baud candidates = {bauds}", flush=True)
    print(f"[scan] ID candidates = {ids}", flush=True)
    if not READONLY_SCAN_FULL_BAUD_SCAN:
        print("[scan] full baud scan disabled for speed. Set FEETECH_READONLY_SCAN_FULL_BAUD_SCAN = True for slow exhaustive baud scan.", flush=True)
    print("", flush=True)

    found: list[tuple[int, int, int | None, int | None, int | None]] = []
    for baud in bauds:
        print(f"[scan] scanning baudrate {baud}...", flush=True)
        bus = None
        try:
            bus = StableFeetechBus(port, int(baud)) if STABLE_BUS_ENABLED else None
        except Exception as exc:
            print(f"[scan]   could not open {port} at {baud}: {exc}", flush=True)
            continue
        try:
            for sid in ids:
                if any(existing_sid == int(sid) for existing_sid, *_ in found):
                    continue
                pos = None
                pos_addr = None
                model = None
                if bus is not None:
                    pos, pos_addr = _readonly_scan_read_position(bus, int(sid))
                    model = bus.read_u16(int(sid), FEETECH_MODEL_NUMBER_ADDR, retries=1)
                else:
                    for addr in _stable_position_addr_candidates():
                        pos = _direct_read_u16(port, int(sid), int(addr), int(baud))
                        if pos is not None:
                            pos_addr = int(addr)
                            break
                    model = _direct_read_model(port, int(sid), int(baud))
                if pos is not None or model is not None:
                    found.append((int(sid), int(baud), int(pos) if pos is not None else None, int(pos_addr) if pos_addr is not None else None, int(model) if model is not None else None))
                    print(f"[scan]   FOUND id={int(sid):3d} baud={int(baud):7d} position={str(pos):>6s} pos_addr={str(pos_addr):>4s} model={str(model):>5s}", flush=True)
                elif READONLY_SCAN_SHOW_MISSES:
                    print(f"[scan]   no response id={int(sid):3d}", flush=True)
        finally:
            try:
                if bus is not None:
                    bus.close()
            except Exception:
                pass

    print("", flush=True)
    if not found:
        print("[scan] No responding IDs found in the selected range.", flush=True)
        print("[scan] If the motor has an unknown baudrate, enable FEETECH_READONLY_SCAN_FULL_BAUD_SCAN in values.py and retry.", flush=True)
        return 0

    border = "+" + "-" * 6 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 10 + "+"
    print(border, flush=True)
    print(f"| {'id':>4s} | {'baud':>8s} | {'position':>8s} | {'pos_addr':>8s} | {'model':>8s} |", flush=True)
    print(border, flush=True)
    for sid, baud, pos, pos_addr, model in found:
        print(f"| {sid:4d} | {baud:8d} | {str(pos) if pos is not None else '----':>8s} | {str(pos_addr) if pos_addr is not None else '----':>8s} | {str(model) if model is not None else '----':>8s} |", flush=True)
    print(border, flush=True)
    if len(found) > 1:
        print("[scan] More than one ID responded. If you intended to check one motor, disconnect all other motors.", flush=True)
    return 0


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
            print(f"[calibrate] Failed to connect to robot through direct bus: {exc}")
            return 1
        _best_effort_identify(session)
        return 0
    if mode in {"scan", "readonly_scan", "read_only_scan", "id_scan"}:
        return run_readonly_scan()
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
    print("  4) Identify/verify configured motors only")
    print("  5) Read-only scan connected motor IDs")
    reply = input("Selection [1/2/3/4/5]: ").strip()
    return {"1": "full", "2": "setup", "3": "calibration", "4": "identify", "5": "scan"}.get(reply, "full")


if __name__ == "__main__":
    raise SystemExit(main())