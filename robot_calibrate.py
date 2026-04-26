from __future__ import annotations

import glob
import json
import sys
import time
import threading
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

FEETECH_MODEL_NUMBER_ADDR = int(getattr(val, "FEETECH_MODEL_NUMBER_ADDR", 3))
FEETECH_ID_REGISTER_ADDR = int(getattr(val, "FEETECH_ID_REGISTER_ADDR", 5))
FEETECH_LOCK_REGISTER_ADDR = int(getattr(val, "FEETECH_LOCK_REGISTER_ADDR", 55))
FEETECH_LOCK_UNLOCK_VALUE = int(getattr(val, "FEETECH_LOCK_UNLOCK_VALUE", 0))
FEETECH_LOCK_LOCK_VALUE = int(getattr(val, "FEETECH_LOCK_LOCK_VALUE", 1))
FEETECH_TORQUE_ENABLE_ADDR = int(getattr(val, "FEETECH_TORQUE_ENABLE_ADDR", 40))
FEETECH_GOAL_POSITION_ADDR = int(getattr(val, "FEETECH_GOAL_POSITION_ADDR", 42))
FEETECH_PRESENT_POSITION_ADDR = int(getattr(val, "FEETECH_PRESENT_POSITION_ADDR", 56))
FEETECH_PRESENT_POSITION_LEN = int(getattr(val, "FEETECH_PRESENT_POSITION_LEN", 2))
DEFAULT_BAUDRATE = int(getattr(val, "REAL_ROBOT_BAUDRATE", 1000000))
PACKET_TIMEOUT_S = float(getattr(val, "FEETECH_DIRECT_PACKET_TIMEOUT_S", 0.18))
READ_RETRIES = int(getattr(val, "FEETECH_DIRECT_READ_RETRIES", 3))
WRITE_RETRIES = int(getattr(val, "FEETECH_DIRECT_WRITE_RETRIES", 3))
INTER_PACKET_DELAY_S = float(getattr(val, "FEETECH_STABLE_BUS_INTER_PACKET_DELAY_S", 0.006))

# Safety-critical setup behavior.
SETUP_CONFIRM_REASSIGN_VALID_ID = bool(getattr(val, "FEETECH_SETUP_CONFIRM_REASSIGN_VALID_ID", True))
SETUP_VERIFY_EXACTLY_ONE_ID = bool(getattr(val, "FEETECH_SETUP_VERIFY_EXACTLY_ONE_ID", True))
SETUP_VERIFY_RESCAN_AFTER_WRITE = bool(getattr(val, "FEETECH_SETUP_VERIFY_RESCAN_AFTER_WRITE", True))
SETUP_POWER_CYCLE_AFTER_ID_WRITE = bool(getattr(val, "FEETECH_SETUP_POWER_CYCLE_AFTER_ID_WRITE", False))
SETUP_FULL_BAUD_SCAN = bool(getattr(val, "FEETECH_SETUP_FULL_BAUD_SCAN", False))
SETUP_FULL_ID_SCAN = bool(getattr(val, "FEETECH_SETUP_FULL_ID_SCAN", False))
SETUP_MAX_ID = int(getattr(val, "FEETECH_SETUP_SCAN_MAX_ID", 20))
SETUP_EXTRA_SCAN_IDS = [int(x) for x in getattr(val, "FEETECH_SETUP_EXTRA_SCAN_IDS", [0, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])]
SETUP_READ_RETRIES = int(getattr(val, "FEETECH_SETUP_READ_RETRIES", 2))
SETUP_WRITE_VERIFY_RETRIES = int(getattr(val, "FEETECH_SETUP_WRITE_VERIFY_RETRIES", 8))
SETUP_POST_WRITE_DELAY_S = float(getattr(val, "FEETECH_SETUP_POST_WRITE_DELAY_S", 0.8))
ID_FLASH_VERIFY_RETRIES = int(getattr(val, "FEETECH_ID_FLASH_VERIFY_RETRIES", 10))
ID_FLASH_VERIFY_DELAY_S = float(getattr(val, "FEETECH_ID_FLASH_VERIFY_DELAY_S", 0.30))
ID_FLASH_REQUIRE_ID_REGISTER_MATCH = bool(getattr(val, "FEETECH_ID_FLASH_REQUIRE_ID_REGISTER_MATCH", True))
ID_FLASH_LOCK_EEPROM_AFTER_WRITE = bool(getattr(val, "FEETECH_ID_FLASH_LOCK_EEPROM_AFTER_WRITE", True))

IDENTIFY_ATTEMPTS = int(getattr(val, "FEETECH_STABLE_IDENTIFY_ATTEMPTS", 3))
IDENTIFY_FULL_BAUD_SCAN = bool(getattr(val, "FEETECH_IDENTIFY_FULL_BAUD_SCAN", False))
READONLY_SCAN_MAX_ID = int(getattr(val, "FEETECH_READONLY_SCAN_MAX_ID", 20))
READONLY_SCAN_EXTRA_IDS = [int(x) for x in getattr(val, "FEETECH_READONLY_SCAN_EXTRA_IDS", [0])]
READONLY_SCAN_FULL_BAUD_SCAN = bool(getattr(val, "FEETECH_READONLY_SCAN_FULL_BAUD_SCAN", False))
READONLY_SCAN_SHOW_MISSES = bool(getattr(val, "FEETECH_READONLY_SCAN_SHOW_MISSES", False))
LIVE_TABLE_PERIOD_S = float(getattr(val, "FEETECH_LIVE_TABLE_PERIOD_S", 0.5))
LIVE_TABLE_MAX_SECONDS = float(getattr(val, "FEETECH_LIVE_TABLE_MAX_SECONDS", 0.0))


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
    bus: Any = None

    def __post_init__(self) -> None:
        if not self.motor_baudrates or len(self.motor_baudrates) != len(self.motor_ids):
            self.motor_baudrates = [DEFAULT_BAUDRATE] * len(self.motor_ids)

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
        if self.bus is not None:
            try:
                self.bus.close()
            except Exception:
                pass
            self.bus = None

    def read(self, register: str, motor_names: Sequence[str] | str | None = None) -> list[int] | int:
        names = _normalize_motor_selection(motor_names, self.motor_names)
        reg = str(register).lower().replace("_", "")
        out: list[int] = []
        for name in names:
            idx = self.motor_names.index(name)
            sid = int(self.motor_ids[idx])
            if reg in {"id", "ids"}:
                out.append(sid)
            elif reg in {"model", "modelnumber"}:
                model = _read_model_any_baud(self.port, sid, [self.baud_for_id(sid)])
                out.append(int(model) if model is not None else int(self.model_numbers[idx]))
            elif reg in {"presentposition", "position"}:
                pos, baud, _addr = _read_position_any_baud(self.port, sid, [self.baud_for_id(sid)])
                if pos is None:
                    raise RuntimeError(f"No Present_Position response from {name} / ID {sid}")
                self.set_baud_for_id(sid, baud)
                out.append(int(pos))
            else:
                raise RuntimeError(f"Unsupported direct register read: {register}")
        return out[0] if isinstance(motor_names, str) and len(out) == 1 else out

    def write(self, register: str, value: Any, motor_names: Sequence[str] | str | None = None) -> None:
        names = _normalize_motor_selection(motor_names, self.motor_names)
        reg = str(register).lower().replace("_", "")
        if isinstance(value, dict):
            items = [(name, int(value[name])) for name in names if name in value]
        elif isinstance(value, (list, tuple)):
            items = [(name, int(v)) for name, v in zip(names, value, strict=False)]
        else:
            items = [(name, int(value)) for name in names]
        bus = _get_session_bus(self)
        for name, v in items:
            idx = self.motor_names.index(name)
            sid = int(self.motor_ids[idx])
            if reg in {"torqueenable", "torque"}:
                ok = bus.write_u8(sid, FEETECH_TORQUE_ENABLE_ADDR, v)
            elif reg in {"goalposition"}:
                ok = bus.write_u16(sid, FEETECH_GOAL_POSITION_ADDR, v)
            else:
                raise RuntimeError(f"Unsupported direct register write: {register}")
            if not ok:
                raise RuntimeError(f"Write {register} failed for {name} / ID {sid}")

    sync_write = write


class StableFeetechBus:
    def __init__(self, port: str, baudrate: int):
        if serial is None:
            raise RuntimeError("pyserial is not installed/importable; install pyserial to use direct calibration.")
        self.port = str(port)
        self.baudrate = int(baudrate)
        self.ser = serial.Serial(self.port, self.baudrate, timeout=0.04, write_timeout=0.25)
        self.lock = threading.RLock()
        self._drain()
        time.sleep(0.02)

    def close(self) -> None:
        with self.lock:
            try:
                self.ser.close()
            except Exception:
                pass

    def _drain(self) -> None:
        try:
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
        except Exception:
            pass

    def txrx(self, servo_id: int, instruction: int, params: Sequence[int] = (), timeout_s: float = PACKET_TIMEOUT_S) -> tuple[int, int, list[int]] | None:
        with self.lock:
            try:
                self._drain()
                self.ser.write(_feetech_packet(int(servo_id), int(instruction), params))
                self.ser.flush()
                status = _read_status(self.ser, timeout_s=timeout_s)
                time.sleep(INTER_PACKET_DELAY_S)
                return status
            except Exception:
                return None

    def ping(self, servo_id: int, retries: int = 1) -> bool:
        for _ in range(max(1, int(retries))):
            st = self.txrx(int(servo_id), 0x01, (), timeout_s=PACKET_TIMEOUT_S)
            if st is not None and int(st[0]) == int(servo_id):
                return True
        return False

    def read_reg(self, servo_id: int, addr: int, length: int, retries: int = READ_RETRIES) -> list[int] | None:
        for _ in range(max(1, int(retries))):
            st = self.txrx(int(servo_id), 0x02, (int(addr), int(length)), timeout_s=PACKET_TIMEOUT_S)
            if st is not None and int(st[0]) == int(servo_id) and int(st[1]) == 0:
                params = list(st[2])
                if len(params) >= int(length):
                    return params[: int(length)]
        return None

    def read_u8(self, servo_id: int, addr: int, retries: int = READ_RETRIES) -> int | None:
        data = self.read_reg(int(servo_id), int(addr), 1, retries=retries)
        return int(data[0]) if data else None

    def read_u16(self, servo_id: int, addr: int, retries: int = READ_RETRIES) -> int | None:
        data = self.read_reg(int(servo_id), int(addr), 2, retries=retries)
        if data is None or len(data) < 2:
            return None
        return int(data[0]) | (int(data[1]) << 8)

    def write_reg(self, servo_id: int, addr: int, data: Sequence[int], retries: int = WRITE_RETRIES) -> bool:
        # Some Feetech buses do not return status for write packets, so the return
        # value only means the packet was sent without a serial exception. ID-write
        # setup never trusts this alone; it always rescans the bus afterwards.
        payload = (int(addr), *[int(x) & 0xFF for x in data])
        sent = False
        for _ in range(max(1, int(retries))):
            st = self.txrx(int(servo_id), 0x03, payload, timeout_s=PACKET_TIMEOUT_S)
            if st is None:
                sent = True
            elif int(st[0]) == int(servo_id) and int(st[1]) == 0:
                return True
            time.sleep(INTER_PACKET_DELAY_S)
        return sent

    def write_u8(self, servo_id: int, addr: int, value: int, retries: int = WRITE_RETRIES) -> bool:
        return self.write_reg(int(servo_id), int(addr), [int(value) & 0xFF], retries=retries)

    def write_u16(self, servo_id: int, addr: int, value: int, retries: int = WRITE_RETRIES) -> bool:
        v = int(value)
        return self.write_reg(int(servo_id), int(addr), [v & 0xFF, (v >> 8) & 0xFF], retries=retries)


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


def _get_configured_motor_names() -> list[str]:
    names = list(getattr(val, "REAL_ROBOT_MOTOR_NAMES", []))
    if names:
        return [str(x) for x in names]
    return ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "wrist_pitch", "gripper"]


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


def _position_addr_candidates() -> list[int]:
    raw = getattr(val, "FEETECH_PRESENT_POSITION_ADDR_CANDIDATES", None)
    if raw is None:
        raw = getattr(val, "FEETECH_CAPTURE_POSITION_ADDR_CANDIDATES", [FEETECH_PRESENT_POSITION_ADDR, 56])
    out: list[int] = []
    for addr in [FEETECH_PRESENT_POSITION_ADDR, *list(raw), 56]:
        try:
            addr = int(addr)
        except Exception:
            continue
        if 0 <= addr <= 255 and addr not in out:
            out.append(addr)
    return out or [FEETECH_PRESENT_POSITION_ADDR]


def _get_scan_baudrates(extra: Sequence[int] | None = None) -> list[int]:
    raw = list(getattr(val, "REAL_ROBOT_SCAN_BAUDRATES", []))
    candidates = list(extra or []) + raw + [DEFAULT_BAUDRATE, 1000000, 500000, 250000, 128000, 115200, 57600, 38400, 19200]
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
    for pattern in ["/dev/cu.usbmodem*", "/dev/cu.usbserial*", "/dev/tty.usbmodem*", "/dev/tty.usbserial*"]:
        ports.extend(glob.glob(pattern))
    return sorted(dict.fromkeys(ports))


def _find_robot_port() -> str:
    ports = _candidate_robot_ports()
    if not ports:
        raise RuntimeError("Could not auto-detect the robot serial port. Set values.REAL_ROBOT_PORT manually.")
    return ports[0]


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


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        return None
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
    top = payload.get("baudrate")
    out: list[int] = []
    for name in motor_names:
        entry = payload.get(str(name))
        if isinstance(entry, dict) and entry.get("baudrate") is not None:
            try:
                out.append(int(entry["baudrate"]))
                continue
            except Exception:
                pass
        if top is not None:
            try:
                out.append(int(top))
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
            if motor_baudrates is None:
                motor_baudrates = [DEFAULT_BAUDRATE] * len(motor_names)
            return SetupStatus(True, str(path), list(motor_names), list(motor_ids), list(motor_baudrates))
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
        return CalibrationStatus(False, None, motor_names)
    neutral = payload.get("neutral_pos")
    min_pos = payload.get("min_pos")
    max_pos = payload.get("max_pos")
    if isinstance(neutral, list) and isinstance(min_pos, list) and isinstance(max_pos, list):
        if len(neutral) == len(min_pos) == len(max_pos) == len(configured_names):
            return CalibrationStatus(True, str(PROJECT_JSON_PATH), motor_names)
    return CalibrationStatus(False, None, motor_names)


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
    time.sleep(0.02)
    return ser


def _direct_read_u16(port: str, servo_id: int, addr: int, baudrate: int, retries: int = READ_RETRIES) -> int | None:
    try:
        bus = StableFeetechBus(str(port), int(baudrate))
        try:
            return bus.read_u16(int(servo_id), int(addr), retries=retries)
        finally:
            bus.close()
    except Exception:
        return None


def _direct_read_model(port: str, servo_id: int, baudrate: int) -> int | None:
    return _direct_read_u16(port, int(servo_id), FEETECH_MODEL_NUMBER_ADDR, int(baudrate), retries=1)


def _read_position_with_bus(bus: StableFeetechBus, servo_id: int, retries: int = READ_RETRIES) -> tuple[int | None, int | None]:
    for addr in _position_addr_candidates():
        pos = bus.read_u16(int(servo_id), int(addr), retries=retries)
        if pos is not None:
            return int(pos), int(addr)
    return None, None


def _read_position_any_baud(port: str, servo_id: int, preferred_bauds: Sequence[int] | None = None) -> tuple[int | None, int, int | None]:
    for baud in _get_scan_baudrates(preferred_bauds):
        try:
            bus = StableFeetechBus(str(port), int(baud))
            try:
                pos, addr = _read_position_with_bus(bus, int(servo_id), retries=READ_RETRIES)
                if pos is not None:
                    return int(pos), int(baud), int(addr) if addr is not None else None
            finally:
                bus.close()
        except Exception:
            continue
    fallback = int(_get_scan_baudrates(preferred_bauds)[0])
    return None, fallback, None


def _read_model_any_baud(port: str, servo_id: int, preferred_bauds: Sequence[int] | None = None) -> int | None:
    for baud in _get_scan_baudrates(preferred_bauds):
        model = _direct_read_model(port, int(servo_id), int(baud))
        if model is not None:
            return int(model)
    return None


def _get_session_bus(session: DirectCalibrationSession) -> StableFeetechBus:
    if session.bus is None or not isinstance(session.bus, StableFeetechBus) or int(session.bus.baudrate) != int(session.baudrate):
        if session.bus is not None:
            try:
                session.bus.close()
            except Exception:
                pass
        session.bus = StableFeetechBus(session.port, session.baudrate)
    return session.bus


def _id_candidates(max_id: int, extra_ids: Sequence[int] = ()) -> list[int]:
    ids = list(range(1, int(max_id) + 1)) + [int(x) for x in extra_ids]
    if SETUP_FULL_ID_SCAN or max_id >= 253:
        ids.extend(range(0, 254))
    out: list[int] = []
    for sid in ids:
        try:
            sid = int(sid)
        except Exception:
            continue
        if 0 <= sid <= 253 and sid not in out:
            out.append(sid)
    return out


def _setup_baud_candidates() -> list[int]:
    return _get_scan_baudrates([DEFAULT_BAUDRATE, 1000000]) if SETUP_FULL_BAUD_SCAN else list(dict.fromkeys([DEFAULT_BAUDRATE, 1000000]))


def _scan_ids_readonly(
    port: str,
    ids: Sequence[int],
    bauds: Sequence[int],
    retries: int = 1,
    show_misses: bool = False,
) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for baud in bauds:
        print(f"[scan] scanning baudrate {baud}...", flush=True)
        try:
            bus = StableFeetechBus(str(port), int(baud))
        except Exception as exc:
            print(f"[scan]   could not open {port} at {baud}: {exc}", flush=True)
            continue
        try:
            for sid in ids:
                sid = int(sid)
                if sid in seen_ids:
                    continue
                pos, pos_addr = _read_position_with_bus(bus, sid, retries=max(1, retries))
                model = bus.read_u16(sid, FEETECH_MODEL_NUMBER_ADDR, retries=1)
                id_reg = bus.read_u8(sid, FEETECH_ID_REGISTER_ADDR, retries=1)
                if pos is not None or model is not None or id_reg is not None:
                    seen_ids.add(sid)
                    item = {
                        "id": sid,
                        "baud": int(baud),
                        "position": int(pos) if pos is not None else None,
                        "pos_addr": int(pos_addr) if pos_addr is not None else None,
                        "model": int(model) if model is not None else None,
                        "id_register": int(id_reg) if id_reg is not None else None,
                    }
                    found.append(item)
                    print(
                        f"[scan]   FOUND id={sid:3d} baud={int(baud):7d} "
                        f"position={str(pos):>6s} pos_addr={str(pos_addr):>4s} "
                        f"model={str(model):>5s} id_reg={str(id_reg):>4s}",
                        flush=True,
                    )
                elif show_misses:
                    print(f"[scan]   no response id={sid:3d}", flush=True)
        finally:
            bus.close()
    return found


def _scan_single_connected_motor(port: str, target_id: int | None = None) -> tuple[int, int] | None:
    ids = []
    if target_id is not None:
        ids.append(int(target_id))
    ids.extend(_get_configured_motor_ids(_get_configured_motor_names()))
    ids.extend(SETUP_EXTRA_SCAN_IDS)
    ids.extend(_id_candidates(SETUP_MAX_ID))
    ids = list(dict.fromkeys(int(x) for x in ids if 0 <= int(x) <= 253))
    bauds = _setup_baud_candidates()
    print("[calibrate] Read-only setup scan for the single connected motor...", flush=True)
    print(f"[calibrate]   baud candidates: {bauds}", flush=True)
    print(f"[calibrate]   id candidates: {ids}", flush=True)
    print("[calibrate]   no IDs are written during this scan.", flush=True)
    found = _scan_ids_readonly(port, ids, bauds, retries=SETUP_READ_RETRIES, show_misses=False)
    if len(found) != 1:
        if not found:
            print("[calibrate] No motor responded. Check data cable, power, port, or enable full ID/baud scan.", flush=True)
        else:
            print("[calibrate] More than one ID responded. Connect ONLY the requested motor before setup.", flush=True)
        return None
    return int(found[0]["id"]), int(found[0]["baud"])


def _verify_single_connected_id_by_readonly_scan(
    port: str,
    target_id: int,
    baudrate: int,
    old_id: int | None = None,
    attempts: int | None = None,
    delay_s: float | None = None,
) -> bool:
    """Verify an ID change using the same raw read-only scan method as option 5.

    This is the safety-critical check. It does not trust write acknowledgements.
    A flash is accepted only when a read-only scan sees exactly one responding
    servo, the responding packet ID is target_id, and the ID register read from
    that servo also reports target_id.
    """
    ids: list[int] = [int(target_id)]
    if old_id is not None:
        ids.append(int(old_id))
    ids.extend(_get_configured_motor_ids(_get_configured_motor_names()))
    ids.extend(SETUP_EXTRA_SCAN_IDS)
    ids.extend(_id_candidates(max(SETUP_MAX_ID, READONLY_SCAN_MAX_ID), READONLY_SCAN_EXTRA_IDS))
    ids = list(dict.fromkeys(int(x) for x in ids if 0 <= int(x) <= 253))
    attempts = max(1, int(attempts if attempts is not None else ID_FLASH_VERIFY_RETRIES))
    delay_s = float(delay_s if delay_s is not None else ID_FLASH_VERIFY_DELAY_S)

    for attempt in range(1, attempts + 1):
        print(f"[flash] verify by read-only scan attempt {attempt}/{attempts}...", flush=True)
        found = _scan_ids_readonly(port, ids, [int(baudrate)], retries=max(1, SETUP_READ_RETRIES), show_misses=False)
        found_ids = [int(x["id"]) for x in found]
        if len(found) == 1:
            item = found[0]
            packet_id = int(item["id"])
            id_reg = item.get("id_register")
            id_reg_ok = (id_reg is not None and int(id_reg) == int(target_id))
            if packet_id == int(target_id) and (id_reg_ok or not ID_FLASH_REQUIRE_ID_REGISTER_MATCH):
                print(
                    f"[flash] VERIFIED: exactly one motor responded as ID{target_id}; "
                    f"id_reg={id_reg}, baud={int(item['baud'])}, position={item.get('position')}",
                    flush=True,
                )
                return True
        print(f"[flash] verification saw IDs {found_ids}; target ID{target_id} not proven yet.", flush=True)
        time.sleep(delay_s)
    return False


def _verify_unique_id_after_write(port: str, target_id: int, baudrate: int, old_id: int | None = None) -> bool:
    return _verify_single_connected_id_by_readonly_scan(port, target_id, baudrate, old_id=old_id)


def _write_feetech_id_safely(port: str, current_id: int, target_id: int, baudrate: int) -> bool:
    """Safely flash one connected Feetech/ST servo ID.

    This follows the LeRobot-style EEPROM sequence: torque off, unlock EEPROM,
    write ID, close/reopen the bus, then verify with a read-only scan. The write
    packet return value is never treated as proof of success.
    """
    current_id = int(current_id)
    target_id = int(target_id)
    baudrate = int(baudrate)
    if not (0 <= target_id <= 253):
        print(f"[flash] Invalid target ID {target_id}. Must be 0..253.", flush=True)
        return False

    print(f"[flash] Preparing to write ID {target_id} to the single connected motor currently at ID {current_id}.", flush=True)
    print("[flash] Step 1/4: torque off and EEPROM unlock...", flush=True)
    try:
        bus = StableFeetechBus(port, baudrate)
    except Exception as exc:
        print(f"[flash] Could not open bus for ID write: {exc}", flush=True)
        return False

    try:
        bus.write_u8(current_id, FEETECH_TORQUE_ENABLE_ADDR, 0, retries=WRITE_RETRIES)
        time.sleep(0.05)
        bus.write_u8(current_id, FEETECH_LOCK_REGISTER_ADDR, FEETECH_LOCK_UNLOCK_VALUE, retries=WRITE_RETRIES)
        time.sleep(0.08)
        print(f"[flash] Step 2/4: writing ID register addr {FEETECH_ID_REGISTER_ADDR} = {target_id}...", flush=True)
        bus.write_u8(current_id, FEETECH_ID_REGISTER_ADDR, target_id, retries=WRITE_RETRIES)
        time.sleep(0.10)
    finally:
        bus.close()

    print("[flash] Step 3/4: waiting for servo to apply new ID...", flush=True)
    time.sleep(SETUP_POST_WRITE_DELAY_S)
    if SETUP_POWER_CYCLE_AFTER_ID_WRITE:
        input("Power-cycle the servo bus now, then press Enter to verify the new ID... ")

    print("[flash] Step 4/4: verifying with the exact same read-only scan method as option 5...", flush=True)
    verified = _verify_single_connected_id_by_readonly_scan(port, target_id, baudrate, old_id=current_id)
    if not verified:
        print("[flash] FAILED: read-only scan did not prove the new ID. The motor ID was not trusted as changed.", flush=True)
        return False

    if ID_FLASH_LOCK_EEPROM_AFTER_WRITE:
        try:
            bus = StableFeetechBus(port, baudrate)
            try:
                bus.write_u8(target_id, FEETECH_LOCK_REGISTER_ADDR, FEETECH_LOCK_LOCK_VALUE, retries=WRITE_RETRIES)
            finally:
                bus.close()
            print(f"[flash] EEPROM lock restored using ID{target_id}.", flush=True)
        except Exception as exc:
            print(f"[flash] Warning: ID was verified, but EEPROM relock write failed: {exc}", flush=True)
    return True

def _manual_setup_one_motor(port: str, name: str, target_id: int) -> int | None:
    _print_header(f"Set motor ID {target_id}: {name}")
    print(
        f"Disconnect the full daisy chain. Connect ONLY the physical '{name}' motor to the controller board.\n"
        "Do not connect any other servos. Power-cycle the controller/servo bus after changing connections.\n"
        "This setup now verifies by read-only scan before and after writing, so false-positive ID writes are rejected."
    )
    input("Ready? ")
    scanned = _scan_single_connected_motor(port, int(target_id))
    if scanned is None:
        return None
    current_id, baudrate = scanned
    configured_id_to_name = dict(zip(_get_configured_motor_ids(_get_configured_motor_names()), _get_configured_motor_names(), strict=True))
    if current_id == int(target_id):
        print(f"[calibrate] '{name}' already has correct ID {target_id} at baudrate {baudrate}.", flush=True)
        if SETUP_VERIFY_EXACTLY_ONE_ID and not _verify_unique_id_after_write(port, target_id, baudrate, old_id=None):
            print(f"[calibrate] ID {target_id} did not pass exact-one verification.", flush=True)
            return None
        return int(baudrate)
    if SETUP_CONFIRM_REASSIGN_VALID_ID and current_id in configured_id_to_name:
        print("\n[calibrate] SAFETY WARNING:", flush=True)
        print(f"  You are setting {name}/ID{target_id}, but the connected motor currently responds as ID{current_id}.", flush=True)
        print(f"  ID{current_id} is configured as '{configured_id_to_name[current_id]}'.", flush=True)
        print("  This can happen if this physical motor was overwritten earlier, but it can also mean the wrong motor is connected.", flush=True)
        reply = input("Type YES to rewrite this connected motor's ID anyway: ").strip()
        if reply != "YES":
            print("[calibrate] ID rewrite cancelled. No setup metadata was changed.", flush=True)
            return None
    if not _write_feetech_id_safely(port, current_id, int(target_id), int(baudrate)):
        print(f"[calibrate] Failed to safely write/verify ID {target_id}. This setup step is NOT trusted.", flush=True)
        return None
    print(f"[calibrate] Verified '{name}' as the only connected ID {target_id} at baudrate {baudrate}.", flush=True)
    return int(baudrate)


def _run_manual_motor_setup(port: str, motor_names: Sequence[str], motor_ids: Sequence[int]) -> list[int] | None:
    print("[calibrate] Manual 8-motor setup order:")
    for name, mid in zip(motor_names, motor_ids, strict=True):
        print(f"  {mid}: {name}")
    print("\nEach prompt assigns/verifies exactly the configured ID shown above.")
    bauds: list[int] = []
    for name, mid in zip(motor_names, motor_ids, strict=True):
        baud = _manual_setup_one_motor(port, str(name), int(mid))
        if baud is None:
            return None
        bauds.append(int(baud))
    return bauds


def _build_setup_payload_from_config(motor_names: Sequence[str], motor_ids: Sequence[int], motor_baudrates: Sequence[int] | None = None) -> dict[str, Any]:
    if motor_baudrates is None or len(motor_baudrates) != len(motor_names):
        motor_baudrates = [DEFAULT_BAUDRATE] * len(motor_names)
    payload: dict[str, Any] = {
        "created_at_unix": time.time(),
        "motor_names": list(motor_names),
        "motor_ids": [int(x) for x in motor_ids],
        "baudrate": int(motor_baudrates[0]) if motor_baudrates else DEFAULT_BAUDRATE,
        "motor_baudrates": [int(x) for x in motor_baudrates],
        "setup_only": True,
        "safety": {
            "exactly_one_id_verified_after_each_write": bool(SETUP_VERIFY_EXACTLY_ONE_ID),
            "rewrite_valid_configured_ids_requires_confirmation": bool(SETUP_CONFIRM_REASSIGN_VALID_ID),
            "write_result_not_trusted_without_rescan": True,
        },
        "notes": {
            "purpose": "Records that motor IDs were configured and verified separately from joint calibration.",
            "workflow": "Each motor was connected one at a time and verified by read-only scan before/after ID writes.",
        },
    }
    for name, mid, baud in zip(motor_names, motor_ids, motor_baudrates, strict=True):
        payload[str(name)] = {"name": str(name), "id": int(mid), "baudrate": int(baud)}
    return payload


def write_setup_output(payload: dict[str, Any]) -> Path:
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    SETUP_JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    return SETUP_JSON_PATH


def run_motor_setup_only() -> int:
    _print_header("Robot motor setup")
    print("This setup is read-verify-write-verify. It rejects false-positive ID writes.")
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
        print("[calibrate] Motor setup did not complete. No setup file was written.")
        return 1
    path = write_setup_output(_build_setup_payload_from_config(motor_names, motor_ids, motor_baudrates))
    print(f"[calibrate] Saved motor-setup metadata to: {path}")
    print("[calibrate] Motor setup complete. Reconnect the full daisy chain, then run option 5 or 4 before calibration.")
    return 0


def connect_session() -> DirectCalibrationSession:
    ports = _candidate_robot_ports()
    if not ports:
        raise RuntimeError("Could not auto-detect the robot serial port. Set values.REAL_ROBOT_PORT manually.")
    setup_status = get_motor_setup_status()
    if setup_status.configured:
        motor_names = list(setup_status.motor_names)
        motor_ids = list(setup_status.motor_ids or _get_configured_motor_ids(motor_names))
        motor_baudrates = list(setup_status.motor_baudrates or [DEFAULT_BAUDRATE] * len(motor_names))
    else:
        motor_names = _get_configured_motor_names()
        motor_ids = _get_configured_motor_ids(motor_names)
        motor_baudrates = [DEFAULT_BAUDRATE] * len(motor_names)
    model_numbers = _get_configured_motor_model_numbers(motor_names)
    port = str(ports[0])
    print(f"[calibrate] using direct port = {port}", flush=True)
    if setup_status.configured:
        print(f"[calibrate] using setup metadata = {setup_status.source}", flush=True)
    print("[calibrate] using stable persistent serial motor-bus session.", flush=True)
    for name, sid, baud in zip(motor_names, motor_ids, motor_baudrates, strict=True):
        print(f"  {name:16s} id={int(sid):3d} baud={int(baud)}", flush=True)
    session = DirectCalibrationSession(port, list(motor_names), list(motor_ids), list(model_numbers), list(motor_baudrates))
    return session


def _baud_candidates_for_session(session: DirectCalibrationSession, servo_id: int, full_scan: bool = False) -> list[int]:
    base = [session.baud_for_id(int(servo_id)), DEFAULT_BAUDRATE, 1000000]
    return _get_scan_baudrates(base) if full_scan else list(dict.fromkeys(int(b) for b in base if int(b) > 0))


def _read_position_for_session(session: DirectCalibrationSession, servo_id: int, attempts: int = 1, full_baud_scan: bool = False) -> tuple[int | None, int, int | None, int, int]:
    last_baud = session.baud_for_id(int(servo_id))
    total = 0
    ok = 0
    last_pos = None
    last_addr = None
    for baud in _baud_candidates_for_session(session, int(servo_id), full_scan=full_baud_scan):
        try:
            bus = StableFeetechBus(session.port, int(baud))
        except Exception:
            continue
        try:
            for _ in range(max(1, int(attempts))):
                total += 1
                pos, addr = _read_position_with_bus(bus, int(servo_id), retries=1)
                if pos is not None:
                    ok += 1
                    last_pos = int(pos)
                    last_addr = int(addr) if addr is not None else None
                    last_baud = int(baud)
            if ok > 0:
                return last_pos, last_baud, last_addr, ok, total
        finally:
            bus.close()
    return None, last_baud, None, ok, total or max(1, int(attempts))


def read_positions(session: DirectCalibrationSession) -> dict[str, int]:
    positions: dict[str, int] = {}
    failures: list[str] = []
    print("[calibrate] Reading current positions from all configured motors...", flush=True)
    for idx, (name, sid) in enumerate(zip(session.motor_names, session.motor_ids, strict=True), start=1):
        pos, baud, addr, ok, tries = _read_position_for_session(session, int(sid), attempts=max(1, IDENTIFY_ATTEMPTS), full_baud_scan=IDENTIFY_FULL_BAUD_SCAN)
        if pos is None:
            failures.append(f"{name}/ID{int(sid)}")
            print(f"[calibrate]   ({idx}/{len(session.motor_ids)}) {name}/ID{int(sid)}: FAILED ({ok}/{tries})", flush=True)
            continue
        session.set_baud_for_id(int(sid), int(baud))
        positions[name] = int(pos)
        print(f"[calibrate]   ({idx}/{len(session.motor_ids)}) {name}/ID{int(sid)}: {pos} baud={baud} pos_addr={addr} ({ok}/{tries})", flush=True)
    if failures:
        raise RuntimeError("Could not read Present_Position from: " + ", ".join(failures))
    return positions


def set_torque(session: DirectCalibrationSession, enabled: bool) -> None:
    value = 1 if enabled else 0
    failed: list[str] = []
    # Torque writes are best effort because some servos/buses do not return write status.
    for name, sid in zip(session.motor_names, session.motor_ids, strict=True):
        try:
            bus = StableFeetechBus(session.port, session.baud_for_id(int(sid)))
            try:
                bus.write_u8(int(sid), FEETECH_TORQUE_ENABLE_ADDR, value, retries=1)
            finally:
                bus.close()
        except Exception:
            failed.append(f"{name}/ID{int(sid)}")
    if failed:
        print(f"[calibrate] Torque {'enable' if enabled else 'disable'} write attempted; no response from: {', '.join(failed)}")
    else:
        print(f"[calibrate] Torque {'ENABLED' if enabled else 'DISABLED'} for {len(session.motor_names)} motors.")


def _best_effort_identify(session: DirectCalibrationSession) -> None:
    _print_header("Identify / verify configured motors")
    print("Stable verification table. A position of ---- means that ID did not return Present_Position in this pass.\n", flush=True)
    border = "+" + "-" * 18 + "+" + "-" * 6 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 10 + "+"
    print(border)
    print(f"| {'motor':16s} | {'id':>4s} | {'baud':>8s} | {'position':>8s} | {'pos_addr':>8s} | {'ok/try':>8s} |")
    print(border)
    failures: list[str] = []
    for name, sid in zip(session.motor_names, session.motor_ids, strict=True):
        print(f"[calibrate] testing {name}/ID{int(sid)}...", flush=True)
        pos, baud, addr, ok, tries = _read_position_for_session(session, int(sid), attempts=IDENTIFY_ATTEMPTS, full_baud_scan=IDENTIFY_FULL_BAUD_SCAN)
        if pos is None:
            failures.append(f"{name}/ID{int(sid)}")
            pos_txt = "----"
            addr_txt = "----"
        else:
            session.set_baud_for_id(int(sid), int(baud))
            pos_txt = str(int(pos))
            addr_txt = str(int(addr)) if addr is not None else "----"
        print(f"| {name:16s} | {int(sid):4d} | {session.baud_for_id(int(sid)):8d} | {pos_txt:>8s} | {addr_txt:>8s} | {f'{ok}/{tries}':>8s} |", flush=True)
    print(border)
    if failures:
        print("Unreadable: " + ", ".join(failures), flush=True)


def prompt_capture(session: DirectCalibrationSession, title: str, instructions: str) -> dict[str, int]:
    _print_header(title)
    print(instructions)
    input("READY TO CAPTURE? Press Enter now... ")
    return read_positions(session)


def _wait_for_enter_event(prompt: str = "Press Enter when finished moving through all safe ranges... ") -> threading.Event:
    done = threading.Event()
    def _reader() -> None:
        try:
            input(prompt)
        except Exception:
            pass
        done.set()
    threading.Thread(target=_reader, daemon=True).start()
    return done


def _format_live_table(session: DirectCalibrationSession, current: dict[str, int | None], min_pos: dict[str, int | None], max_pos: dict[str, int | None], failures: Sequence[str], elapsed_s: float) -> str:
    lines = ["\nLeRobot-style range table", "Move every joint through its full SAFE range. Press Enter when done.", f"Elapsed: {elapsed_s:6.1f}s"]
    border = "+" + "-" * 18 + "+" + "-" * 6 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 10 + "+"
    lines.append(border)
    lines.append(f"| {'motor':16s} | {'id':>4s} | {'current':>8s} | {'min':>8s} | {'max':>8s} | {'span':>8s} |")
    lines.append(border)
    def fmt(v: int | None) -> str:
        return "----" if v is None else str(int(v))
    for name, sid in zip(session.motor_names, session.motor_ids, strict=True):
        cur = current.get(name)
        mn = min_pos.get(name)
        mx = max_pos.get(name)
        span = None if mn is None or mx is None else int(mx) - int(mn)
        lines.append(f"| {name:16s} | {int(sid):4d} | {fmt(cur):>8s} | {fmt(mn):>8s} | {fmt(mx):>8s} | {fmt(span):>8s} |")
    lines.append(border)
    if failures:
        lines.append("Unreadable this cycle: " + ", ".join(failures))
    return "\n".join(lines)


def capture_joint_limits(session: DirectCalibrationSession) -> tuple[dict[str, int], dict[str, int]]:
    _print_header("Capture MIN/MAX ranges for all motors")
    print("Move the arm slowly through the full SAFE range for every joint. Include gripper open/closed. Do not force hard stops.")
    current: dict[str, int | None] = {name: None for name in session.motor_names}
    min_pos: dict[str, int | None] = {name: None for name in session.motor_names}
    max_pos: dict[str, int | None] = {name: None for name in session.motor_names}
    done = _wait_for_enter_event()
    start = time.monotonic()
    last_print = 0.0
    while not done.is_set():
        failures: list[str] = []
        for name, sid in zip(session.motor_names, session.motor_ids, strict=True):
            pos, baud, _addr, ok, _tries = _read_position_for_session(session, int(sid), attempts=1, full_baud_scan=False)
            if pos is None:
                failures.append(f"{name}/ID{int(sid)}")
                continue
            session.set_baud_for_id(int(sid), int(baud))
            current[name] = int(pos)
            min_pos[name] = int(pos) if min_pos[name] is None else min(int(min_pos[name]), int(pos))
            max_pos[name] = int(pos) if max_pos[name] is None else max(int(max_pos[name]), int(pos))
        now = time.monotonic()
        elapsed = now - start
        if now - last_print >= LIVE_TABLE_PERIOD_S:
            print(_format_live_table(session, current, min_pos, max_pos, failures, elapsed), flush=True)
            last_print = now
        if LIVE_TABLE_MAX_SECONDS > 0 and elapsed >= LIVE_TABLE_MAX_SECONDS:
            break
        time.sleep(0.02)
    missing = [name for name in session.motor_names if min_pos[name] is None or max_pos[name] is None]
    if missing:
        raise RuntimeError("No usable min/max data was captured for: " + ", ".join(missing))
    return {name: int(min_pos[name]) for name in session.motor_names}, {name: int(max_pos[name]) for name in session.motor_names}


def infer_drive_mode(neutral: dict[str, int], max_pos: dict[str, int]) -> dict[str, int]:
    return {name: 0 if _is_gripper(name) else int(max_pos[name] < neutral[name]) for name in neutral}


def infer_homing_offset(neutral: dict[str, int], drive_mode: dict[str, int]) -> dict[str, int]:
    return {name: int(neutral_pos if drive_mode[name] else -neutral_pos) for name, neutral_pos in neutral.items()}


def build_calibration_payload(session: DirectCalibrationSession, neutral: dict[str, int], min_pos: dict[str, int], max_pos: dict[str, int], motor_ids: list[int] | None) -> dict[str, Any]:
    drive_mode = infer_drive_mode(neutral, max_pos)
    homing_offset = infer_homing_offset(neutral, drive_mode)
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
        "calib_mode": ["LINEAR" if _is_gripper(name) else "DEGREE" for name in session.motor_names],
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
    paths = _get_output_json_paths()
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["Robot joint calibration summary", "=" * 40, "JSON files:"]
    for path in paths:
        lines.append(f"  {path}")
    lines.append("")
    for i, name in enumerate(payload["motor_names"]):
        lines += [
            f"{name}:",
            f"  id            = {payload['motor_ids'][i]}",
            f"  baudrate      = {payload['motor_baudrates'][i]}",
            f"  neutral_pos   = {payload['neutral_pos'][i]}",
            f"  min_pos       = {payload['min_pos'][i]}",
            f"  max_pos       = {payload['max_pos'][i]}",
            f"  homing_offset = {payload['homing_offset'][i]}",
            f"  drive_mode    = {payload['drive_mode'][i]}",
            "",
        ]
    TXT_PATH.write_text("\n".join(lines) + "\n")
    return paths


def run_calibration_only() -> int:
    setup_status = get_motor_setup_status()
    if not setup_status.configured:
        print("[calibrate] Motor-ID setup was not detected. Run setup first.")
        return 1
    _print_header("Interactive robot calibration")
    print("This stage uses a stable persistent Feetech motor-bus session.")
    try:
        session = connect_session()
    except Exception as exc:
        print(f"[calibrate] Failed to connect to robot through direct bus: {exc}")
        return 1
    try:
        set_torque(session, enabled=False)
        neutral = prompt_capture(session, "Capture NEUTRAL pose", "Move the arm into your desired neutral/zero pose.")
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
        session.disconnect()


def run_readonly_scan() -> int:
    _print_header("Read-only motor ID scan")
    print("This mode does NOT write, flash, or change servo IDs.")
    ports = _candidate_robot_ports()
    if not ports:
        print("[scan] Could not auto-detect the robot serial port. Set values.REAL_ROBOT_PORT manually.")
        return 1
    port = str(ports[0])
    ids = _id_candidates(READONLY_SCAN_MAX_ID, READONLY_SCAN_EXTRA_IDS)
    bauds = _get_scan_baudrates([DEFAULT_BAUDRATE, 1000000]) if READONLY_SCAN_FULL_BAUD_SCAN else list(dict.fromkeys([DEFAULT_BAUDRATE, 1000000]))
    print(f"[scan] port = {port}")
    print(f"[scan] baud candidates = {bauds}")
    print(f"[scan] ID candidates = {ids}")
    if not READONLY_SCAN_FULL_BAUD_SCAN:
        print("[scan] full baud scan disabled for speed. Set FEETECH_READONLY_SCAN_FULL_BAUD_SCAN = True for slow exhaustive baud scan.")
    print("")
    found = _scan_ids_readonly(port, ids, bauds, retries=max(1, int(getattr(val, "FEETECH_READONLY_SCAN_READ_RETRIES", 2))), show_misses=READONLY_SCAN_SHOW_MISSES)
    print("")
    if not found:
        print("[scan] No responding IDs found in the selected range.")
        return 0
    border = "+" + "-" * 6 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 10 + "+"
    print(border)
    print(f"| {'id':>4s} | {'baud':>8s} | {'position':>8s} | {'pos_addr':>8s} | {'model':>8s} | {'id_reg':>8s} |")
    print(border)
    for item in found:
        print(
            f"| {item['id']:4d} | {item['baud']:8d} | "
            f"{str(item['position']) if item['position'] is not None else '----':>8s} | "
            f"{str(item['pos_addr']) if item['pos_addr'] is not None else '----':>8s} | "
            f"{str(item['model']) if item['model'] is not None else '----':>8s} | "
            f"{str(item['id_register']) if item['id_register'] is not None else '----':>8s} |"
        )
    print(border)
    if len(found) > 1:
        print("[scan] More than one ID responded. If you intended to check one motor, disconnect all other motors.")
    return 0


def run_flash_one_motor_id() -> int:
    _print_header("Flash one connected motor ID")
    print("This mode changes exactly one connected servo ID.")
    print("Disconnect every other servo before continuing. The script will reject multiple responding IDs.")
    ports = _candidate_robot_ports()
    if not ports:
        print("[flash] Could not auto-detect the robot serial port. Set values.REAL_ROBOT_PORT manually.")
        return 1
    port = str(ports[0])
    print(f"[flash] port = {port}")
    raw = input("Enter target ID to flash this single connected motor to [0-253]: ").strip()
    try:
        target_id = int(raw)
    except Exception:
        print(f"[flash] Invalid target ID: {raw!r}")
        return 1
    if not (0 <= target_id <= 253):
        print("[flash] Invalid target ID. Must be in range 0..253.")
        return 1

    input("Connect ONLY the one motor to be flashed, power-cycle the bus, then press Enter... ")
    print("[flash] Pre-write read-only scan. No IDs are written during this scan.", flush=True)
    scanned = _scan_single_connected_motor(port, target_id=target_id)
    if scanned is None:
        print("[flash] Aborting: could not prove that exactly one motor is connected.", flush=True)
        return 1
    current_id, baudrate = scanned
    print(f"[flash] Single connected motor currently responds as ID{current_id} at baud {baudrate}.", flush=True)

    if int(current_id) == int(target_id):
        print(f"[flash] Motor is already ID{target_id}. Verifying with option-5-style read-only scan before returning success...", flush=True)
        if _verify_single_connected_id_by_readonly_scan(port, target_id, baudrate, old_id=None):
            print(f"[flash] SUCCESS: single connected motor is verified as ID{target_id}.", flush=True)
            return 0
        print(f"[flash] FAILED: motor appeared as ID{target_id}, but exact read-only verification failed.", flush=True)
        return 1

    configured_id_to_name = dict(zip(_get_configured_motor_ids(_get_configured_motor_names()), _get_configured_motor_names(), strict=True))
    if current_id in configured_id_to_name:
        print("\n[flash] SAFETY WARNING:", flush=True)
        print(f"  The connected motor currently responds as ID{current_id}, configured as '{configured_id_to_name[current_id]}'.", flush=True)
        print(f"  You are about to rewrite it to ID{target_id}.", flush=True)
        print("  Continue only if this is the physical motor you intend to reassign.", flush=True)
        reply = input("Type YES to flash this connected motor anyway: ").strip()
        if reply != "YES":
            print("[flash] Cancelled. No write attempted.", flush=True)
            return 1
    else:
        reply = input(f"Type YES to flash connected motor ID{current_id} -> ID{target_id}: ").strip()
        if reply != "YES":
            print("[flash] Cancelled. No write attempted.", flush=True)
            return 1

    if not _write_feetech_id_safely(port, current_id, target_id, baudrate):
        print("[flash] FAILED: ID flash was not verified. Do not trust that the motor changed IDs.", flush=True)
        return 1
    print(f"[flash] SUCCESS: motor was verified as ID{target_id} by read-only scan.", flush=True)
    return 0

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
        try:
            _best_effort_identify(session)
            return 0
        finally:
            session.disconnect()
    if mode in {"scan", "readonly_scan", "read_only_scan", "id_scan"}:
        return run_readonly_scan()
    if mode in {"flash", "flash_one", "flash_one_motor", "set_id", "write_id"}:
        return run_flash_one_motor_id()
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
    print("  6) Flash one connected motor ID")
    reply = input("Selection [1/2/3/4/5/6]: ").strip()
    return {"1": "full", "2": "setup", "3": "calibration", "4": "identify", "5": "scan", "6": "flash"}.get(reply, "full")


def main(mode: str | None = None) -> int:
    if mode is None and len(sys.argv) > 1:
        mode = sys.argv[1]
    if mode is None:
        mode = _interactive_menu_choice()
    return run_workflow(mode)


if __name__ == "__main__":
    raise SystemExit(main())