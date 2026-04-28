from __future__ import annotations

import importlib
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
import json
from pathlib import Path

import numpy as np
import serial
from serial.tools import list_ports

import values as val


class RealRobotUnavailableError(RuntimeError):
    pass


@dataclass
class JointCommand:
    shoulder_pan: float
    shoulder_lift: float
    elbow_flex: float
    wrist_flex: float
    wrist_yaw: float
    wrist_roll: float
    wrist_pitch: float
    gripper_open01: float


class _DirectFeetechBus:
    """Persistent direct Feetech/ST bus for the project calibration JSON."""

    def __init__(self, port: str, calibration: dict[str, Any]):
        self.port = str(port)
        self.calibration = calibration
        self.motor_names = [str(x) for x in calibration.get("motor_names", [])]
        self.motor_ids = [int(x) for x in calibration.get("motor_ids", [])]
        if not self.motor_names or len(self.motor_names) != len(self.motor_ids):
            self.motor_names = [str(x) for x in getattr(val, "REAL_ROBOT_MOTOR_NAMES", [])]
            self.motor_ids = [int(x) for x in getattr(val, "REAL_ROBOT_MOTOR_IDS", [])]
        raw_bauds = calibration.get("motor_baudrates", [])
        if isinstance(raw_bauds, list) and len(raw_bauds) == len(self.motor_ids):
            self.motor_baudrates = [int(x) for x in raw_bauds]
        else:
            self.motor_baudrates = [int(getattr(val, "REAL_ROBOT_BAUDRATE", 1000000))] * len(self.motor_ids)
        self.baudrate = int(self.motor_baudrates[0]) if self.motor_baudrates else int(getattr(val, "REAL_ROBOT_BAUDRATE", 1000000))
        self._ser = None
        self._entries = self._build_entries()

    @staticmethod
    def _checksum(tail: list[int]) -> int:
        return (~sum(int(x) & 0xFF for x in tail)) & 0xFF

    @classmethod
    def _packet(cls, servo_id: int, instruction: int, params=()) -> bytes:
        params_b = [int(x) & 0xFF for x in params]
        tail = [int(servo_id) & 0xFF, len(params_b) + 2, int(instruction) & 0xFF, *params_b]
        return bytes([0xFF, 0xFF, *tail, cls._checksum(tail)])

    def connect(self) -> None:
        if self._ser is not None and getattr(self._ser, "is_open", False):
            return
        self._ser = serial.Serial(self.port, self.baudrate, timeout=0.03, write_timeout=0.05)
        try:
            self._ser.reset_input_buffer()
            self._ser.reset_output_buffer()
        except Exception:
            pass
        time.sleep(0.05)
        # Stale Goal_Position registers from a prior session would make every
        # motor snap to the old target the moment torque enables. On a current-
        # limited supply this trips the rail. Disable torque, copy each motor's
        # present position into its goal register, clamp torque to a safe floor,
        # then re-enable.
        self.enable_torque(False)
        self._snapshot_positions_to_goals()
        safe_pct = float(getattr(val, "REAL_ROBOT_SAFE_INITIAL_TORQUE_PERCENT", 30.0))
        self.set_torque_limits_percent({name: safe_pct for name in self.motor_names})
        self.enable_torque(True)

    def _snapshot_positions_to_goals(self) -> None:
        pos_addr = int(getattr(val, "FEETECH_PRESENT_POSITION_ADDR", 56))
        goal_addr = int(getattr(val, "FEETECH_GOAL_POSITION_ADDR", 42))
        for sid in self.motor_ids:
            present = self._read_u16(int(sid), pos_addr)
            if present is None:
                continue
            self._write_u16(int(sid), goal_addr, int(present))

    def disconnect(self) -> None:
        try:
            self.enable_torque(False)
        except Exception:
            pass
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
        self._ser = None

    def _ensure_open(self) -> None:
        if self._ser is None or not getattr(self._ser, "is_open", False):
            self.connect()

    def _write_register(self, servo_id: int, addr: int, data: list[int]) -> bool:
        self._ensure_open()
        try:
            self._ser.write(self._packet(int(servo_id), 0x03, [int(addr), *[int(x) & 0xFF for x in data]]))
            self._ser.flush()
            time.sleep(float(getattr(val, "DIRECT_FEETECH_INTER_WRITE_DELAY_S", 0.004)))
            return True
        except Exception as exc:
            print(f"[robot_controller] direct write failed for ID{servo_id} addr={addr}: {exc}")
            return False

    def _read_status(self, timeout_s: float = 0.06):
        self._ensure_open()
        deadline = time.monotonic() + float(timeout_s)
        window = bytearray()
        while time.monotonic() < deadline:
            b = self._ser.read(1)
            if not b:
                continue
            window += b
            if len(window) > 2:
                window[:] = window[-2:]
            if len(window) == 2 and window[0] == 0xFF and window[1] == 0xFF:
                header = self._ser.read(3)
                if len(header) != 3:
                    return None
                sid, length, error = int(header[0]), int(header[1]), int(header[2])
                rest = self._ser.read(max(0, length - 1))
                if len(rest) != max(0, length - 1) or not rest:
                    return None
                params = list(rest[:-1])
                checksum = int(rest[-1])
                if checksum != self._checksum([sid, length, error, *params]):
                    return None
                return sid, error, params
        return None

    def _read_register(self, servo_id: int, addr: int, length: int):
        self._ensure_open()
        try:
            self._ser.reset_input_buffer()
        except Exception:
            pass
        try:
            self._ser.write(self._packet(int(servo_id), 0x02, [int(addr), int(length)]))
            self._ser.flush()
            status = self._read_status(timeout_s=float(getattr(val, "DIRECT_FEETECH_READ_TIMEOUT_S", 0.08)))
            if status is None or int(status[0]) != int(servo_id) or int(status[1]) != 0:
                return None
            params = list(status[2])
            return params[: int(length)] if len(params) >= int(length) else None
        except Exception:
            return None

    def _read_u16(self, servo_id: int, addr: int):
        data = self._read_register(servo_id, addr, 2)
        if data is None or len(data) < 2:
            return None
        return int(data[0]) | (int(data[1]) << 8)

    def _write_u8(self, servo_id: int, addr: int, value: int) -> bool:
        return self._write_register(servo_id, addr, [int(value) & 0xFF])

    def _write_u16(self, servo_id: int, addr: int, value: int) -> bool:
        value = int(round(value))
        return self._write_register(servo_id, addr, [value & 0xFF, (value >> 8) & 0xFF])

    def _motor_name_to_id(self, name: str) -> Optional[int]:
        if name not in self.motor_names:
            return None
        return int(self.motor_ids[self.motor_names.index(name)])

    def set_torque_limits_percent(self, limits_by_name: Dict[str, float]) -> bool:
        """Best-effort runtime torque/current limit write for Feetech/ST servos."""
        addr = int(getattr(val, "FEETECH_TORQUE_LIMIT_ADDR", 48))
        raw_max = int(getattr(val, "FEETECH_TORQUE_LIMIT_RAW_MAX", 1000))
        ok = True
        for name, pct in limits_by_name.items():
            sid = self._motor_name_to_id(str(name))
            if sid is None:
                continue
            pct = max(0.0, min(100.0, float(pct)))
            raw = int(round((pct / 100.0) * raw_max))
            if not self._write_u16(int(sid), addr, raw):
                ok = False
        return ok

    def enable_torque(self, enabled: bool) -> None:
        addr = int(getattr(val, "FEETECH_TORQUE_ENABLE_ADDR", 40))
        value = 1 if enabled else 0
        for sid in self.motor_ids:
            self._write_u8(int(sid), addr, value)

    def write(self, register: str, values) -> None:
        reg = str(register).lower().replace("_", "").replace("-", "")
        if reg in {"torqueenable", "torque"}:
            if isinstance(values, dict):
                for name, value in values.items():
                    if name in self.motor_names:
                        self._write_u8(self.motor_ids[self.motor_names.index(name)], int(getattr(val, "FEETECH_TORQUE_ENABLE_ADDR", 40)), int(value))
            else:
                self.enable_torque(bool(values))
            return
        if reg in {"torquelimit", "maxtorque", "currentlimit", "maxcurrent"}:
            if isinstance(values, dict):
                limits = {str(name): float(value) for name, value in values.items()}
            else:
                limits = {name: float(values) for name in self.motor_names}
            if not self.set_torque_limits_percent(limits):
                raise RuntimeError(f"Some direct torque/current limit writes failed for register {register}")
            return
        raise RuntimeError(f"Unsupported direct bus write register: {register}")

    sync_write = write

    def _build_entries(self) -> dict[str, dict[str, float]]:
        entries = {}
        for i, name in enumerate(self.motor_names):
            item = self.calibration.get(name, {}) if isinstance(self.calibration.get(name, {}), dict) else {}
            neutral_list = self.calibration.get("neutral_pos", [])
            min_list = self.calibration.get("min_pos", [])
            max_list = self.calibration.get("max_pos", [])
            drive_list = self.calibration.get("drive_mode", [])
            neutral = item.get("neutral", neutral_list[i] if i < len(neutral_list) else 2048)
            min_pos = item.get("range_min", item.get("recorded_min", min_list[i] if i < len(min_list) else 0))
            max_pos = item.get("range_max", item.get("recorded_max", max_list[i] if i < len(max_list) else 4095))
            drive_mode = item.get("drive_mode", drive_list[i] if i < len(drive_list) else 0)
            entries[str(name)] = {
                "id": float(self.motor_ids[i]),
                "neutral": float(neutral),
                "range_min": float(min(min_pos, max_pos)),
                "range_max": float(max(min_pos, max_pos)),
                "drive_mode": float(drive_mode),
            }
        return entries

    def _goal_to_raw(self, motor_name: str, target: float) -> int:
        e = self._entries[motor_name]
        if motor_name.lower() == "gripper":
            pct = max(0.0, min(100.0, float(target)))
            raw = e["range_min"] + (pct / 100.0) * (e["range_max"] - e["range_min"])
        else:
            scale = 4096.0 / 360.0
            sign = -1.0 if int(e.get("drive_mode", 0)) else 1.0
            raw = e["neutral"] + sign * float(target) * scale
        raw = max(e["range_min"], min(e["range_max"], raw))
        return int(round(raw))

    def send_action(self, action: Dict[str, float]) -> Dict[str, float]:
        goal_addr = int(getattr(val, "FEETECH_GOAL_POSITION_ADDR", 42))
        sent = {}
        for key, target in action.items():
            motor_name = str(key).removesuffix(".pos")
            if motor_name not in self.motor_names:
                continue
            raw = self._goal_to_raw(motor_name, float(target))
            sid = self.motor_ids[self.motor_names.index(motor_name)]
            self._write_u16(int(sid), goal_addr, raw)
            sent[key] = float(target)
        return sent

    def get_observation(self) -> Dict[str, float]:
        pos_addr = int(getattr(val, "FEETECH_PRESENT_POSITION_ADDR", 56))
        obs = {}
        for name, sid in zip(self.motor_names, self.motor_ids, strict=True):
            raw = self._read_u16(int(sid), pos_addr)
            if raw is None:
                continue
            e = self._entries[name]
            if name.lower() == "gripper":
                denom = max(1.0, e["range_max"] - e["range_min"])
                val_pct = 100.0 * (float(raw) - e["range_min"]) / denom
                obs[f"{name}.pos"] = float(max(0.0, min(100.0, val_pct)))
            else:
                sign = -1.0 if int(e.get("drive_mode", 0)) else 1.0
                obs[f"{name}.pos"] = float(sign * (float(raw) - e["neutral"]) * 360.0 / 4096.0)
        return obs


class _DirectFeetechRobot:
    def __init__(self, port: str, calibration: dict[str, Any]):
        self.bus = _DirectFeetechBus(port, calibration)
        self.action_features = {f"{name}.pos": float for name in self.bus.motor_names}

    def connect(self) -> None:
        self.bus.connect()

    def disconnect(self) -> None:
        self.bus.disconnect()

    def send_action(self, action: Dict[str, float]) -> Dict[str, float]:
        return self.bus.send_action(action)

    def get_observation(self) -> Dict[str, float]:
        return self.bus.get_observation()


class SOArmHardwareController:
    def __init__(self):
        self.robot = None
        self.last_send_time = 0.0
        self.connected = False
        self._last_action = None
        self._last_limited_action = None
        self._last_velocity = {
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": 0.0,
            "elbow_flex.pos": 0.0,
            "wrist_flex.pos": 0.0,
            "wrist_yaw.pos": 0.0,
            "wrist_roll.pos": 0.0,
            "wrist_pitch.pos": 0.0,
            "gripper.pos": 0.0,
        }
        self._pid_integral: Dict[str, float] = {}
        self._pid_prev_error: Dict[str, float] = {}
        self._pid_derivative: Dict[str, float] = {}
        self._pid_prev_time: Optional[float] = None
        self._last_present_joints_rad: Optional[Dict[str, float]] = None
        self._last_present_read_time = 0.0
        self.last_command_diagnostics: Dict[str, Any] = {}
        self._async_lock = threading.Lock()
        self._async_stop = threading.Event()
        self._async_thread: Optional[threading.Thread] = None
        self._pending_cmd: Optional[JointCommand] = None
        self._last_serial_write_ms = 0.0
        self._async_dropped_commands = 0
        self._speed_diag_printed = False

    def _find_candidate_ports(self) -> List[str]:
        ports = list(list_ports.comports())
        candidates = []

        preferred_vid_pid = {
            (0x1A86, 0x7523),
            (0x10C4, 0xEA60),
            (0x0403, 0x6001),
            (0x2341, 0x0043),
            (0x2341, 0x0001),
            (0x2E8A, 0x000A),
        }

        for p in ports:
            device = (p.device or "").strip()
            desc = (p.description or "").lower()
            manu = (p.manufacturer or "").lower()
            hwid = (p.hwid or "").lower()
            vid = p.vid
            pid = p.pid

            score = 0

            if device.startswith("/dev/tty.usb") or device.startswith("/dev/cu.usb"):
                score += 3
            if "usb" in desc or "usb" in hwid:
                score += 2
            if "serial" in desc or "uart" in desc:
                score += 2
            if "feetech" in desc or "waveshare" in desc or "servo" in desc:
                score += 4
            if "wch" in manu or "silicon labs" in manu or "ftdi" in manu or "arduino" in manu:
                score += 2
            if vid is not None and pid is not None and (vid, pid) in preferred_vid_pid:
                score += 5

            if score > 0:
                candidates.append((score, device))

        candidates.sort(key=lambda x: (-x[0], x[1]))
        return [device for _, device in candidates]

    def _auto_detect_port(self) -> Optional[str]:
        candidates = self._find_candidate_ports()
        print(f"[robot_controller] candidate ports = {candidates}")
        if not candidates:
            return None
        return candidates[0]

    def _import_symbol(self, module_path: str, symbol_name: str):
        module = importlib.import_module(module_path)
        return getattr(module, symbol_name)



    def _resolve_project_calibration_file(self) -> Optional[Path]:
        raw = str(getattr(val, "ROBOT_JOINT_CALIBRATION_FILE", "")).strip()
        if not raw:
            return None
        path = Path(raw)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        return path

    def _load_project_calibration_metadata(self) -> Optional[dict]:
        path = self._resolve_project_calibration_file()
        if path is None or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            print(f"[robot_controller] warning: failed to read calibration JSON {path}: {exc}")
            return None

        motor_names = list(payload.get("motor_names", []))
        configured_names = list(getattr(val, "REAL_ROBOT_MOTOR_NAMES", []))
        if motor_names and configured_names and motor_names != configured_names:
            print(
                "[robot_controller] warning: project calibration motor order does not match "
                f"REAL_ROBOT_MOTOR_NAMES. calibration={motor_names}, configured={configured_names}"
            )
        return payload

    def _import_lerobot_so_follower(self):
        attempts = [
            # New LeRobot API: generic SO follower implementation.
            # IMPORTANT: use SO101FollowerConfig/SOFollowerRobotConfig, not the
            # plain SOFollowerConfig mixin, because the plain mixin does not
            # include RobotConfig fields such as `id` and `calibration_dir`.
            (
                "lerobot.robots.so_follower.so_follower",
                "SOFollower",
                "lerobot.robots.so_follower.config_so_follower",
                "SO101FollowerConfig",
            ),
            (
                "lerobot.robots.so_follower.so_follower",
                "SOFollower",
                "lerobot.robots.so_follower.config_so_follower",
                "SOFollowerRobotConfig",
            ),
            # Older LeRobot API fallback.
            (
                "lerobot.robots.so101_follower.so101_follower",
                "SO101Follower",
                "lerobot.robots.so101_follower.config_so101_follower",
                "SO101FollowerConfig",
            ),
        ]

        import_errors = []

        for follower_module, follower_symbol, config_module, config_symbol in attempts:
            try:
                follower_cls = self._import_symbol(follower_module, follower_symbol)
                config_cls = self._import_symbol(config_module, config_symbol)
                print(f"[robot_controller] imported {follower_module}")
                return follower_cls, config_cls
            except Exception as e:
                import_errors.append(f"{follower_module} failed: {e}")

        raise RealRobotUnavailableError(
            "Could not import a compatible SO follower driver from LeRobot. "
            + " | ".join(import_errors)
        )

    def _build_lerobot_config(self, config_cls, port: str, robot_id: str):
        # Different LeRobot releases expose slightly different config classes.
        # Build kwargs defensively so this works with the new generic SOFollower
        # API and with older SO101-specific APIs.
        import inspect

        max_relative_target = self.get_effective_relative_target_deg(1)
        try:
            max_relative_target = float(max_relative_target)
        except Exception:
            max_relative_target = 2.0

        kwargs = {
            "port": port,
            "id": robot_id,
            "use_degrees": True,
            "max_relative_target": max_relative_target,
        }

        driver_calibration_file = str(getattr(val, "LEROBOT_DRIVER_CALIBRATION_FILE", "")).strip()
        if driver_calibration_file:
            calibration_path = Path(driver_calibration_file).expanduser()
            if calibration_path.suffix.lower() == ".json":
                kwargs["calibration_dir"] = calibration_path.parent
            else:
                kwargs["calibration_dir"] = calibration_path

        try:
            sig = inspect.signature(config_cls)
            accepted = set(sig.parameters.keys())
            filtered = {k: v for k, v in kwargs.items() if k in accepted}
        except Exception:
            filtered = kwargs

        try:
            return config_cls(**filtered)
        except TypeError:
            # Last-resort compatibility path: construct the config with the
            # required port only, then attach optional attributes if the object
            # supports them. This intentionally avoids passing unknown kwargs.
            cfg = config_cls(port=port)
            for key, value in kwargs.items():
                if key == "port":
                    continue
                try:
                    setattr(cfg, key, value)
                except Exception:
                    pass
            return cfg

    def _configured_motor_names(self) -> list[str]:
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

    def _configured_motor_ids(self, motor_names: list[str]) -> list[int]:
        ids = list(getattr(val, "REAL_ROBOT_MOTOR_IDS", []))
        if len(ids) == len(motor_names):
            return [int(x) for x in ids]
        return list(range(1, len(motor_names) + 1))

    def get_effective_arm_speed_percent(self) -> float:
        try:
            pct = float(getattr(val, "REAL_ROBOT_ARM_SPEED_PERCENT", 100.0))
        except Exception:
            pct = 100.0
        try:
            min_pct = float(getattr(val, "REAL_ROBOT_MIN_ARM_SPEED_PERCENT", 1.0))
        except Exception:
            min_pct = 1.0
        return max(min_pct, min(100.0, pct))

    def _motor_id_for_key(self, key: str) -> Optional[int]:
        name = str(key).removesuffix(".pos")
        names = self._configured_motor_names()
        ids = self._configured_motor_ids(names)
        if name not in names:
            return None
        idx = names.index(name)
        return int(ids[idx]) if idx < len(ids) else None

    def _is_gripper_motor_id(self, motor_id: Optional[int]) -> bool:
        return int(motor_id or -1) == 8

    def _arm_speed_scale_for_motor(self, motor_id: Optional[int]) -> float:
        if self._is_gripper_motor_id(motor_id):
            return 1.0
        return self.get_effective_arm_speed_percent() / 100.0

    def _rate_scale(self) -> float:
        if not bool(getattr(val, "REAL_ROBOT_APPLY_SPEED_PERCENT_TO_RATES", False)):
            return 1.0
        return self.get_effective_arm_speed_percent() / 100.0

    def get_effective_rate_hz(self, base_value, default: float) -> float:
        try:
            base = float(base_value)
        except Exception:
            base = float(default)
        return max(1e-6, base * self._rate_scale())

    def get_effective_robot_hz(self) -> float:
        base = getattr(val, "REAL_ROBOT_BASE_HZ", getattr(val, "REAL_ROBOT_HZ", 20.0))
        return self.get_effective_rate_hz(base, 20.0)

    def get_effective_command_max_hz(self) -> float:
        base = getattr(val, "REAL_ROBOT_BASE_COMMAND_MAX_HZ", getattr(val, "ROBOT_COMMAND_MAX_HZ", getattr(val, "REAL_ROBOT_HZ", 20.0)))
        return self.get_effective_rate_hz(base, 20.0)

    def get_effective_velocity_deg(self, motor_id: int) -> float:
        base = float(getattr(val, "REAL_ROBOT_GRIPPER_MAX_VELOCITY_DEG", getattr(val, "REAL_ROBOT_MAX_VELOCITY_DEG", 25.0))) if self._is_gripper_motor_id(motor_id) else float(getattr(val, "REAL_ROBOT_MAX_VELOCITY_DEG", 25.0))
        return float(base * self._arm_speed_scale_for_motor(motor_id))

    def get_effective_acceleration_deg(self, motor_id: int) -> float:
        base = float(getattr(val, "REAL_ROBOT_GRIPPER_MAX_ACCELERATION_DEG", getattr(val, "REAL_ROBOT_MAX_ACCELERATION_DEG", 20.0))) if self._is_gripper_motor_id(motor_id) else float(getattr(val, "REAL_ROBOT_MAX_ACCELERATION_DEG", 20.0))
        return float(base * self._arm_speed_scale_for_motor(motor_id))

    def get_effective_relative_target_deg(self, motor_id: int) -> float:
        base = float(getattr(val, "REAL_ROBOT_MAX_RELATIVE_TARGET_DEG", 360.0))
        return float(base * self._arm_speed_scale_for_motor(motor_id))

    def get_effective_torque_percent(self, motor_id: int, base_percent: Optional[float] = None) -> float:
        base = float(getattr(val, "REAL_ROBOT_TORQUE_LIMIT_PERCENT", 100.0) if base_percent is None else base_percent)
        if self._is_gripper_motor_id(motor_id) or not bool(getattr(val, "REAL_ROBOT_APPLY_SPEED_PERCENT_TO_TORQUE", True)):
            return max(0.0, min(100.0, base))
        return max(0.0, min(100.0, base * self._arm_speed_scale_for_motor(motor_id)))

    def _configured_motor_models(self, motor_names: list[str]) -> list[str]:
        models = list(getattr(val, "REAL_ROBOT_MOTOR_MODELS", []))
        if len(models) == len(motor_names):
            return [str(x) for x in models]
        model = str(getattr(val, "REAL_ROBOT_MOTOR_MODEL", "sts3215"))
        return [model for _ in motor_names]

    def _override_lerobot_bus_for_configured_motors(self, robot, cfg) -> None:
        """Replace LeRobot's stock 6-motor SO bus with this project's configured bus.

        The upstream generic SOFollower currently hard-codes the standard six-motor
        SO100/SO101 follower bus inside SOFollower.__init__(). This project uses an
        8-motor chain: seven arm joints plus the gripper. Replacing the bus before
        robot.connect() makes LeRobot's normal connect/setup/calibrate/send_action
        methods operate on the configured 8 motors instead of failing with the
        stock expected IDs {1..6}.
        """
        motor_names = self._configured_motor_names()
        motor_ids = self._configured_motor_ids(motor_names)
        motor_models = self._configured_motor_models(motor_names)

        try:
            from lerobot.motors import Motor, MotorNormMode
            from lerobot.motors.feetech import FeetechMotorsBus
        except Exception as exc:
            print(f"[robot_controller] warning: could not import LeRobot motor bus classes for custom 8-motor setup: {exc}")
            return

        use_degrees = bool(getattr(cfg, "use_degrees", True))
        body_mode = MotorNormMode.DEGREES if use_degrees else MotorNormMode.RANGE_M100_100
        gripper_mode = MotorNormMode.RANGE_0_100

        motors = {}
        for name, motor_id, model in zip(motor_names, motor_ids, motor_models, strict=True):
            norm_mode = gripper_mode if name.lower() == "gripper" else body_mode
            motors[name] = Motor(int(motor_id), str(model), norm_mode)

        try:
            robot.bus = FeetechMotorsBus(
                port=getattr(cfg, "port", None),
                motors=motors,
                calibration=getattr(robot, "calibration", None),
            )
            print(
                "[robot_controller] configured custom motor bus: "
                + str({name: int(m.id) for name, m in motors.items()})
            )
        except Exception as exc:
            print(f"[robot_controller] warning: failed to install custom 8-motor LeRobot bus: {exc}")

    def _connect_direct_project_controller(self, port: str) -> bool:
        project_cal = self._load_project_calibration_metadata()
        if project_cal is None:
            path = self._resolve_project_calibration_file()
            print(f"[robot_controller] direct controller unavailable: calibration JSON not found at {path}")
            return False
        try:
            self.robot = _DirectFeetechRobot(port, project_cal)
            self.robot.connect()
            self.connected = True
            self.last_send_time = 0.0
            self._last_action = None
            self._last_limited_action = None
            for k in self._last_velocity:
                self._last_velocity[k] = 0.0
            self._apply_torque_limits()
            self._print_speed_percent_diagnostics()
            print("[robot_controller] connected using direct project Feetech controller")
            print("[robot_controller] bypassed LeRobot calibration loader to avoid incompatible project JSON calibration formats")
            return True
        except Exception as exc:
            print(f"[robot_controller] direct project controller failed: {exc}")
            self.robot = None
            self.connected = False
            return False

    def connect(self):
        SOFollower, SOFollowerConfig = self._import_lerobot_so_follower()

        port = getattr(val, "REAL_ROBOT_PORT", "").strip()

        if not port:
            port = self._auto_detect_port()

        if not port:
            visible_ports = [p.device for p in list_ports.comports()]
            raise RealRobotUnavailableError(
                "Could not auto-detect the real robot serial port. "
                f"Visible ports: {visible_ports}. "
                "Set values.REAL_ROBOT_PORT manually."
            )

        print(f"[robot_controller] using port = {port}")

        if bool(getattr(val, "REAL_ROBOT_USE_DIRECT_PROJECT_CONTROLLER", True)):
            if self._connect_direct_project_controller(port):
                return
            print("[robot_controller] falling back to LeRobot SO follower driver")

        robot_id = getattr(val, "REAL_ROBOT_ID", "my_awesome_follower_arm")

        cfg = self._build_lerobot_config(SOFollowerConfig, port, robot_id)

        self.robot = SOFollower(cfg)
        self._override_lerobot_bus_for_configured_motors(self.robot, cfg)
        try:
            self.robot.connect(calibrate=getattr(val, "REAL_ROBOT_AUTO_CALIBRATE", False))
        except AttributeError as exc:
            if "copy" in str(exc) and self._connect_direct_project_controller(port):
                return
            raise
        self.connected = True
        self.last_send_time = 0.0
        self._last_action = None
        self._last_limited_action = None
        for k in self._last_velocity:
            self._last_velocity[k] = 0.0

        self._apply_torque_limits()
        self._print_speed_percent_diagnostics()

        project_cal = self._load_project_calibration_metadata()
        if project_cal is not None:
            path = self._resolve_project_calibration_file()
            print(f"[robot_controller] loaded project calibration metadata from {path}")
            neutral = project_cal.get("neutral_pos")
            mins = project_cal.get("min_pos")
            maxs = project_cal.get("max_pos")
            if neutral is not None and mins is not None and maxs is not None:
                count = len(project_cal.get("motor_names", []))
                print(f"[robot_controller] calibration neutral/min/max available for {count} motors")
        else:
            path = self._resolve_project_calibration_file()
            if path is not None:
                print(f"[robot_controller] project calibration file not found at {path}; run calibrate_robot_joints.py first")

        print(f"[robot_controller] connected on {port} with id={robot_id}")

    def disconnect(self):
        self.stop_async_sender()
        if self.robot is not None and self.connected:
            try:
                self.robot.disconnect()
            except Exception as e:
                print(f"[robot_controller] disconnect warning: {e}")
        self.connected = False
        self.robot = None

    def _motor_limit_percent_by_name(self) -> Dict[str, float]:
        default_percent = float(getattr(val, "REAL_ROBOT_TORQUE_LIMIT_PERCENT", 50.0))
        default_percent = max(0.0, min(100.0, default_percent))
        motor_names = self._configured_motor_names()
        motor_ids = self._configured_motor_ids(motor_names)

        groups = getattr(val, "REAL_ROBOT_TORQUE_LIMIT_GROUPS", None)
        by_id = getattr(val, "REAL_ROBOT_TORQUE_LIMIT_BY_MOTOR_ID", {})
        if not isinstance(by_id, dict):
            by_id = {}

        out: Dict[str, float] = {}
        for name, motor_id in zip(motor_names, motor_ids, strict=True):
            value = by_id.get(int(motor_id), by_id.get(str(int(motor_id)), None))
            if value is None and isinstance(groups, dict):
                for group in groups.values():
                    if not isinstance(group, dict):
                        continue
                    ids = {int(x) for x in group.get("motor_ids", [])}
                    names = {str(x) for x in group.get("motor_names", [])}
                    if int(motor_id) in ids or str(name) in names:
                        value = group.get("percent", default_percent)
                        break
            if value is None:
                if int(motor_id) in {1, 2, 3, 4}:
                    value = getattr(val, "REAL_ROBOT_TORQUE_LIMIT_HIGH_LOAD_PERCENT", 75.0)
                elif int(motor_id) in {5, 6, 7, 8}:
                    value = getattr(val, "REAL_ROBOT_TORQUE_LIMIT_LOW_LOAD_PERCENT", 25.0)
                else:
                    value = default_percent
            try:
                pct = float(value)
            except Exception:
                pct = default_percent
            out[str(name)] = self.get_effective_torque_percent(int(motor_id), pct)
        return out

    def _apply_torque_limits(self):
        if not getattr(val, "REAL_ROBOT_ENABLE_TORQUE_LIMIT", True):
            return
        if self.robot is None:
            return

        motor_limits = self._motor_limit_percent_by_name()
        if not motor_limits:
            return

        bus = getattr(self.robot, "bus", None)
        if bus is None:
            print("[robot_controller] torque/current limit skipped: no robot.bus")
            return

        for register_name in ("Torque_Limit", "Max_Torque", "Current_Limit", "Max_Current"):
            try:
                if hasattr(bus, "write"):
                    bus.write(register_name, motor_limits)
                elif hasattr(bus, "sync_write"):
                    bus.sync_write(register_name, motor_limits)
                else:
                    continue
                if bool(getattr(val, "REAL_ROBOT_VERBOSE_ACTION_LOG", False)):
                    print(f"[robot_controller] torque/current limits applied using {register_name}: {motor_limits}")
                else:
                    print("[robot_controller] torque/current limits applied")
                if bool(getattr(val, "REAL_ROBOT_SPEED_PERCENT_VERBOSE", True)):
                    self._print_speed_percent_diagnostics(motor_limits)
                return
            except Exception:
                continue

        print("[robot_controller] torque/current limit register not applied; using motion limits only")
        self._print_speed_percent_diagnostics(motor_limits)

    def _print_speed_percent_diagnostics(self, motor_limits: Optional[Dict[str, float]] = None) -> None:
        if self._speed_diag_printed or not bool(getattr(val, "REAL_ROBOT_SPEED_PERCENT_VERBOSE", True)):
            return
        self._speed_diag_printed = True
        motor_names = self._configured_motor_names()
        motor_ids = self._configured_motor_ids(motor_names)
        if motor_limits is None:
            motor_limits = self._motor_limit_percent_by_name()
        print("[robot_controller] speed-percent diagnostics:")
        print(f"  arm_speed_percent = {self.get_effective_arm_speed_percent():.1f}")
        print(f"  torque_scaled = {bool(getattr(val, 'REAL_ROBOT_APPLY_SPEED_PERCENT_TO_TORQUE', True))}")
        print(f"  rate_scaled = {bool(getattr(val, 'REAL_ROBOT_APPLY_SPEED_PERCENT_TO_RATES', False))}")
        for name, sid in zip(motor_names, motor_ids, strict=True):
            pct = float(motor_limits.get(name, self.get_effective_torque_percent(int(sid)))) if isinstance(motor_limits, dict) else self.get_effective_torque_percent(int(sid))
            print(f"  motor {int(sid)} {name}: effective_torque_current_percent={pct:.1f}")
        print(
            "  motor 1 effective: "
            f"velocity={self.get_effective_velocity_deg(1):.1f} deg/s, "
            f"accel={self.get_effective_acceleration_deg(1):.1f} deg/s^2, "
            f"relative_step={self.get_effective_relative_target_deg(1):.1f} deg"
        )
        print(
            "  gripper motor 8 effective: "
            f"velocity={self.get_effective_velocity_deg(8):.1f}, "
            f"accel={self.get_effective_acceleration_deg(8):.1f}, "
            "gripper_unscaled=True"
        )
        print(
            "[robot_controller] Hardware servo profile speed/accel registers are not configured by this controller; "
            "values.py speed limits are software-side command shaping only."
        )

    def _supported_action_keys(self) -> set[str] | None:
        if self.robot is None:
            return None
        try:
            features = getattr(self.robot, "action_features", None)
            if isinstance(features, dict) and features:
                return set(features.keys())
        except Exception:
            pass
        bus = getattr(self.robot, "bus", None)
        motors = getattr(bus, "motors", None)
        if isinstance(motors, dict) and motors:
            return {f"{name}.pos" for name in motors.keys()}
        motor_names = getattr(bus, "motor_names", None)
        if motor_names:
            return {f"{name}.pos" for name in motor_names}
        return None

    def _filter_action_to_supported_keys(self, action: Dict[str, float]) -> Dict[str, float]:
        supported = self._supported_action_keys()
        if not supported:
            return dict(action)
        out = {k: v for k, v in action.items() if k in supported}
        missing = sorted(set(action.keys()) - set(out.keys()))
        if missing and not getattr(self, "_warned_unsupported_action_keys", False):
            print(
                "[robot_controller] warning: LeRobot driver does not expose these action keys, "
                f"so they will not be sent: {missing}"
            )
            self._warned_unsupported_action_keys = True
        return out

    def ready_to_send(self) -> bool:
        now = time.time()
        period = 1.0 / self.get_effective_command_max_hz()
        return (now - self.last_send_time) >= period

    def send_if_due(self, cmd: JointCommand):
        if not self.connected or self.robot is None:
            print("[robot_controller] send skipped: not connected")
            return None

        if not self.ready_to_send():
            return None

        raw_action = self._filter_action_to_supported_keys(self._joint_command_to_action(cmd))
        if not raw_action:
            print("[robot_controller] send skipped: no command keys match this LeRobot driver")
            return None
        limited_action = self._apply_velocity_and_acceleration_limits(raw_action)
        control_action = self._apply_outer_pid(limited_action)

        if not self._tracking_watchdog_allows_action(control_action):
            return None

        if bool(getattr(val, "REAL_ROBOT_VERBOSE_ACTION_LOG", False)):
            print(f"[robot_controller] raw_action = {raw_action}")
            print(f"[robot_controller] limited_action = {limited_action}")
            if control_action != limited_action:
                print(f"[robot_controller] pid_action = {control_action}")

        if self._last_action is not None:
            deadband = float(getattr(val, "REAL_ROBOT_ACTION_DEADBAND_DEG", 0.5))
            moved = any(
                abs(float(control_action[k]) - float(self._last_action[k])) >= deadband
                for k in control_action
            )
            if not moved:
                return None

        t0 = time.perf_counter()
        sent = self.robot.send_action(control_action)
        self._last_serial_write_ms = (time.perf_counter() - t0) * 1000.0
        if bool(getattr(val, "REAL_ROBOT_VERBOSE_ACTION_LOG", False)):
            print(f"[robot_controller] sent = {sent}")
        self._last_action = dict(sent)
        self.last_send_time = time.time()
        self.last_command_diagnostics.update({
            "serial_write_ms": self._last_serial_write_ms,
            "async_dropped_commands": self._async_dropped_commands,
        })
        return sent

    def start_async_sender(self) -> None:
        """Start a latest-command sender thread. Safe to call even if disabled."""
        if not bool(getattr(val, "REAL_ROBOT_ASYNC_COMMAND_SENDER", True)):
            return
        if self._async_thread is not None and self._async_thread.is_alive():
            return
        self._async_stop.clear()
        self._async_thread = threading.Thread(target=self._async_sender_loop, name="robot-command-sender", daemon=True)
        self._async_thread.start()
        print("[main] Robot command sender running asynchronously")

    def stop_async_sender(self) -> None:
        self._async_stop.set()
        th = self._async_thread
        if th is not None and th.is_alive():
            th.join(timeout=1.0)
        self._async_thread = None

    def submit_latest_command(self, cmd: JointCommand):
        """Submit latest command without blocking the camera loop."""
        if not bool(getattr(val, "REAL_ROBOT_ASYNC_COMMAND_SENDER", True)):
            return self.send_if_due(cmd)
        with self._async_lock:
            if self._pending_cmd is not None and bool(getattr(val, "ROBOT_COMMAND_DROP_OLD", True)):
                self._async_dropped_commands += 1
            self._pending_cmd = cmd
        return None

    def _async_sender_loop(self) -> None:
        while not self._async_stop.is_set():
            cmd = None
            with self._async_lock:
                if self._pending_cmd is not None:
                    cmd = self._pending_cmd
                    self._pending_cmd = None
            if cmd is not None:
                try:
                    self.send_if_due(cmd)
                except Exception as exc:
                    print(f"[robot_controller] async sender warning: {exc}")
            period = 1.0 / self.get_effective_command_max_hz()
            time.sleep(min(0.02, max(0.001, 0.5 * period)))

    def _cached_present_joints_rad(self) -> Optional[Dict[str, float]]:
        max_age = max(0.0, float(getattr(val, "REAL_ROBOT_FEEDBACK_CACHE_S", 0.10)))
        now = time.time()
        if isinstance(self._last_present_joints_rad, dict) and (now - self._last_present_read_time) <= max_age:
            return dict(self._last_present_joints_rad)
        present = self.read_present_joints_rad()
        if isinstance(present, dict):
            self._last_present_joints_rad = dict(present)
            self._last_present_read_time = now
            return dict(present)
        return None

    def _present_action_degrees(self) -> Optional[Dict[str, float]]:
        present = self._cached_present_joints_rad()
        if not isinstance(present, dict):
            return None
        names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "wrist_pitch"]
        try:
            measured_adjusted = apply_joint_direction_conventions([present[name] for name in names])
        except Exception:
            return None
        out = {f"{name}.pos": self._rad_to_deg(measured_adjusted[i]) for i, name in enumerate(names)}
        try:
            gripper_value = float(present.get("gripper_open01", float("nan")))
        except Exception:
            gripper_value = float("nan")
        if not math.isnan(gripper_value):
            out["gripper.pos"] = self._gripper_open01_to_percent(gripper_value)
        return out

    def _pid_gain(self, base_name: str, key: str, default: float) -> float:
        raw = getattr(val, base_name, default)
        index_map = {
            "shoulder_pan.pos": 0, "shoulder_lift.pos": 1, "elbow_flex.pos": 2, "wrist_flex.pos": 3,
            "wrist_yaw.pos": 4, "wrist_roll.pos": 5, "wrist_pitch.pos": 6, "gripper.pos": 7,
        }
        try:
            if isinstance(raw, dict):
                joint_name = key.removesuffix(".pos")
                return float(raw.get(key, raw.get(joint_name, default)))
            if isinstance(raw, (list, tuple)):
                idx = index_map.get(key)
                if idx is not None and idx < len(raw):
                    return float(raw[idx])
                return float(default)
            return float(raw)
        except Exception:
            return float(default)

    def reset_outer_pid(self) -> None:
        self._pid_integral.clear()
        self._pid_prev_error.clear()
        self._pid_derivative.clear()
        self._pid_prev_time = None

    def _apply_outer_pid(self, action: Dict[str, float]) -> Dict[str, float]:
        if not bool(getattr(val, "REAL_ROBOT_ENABLE_OUTER_PID", getattr(val, "REAL_ROBOT_PID_ENABLED", False))):
            return dict(action)
        measured = self._present_action_degrees()
        if not isinstance(measured, dict):
            return dict(action)

        now = time.time()
        if self._pid_prev_time is None:
            dt = 1.0 / self.get_effective_robot_hz()
        else:
            dt = max(1e-3, min(0.5, now - float(self._pid_prev_time)))
        self._pid_prev_time = now

        out = dict(action)
        max_correction = abs(float(getattr(val, "REAL_ROBOT_PID_MAX_CORRECTION_DEG", 2.0)))
        integral_limit = abs(float(getattr(val, "REAL_ROBOT_PID_INTEGRAL_LIMIT_DEG_S", 8.0)))
        deadband = abs(float(getattr(val, "REAL_ROBOT_PID_DEADBAND_DEG", 0.4)))
        alpha = max(0.0, min(1.0, float(getattr(val, "REAL_ROBOT_PID_DERIVATIVE_FILTER_ALPHA", 0.25))))
        include_gripper = bool(getattr(val, "REAL_ROBOT_PID_APPLY_TO_GRIPPER", False))

        for key, target in action.items():
            if key == "gripper.pos" and not include_gripper:
                continue
            if key not in measured:
                continue
            target_deg = float(target)
            measured_deg = float(measured[key])
            error = target_deg - measured_deg
            if abs(error) < deadband:
                error = 0.0
            kp = self._pid_gain("REAL_ROBOT_PID_KP", key, 0.0)
            ki = self._pid_gain("REAL_ROBOT_PID_KI", key, 0.0)
            kd = self._pid_gain("REAL_ROBOT_PID_KD", key, 0.0)
            integral = float(self._pid_integral.get(key, 0.0)) + error * dt
            integral = max(-integral_limit, min(integral_limit, integral))
            self._pid_integral[key] = integral
            prev_error = float(self._pid_prev_error.get(key, error))
            raw_derivative = (error - prev_error) / dt
            prev_derivative = float(self._pid_derivative.get(key, raw_derivative))
            derivative = (1.0 - alpha) * prev_derivative + alpha * raw_derivative
            self._pid_derivative[key] = derivative
            self._pid_prev_error[key] = error
            correction = kp * error + ki * integral + kd * derivative
            correction = max(-max_correction, min(max_correction, correction))
            out[key] = float(target_deg + correction)

        if bool(getattr(val, "REAL_ROBOT_PID_VERBOSE", False)):
            print(f"[robot_controller] outer PID action = {out}")
        return out

    def _watchdog_offender_info(self, name: str, target_deg: float, present_deg: float, err_deg: float) -> str:
        motor_names = list(getattr(val, "REAL_ROBOT_MOTOR_NAMES", []))
        motor_ids = list(getattr(val, "REAL_ROBOT_MOTOR_IDS", []))
        mid = "?"
        if name in motor_names:
            idx = motor_names.index(name)
            if idx < len(motor_ids):
                mid = str(int(motor_ids[idx]))
        return f"{name}/M{mid} cmd={target_deg:+.1f} meas={present_deg:+.1f} err={err_deg:.1f}"

    def _tracking_watchdog_allows_action(self, action: Dict[str, float]) -> bool:
        if not bool(getattr(val, "REAL_ROBOT_ENABLE_TRACKING_WATCHDOG", True)):
            return True
        present = self._present_action_degrees()
        if not isinstance(present, dict):
            return True
        warn_deg = float(getattr(val, "REAL_ROBOT_TRACKING_WARN_DEG", 10.0))
        abort_deg = float(getattr(val, "REAL_ROBOT_TRACKING_ABORT_DEG", 22.0))
        max_err_deg = 0.0
        offenders = []
        for key, target in action.items():
            if key == "gripper.pos" or key not in present:
                continue
            try:
                target_f = float(target)
                present_f = float(present[key])
                err_deg = abs(target_f - present_f)
            except Exception:
                continue
            max_err_deg = max(max_err_deg, err_deg)
            if err_deg >= warn_deg:
                jname = key[:-4] if key.endswith(".pos") else key
                offenders.append((jname, target_f, present_f, err_deg))
        if max_err_deg >= abort_deg:
            details = "; ".join(self._watchdog_offender_info(n, t, p, e) for n, t, p, e in offenders if e >= abort_deg)
            print(f"[robot_controller] tracking watchdog ABORT (threshold {abort_deg:.0f} deg): blocked. max_err={max_err_deg:.1f} deg. offenders: {details}")
            return False
        if max_err_deg >= warn_deg:
            details = "; ".join(self._watchdog_offender_info(n, t, p, e) for n, t, p, e in offenders)
            print(f"[robot_controller] tracking watchdog WARN (threshold {warn_deg:.0f} deg): max_err={max_err_deg:.1f} deg. offenders: {details}")
        return True

    def _tracking_watchdog_allows(self, cmd: JointCommand) -> bool:
        if not bool(getattr(val, "REAL_ROBOT_ENABLE_TRACKING_WATCHDOG", True)):
            return True
        present = self.read_present_joints_rad()
        if not isinstance(present, dict):
            return True
        targets = {
            "shoulder_pan": cmd.shoulder_pan,
            "shoulder_lift": cmd.shoulder_lift,
            "elbow_flex": cmd.elbow_flex,
            "wrist_flex": cmd.wrist_flex,
            "wrist_yaw": cmd.wrist_yaw,
            "wrist_roll": cmd.wrist_roll,
            "wrist_pitch": cmd.wrist_pitch,
        }
        warn_deg = float(getattr(val, "REAL_ROBOT_TRACKING_WARN_DEG", 10.0))
        abort_deg = float(getattr(val, "REAL_ROBOT_TRACKING_ABORT_DEG", 22.0))
        max_err_deg = 0.0
        offenders = []
        for name, target in targets.items():
            if name not in present:
                continue
            try:
                target_deg = math.degrees(float(target))
                present_deg = math.degrees(float(present[name]))
                err_deg = abs(target_deg - present_deg)
            except Exception:
                continue
            max_err_deg = max(max_err_deg, err_deg)
            if err_deg >= warn_deg:
                offenders.append((name, target_deg, present_deg, err_deg))
        if max_err_deg >= abort_deg:
            details = "; ".join(self._watchdog_offender_info(n, t, p, e) for n, t, p, e in offenders if e >= abort_deg)
            print(f"[robot_controller] tracking watchdog ABORT (threshold {abort_deg:.0f} deg): blocked. max_err={max_err_deg:.1f} deg. offenders: {details}")
            return False
        if max_err_deg >= warn_deg:
            details = "; ".join(self._watchdog_offender_info(n, t, p, e) for n, t, p, e in offenders)
            print(f"[robot_controller] tracking watchdog WARN (threshold {warn_deg:.0f} deg): max_err={max_err_deg:.1f} deg. offenders: {details}")
        return True

    def _apply_velocity_and_acceleration_limits(self, action: Dict[str, float]) -> Dict[str, float]:
        if self._last_limited_action is None:
            # Seed the cursor at the servo's actual present position. Without
            # this it starts at 0; the watchdog then blocks every command for
            # ~12 s while the limiter ramps from 0 toward the target, and the
            # moment it crosses the watchdog threshold all eight motors get
            # their first non-trivial goal-position write simultaneously,
            # spiking the supply past its current limit.
            present = self._present_action_degrees()
            self._last_limited_action = {}
            for key, target in action.items():
                if isinstance(present, dict) and key in present:
                    self._last_limited_action[key] = float(present[key])
                elif key == "gripper.pos":
                    self._last_limited_action[key] = 50.0
                else:
                    self._last_limited_action[key] = 0.0
                self._last_velocity.setdefault(key, 0.0)
            return self._apply_velocity_and_acceleration_limits(action)

        for key, target in action.items():
            if key not in self._last_limited_action:
                self._last_limited_action[key] = 50.0 if key == "gripper.pos" else float(target)
            self._last_velocity.setdefault(key, 0.0)

        dt = 1.0 / self.get_effective_robot_hz()
        out = dict(self._last_limited_action)

        for key, target in action.items():
            motor_id = self._motor_id_for_key(key)
            vmax = self.get_effective_velocity_deg(int(motor_id or 0))
            amax = self.get_effective_acceleration_deg(int(motor_id or 0))

            prev_pos = float(self._last_limited_action[key])
            prev_vel = float(self._last_velocity.get(key, 0.0))
            target_f = float(target)
            max_step = abs(self.get_effective_relative_target_deg(int(motor_id or 0)))
            if max_step > 0.0:
                target_f = max(prev_pos - max_step, min(prev_pos + max_step, target_f))

            desired_vel = (target_f - prev_pos) / dt
            desired_vel = max(-vmax, min(vmax, desired_vel))

            accel = (desired_vel - prev_vel) / dt
            accel = max(-amax, min(amax, accel))

            new_vel = prev_vel + accel * dt
            new_vel = max(-vmax, min(vmax, new_vel))

            new_pos = prev_pos + new_vel * dt

            if (target_f - prev_pos) > 0.0:
                new_pos = min(new_pos, target_f)
            else:
                new_pos = max(new_pos, target_f)

            out[key] = float(new_pos)
            self._last_velocity[key] = float(new_vel)

        self._last_limited_action = dict(out)
        self.last_command_diagnostics.update({
            "arm_speed_percent": self.get_effective_arm_speed_percent(),
            "effective_arm_velocity_deg": self.get_effective_velocity_deg(1),
            "effective_arm_acceleration_deg": self.get_effective_acceleration_deg(1),
            "effective_arm_relative_target_deg": self.get_effective_relative_target_deg(1),
            "gripper_unscaled": True,
        })
        return out

    def _joint_command_to_action(self, cmd: JointCommand) -> Dict[str, float]:
        raw_joints = [
            cmd.shoulder_pan,
            cmd.shoulder_lift,
            cmd.elbow_flex,
            cmd.wrist_flex,
            cmd.wrist_yaw,
            cmd.wrist_roll,
            cmd.wrist_pitch,
        ]
        names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "wrist_pitch"]
        fallback = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_yaw": 0.0,
            "wrist_roll": 0.0,
            "wrist_pitch": 0.0,
        }
        if isinstance(self._last_present_joints_rad, dict):
            fallback.update({k: float(v) for k, v in self._last_present_joints_rad.items() if k in fallback and math.isfinite(float(v))})
        clean = []
        clamped = {}
        omitted = []
        for name, raw in zip(names, raw_joints, strict=True):
            try:
                x = float(raw)
            except Exception:
                x = fallback[name]
                omitted.append(name)
            if not math.isfinite(x):
                x = fallback[name]
                omitted.append(name)
            lo = float(getattr(val, f"{name.upper()}_MIN", -math.pi))
            hi = float(getattr(val, f"{name.upper()}_MAX", math.pi))
            if name == "shoulder_pan":
                lo, hi = float(getattr(val, "BASE_PAN_MIN", lo)), float(getattr(val, "BASE_PAN_MAX", hi))
            elif name == "shoulder_lift":
                lo, hi = float(getattr(val, "SHOULDER_LIFT_MIN", lo)), float(getattr(val, "SHOULDER_LIFT_MAX", hi))
            elif name == "elbow_flex":
                lo, hi = float(getattr(val, "ELBOW_MIN", lo)), float(getattr(val, "ELBOW_MAX", hi))
            if hi < lo:
                lo, hi = hi, lo
            x_clamped = max(lo, min(hi, x))
            clean.append(x_clamped)
            clamped[name] = {"requested_rad": x, "clamped_rad": x_clamped}

        adjusted = apply_joint_direction_conventions(clean)
        motor_names = list(getattr(val, "REAL_ROBOT_MOTOR_NAMES", names + ["gripper"]))
        motor_ids = list(getattr(val, "REAL_ROBOT_MOTOR_IDS", list(range(1, len(motor_names) + 1))))
        motor_id_map = {name: int(motor_ids[i]) for i, name in enumerate(motor_names) if i < len(motor_ids)}
        try:
            gripper_open01 = float(cmd.gripper_open01)
        except Exception:
            gripper_open01 = 1.0
        if not math.isfinite(gripper_open01):
            gripper_open01 = 1.0
        gripper_open01 = float(np.clip(gripper_open01, 0.0, 1.0))
        self.last_command_diagnostics = {
            "desired_joint_angles_rad": {name: float(v) for name, v in zip(names, raw_joints, strict=True) if isinstance(v, (int, float)) and math.isfinite(float(v))},
            "clamped_joint_angles_rad": {name: clamped[name]["clamped_rad"] for name in names},
            "motor_ids_used": motor_id_map,
            "wrist_omitted_or_held": [name for name in omitted if name in {"wrist_yaw", "wrist_roll", "wrist_pitch"}],
        }
        return {
            "shoulder_pan.pos": self._rad_to_deg(adjusted[0]),
            "shoulder_lift.pos": self._rad_to_deg(adjusted[1]),
            "elbow_flex.pos": self._rad_to_deg(adjusted[2]),
            "wrist_flex.pos": self._rad_to_deg(adjusted[3]),
            "wrist_yaw.pos": self._rad_to_deg(adjusted[4]),
            "wrist_roll.pos": self._rad_to_deg(adjusted[5]),
            "wrist_pitch.pos": self._rad_to_deg(adjusted[6]),
            "gripper.pos": self._gripper_open01_to_percent(gripper_open01),
        }

    @staticmethod
    def _rad_to_deg(x: float) -> float:
        return float(np.rad2deg(float(x)))

    def _gripper_open01_to_percent(self, open01: float) -> float:
        open01 = float(np.clip(open01, 0.0, 1.0))
        if getattr(val, "INVERT_GRIPPER", False):
            open01 = 1.0 - open01
        return 100.0 * open01

    def read_present_joints_rad(self) -> Optional[Dict[str, float]]:
        """Return the current arm state in the same convention as `JointCommand`
        (radians for joints, 0-1 for gripper). Returns None on any error, so
        callers can fall back to time-based waypoint arrival.

        The LeRobot SO follower driver, when configured with `use_degrees=True`,
        exposes `{name}.pos` observation keys in degrees. We convert to radians
        and then invert both `apply_joint_direction_conventions` (offsets + sign
        flips) and the gripper-percent mapping so the returned dict matches what
        would have been passed into `JointCommand`.
        """
        if not self.connected or self.robot is None:
            return None

        obs = None
        for fn_name in ("get_observation", "capture_observation", "read_observation"):
            fn = getattr(self.robot, fn_name, None)
            if fn is None:
                continue
            try:
                obs = fn()
            except Exception:
                obs = None
                continue
            if isinstance(obs, dict):
                break
        if not isinstance(obs, dict):
            return None

        def _read(name: str) -> Optional[float]:
            for key in (f"{name}.pos", name, f"observation.{name}.pos"):
                if key in obs:
                    try:
                        return float(obs[key])
                    except Exception:
                        return None
            return None

        names = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                 "wrist_flex", "wrist_yaw", "wrist_roll", "wrist_pitch"]
        deg_vals: List[float] = []
        for name in names:
            raw = _read(name)
            if raw is None:
                return None
            deg_vals.append(float(raw))

        rad_vals = [float(np.deg2rad(d)) for d in deg_vals]

        # Reverse apply_joint_direction_conventions: subtract the software
        # offsets first, then undo the sign inversions (sign flip is its own
        # inverse).
        offsets_deg = getattr(val, "REAL_ROBOT_JOINT_OFFSETS_DEG", [0.0] * 7)
        offsets_rad = [math.radians(float(x)) for x in offsets_deg]
        if len(offsets_rad) != 7:
            return None
        rad_vals = [a - b for a, b in zip(rad_vals, offsets_rad)]

        if getattr(val, "INVERT_BASE_PAN", False):
            rad_vals[0] = -rad_vals[0]
        if getattr(val, "INVERT_SHOULDER_LIFT", False):
            rad_vals[1] = -rad_vals[1]
        if getattr(val, "INVERT_ELBOW", False):
            rad_vals[2] = -rad_vals[2]
        if getattr(val, "INVERT_WRIST_FLEX", False):
            rad_vals[3] = -rad_vals[3]
        if getattr(val, "INVERT_WRIST_YAW", False):
            rad_vals[4] = -rad_vals[4]
        if getattr(val, "INVERT_WRIST_ROLL", False):
            rad_vals[5] = -rad_vals[5]
        if getattr(val, "INVERT_WRIST_PITCH", False):
            rad_vals[6] = -rad_vals[6]

        gripper_raw = _read("gripper")
        if gripper_raw is None:
            gripper_open01 = float("nan")
        else:
            g = float(np.clip(gripper_raw / 100.0, 0.0, 1.0))
            if getattr(val, "INVERT_GRIPPER", False):
                g = 1.0 - g
            gripper_open01 = g

        return {
            "shoulder_pan": rad_vals[0],
            "shoulder_lift": rad_vals[1],
            "elbow_flex": rad_vals[2],
            "wrist_flex": rad_vals[3],
            "wrist_yaw": rad_vals[4],
            "wrist_roll": rad_vals[5],
            "wrist_pitch": rad_vals[6],
            "gripper_open01": gripper_open01,
        }


def apply_joint_direction_conventions(jvals: Iterable[float]):
    j = list(map(float, jvals))
    if len(j) != 7:
        raise ValueError(f"Expected 7 arm joints, got {len(j)}")

    if getattr(val, "INVERT_BASE_PAN", False):
        j[0] = -j[0]
    if getattr(val, "INVERT_SHOULDER_LIFT", False):
        j[1] = -j[1]
    if getattr(val, "INVERT_ELBOW", False):
        j[2] = -j[2]
    if getattr(val, "INVERT_WRIST_FLEX", False):
        j[3] = -j[3]
    if getattr(val, "INVERT_WRIST_YAW", False):
        j[4] = -j[4]
    if getattr(val, "INVERT_WRIST_ROLL", False):
        j[5] = -j[5]
    if getattr(val, "INVERT_WRIST_PITCH", False):
        j[6] = -j[6]

    offsets_deg = getattr(val, "REAL_ROBOT_JOINT_OFFSETS_DEG", [0, 0, 0, 0, 0, 0, 0])
    if len(offsets_deg) != 7:
        raise ValueError("REAL_ROBOT_JOINT_OFFSETS_DEG must have length 7")

    offsets_rad = [math.radians(float(x)) for x in offsets_deg]
    return [a + b for a, b in zip(j, offsets_rad)]
