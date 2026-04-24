from __future__ import annotations

import importlib
import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
import json
from pathlib import Path

import numpy as np
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

        kwargs = {
            "port": port,
            "id": robot_id,
            "use_degrees": True,
            "max_relative_target": float(getattr(val, "REAL_ROBOT_MAX_RELATIVE_TARGET_DEG", 2.0)),
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

        robot_id = getattr(val, "REAL_ROBOT_ID", "my_awesome_follower_arm")

        cfg = self._build_lerobot_config(SOFollowerConfig, port, robot_id)

        self.robot = SOFollower(cfg)
        self.robot.connect(calibrate=getattr(val, "REAL_ROBOT_AUTO_CALIBRATE", False))
        self.connected = True
        self.last_send_time = 0.0
        self._last_action = None
        self._last_limited_action = None
        for k in self._last_velocity:
            self._last_velocity[k] = 0.0

        self._apply_torque_limits()

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
        if self.robot is not None and self.connected:
            try:
                self.robot.disconnect()
            except Exception as e:
                print(f"[robot_controller] disconnect warning: {e}")
        self.connected = False
        self.robot = None

    def _apply_torque_limits(self):
        if not getattr(val, "REAL_ROBOT_ENABLE_TORQUE_LIMIT", True):
            return
        if self.robot is None:
            return

        percent = float(getattr(val, "REAL_ROBOT_TORQUE_LIMIT_PERCENT", 20.0))
        percent = max(1.0, min(100.0, percent))

        motor_names = list(getattr(val, "REAL_ROBOT_MOTOR_NAMES", [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_yaw",
            "wrist_roll",
            "wrist_pitch",
            "gripper",
        ]))

        register_candidates = [
            "Torque_Limit",
            "Max_Torque",
        ]

        value_candidates = [
            int(round(percent)),
            int(round(percent * 10.23)),
        ]

        applied = False

        bus = getattr(self.robot, "bus", None)
        if bus is None:
            print("[robot_controller] torque limit skipped: no robot.bus")
            return

        for register_name in register_candidates:
            for value in value_candidates:
                try:
                    ids_values = {name: value for name in motor_names}
                    if hasattr(bus, "write"):
                        bus.write(register_name, ids_values)
                    elif hasattr(bus, "sync_write"):
                        bus.sync_write(register_name, ids_values)
                    else:
                        continue
                    print(f"[robot_controller] torque limit applied using {register_name} = {value}")
                    applied = True
                    break
                except Exception:
                    continue
            if applied:
                break

        if not applied:
            print("[robot_controller] torque limit register not applied; using motion limits only")

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
        period = 1.0 / max(float(getattr(val, "REAL_ROBOT_HZ", 20.0)), 1e-6)
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

        print(f"[robot_controller] raw_action = {raw_action}")
        print(f"[robot_controller] limited_action = {limited_action}")

        if self._last_action is not None:
            deadband = float(getattr(val, "REAL_ROBOT_ACTION_DEADBAND_DEG", 0.5))
            moved = any(
                abs(float(limited_action[k]) - float(self._last_action[k])) >= deadband
                for k in limited_action
            )
            if not moved:
                return None

        sent = self.robot.send_action(limited_action)
        print(f"[robot_controller] sent = {sent}")
        self._last_action = dict(sent)
        self.last_send_time = time.time()
        return sent

    def _apply_velocity_and_acceleration_limits(self, action: Dict[str, float]) -> Dict[str, float]:
        if self._last_limited_action is None:
            self._last_limited_action = {}
            for key, target in action.items():
                if key == "gripper.pos":
                    self._last_limited_action[key] = 50.0
                else:
                    self._last_limited_action[key] = 0.0
                self._last_velocity.setdefault(key, 0.0)
            return self._apply_velocity_and_acceleration_limits(action)

        for key, target in action.items():
            if key not in self._last_limited_action:
                self._last_limited_action[key] = 50.0 if key == "gripper.pos" else float(target)
            self._last_velocity.setdefault(key, 0.0)

        dt = 1.0 / max(float(getattr(val, "REAL_ROBOT_HZ", 20.0)), 1e-6)
        vmax = float(getattr(val, "REAL_ROBOT_MAX_VELOCITY_DEG", 25.0))
        amax = float(getattr(val, "REAL_ROBOT_MAX_ACCELERATION_DEG", 20.0))

        out = dict(self._last_limited_action)

        for key, target in action.items():
            prev_pos = float(self._last_limited_action[key])
            prev_vel = float(self._last_velocity.get(key, 0.0))

            desired_vel = (float(target) - prev_pos) / dt
            desired_vel = max(-vmax, min(vmax, desired_vel))

            accel = (desired_vel - prev_vel) / dt
            accel = max(-amax, min(amax, accel))

            new_vel = prev_vel + accel * dt
            new_vel = max(-vmax, min(vmax, new_vel))

            new_pos = prev_pos + new_vel * dt

            if (target - prev_pos) > 0.0:
                new_pos = min(new_pos, float(target))
            else:
                new_pos = max(new_pos, float(target))

            out[key] = float(new_pos)
            self._last_velocity[key] = float(new_vel)

        self._last_limited_action = dict(out)
        return out

    def _joint_command_to_action(self, cmd: JointCommand) -> Dict[str, float]:
        adjusted = apply_joint_direction_conventions([
            cmd.shoulder_pan,
            cmd.shoulder_lift,
            cmd.elbow_flex,
            cmd.wrist_flex,
            cmd.wrist_yaw,
            cmd.wrist_roll,
            cmd.wrist_pitch,
        ])
        return {
            "shoulder_pan.pos": self._rad_to_deg(adjusted[0]),
            "shoulder_lift.pos": self._rad_to_deg(adjusted[1]),
            "elbow_flex.pos": self._rad_to_deg(adjusted[2]),
            "wrist_flex.pos": self._rad_to_deg(adjusted[3]),
            "wrist_yaw.pos": self._rad_to_deg(adjusted[4]),
            "wrist_roll.pos": self._rad_to_deg(adjusted[5]),
            "wrist_pitch.pos": self._rad_to_deg(adjusted[6]),
            "gripper.pos": self._gripper_open01_to_percent(cmd.gripper_open01),
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