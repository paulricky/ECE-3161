from __future__ import annotations

import importlib
import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

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

    def _import_lerobot_so101(self):
        attempts = [
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
            "Could not import the SO101 follower driver from LeRobot. "
            + " | ".join(import_errors)
        )

    def connect(self):
        SO101Follower, SO101FollowerConfig = self._import_lerobot_so101()

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

        cfg = SO101FollowerConfig(
            port=port,
            id=robot_id,
            use_degrees=True,
            max_relative_target=float(getattr(val, "REAL_ROBOT_MAX_RELATIVE_TARGET_DEG", 2.0)),
        )

        self.robot = SO101Follower(cfg)
        self.robot.connect(calibrate=getattr(val, "REAL_ROBOT_AUTO_CALIBRATE", False))
        self.connected = True
        self.last_send_time = 0.0
        self._last_action = None
        self._last_limited_action = None
        for k in self._last_velocity:
            self._last_velocity[k] = 0.0

        self._apply_torque_limits()

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

        motor_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_yaw",
            "wrist_roll",
            "gripper",
        ]

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

        raw_action = self._joint_command_to_action(cmd)
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
            self._last_limited_action = {
                "shoulder_pan.pos": 0.0,
                "shoulder_lift.pos": 0.0,
                "elbow_flex.pos": 0.0,
                "wrist_flex.pos": 0.0,
                "wrist_yaw.pos": 0.0,
                "wrist_roll.pos": 0.0,
                "gripper.pos": 50.0,
            }
            return self._apply_velocity_and_acceleration_limits(action)

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
        return {
            "shoulder_pan.pos": self._rad_to_deg(cmd.shoulder_pan),
            "shoulder_lift.pos": self._rad_to_deg(cmd.shoulder_lift),
            "elbow_flex.pos": self._rad_to_deg(cmd.elbow_flex),
            "wrist_flex.pos": self._rad_to_deg(cmd.wrist_flex),
            "wrist_yaw.pos": self._rad_to_deg(cmd.wrist_yaw),
            "wrist_roll.pos": self._rad_to_deg(cmd.wrist_roll),
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


def apply_joint_direction_conventions(jvals: Iterable[float]):
    j = list(map(float, jvals))
    if len(j) != 6:
        raise ValueError(f"Expected 6 arm joints, got {len(j)}")

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

    offsets_deg = getattr(val, "REAL_ROBOT_JOINT_OFFSETS_DEG", [0, 0, 0, 0, 0, 0])
    if len(offsets_deg) != 6:
        raise ValueError("REAL_ROBOT_JOINT_OFFSETS_DEG must have length 6")

    offsets_rad = [math.radians(float(x)) for x in offsets_deg]
    return [a + b for a, b in zip(j, offsets_rad)]