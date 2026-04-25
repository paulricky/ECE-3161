from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import values as val
from robot.mathmodel import solve_ik_from_target, _fk_arm_position, _ee_geom
from robot.controller import JointCommand
from vision.object_detector import Detection, DetectionFrame
from vision.aruco_marker import MarkerPose
from vision.pixel_to_workspace import PixelToWorkspace, in_bounds_xy


_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "wrist_pitch",
)


class PickPlaceState(Enum):
    IDLE = "idle"
    HOMING = "homing"
    SCANNING = "scanning"
    PICKING = "picking"
    ABORT_TO_HOME = "abort_to_home"
    DONE = "done"
    ERROR = "error"


@dataclass
class Waypoint:
    label: str
    gripper_open01: float
    min_dwell_s: float
    arrival_tol_rad: float
    timeout_s: float
    xyz_ws: Optional[np.ndarray] = None      # Cartesian IK target
    rpy_ws: Optional[np.ndarray] = None
    joint_targets: Optional[Dict[str, float]] = None  # direct joint command
    notes: str = ""


@dataclass
class PickPlaceConfig:
    # geometry
    table_z: float
    pick_z_offset: float
    place_z_offset: float
    transit_z_offset: float
    ik_topdown_pitch_offset: float
    ik_abort_position_err: float
    workspace_x_min: float
    workspace_x_max: float
    workspace_y_min: float
    workspace_y_max: float
    # state machine
    arrival_tol_rad: float
    waypoint_dwell_s: float
    gripper_close_dwell_s: float
    gripper_open_dwell_s: float
    waypoint_timeout_s: float
    max_aborts_before_error: int
    continuous_mode: bool
    required_stable_frames: int
    picking_policy: str
    # home
    home_joints_rad: List[float]
    home_gripper_open01: float
    # robot speed (for fallback-arrival time estimate)
    max_velocity_deg: float

    @classmethod
    def from_values(cls, table_z: Optional[float] = None) -> "PickPlaceConfig":
        return cls(
            table_z=float(table_z if table_z is not None
                          else getattr(val, "PICKPLACE_TABLE_Z_M", 0.02)),
            pick_z_offset=float(getattr(val, "PICKPLACE_PICK_Z_OFFSET_M", 0.005)),
            place_z_offset=float(getattr(val, "PICKPLACE_PLACE_Z_OFFSET_M", 0.015)),
            transit_z_offset=float(getattr(val, "PICKPLACE_TRANSIT_Z_OFFSET_M", 0.08)),
            ik_topdown_pitch_offset=float(getattr(val, "IK_TOPDOWN_APPROACH_OFFSET_RAD", 0.05)),
            ik_abort_position_err=float(getattr(val, "IK_ABORT_POSITION_ERR_M", 0.010)),
            workspace_x_min=float(getattr(val, "WORKSPACE_X_MIN", -0.18)),
            workspace_x_max=float(getattr(val, "WORKSPACE_X_MAX", 0.18)),
            workspace_y_min=float(getattr(val, "WORKSPACE_Y_MIN", 0.12)),
            workspace_y_max=float(getattr(val, "WORKSPACE_Y_MAX", 0.38)),
            arrival_tol_rad=float(getattr(val, "PICKPLACE_ARRIVAL_TOL_RAD", 0.03)),
            waypoint_dwell_s=float(getattr(val, "PICKPLACE_WAYPOINT_DWELL_S", 0.3)),
            gripper_close_dwell_s=float(getattr(val, "PICKPLACE_GRIPPER_CLOSE_DWELL_S", 0.8)),
            gripper_open_dwell_s=float(getattr(val, "PICKPLACE_GRIPPER_OPEN_DWELL_S", 0.5)),
            waypoint_timeout_s=float(getattr(val, "PICKPLACE_WAYPOINT_TIMEOUT_S", 8.0)),
            max_aborts_before_error=int(getattr(val, "PICKPLACE_MAX_ABORTS_BEFORE_ERROR", 3)),
            continuous_mode=bool(getattr(val, "PICKPLACE_CONTINUOUS_MODE", True)),
            required_stable_frames=int(getattr(val, "PICKPLACE_SCAN_DETECTION_REQUIRED_FRAMES", 2)),
            picking_policy=str(getattr(val, "PICKPLACE_PICKING_POLICY", "highest_confidence")),
            home_joints_rad=list(getattr(val, "PICKPLACE_HOME_JOINTS_RAD", [0.0] * 7)),
            home_gripper_open01=float(getattr(val, "PICKPLACE_HOME_GRIPPER_OPEN01", 1.0)),
            max_velocity_deg=float(getattr(val, "REAL_ROBOT_MAX_VELOCITY_DEG", 20.0)),
        )


def _wrap_angle(x: float) -> float:
    return math.atan2(math.sin(float(x)), math.cos(float(x)))


def _angle_diff(a: float, b: float) -> float:
    return _wrap_angle(float(a) - float(b))


def _bbox_iou(a: Sequence[int], b: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, (ax2 - ax1)) * max(0, (ay2 - ay1))
    area_b = max(0, (bx2 - bx1)) * max(0, (by2 - by1))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _compute_top_down_rpy(xyz_ws: np.ndarray, theta_ws: float,
                          pitch_offset: float) -> np.ndarray:
    """Build target rpy for `solve_ik_from_target` such that the EE points
    downward with its tool-axis rotated so the gripper jaws align with
    `theta_ws` (an in-plane angle in the workspace frame).

    With the FK convention `R = Rz(yaw) @ Ry(pitch) @ Rx(roll)` and the tool
    offset along local +x (see mathmodel._fk_chain_details), setting
    `pitch = -pi/2 + eps` tips the tool axis down, and the jaw-opening axis
    (tool's local +y after roll) ends up rotated by `roll + yaw` about the
    world z axis. We want that combined angle to equal `theta_ws`, so
    `roll = theta_ws - yaw`.
    """
    x, y = float(xyz_ws[0]), float(xyz_ws[1])
    yaw = math.atan2(y, x) if (x * x + y * y) > 1e-18 else 0.0
    pitch = -math.pi / 2.0 + float(pitch_offset)
    roll = _wrap_angle(float(theta_ws) - yaw)
    return np.array([roll, pitch, yaw], dtype=np.float64)


@dataclass
class _DetectionCandidate:
    detection: Detection
    xyz_ws: np.ndarray
    theta_ws: float


class PickPlacePlanner:
    def __init__(
        self,
        pixel_to_ws: PixelToWorkspace,
        config: PickPlaceConfig,
        drop_marker_id: int,
        lerobot_calibration: Optional[dict] = None,
    ):
        self.pixel_to_ws = pixel_to_ws
        self.cfg = config
        self.drop_marker_id = int(drop_marker_id)
        self.lerobot_calibration = lerobot_calibration

        self.state: PickPlaceState = PickPlaceState.IDLE
        self.waypoints: List[Waypoint] = []
        self.current_index: int = 0
        self.waypoint_start_time: Optional[float] = None
        self.arrived_since: Optional[float] = None

        self.last_commanded: Optional[JointCommand] = None
        self.last_commanded_joints_rad: Optional[Dict[str, float]] = None
        self.last_feedback_joints_rad: Optional[Dict[str, float]] = None

        self.detection_frame: Optional[DetectionFrame] = None
        self.last_detection_timestamp: float = -1.0
        self.detection_stability: List[Detection] = []

        self.drop_marker_pose: Optional[MarkerPose] = None
        self.abort_count: int = 0
        self.last_chosen_detection: Optional[_DetectionCandidate] = None
        self._has_completed_pick: bool = False

    # ------------------------------------------------------------------
    # External update API
    # ------------------------------------------------------------------

    def update_detections(self, df: Optional[DetectionFrame]) -> None:
        self.detection_frame = df
        if df is None:
            return
        if df.timestamp == self.last_detection_timestamp:
            return
        self.last_detection_timestamp = df.timestamp

        cand = self._choose_detection(df.detections)
        if cand is None:
            self.detection_stability.clear()
            return

        if not self.detection_stability:
            self.detection_stability.append(cand.detection)
            return
        prev = self.detection_stability[-1]
        if (prev.class_name == cand.detection.class_name
                and _bbox_iou(prev.bbox_xyxy, cand.detection.bbox_xyxy) > 0.5):
            self.detection_stability.append(cand.detection)
            if len(self.detection_stability) > max(8, self.cfg.required_stable_frames * 2):
                self.detection_stability = self.detection_stability[-self.cfg.required_stable_frames * 2:]
        else:
            self.detection_stability = [cand.detection]

    def update_drop_marker(self, markers: Dict[int, MarkerPose]) -> None:
        self.drop_marker_pose = markers.get(self.drop_marker_id, None)

    def update_robot_feedback(self, joints_rad: Optional[Dict[str, float]]) -> None:
        self.last_feedback_joints_rad = joints_rad

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick(self, now: Optional[float] = None) -> Optional[JointCommand]:
        now = float(time.time() if now is None else now)

        if self.state in (PickPlaceState.ERROR, PickPlaceState.DONE):
            return None

        if self.state == PickPlaceState.IDLE:
            self._start_sequence([self._home_waypoint("return_home_initial")],
                                 PickPlaceState.HOMING, now)
            return self._enter_current_waypoint(now)

        if self.state in (PickPlaceState.HOMING, PickPlaceState.PICKING,
                          PickPlaceState.ABORT_TO_HOME):
            if not self.waypoints:
                self.state = PickPlaceState.SCANNING
                return None

            wp = self.waypoints[self.current_index]
            if self._arrived(wp, now):
                self._log(f"[planner] arrived at '{wp.label}' "
                          f"after {now - (self.waypoint_start_time or now):.2f}s")
                self.current_index += 1
                self.waypoint_start_time = None
                self.arrived_since = None

                if self.current_index >= len(self.waypoints):
                    return self._on_sequence_complete(now)
                return self._enter_current_waypoint(now)
            return self._execute_current_waypoint(now)

        if self.state == PickPlaceState.SCANNING:
            return self._tick_scanning(now)

        return None

    def _tick_scanning(self, now: float) -> Optional[JointCommand]:
        if self.drop_marker_pose is None:
            return None
        if len(self.detection_stability) < max(1, self.cfg.required_stable_frames):
            return None

        cand = self._choose_detection(self.detection_stability[-self.cfg.required_stable_frames:])
        if cand is None:
            return None

        wps = self._build_pick_place_sequence(cand, self.drop_marker_pose)
        if wps is None:
            self._log("[planner] failed to build pick-and-place sequence; abort")
            return self._abort_to_home(now)

        self.last_chosen_detection = cand
        self._log(f"[planner] PLAN: pick '{cand.detection.class_name}' at "
                  f"xy=({cand.xyz_ws[0]:.3f},{cand.xyz_ws[1]:.3f}) "
                  f"theta={math.degrees(cand.theta_ws):.1f}deg -> drop at "
                  f"xy=({self.drop_marker_pose.xyz_workspace[0]:.3f},"
                  f"{self.drop_marker_pose.xyz_workspace[1]:.3f})")
        self._start_sequence(wps, PickPlaceState.PICKING, now)
        return self._enter_current_waypoint(now)

    def _on_sequence_complete(self, now: float) -> Optional[JointCommand]:
        self.detection_stability.clear()
        if self.state == PickPlaceState.ABORT_TO_HOME:
            self.state = PickPlaceState.SCANNING
            self.waypoints = []
            self.current_index = 0
            return None

        if self.state == PickPlaceState.HOMING:
            if self._has_completed_pick and not self.cfg.continuous_mode:
                self._has_completed_pick = False
                self.state = PickPlaceState.DONE
                self.waypoints = []
                self.current_index = 0
                return None
            self._has_completed_pick = False
            self.state = PickPlaceState.SCANNING
            self.waypoints = []
            self.current_index = 0
            return None

        # PICKING finished successfully -> return home, reset the abort streak.
        self.abort_count = 0
        self._has_completed_pick = True
        self.waypoints = [self._home_waypoint("return_home_after_place")]
        self.current_index = 0
        self.state = PickPlaceState.HOMING
        return self._enter_current_waypoint(now)

    # ------------------------------------------------------------------
    # Waypoint execution
    # ------------------------------------------------------------------

    def _start_sequence(self, wps: List[Waypoint], state: PickPlaceState, now: float) -> None:
        self.waypoints = list(wps)
        self.current_index = 0
        self.waypoint_start_time = None
        self.arrived_since = None
        self.state = state

    def _enter_current_waypoint(self, now: float) -> Optional[JointCommand]:
        if self.current_index >= len(self.waypoints):
            return None
        wp = self.waypoints[self.current_index]
        self.waypoint_start_time = now
        self.arrived_since = None
        self._log(f"[planner] -> '{wp.label}' gripper={wp.gripper_open01:.2f}")
        return self._execute_current_waypoint(now)

    def _execute_current_waypoint(self, now: float) -> Optional[JointCommand]:
        wp = self.waypoints[self.current_index]
        cmd = self._waypoint_to_joint_cmd(wp)
        if cmd is None:
            self._log(f"[planner] cannot solve waypoint '{wp.label}'; abort")
            return self._abort_to_home(now)
        # Safety: if we've committed to a pick-and-place and the drop marker
        # disappears before we reach the transit phase, abort. Once we're
        # descending to place we keep going (we have the cached pose).
        if self.state == PickPlaceState.PICKING and wp.label in (
                "approach_pick", "descend_pick", "close_gripper", "lift_pick"):
            if self.drop_marker_pose is None:
                self._log(f"[planner] drop marker lost at '{wp.label}'; abort")
                return self._abort_to_home(now)

        self.last_commanded = cmd
        self.last_commanded_joints_rad = _cmd_to_joints_dict(cmd)
        return cmd

    def _arrived(self, wp: Waypoint, now: float) -> bool:
        if self.waypoint_start_time is None:
            return False

        elapsed = now - self.waypoint_start_time
        if elapsed > wp.timeout_s:
            self._log(f"[planner] WARN timeout on '{wp.label}' ({elapsed:.2f}s)")
            return True

        if self.last_commanded_joints_rad is None:
            return False

        if self.last_feedback_joints_rad is None:
            estimate = self._time_estimate_for_arrival(wp)
            if elapsed >= estimate:
                return True
            return False

        max_err = 0.0
        for name in _JOINT_NAMES:
            cmd_a = float(self.last_commanded_joints_rad.get(name, 0.0))
            fb_a = float(self.last_feedback_joints_rad.get(name, cmd_a))
            err = abs(_angle_diff(cmd_a, fb_a))
            if err > max_err:
                max_err = err

        if max_err <= wp.arrival_tol_rad:
            if self.arrived_since is None:
                self.arrived_since = now
            return (now - self.arrived_since) >= wp.min_dwell_s
        self.arrived_since = None
        return False

    def _time_estimate_for_arrival(self, wp: Waypoint) -> float:
        if self.last_commanded_joints_rad is None or self.last_feedback_joints_rad is None:
            max_delta_rad = 1.0
        else:
            max_delta_rad = 0.0
            for name in _JOINT_NAMES:
                cmd_a = float(self.last_commanded_joints_rad.get(name, 0.0))
                fb_a = float(self.last_feedback_joints_rad.get(name, cmd_a))
                d = abs(_angle_diff(cmd_a, fb_a))
                if d > max_delta_rad:
                    max_delta_rad = d
        max_vel_rad = max(1e-3, math.radians(self.cfg.max_velocity_deg))
        motion_time = max_delta_rad / max_vel_rad
        return min(wp.timeout_s, motion_time + 2.0 * wp.min_dwell_s)

    def _abort_to_home(self, now: float) -> Optional[JointCommand]:
        self.abort_count += 1
        if self.abort_count > self.cfg.max_aborts_before_error:
            self._log(f"[planner] abort count exceeded ({self.abort_count}); ERROR")
            self.state = PickPlaceState.ERROR
            self.waypoints = []
            return None
        self._start_sequence([self._home_waypoint("abort_home",
                                                  gripper_open01=self.cfg.home_gripper_open01)],
                             PickPlaceState.ABORT_TO_HOME, now)
        return self._enter_current_waypoint(now)

    # ------------------------------------------------------------------
    # IK / commands
    # ------------------------------------------------------------------

    def _waypoint_to_joint_cmd(self, wp: Waypoint) -> Optional[JointCommand]:
        if wp.joint_targets is not None:
            jt = wp.joint_targets
            return JointCommand(
                shoulder_pan=float(jt.get("shoulder_pan", 0.0)),
                shoulder_lift=float(jt.get("shoulder_lift", 0.0)),
                elbow_flex=float(jt.get("elbow_flex", 0.0)),
                wrist_flex=float(jt.get("wrist_flex", 0.0)),
                wrist_yaw=float(jt.get("wrist_yaw", 0.0)),
                wrist_roll=float(jt.get("wrist_roll", 0.0)),
                wrist_pitch=float(jt.get("wrist_pitch", 0.0)),
                gripper_open01=float(wp.gripper_open01),
            )

        if wp.xyz_ws is None or wp.rpy_ws is None:
            return None

        if not in_bounds_xy(wp.xyz_ws[:2],
                            self.cfg.workspace_x_min, self.cfg.workspace_x_max,
                            self.cfg.workspace_y_min, self.cfg.workspace_y_max):
            self._log(f"[planner] '{wp.label}' xy out of bounds: "
                      f"({wp.xyz_ws[0]:.3f}, {wp.xyz_ws[1]:.3f})")
            return None

        try:
            prev = self.last_commanded_joints_rad
            sol = solve_ik_from_target(
                target_xyz=wp.xyz_ws,
                target_rpy=wp.rpy_ws,
                gripper_open01=float(wp.gripper_open01),
                lerobot_calibration=self.lerobot_calibration,
                previous_joints=prev,
                ik_mode="planning",
                strict_reachability=True,
            )
        except Exception as exc:
            self._log(f"[planner] IK exception at '{wp.label}': {exc}")
            return None

        # Post-solve FK verification: the solver silently clamps on unreachable
        # targets, so cross-check with an FK reconstruction.
        try:
            achieved = _fk_arm_position(sol, _ee_geom())
            err = float(np.linalg.norm(achieved - np.asarray(wp.xyz_ws, dtype=np.float64)))
        except Exception as exc:
            self._log(f"[planner] FK verify exception at '{wp.label}': {exc}")
            return None

        if err > self.cfg.ik_abort_position_err:
            self._log(f"[planner] IK unreachable '{wp.label}': "
                      f"target={np.asarray(wp.xyz_ws).tolist()} err={err:.4f}m")
            return None

        return JointCommand(
            shoulder_pan=float(sol["shoulder_pan"]),
            shoulder_lift=float(sol["shoulder_lift"]),
            elbow_flex=float(sol["elbow_flex"]),
            wrist_flex=float(sol["wrist_flex"]),
            wrist_yaw=float(sol["wrist_yaw"]),
            wrist_roll=float(sol["wrist_roll"]),
            wrist_pitch=float(sol.get("wrist_pitch", 0.0)),
            gripper_open01=float(sol.get("gripper_open01", wp.gripper_open01)),
        )

    # ------------------------------------------------------------------
    # Waypoint sequence builder
    # ------------------------------------------------------------------

    def _home_waypoint(self, label: str,
                        gripper_open01: Optional[float] = None) -> Waypoint:
        hj = list(self.cfg.home_joints_rad)
        while len(hj) < 7:
            hj.append(0.0)
        targets = {
            "shoulder_pan": float(hj[0]),
            "shoulder_lift": float(hj[1]),
            "elbow_flex": float(hj[2]),
            "wrist_flex": float(hj[3]),
            "wrist_yaw": float(hj[4]),
            "wrist_roll": float(hj[5]),
            "wrist_pitch": float(hj[6]),
        }
        return Waypoint(
            label=label,
            gripper_open01=(self.cfg.home_gripper_open01 if gripper_open01 is None
                            else float(gripper_open01)),
            min_dwell_s=self.cfg.waypoint_dwell_s,
            arrival_tol_rad=self.cfg.arrival_tol_rad,
            timeout_s=self.cfg.waypoint_timeout_s,
            joint_targets=targets,
            notes="direct joint-space home",
        )

    def _build_pick_place_sequence(
        self,
        cand: _DetectionCandidate,
        drop: MarkerPose,
    ) -> Optional[List[Waypoint]]:
        table_z = self.cfg.table_z
        pick_z = table_z + self.cfg.pick_z_offset
        place_z = table_z + self.cfg.place_z_offset
        transit_z = table_z + self.cfg.transit_z_offset

        x_o, y_o = float(cand.xyz_ws[0]), float(cand.xyz_ws[1])
        theta_o = float(cand.theta_ws)
        x_d, y_d = float(drop.xyz_workspace[0]), float(drop.xyz_workspace[1])

        if not in_bounds_xy((x_o, y_o),
                            self.cfg.workspace_x_min, self.cfg.workspace_x_max,
                            self.cfg.workspace_y_min, self.cfg.workspace_y_max):
            return None
        if not in_bounds_xy((x_d, y_d),
                            self.cfg.workspace_x_min, self.cfg.workspace_x_max,
                            self.cfg.workspace_y_min, self.cfg.workspace_y_max):
            return None

        rpy_pick = _compute_top_down_rpy(np.array([x_o, y_o, pick_z]), theta_o,
                                         self.cfg.ik_topdown_pitch_offset)
        rpy_place = _compute_top_down_rpy(np.array([x_d, y_d, place_z]), theta_o,
                                          self.cfg.ik_topdown_pitch_offset)

        dwell = self.cfg.waypoint_dwell_s
        close_dwell = self.cfg.gripper_close_dwell_s
        open_dwell = self.cfg.gripper_open_dwell_s
        tol = self.cfg.arrival_tol_rad
        to = self.cfg.waypoint_timeout_s

        def mkwp(label: str, x: float, y: float, z: float, rpy: np.ndarray,
                 gripper: float, min_dwell: float) -> Waypoint:
            return Waypoint(
                label=label,
                gripper_open01=gripper,
                min_dwell_s=min_dwell,
                arrival_tol_rad=tol,
                timeout_s=to,
                xyz_ws=np.array([x, y, z], dtype=np.float64),
                rpy_ws=rpy.copy(),
            )

        return [
            mkwp("approach_pick",  x_o, y_o, transit_z, rpy_pick,  1.0, dwell),
            mkwp("descend_pick",   x_o, y_o, pick_z,    rpy_pick,  1.0, dwell),
            mkwp("close_gripper",  x_o, y_o, pick_z,    rpy_pick,  0.0, close_dwell),
            mkwp("lift_pick",      x_o, y_o, transit_z, rpy_pick,  0.0, dwell),
            mkwp("transit",        x_d, y_d, transit_z, rpy_place, 0.0, dwell),
            mkwp("descend_place",  x_d, y_d, place_z,   rpy_place, 0.0, dwell),
            mkwp("open_gripper",   x_d, y_d, place_z,   rpy_place, 1.0, open_dwell),
            mkwp("lift_place",     x_d, y_d, transit_z, rpy_place, 1.0, dwell),
        ]

    # ------------------------------------------------------------------
    # Detection picking
    # ------------------------------------------------------------------

    def _choose_detection(
        self,
        detections: Sequence[Detection],
    ) -> Optional[_DetectionCandidate]:
        candidates: List[_DetectionCandidate] = []
        for det in detections:
            proj = self.pixel_to_ws.project(det.center_px)
            if proj is None:
                continue
            xy = proj[:2]
            if not in_bounds_xy(xy,
                                self.cfg.workspace_x_min, self.cfg.workspace_x_max,
                                self.cfg.workspace_y_min, self.cfg.workspace_y_max):
                continue
            theta = self.pixel_to_ws.pixel_angle_to_yaw(det.center_px, det.pixel_angle_rad)
            if theta is None:
                theta = 0.0
            # Keep the grasp yaw in [-pi/2, pi/2] -- gripper is 180-symmetric.
            theta = math.atan2(math.sin(theta), math.cos(theta))
            if theta > math.pi / 2.0:
                theta -= math.pi
            elif theta < -math.pi / 2.0:
                theta += math.pi
            candidates.append(_DetectionCandidate(
                detection=det,
                xyz_ws=np.array([proj[0], proj[1], self.cfg.table_z],
                                dtype=np.float64),
                theta_ws=float(theta),
            ))

        if not candidates:
            return None

        policy = self.cfg.picking_policy.lower()
        if policy == "closest_to_base":
            return min(candidates,
                       key=lambda c: float(np.hypot(c.xyz_ws[0], c.xyz_ws[1])))
        # default: highest_confidence
        return max(candidates, key=lambda c: c.detection.confidence)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        print(msg)


def _cmd_to_joints_dict(cmd: JointCommand) -> Dict[str, float]:
    return {
        "shoulder_pan": float(cmd.shoulder_pan),
        "shoulder_lift": float(cmd.shoulder_lift),
        "elbow_flex": float(cmd.elbow_flex),
        "wrist_flex": float(cmd.wrist_flex),
        "wrist_yaw": float(cmd.wrist_yaw),
        "wrist_roll": float(cmd.wrist_roll),
        "wrist_pitch": float(cmd.wrist_pitch),
        "gripper_open01": float(cmd.gripper_open01),
    }