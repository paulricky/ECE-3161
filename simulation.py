from __future__ import annotations

import os
import math
import cv2
import pybullet as p
import pybullet_data

import values as val
import mathmodel as mm
import depthcalibrator as dc


_INTR = None
_WS = None
_EXT = None


def set_calibration(intr, ws, ext=None):
    global _INTR, _WS, _EXT
    _INTR = intr
    _WS = ws
    _EXT = ext


def set_workspace(ws):
    global _WS
    _WS = ws


def set_workspace_homography(H):
    global _WS
    if _WS is None:
        _WS = {}
    _WS["H"] = H


def joint_limits_from_info(info):
    lower = float(info[8])
    upper = float(info[9])
    if upper <= lower:
        return None
    return lower, upper


def find_urdf_path():
    # Prefer values.URDF_PATH layout (repo root)
    root = getattr(val, "URDF_PATH", None)
    if root:
        root = os.path.expanduser(str(root))
    candidates = []
    if root:
        candidates += [
            os.path.join(root, "Simulation", "SO101", "so101_new_calib.urdf"),
            os.path.join(root, "Simulation", "SO101", "so101.urdf"),
            os.path.join(root, "Simulation", "SO101", "urdf", "so101.urdf"),
            os.path.join(root, "Simulation", "SO100", "so100.urdf"),
        ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def lift_robot_to_ground(robot_id):
    aabb_min, aabb_max = p.getAABB(robot_id, -1)
    min_z = aabb_min[2]
    for j in range(p.getNumJoints(robot_id)):
        mn, mx = p.getAABB(robot_id, j)
        min_z = min(min_z, mn[2])
    if min_z < 0.0:
        lift = -min_z + 0.002
        pos, orn = p.getBasePositionAndOrientation(robot_id)
        p.resetBasePositionAndOrientation(robot_id, [pos[0], pos[1], pos[2] + lift], orn)


def setup_pybullet():
    p.connect(p.GUI)
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / float(val.SIM_HZ))
    p.setRealTimeSimulation(0)

    p.loadURDF("plane.urdf")

    urdf_path = find_urdf_path()
    if urdf_path is None:
        raise FileNotFoundError(
            "Could not find an SO-ARM100 URDF under values.URDF_PATH. "
            "Check val.URDF_PATH points to the repo root (SO-ARM100)."
        )

    p.setAdditionalSearchPath(os.path.dirname(urdf_path))

    robot_id = p.loadURDF(
        urdf_path,
        basePosition=[0, 0, 0.05],
        useFixedBase=True,
        flags=p.URDF_USE_INERTIA_FROM_FILE,
    )

    name_to_idx = {}
    idx_to_info = {}
    for ji in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, ji)
        jname = info[1].decode("utf-8")
        name_to_idx[jname] = ji
        idx_to_info[ji] = info

    lift_robot_to_ground(robot_id)

    p.resetDebugVisualizerCamera(
        cameraDistance=0.7,
        cameraYaw=60,
        cameraPitch=-25,
        cameraTargetPosition=[0, 0.20, 0.10],
    )

    return robot_id, name_to_idx, idx_to_info, urdf_path


def dump_joints(robot_id):
    print("\n==== JOINT DUMP ====")
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        name = info[1].decode("utf-8")
        jtype = info[2]
        qidx = info[3]
        lo, hi = float(info[8]), float(info[9])
        link = info[12].decode("utf-8")
        axis = tuple(float(x) for x in info[13])
        tname = {
            p.JOINT_REVOLUTE: "REVOLUTE",
            p.JOINT_PRISMATIC: "PRISMATIC",
            p.JOINT_FIXED: "FIXED",
            p.JOINT_SPHERICAL: "SPHERICAL",
            p.JOINT_PLANAR: "PLANAR",
        }.get(jtype, str(jtype))
        print(f"{j:2d}  {name:22s} type={tname:8s} qIndex={qidx:2d} lim=[{lo:.3f},{hi:.3f}] link={link} axis={axis}")


def pick_joint(name_to_idx, keywords):
    names = list(name_to_idx.keys())
    low = {n: n.lower() for n in names}
    for kw in keywords:
        kw = kw.lower()
        for n in names:
            if kw in low[n]:
                return n, name_to_idx[n]
    return None, None


def pick_all_joints(name_to_idx, keywords):
    names = list(name_to_idx.keys())
    low = {n: n.lower() for n in names}
    hits = []
    for n in names:
        for kw in keywords:
            if kw.lower() in low[n]:
                hits.append((n, name_to_idx[n]))
                break
    hits.sort(key=lambda x: x[1])
    return hits


def discover_so101_joints(name_to_idx):
    joints = {}
    joints["shoulder_pan"] = pick_joint(name_to_idx, ["shoulder_pan", "base_yaw", "pan", "yaw", "joint1", "j1"])
    joints["shoulder_lift"] = pick_joint(name_to_idx, ["shoulder_lift", "shoulder_pitch", "lift", "pitch", "joint2", "j2"])
    joints["elbow_flex"] = pick_joint(name_to_idx, ["elbow_flex", "elbow", "joint3", "j3"])
    joints["wrist_flex"] = pick_joint(name_to_idx, ["wrist_flex", "wrist_pitch", "joint4", "j4"])
    joints["wrist_roll"] = pick_joint(name_to_idx, ["wrist_roll", "wrist_rotate", "roll", "rotate", "joint5", "j5"])
    joints["gripper_all"] = pick_all_joints(name_to_idx, ["finger", "gripper", "claw", "jaw"])
    return joints


def find_gripper_link(robot_id):
    best = None
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        link = info[12].decode("utf-8").lower()
        jname = info[1].decode("utf-8").lower()
        if "gripper" in link or "jaw" in link or "gripper" in jname or "jaw" in jname:
            best = j
    if best is None:
        best = p.getNumJoints(robot_id) - 1
    return best


def set_joint_target(robot_id, idx_to_info, jidx, target):
    info = idx_to_info[jidx]
    lim = joint_limits_from_info(info)
    if lim is not None:
        target = mm.clamp(float(target), lim[0], lim[1])

    p.setJointMotorControl2(
        robot_id,
        jidx,
        p.POSITION_CONTROL,
        targetPosition=float(target),
        force=val.MOTOR_FORCE,
        maxVelocity=2.0,
        positionGain=val.POS_GAIN,
        velocityGain=val.VEL_GAIN,
    )


def set_gripper_targets(robot_id, idx_to_info, gripper_joints, open01):
    open01 = mm.sat01(float(open01))
    for name, jidx in gripper_joints:
        info = idx_to_info[jidx]
        lim = joint_limits_from_info(info)
        if lim is not None:
            lo, hi = lim
            target = lo + (hi - lo) * open01
        else:
            target = mm.lerp(-0.5, 0.5, open01)

        p.setJointMotorControl2(
            robot_id,
            jidx,
            p.POSITION_CONTROL,
            targetPosition=float(target),
            force=val.MOTOR_FORCE,
            positionGain=val.POS_GAIN,
            velocityGain=val.VEL_GAIN,
        )


def spawn_objects():
    ids = []
    col_box = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
    vis_box = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02], rgbaColor=[0.9, 0.2, 0.2, 1.0])
    for k in range(4):
        x = 0.12 + 0.05 * k
        y = 0.25
        z = 0.02
        bid = p.createMultiBody(
            baseMass=0.05,
            baseCollisionShapeIndex=col_box,
            baseVisualShapeIndex=vis_box,
            basePosition=[x, y, z],
        )
        ids.append(bid)
    return ids


def hand_to_ee_pose(hand_lms, depth_cal):
    cx, cy = mm.hand_center_xy(hand_lms)
    tx = mm.sat01(cx)
    ty = mm.sat01(1.0 - cy)

    # depth proxy: inverse hand size (bigger = closer)
    size = mm.hand_size_proxy(hand_lms)
    proxy = 1.0 / max(size, 1e-4)

    # Support multiple DepthCalibrator API variants
    if hasattr(depth_cal, "normalize01"):
        depth01 = float(depth_cal.normalize01(proxy))
    else:
        if hasattr(depth_cal, "update"):
            depth_cal.update(proxy)
        if hasattr(depth_cal, "get_minmax"):
            dmin, dmax = depth_cal.get_minmax()
        elif hasattr(depth_cal, "minmax"):
            dmin, dmax = depth_cal.minmax()
        else:
            dmin, dmax = (0.0, 1.0)
        depth01 = mm.sat01((proxy - float(dmin)) / max(float(dmax) - float(dmin), 1e-9))

    x = mm.lerp(val.WORKSPACE_X_MIN, val.WORKSPACE_X_MAX, tx)
    y = mm.lerp(val.WORKSPACE_Y_MIN, val.WORKSPACE_Y_MAX, depth01)
    z = mm.lerp(val.WORKSPACE_Z_MIN, val.WORKSPACE_Z_MAX, ty)

    q = mm.palm_quat_world_from_landmarks(hand_lms)

    open01 = 0.5
    return [x, y, z], q, open01, depth01


def solve_ik(robot_id, idx_to_info, ee_link_idx, target_pos, target_orn=None, rest_pose=None):
    n = p.getNumJoints(robot_id)

    lower = []
    upper = []
    ranges = []
    for j in range(n):
        info = idx_to_info[j]
        if info[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            lim = joint_limits_from_info(info)
            if lim is None:
                lo, hi = -3.0, 3.0
            else:
                lo, hi = lim
            lower.append(lo)
            upper.append(hi)
            ranges.append(hi - lo)
        else:
            lower.append(0.0)
            upper.append(0.0)
            ranges.append(0.0)

    if rest_pose is None:
        rest_pose = [0.0] * n

    kwargs = dict(
        lowerLimits=lower,
        upperLimits=upper,
        jointRanges=ranges,
        restPoses=rest_pose,
        maxNumIterations=val.IK_MAX_ITERS,
        residualThreshold=val.IK_RESIDUAL_THRESH,
    )

    if target_orn is None:
        sol = p.calculateInverseKinematics(
            robot_id,
            ee_link_idx,
            targetPosition=list(map(float, target_pos)),
            **kwargs,
        )
    else:
        sol = p.calculateInverseKinematics(
            robot_id,
            ee_link_idx,
            targetPosition=list(map(float, target_pos)),
            targetOrientation=target_orn,
            **kwargs,
        )

    return list(sol)


def self_collision_contacts(robot_id):
    pts = p.getContactPoints(bodyA=robot_id, bodyB=robot_id)
    return [c for c in pts if abs(c[9]) > 1e-6]


def update_grasp(robot_id, gripper_link_idx, grip01, object_ids, state, close_thresh, open_thresh, pick_radius):
    grip01 = float(grip01)
    held = state.get("held_id", None)
    cid = state.get("cid", None)

    ee_pos, ee_orn = p.getLinkState(robot_id, gripper_link_idx)[:2]

    if held is None and grip01 < close_thresh:
        best = None
        best_d = 1e9
        for oid in object_ids:
            pos, orn = p.getBasePositionAndOrientation(oid)
            d = mm.dist(pos, ee_pos)
            if d < best_d:
                best_d = d
                best = oid
        if best is not None and best_d < pick_radius:
            cid = p.createConstraint(
                parentBodyUniqueId=robot_id,
                parentLinkIndex=gripper_link_idx,
                childBodyUniqueId=best,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0],
            )
            state["held_id"] = best
            state["cid"] = cid

    if held is not None and grip01 > open_thresh:
        if cid is not None:
            try:
                p.removeConstraint(cid)
            except Exception:
                pass
        state["held_id"] = None
        state["cid"] = None
