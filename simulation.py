from __future__ import annotations

import os
import numpy as np
import pybullet as p
import pybullet_data

import mathmodel as mm
import values as val


URDF_CANDIDATES = [
    os.path.join(val.URDF_PATH, "Simulation", "SO101", "so101_new_calib.urdf"),
    os.path.join(val.URDF_PATH, "Simulation", "SO101", "so101.urdf"),
    os.path.join(val.URDF_PATH, "Simulation", "SO101", "urdf", "so101.urdf"),
    os.path.join(val.URDF_PATH, "Simulation", "SO100", "so100.urdf"),
]


def find_urdf_path():
    for path in URDF_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def joint_limits_from_info(info):
    lo = float(info[8])
    hi = float(info[9])
    if hi <= lo:
        return None
    return lo, hi


def setup_pybullet():
    p.connect(p.GUI)
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / val.SIM_HZ)
    p.setRealTimeSimulation(0)

    p.loadURDF("plane.urdf")

    urdf_path = find_urdf_path()
    if urdf_path is None:
        raise FileNotFoundError("Could not find SO-ARM100 URDF.\nTried:\n - " + "\n - ".join(URDF_CANDIDATES))

    p.setAdditionalSearchPath(os.path.dirname(urdf_path))

    flags = (
        p.URDF_USE_INERTIA_FROM_FILE
        | p.URDF_USE_SELF_COLLISION
        | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
    )

    robot_id = p.loadURDF(
        urdf_path,
        basePosition=[0, 0, 0.05],
        useFixedBase=True,
        flags=flags,
    )

    name_to_idx = {}
    idx_to_info = {}
    for ji in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, ji)
        name_to_idx[info[1].decode("utf-8")] = ji
        idx_to_info[ji] = info

    lift_robot_to_ground(robot_id)
    setup_collision_filters(robot_id)

    p.resetDebugVisualizerCamera(
        cameraDistance=0.7,
        cameraYaw=60,
        cameraPitch=-25,
        cameraTargetPosition=[0, 0.20, 0.12],
    )

    return robot_id, name_to_idx, idx_to_info, urdf_path


def lift_robot_to_ground(robot_id):
    min_z = 1e9
    mn, _mx = p.getAABB(robot_id, -1)
    min_z = min(min_z, mn[2])
    for j in range(p.getNumJoints(robot_id)):
        mn, _mx = p.getAABB(robot_id, j)
        min_z = min(min_z, mn[2])
    if min_z < 0.0:
        lift = -min_z + 0.002
        pos, orn = p.getBasePositionAndOrientation(robot_id)
        p.resetBasePositionAndOrientation(robot_id, [pos[0], pos[1], pos[2] + lift], orn)


def setup_collision_filters(robot_id):
    # Stop “default pose constant collision” by disabling collisions on adjacent links
    n = p.getNumJoints(robot_id)
    for i in range(-1, n):
        for j in range(-1, n):
            if i == j:
                continue
            if abs(i - j) <= 1:
                p.setCollisionFilterPair(robot_id, robot_id, i, j, enableCollision=0)


def dump_joints(robot_id):
    print("\n==== JOINT DUMP ====")
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        jname = info[1].decode("utf-8")
        jtype = info[2]
        qIndex = info[3]
        lo, hi = float(info[8]), float(info[9])
        link = info[12].decode("utf-8")
        axis = info[13]
        tname = {p.JOINT_REVOLUTE: "REVOLUTE", p.JOINT_PRISMATIC: "PRISMATIC", p.JOINT_FIXED: "FIXED"}.get(jtype, str(jtype))
        print(f"{j:2d}  {jname:22s} type={tname:8s} qIndex={qIndex:2d} lim=[{lo:.3f},{hi:.3f}] link={link} axis={axis}")
    print("")


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
    joints["shoulder_lift"] = pick_joint(name_to_idx, ["shoulder_lift", "lift", "pitch", "joint2", "j2"])
    joints["elbow_flex"] = pick_joint(name_to_idx, ["elbow_flex", "elbow", "joint3", "j3"])
    joints["wrist_flex"] = pick_joint(name_to_idx, ["wrist_flex", "wrist_pitch", "joint4", "j4"])
    joints["wrist_roll"] = pick_joint(name_to_idx, ["wrist_roll", "wrist_rotate", "roll", "rotate", "joint5", "j5"])
    joints["gripper_all"] = pick_all_joints(name_to_idx, ["finger", "gripper", "claw", "jaw"])
    return joints


def find_gripper_link(robot_id):
    # Prefer the moving jaw/gripper link if present
    for j in range(p.getNumJoints(robot_id) - 1, -1, -1):
        info = p.getJointInfo(robot_id, j)
        link = info[12].decode("utf-8").lower()
        if "gripper" in link or "jaw" in link:
            return j
    return p.getNumJoints(robot_id) - 1


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
    open01 = mm.sat01(open01)
    for _name, jidx in gripper_joints:
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
            maxVelocity=2.0,
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
        bid = p.createMultiBody(baseMass=0.05, baseCollisionShapeIndex=col_box, baseVisualShapeIndex=vis_box, basePosition=[x, y, z])
        ids.append(bid)
    return ids


def hand_to_ee_pose(hand_lms, depth_cal):
    cx, cy = mm.hand_center_xy(hand_lms)
    tx = mm.sat01(cx)
    ty = mm.sat01(1.0 - cy)

    # depth proxy: inverse hand size (bigger = closer)
    size = mm.hand_size_proxy(hand_lms)
    proxy = 1.0 / max(size, 1e-4)
    depth01 = depth_cal.normalize01(proxy)

    x = mm.lerp(val.WORKSPACE_X_MIN, val.WORKSPACE_X_MAX, tx)
    y = mm.lerp(val.WORKSPACE_Y_MIN, val.WORKSPACE_Y_MAX, depth01)
    z = mm.lerp(val.WORKSPACE_Z_MIN, val.WORKSPACE_Z_MAX, ty)

    # palm orientation in world-ish frame (stable basis + SVD)
    q = mm.palm_quat_world_from_landmarks(hand_lms)

    # open/close based on full hand (not pinch)
    # 0=open, 1=closed for gripper targets in main logic
    # We'll output open01 so main can use directly
    # Use a smoother proxy: average finger curl -> open fraction
    # (fallback: 2D distance thumb-index if you want)
    # Here: treat "hand size" + palm normal stability as enough; prefer finger state in handtracking
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

    if target_orn is None:
        sol = p.calculateInverseKinematics(
            robot_id,
            ee_link_idx,
            targetPosition=list(map(float, target_pos)),
            lowerLimits=lower,
            upperLimits=upper,
            jointRanges=ranges,
            restPoses=rest_pose,
            maxNumIterations=val.IK_MAX_ITERS,
            residualThreshold=val.IK_RESIDUAL_THRESH,
        )
    else:
        sol = p.calculateInverseKinematics(
            robot_id,
            ee_link_idx,
            targetPosition=list(map(float, target_pos)),
            targetOrientation=target_orn,
            lowerLimits=lower,
            upperLimits=upper,
            jointRanges=ranges,
            restPoses=rest_pose,
            maxNumIterations=val.IK_MAX_ITERS,
            residualThreshold=val.IK_RESIDUAL_THRESH,
        )

    return list(sol)


def self_collision_contacts(robot_id):
    pts = p.getContactPoints(bodyA=robot_id, bodyB=robot_id)
    # ignore trivial/near-zero normal force contacts
    return [c for c in pts if abs(c[9]) > 1e-6]


def update_grasp(robot_id, gripper_link_idx, grip01, object_ids, state, close_thresh, open_thresh, pick_radius):
    held = state.get("held_id")
    cid = state.get("cid")

    ee_pos, ee_orn = p.getLinkState(robot_id, gripper_link_idx)[:2]

    if held is None and grip01 < close_thresh:
        best = None
        best_d = 1e9
        for oid in object_ids:
            op, _ = p.getBasePositionAndOrientation(oid)
            d = mm.dist(op, ee_pos)
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
            p.removeConstraint(cid)
        state["held_id"] = None
        state["cid"] = None
