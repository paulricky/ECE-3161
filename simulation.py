import os
import math
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


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def find_urdf_path():
    for path in URDF_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def joint_limits_from_info(info):
    lower = float(info[8])
    upper = float(info[9])
    if upper <= lower:
        return None
    return lower, upper


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
        raise FileNotFoundError(
            "Could not find an SO-ARM100 URDF.\nTried:\n  - " + "\n  - ".join(URDF_CANDIDATES)
        )

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

    # build joint map
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

        tname = {
            p.JOINT_REVOLUTE: "REVOLUTE",
            p.JOINT_PRISMATIC: "PRISMATIC",
            p.JOINT_FIXED: "FIXED",
            p.JOINT_SPHERICAL: "SPHERICAL",
            p.JOINT_PLANAR: "PLANAR",
        }.get(jtype, str(jtype))

        print(
            f"{j:2d}  {jname:22s} type={tname:8s} qIndex={qIndex:2d} "
            f"lim=[{lo:.3f},{hi:.3f}] link={link} axis={axis}"
        )
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

    joints["shoulder_pan"] = pick_joint(
        name_to_idx, ["shoulder_pan", "base_yaw", "pan", "yaw", "joint1", "j1"]
    )
    joints["shoulder_lift"] = pick_joint(
        name_to_idx, ["shoulder_lift", "lift", "pitch", "joint2", "j2"]
    )
    joints["elbow_flex"] = pick_joint(
        name_to_idx, ["elbow_flex", "elbow", "joint3", "j3"]
    )
    joints["wrist_flex"] = pick_joint(
        name_to_idx, ["wrist_flex", "wrist_pitch", "joint4", "j4"]
    )
    joints["wrist_roll"] = pick_joint(
        name_to_idx, ["wrist_roll", "wrist_rotate", "roll", "rotate", "joint5", "j5"]
    )

    joints["gripper_all"] = pick_all_joints(
        name_to_idx, ["finger", "gripper", "claw", "jaw"]
    )

    return joints


def find_ee_link_for_ik(robot_id):
    """
    Prefer the fixed gripper frame link if present; else fall back to a late wrist/gripper link.
    """
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        jname = info[1].decode("utf-8").lower()
        link = info[12].decode("utf-8").lower()
        if "gripper_frame" in jname or "gripper_frame" in link:
            return j

    for j in range(p.getNumJoints(robot_id) - 1, -1, -1):
        info = p.getJointInfo(robot_id, j)
        link = info[12].decode("utf-8").lower()
        if "gripper_link" in link or "wrist" in link:
            return j

    return p.getNumJoints(robot_id) - 1


def set_joint_target(robot_id, idx_to_info, jidx, target):
    info = idx_to_info[jidx]
    lim = joint_limits_from_info(info)
    if lim is not None:
        target = clamp(target, lim[0], lim[1])

    p.setJointMotorControl2(
        robot_id,
        jidx,
        p.POSITION_CONTROL,
        targetPosition=float(target),
        force=val.MOTOR_FORCE,
        maxVelocity=3.0,
        positionGain=val.POS_GAIN,
        velocityGain=val.VEL_GAIN,
    )


def set_gripper_targets(robot_id, idx_to_info, gripper_joints, open01):
    open01 = mm.clamp01(open01)
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
            positionGain=val.POS_GAIN,
            velocityGain=val.VEL_GAIN,
        )


def spawn_objects():
    """
    Spawn simple manipulable objects.
    """
    ids = []
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    cube_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
    cube_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])

    cyl_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.015, height=0.05)
    cyl_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.015, length=0.05)

    positions = [
        [0.15, 0.25, 0.02],
        [0.10, 0.30, 0.02],
        [0.20, 0.30, 0.02],
        [0.18, 0.22, 0.02],
    ]

    for i, pos in enumerate(positions):
        if i % 2 == 0:
            bid = p.createMultiBody(
                baseMass=0.05,
                baseCollisionShapeIndex=cube_col,
                baseVisualShapeIndex=cube_vis,
                basePosition=pos,
            )
        else:
            bid = p.createMultiBody(
                baseMass=0.05,
                baseCollisionShapeIndex=cyl_col,
                baseVisualShapeIndex=cyl_vis,
                basePosition=pos,
            )

        p.changeDynamics(bid, -1, lateralFriction=1.2, spinningFriction=0.1, rollingFriction=0.05)
        ids.append(bid)

    return ids


def update_grasp(robot_id, gripper_link_idx, grip01, object_ids, state,
                 close_thresh=0.25, open_thresh=0.55, pick_radius=0.06):
    """
    Constraint-based grasp.
    """
    held = state.get("held_id", None)
    cid = state.get("cid", None)

    # release
    if held is not None and grip01 > open_thresh:
        try:
            p.removeConstraint(cid)
        except Exception:
            pass
        state["held_id"] = None
        state["cid"] = None
        return

    if held is not None:
        return

    # attempt pick
    if grip01 < close_thresh:
        ee_pos, _ee_orn = p.getLinkState(robot_id, gripper_link_idx)[:2]

        best_id = None
        best_d = 1e9
        for oid in object_ids:
            opos, _ = p.getBasePositionAndOrientation(oid)
            d = math.dist(ee_pos, opos)
            if d < best_d:
                best_d = d
                best_id = oid

        if best_id is not None and best_d < pick_radius:
            cid = p.createConstraint(
                parentBodyUniqueId=robot_id,
                parentLinkIndex=gripper_link_idx,
                childBodyUniqueId=best_id,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0],
            )
            state["held_id"] = best_id
            state["cid"] = cid


def solve_ik(robot_id, idx_to_info, ee_link_idx, target_pos, target_orn, rest_pose):
    """
    Nullspace IK with joint limits/damping.
    Uses values.py IK_MAX_ITERS and IK_RESIDUAL_THRESH.
    """
    n = p.getNumJoints(robot_id)

    lower = []
    upper = []
    ranges = []
    rest = []
    damping = []

    for j in range(n):
        info = idx_to_info[j]
        jtype = info[2]
        qIndex = info[3]

        if qIndex == -1 or jtype == p.JOINT_FIXED:
            lower.append(0.0)
            upper.append(0.0)
            ranges.append(0.0)
            rest.append(0.0)
            damping.append(0.8)
            continue

        lim = joint_limits_from_info(info)
        if lim is None:
            lo, hi = -math.pi, math.pi
        else:
            lo, hi = lim

        lower.append(lo)
        upper.append(hi)
        ranges.append(hi - lo)

        rp = rest_pose[j] if j < len(rest_pose) else 0.0
        rest.append(float(rp))

        damping.append(0.8 if j >= 3 else 0.4)

    if target_orn is None:
        sol = p.calculateInverseKinematics(
            robot_id,
            ee_link_idx,
            targetPosition=list(target_pos),
            lowerLimits=lower,
            upperLimits=upper,
            jointRanges=ranges,
            restPoses=rest,
            jointDamping=damping,
            maxNumIterations=val.IK_MAX_ITERS,
            residualThreshold=val.IK_RESIDUAL_THRESH,
        )
    else:
        sol = p.calculateInverseKinematics(
            robot_id,
            ee_link_idx,
            targetPosition=list(target_pos),
            targetOrientation=target_orn,
            lowerLimits=lower,
            upperLimits=upper,
            jointRanges=ranges,
            restPoses=rest,
            jointDamping=damping,
            maxNumIterations=val.IK_MAX_ITERS,
            residualThreshold=val.IK_RESIDUAL_THRESH,
        )

    return sol


def hand_to_ee_pose(hand_lms, depth_cal):
    lm = hand_lms.landmark

    # palm center
    idxs = [0, 5, 9, 13, 17]
    cx = sum(lm[i].x for i in idxs) / len(idxs)
    cy = sum(lm[i].y for i in idxs) / len(idxs)

    # X from screen left/right
    x01 = mm.clamp01(cx)
    x = mm.lerp(val.WORKSPACE_X_MIN, val.WORKSPACE_X_MAX, x01)

    # Z from screen up/down (up -> larger z)
    y01 = mm.clamp01(1.0 - cy)
    z = mm.lerp(val.WORKSPACE_Z_MIN, val.WORKSPACE_Z_MAX, y01)

    # Depth proxy (farther => larger depth01)
    wrist = (lm[0].x, lm[0].y, lm[0].z)
    hand_size = math.hypot(lm[0].x - lm[5].x, lm[0].y - lm[5].y)
    size_proxy = 1.0 / max(hand_size, 1e-4)
    depth_proxy = 0.6 * size_proxy + 0.4 * wrist[2]

    depth_cal.update(depth_proxy)
    dmin, dmax = depth_cal.get_minmax()
    depth01 = mm.clamp01((depth_proxy - dmin) / (dmax - dmin))

    # Y from depth
    y = mm.lerp(val.WORKSPACE_Y_MIN, val.WORKSPACE_Y_MAX, depth01)

    # Clamp to workspace bounds
    x = float(mm.clamp(x, val.WORKSPACE_X_MIN, val.WORKSPACE_X_MAX))
    y = float(mm.clamp(y, val.WORKSPACE_Y_MIN, val.WORKSPACE_Y_MAX))
    z = float(mm.clamp(z, val.WORKSPACE_Z_MIN, val.WORKSPACE_Z_MAX))

    ee_pos = [x, y, z]

    ee_orn = mm.palm_quat_from_landmarks(hand_lms)

    # Pinch grip (fallback)
    thumb_tip = (lm[4].x, lm[4].y)
    index_tip = (lm[8].x, lm[8].y)
    pinch = mm.dist(thumb_tip, index_tip)
    grip01 = mm.normalize01(pinch, val.PINCH_MIN, val.PINCH_MAX)

    if val.INVERT_GRIPPER:
        grip01 = 1.0 - grip01

    return ee_pos, ee_orn, float(grip01), float(depth01)
