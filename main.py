import math
import time
import cv2
import pybullet as p

import depthcalibrator as dc
import mathmodel as mm
import simulation as sim
import handtracking as hands
import values as val


def quat_to_euler(q):
    return p.getEulerFromQuaternion(q)


def euler_to_quat(rpy):
    return p.getQuaternionFromEuler([rpy[0], rpy[1], rpy[2]])


def ema_vec3(prev, new, alpha):
    if prev is None:
        return [new[0], new[1], new[2]]
    return [
        mm.ema(prev[0], new[0], alpha),
        mm.ema(prev[1], new[1], alpha),
        mm.ema(prev[2], new[2], alpha),
    ]


def wrap_pi(a):
    while a > math.pi:
        a -= math.pi*2
    while a < math.pi*-1:
        a += math.pi*2
    return a


def ema_euler(prev, new, alpha):
    if prev is None:
        return [wrap_pi(new[0]), wrap_pi(new[1]), wrap_pi(new[2])]
    out = []
    for i in range(3):
        a0 = wrap_pi(prev[i])
        a1 = wrap_pi(new[i])
        d = wrap_pi(a1 - a0)
        out.append(wrap_pi(a0 + alpha * d))
    return out


def get_movable_joint_indices(robot_id, idx_to_info):
    jidxs = []
    for j in range(p.getNumJoints(robot_id)):
        info = idx_to_info[j]
        if info[3] != -1 and info[2] != p.JOINT_FIXED:
            jidxs.append(j)
    return jidxs


def snapshot_joint_positions(robot_id, jidxs):
    return [p.getJointState(robot_id, j)[0] for j in jidxs]


def restore_joint_positions(robot_id, jidxs, q):
    for j, v in zip(jidxs, q):
        p.resetJointState(robot_id, j, v)


def blend_joint_positions(q_safe, q_now, t):
    return [(1.0 - t) * a + t * b for a, b in zip(q_safe, q_now)]


def build_baseline_penetration_pairs(robot_id, tol=0.004, span_ignore=2):
    baseline = set()
    pts = p.getClosestPoints(robot_id, robot_id, distance=0.02)
    for pt in pts:
        a, b = pt[3], pt[4]
        if a == b:
            continue
        if a != -1 and b != -1 and abs(a - b) <= span_ignore:
            continue
        if pt[8] < -tol:
            baseline.add((min(a, b), max(a, b)))
    return baseline


def has_new_self_penetration(robot_id, baseline_pairs, penetration_tol=0.006, ignore_link_span=2):
    pts = p.getClosestPoints(robot_id, robot_id, distance=0.02)
    for pt in pts:
        a, b = pt[3], pt[4]
        if a == b:
            continue
        if a != -1 and b != -1 and abs(a - b) <= ignore_link_span:
            continue

        pair = (min(a, b), max(a, b))
        if pair in baseline_pairs:
            continue

        if pt[8] < -penetration_tol:
            return True
    return False


def apply_reachup_rest_pose_bias(rest_pose, discovered):
    """
    Bias IK toward extension. Uses joint indices discovered from URDF.
    If your lift is inverted, flip the sign of 1.40.
    """
    targets = [
        ("shoulder_lift", 1.40),
        ("elbow_flex", 0.00),
        ("wrist_flex", 0.00),
        ("wrist_roll", 0.00),
    ]
    for logical, value in targets:
        _, jidx = discovered.get(logical, (None, None))
        if jidx is not None and jidx < len(rest_pose):
            rest_pose[jidx] = float(value)

def ema_list(prev, new, alpha):
    if prev is None:
        return list(new)
    return [(1 - alpha) * p + alpha * n for p, n in zip(prev, new)]


def wrap_pi(a):
    while a > 3.141592653589793:
        a -= 6.283185307179586
    while a < -3.141592653589793:
        a += 6.283185307179586
    return a


def ema_angles(prev, new, alpha):
    # EMA for angles (wrap-aware)
    if prev is None:
        return [wrap_pi(x) for x in new]
    out = []
    for p, n in zip(prev, new):
        p = wrap_pi(p)
        n = wrap_pi(n)
        d = wrap_pi(n - p)
        out.append(wrap_pi(p + alpha * d))
    return out


def rate_limit_angles(prev, new, max_rate_rad_s, dt):
    # Limit per-joint change to max_rate_rad_s*dt
    if prev is None:
        return list(new)
    max_step = max_rate_rad_s * dt
    out = []
    for p, n in zip(prev, new):
        p = wrap_pi(p)
        n = wrap_pi(n)
        d = wrap_pi(n - p)
        if d > max_step:
            d = max_step
        elif d < -max_step:
            d = -max_step
        out.append(wrap_pi(p + d))
    return out



def main():
    robot_id, name_to_idx, idx_to_info, urdf_path = sim.setup_pybullet()
    print("Loaded URDF:", urdf_path)

    sim.dump_joints(robot_id)

    object_ids = sim.spawn_objects()

    ee_link_idx = sim.find_ee_link_for_ik(robot_id)
    print("IK end-effector link index:", ee_link_idx)

    grasp_state = {"held_id": None, "cid": None}
    depth_cal = dc.DepthCalibrator(window=240)

    discovered = sim.discover_so101_joints(name_to_idx)

    # Keep only movable gripper joints
    movable_types = {p.JOINT_REVOLUTE, p.JOINT_PRISMATIC}
    discovered["gripper_all"] = [
        (name, jidx) for (name, jidx) in discovered["gripper_all"]
        if idx_to_info[jidx][2] in movable_types and idx_to_info[jidx][3] != -1
    ]

    # Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, val.CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, val.CAM_H)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try VideoCapture(1) or check permissions.")

    prev_time = time.time()
    p_time = time.time()

    ee_pos_sm = None
    ee_rpy_sm = None
    grip01_sm = 0.5

    # Rest pose for IK
    rest_pose = [0.0] * p.getNumJoints(robot_id)
    apply_reachup_rest_pose_bias(rest_pose, discovered)

    # Filtered joint targets to stop IK chatter
    joints_filtered = None

    # Reasonable max joint speed (rad/s) to prevent spasm
    MAX_JOINT_RATE = 1.2

    # Stronger smoothing than pose smoothing (this matters more)
    IK_SMOOTH_ALPHA = 0.12

    wf_idx = discovered.get("wrist_flex", (None, None))[1]
    wr_idx = discovered.get("wrist_roll", (None, None))[1]

    # Self-collision rollback state
    movable_jidxs = get_movable_joint_indices(robot_id, idx_to_info)
    last_safe_q = snapshot_joint_positions(robot_id, movable_jidxs)

    rollback_enabled = False
    rollback_enable_time = time.time() + 1.0

    # Settle + baseline pairs
    for _ in range(240):
        p.stepSimulation()

    baseline_pairs = build_baseline_penetration_pairs(robot_id, tol=0.004, span_ignore=2)
    print("Baseline penetrating pairs:", len(baseline_pairs))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.hands.process(rgb)

        now = time.time()
        dt = max(now - prev_time, 1e-6)
        prev_time = now

        detected_hands = hands.build_detected_hands(results)

        clap_event, per_hand = hands.draw_and_update_gestures(frame, detected_hands, now, dt)

        driver = hands.choose_driver(detected_hands)

        if driver is not None:
            hand_lms, driver_label, _score = driver

            ee_pos, ee_orn, pinch_grip01, depth01 = sim.hand_to_ee_pose(hand_lms, depth_cal)

            ee_pos_sm = ema_vec3(ee_pos_sm, ee_pos, alpha=val.POSE_SMOOTH_ALPHA)

            ee_rpy = quat_to_euler(ee_orn)
            ee_rpy_sm = ema_euler(ee_rpy_sm, ee_rpy, alpha=val.POSE_SMOOTH_ALPHA)
            ee_orn_sm = euler_to_quat(ee_rpy_sm)

            # Gripper mapping: OPEN/CLOSED from hand state, fallback pinch
            target_grip01 = pinch_grip01
            if per_hand is not None:
                info = per_hand.get(driver_label, None)
                if info is not None:
                    open_state = info.get("open_state", None)
                    if open_state == "OPEN":
                        target_grip01 = 1.0
                    elif open_state == "CLOSED":
                        target_grip01 = 0.0
                    elif open_state == "PARTIAL":
                        target_grip01 = 0.35 * pinch_grip01 + 0.65 * 0.5

            grip01_sm = mm.ema(grip01_sm, target_grip01, alpha=val.JOINT_SMOOTH_ALPHA)

            # Debug overlay
            cv2.putText(
                frame,
                f"EE tgt x:{ee_pos_sm[0]:.3f} y:{ee_pos_sm[1]:.3f} z:{ee_pos_sm[2]:.3f} depth01:{depth01:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            sol = sim.solve_ik(
                robot_id=robot_id,
                idx_to_info=idx_to_info,
                ee_link_idx=ee_link_idx,
                target_pos=ee_pos_sm,
                target_orn=None,  # keep position-only IK for stability
                rest_pose=rest_pose,
            )

            # Build a 5-joint vector in consistent order
            j_order = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
            j_indices = []
            j_values = []
            for logical in j_order:
                _, jidx = discovered.get(logical, (None, None))
                j_indices.append(jidx)
                j_values.append(sol[jidx] if jidx is not None else 0.0)

            # Smooth + rate-limit the IK joint outputs (THIS stops spasms)
            joints_filtered = ema_angles(joints_filtered, j_values, IK_SMOOTH_ALPHA)
            joints_filtered = rate_limit_angles(joints_filtered, joints_filtered, MAX_JOINT_RATE, dt)

            # Apply filtered targets
            for jidx, targ in zip(j_indices, joints_filtered):
                if jidx is not None:
                    sim.set_joint_target(robot_id, idx_to_info, jidx, targ)

            # Apply IK to arm joints
            for logical in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]:
                _, jidx = discovered.get(logical, (None, None))
                if jidx is not None:
                    sim.set_joint_target(robot_id, idx_to_info, jidx, sol[jidx])

            # Mild wrist correction to match palm angle
            if wf_idx is not None and wr_idx is not None and driver is not None:
                roll, pitch, _yaw = mm.palm_euler_from_landmarks(hand_lms)
                roll = mm.clamp(roll, -1.0, 1.0)
                pitch = mm.clamp(pitch, -1.0, 1.0)

                # Blend lightly so IK dominates
                blend = 0.10
                # Apply as a small offset from current filtered wrist targets
                try:
                    wf_i = j_order.index("wrist_flex")
                    wr_i = j_order.index("wrist_roll")
                    joints_filtered[wr_i] = wrap_pi((1 - blend) * joints_filtered[wr_i] + blend * roll)
                    joints_filtered[wf_i] = wrap_pi((1 - blend) * joints_filtered[wf_i] + blend * pitch)
                except Exception:
                    pass

        # Gripper drive
        if discovered["gripper_all"]:
            sim.set_gripper_targets(robot_id, idx_to_info, discovered["gripper_all"], grip01_sm)

        # Pick/place
        sim.update_grasp(
            robot_id=robot_id,
            gripper_link_idx=ee_link_idx,
            grip01=grip01_sm,
            object_ids=object_ids,
            state=grasp_state,
            close_thresh=0.25,
            open_thresh=0.55,
            pick_radius=0.06,
        )

        # Physics stepping
        steps = max(1, min(60, int(val.SIM_HZ * dt)))
        for _ in range(steps):
            p.stepSimulation()

        # Enable rollback after warmup
        if not rollback_enabled and time.time() > rollback_enable_time:
            rollback_enabled = True

        # Self-collision blend-back
        if rollback_enabled:
            if has_new_self_penetration(robot_id, baseline_pairs, penetration_tol=0.006, ignore_link_span=2):
                q_now = snapshot_joint_positions(robot_id, movable_jidxs)
                q_blend = blend_joint_positions(last_safe_q, q_now, 0.20)
                restore_joint_positions(robot_id, movable_jidxs, q_blend)
                p.stepSimulation()
                try:
                    hands.log_event("SELF-COLLISION (blend-back)")
                except Exception:
                    pass
            else:
                last_safe_q = snapshot_joint_positions(robot_id, movable_jidxs)
        else:
            last_safe_q = snapshot_joint_positions(robot_id, movable_jidxs)

        # FPS
        c_time = time.time()
        fps = 1.0 / (c_time - p_time) if (c_time - p_time) > 0 else 0.0
        p_time = c_time
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Log pruning + draw (uses val.LOG_DURATION)
        now2 = time.time()
        while hands.action_log and now2 - hands.action_log[0][0] > val.LOG_DURATION:
            hands.action_log.popleft()

        x0, y0 = 10, frame.shape[0] - 10
        line_h = 22
        for i, (_, txt) in enumerate(reversed(hands.action_log)):
            y = y0 - i * line_h
            cv2.putText(
                frame,
                txt,
                (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Hand Tracking -> SO-101 IK (EE mapped)", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.hands.close()
    p.disconnect()


if __name__ == "__main__":
    main()
