from __future__ import annotations

import time
import cv2
import pybullet as p

import depthcalibrator as dc
import mathmodel as mm
import simulation as sim
import handtracking as hands
import values as val

try:
    import calib as calib
except Exception:
    import cailb as calib


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, val.CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, val.CAM_H)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    intr = None
    ws = None
    if hasattr(calib, "ensure_calibration"):
        intr, ws, ext = calib.ensure_calibration(
            cap,
            use_charuco_intrinsics=True,
            intrinsics_target_frames=25,
            phone_ids=(10, 11, 12, 13),
        )

    else:
        if hasattr(calib, "load_intrinsics"):
            intr = calib.load_intrinsics()
        if hasattr(calib, "load_workspace"):
            ws = calib.load_workspace()

    if hasattr(sim, "set_calibration"):
        sim.set_calibration(intr, ws, ext)
    elif hasattr(sim, "set_workspace"):
        sim.set_workspace(ws)
    elif hasattr(sim, "set_workspace_homography") and ws is not None and "H" in ws:
        sim.set_workspace_homography(ws["H"])

    robot_id, name_to_idx, idx_to_info, urdf_path = sim.setup_pybullet()
    print("Loaded URDF:", urdf_path)

    if hasattr(sim, "dump_joints"):
        sim.dump_joints(robot_id)

    object_ids = sim.spawn_objects() if hasattr(sim, "spawn_objects") else []
    ee_link_idx = sim.find_gripper_link(robot_id) if hasattr(sim, "find_gripper_link") else (p.getNumJoints(robot_id) - 1)
    print("Gripper link index:", ee_link_idx)

    grasp_state = {"held_id": None, "cid": None}
    depth_cal = dc.DepthCalibrator(window=240)

    discovered = sim.discover_so101_joints(name_to_idx)

    movable_types = {p.JOINT_REVOLUTE, p.JOINT_PRISMATIC}
    discovered["gripper_all"] = [
        (name, jidx) for (name, jidx) in discovered["gripper_all"]
        if idx_to_info[jidx][2] in movable_types
    ]

    prev_time = time.time()
    p_time = time.time()

    ee_pos_sm = None
    ee_q_sm = None

    j_order = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    joints_prev = None

    rest_pose = [0.0] * p.getNumJoints(robot_id)
    for logical, bias in [("shoulder_lift", 1.2), ("elbow_flex", 0.0)]:
        _, jidx = discovered.get(logical, (None, None))
        if jidx is not None:
            rest_pose[jidx] = float(bias)

    grip01_sm = 0.8

    max_joint_rate = 1.2
    ik_alpha = 0.12

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
        _clap_event, per_hand = hands.draw_and_update_gestures(frame, detected_hands, now, dt)
        driver = hands.choose_driver(detected_hands)

        if driver is not None:
            hand_lms, label, _score = driver

            if hasattr(sim, "hand_to_ee_pose"):
                ee_pos, ee_q, _open01_from_sim, depth01 = sim.hand_to_ee_pose(hand_lms, depth_cal)
            else:
                ee_pos, ee_q, _open01_from_sim, depth01 = [0.0, 0.2, 0.1], (0.0, 0.0, 0.0, 1.0), 0.5, 0.5

            open01 = per_hand.get(label, {}).get("open01", 0.7)
            grip01 = 1.0 - open01
            grip01_sm = mm.ema(grip01_sm, grip01, alpha=val.JOINT_SMOOTH_ALPHA)

            ee_pos_sm = mm.ema_vec(ee_pos_sm, ee_pos, alpha=val.POSE_SMOOTH_ALPHA)
            if ee_q_sm is None:
                ee_q_sm = ee_q
            else:
                r0 = mm.euler_from_quat_xyzw(ee_q_sm, "xyz")
                r1 = mm.euler_from_quat_xyzw(ee_q, "xyz")
                r_sm = mm.ema_vec(r0, r1, alpha=val.POSE_SMOOTH_ALPHA)
                ee_q_sm = mm.quat_from_euler(r_sm, "xyz")

            if ee_pos_sm is not None:
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
                target_orn=None,
                rest_pose=rest_pose,
            )

            jidxs = []
            jvals = []
            for logical in j_order:
                _, jidx = discovered.get(logical, (None, None))
                jidxs.append(jidx)
                jvals.append(sol[jidx] if jidx is not None else 0.0)

            jvals = mm.ema_angles(joints_prev, jvals, ik_alpha)
            jvals = mm.rate_limit_angles(joints_prev, jvals, max_joint_rate, dt)

            for jidx, targ in zip(jidxs, jvals):
                if jidx is not None:
                    sim.set_joint_target(robot_id, idx_to_info, jidx, targ)

            contacts = sim.self_collision_contacts(robot_id) if hasattr(sim, "self_collision_contacts") else []
            if len(contacts) > 8 and joints_prev is not None:
                for jidx, targ in zip(jidxs, joints_prev):
                    if jidx is not None:
                        sim.set_joint_target(robot_id, idx_to_info, jidx, targ)
            else:
                joints_prev = jvals

            if discovered["gripper_all"]:
                sim.set_gripper_targets(robot_id, idx_to_info, discovered["gripper_all"], open01)

            if hasattr(sim, "update_grasp"):
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

        steps = max(1, min(60, int(val.SIM_HZ * dt)))
        for _ in range(steps):
            p.stepSimulation()

        c_time = time.time()
        fps = 1.0 / (c_time - p_time) if (c_time - p_time) > 0 else 0.0
        p_time = c_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        now2 = time.time()
        while hands.action_log and now2 - hands.action_log[0][0] > val.LOG_DURATION:
            hands.action_log.popleft()

        x0, y0 = 10, frame.shape[0] - 10
        line_h = 22
        for i, (_, txt) in enumerate(reversed(hands.action_log)):
            y = y0 - i * line_h
            cv2.putText(frame, txt, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Hand Tracking -> SO-101 IK (Calibrated)", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.hands.close()
    p.disconnect()


if __name__ == "__main__":
    main()
