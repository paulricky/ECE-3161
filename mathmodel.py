from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


EPS = 1e-12

def clamp(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


def sat01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def lerp(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * t)


def inv_lerp(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return sat01((x - lo) / (hi - lo))


def ema(prev, new: float, alpha: float) -> float:
    if prev is None:
        return float(new)
    return float((1.0 - alpha) * float(prev) + alpha * float(new))


def ema_vec(prev, new, alpha: float):
    new = np.asarray(new, dtype=np.float64)
    if prev is None:
        return new.copy()
    prev = np.asarray(prev, dtype=np.float64)
    return (1.0 - alpha) * prev + alpha * new


def wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def ema_angles(prev, new, alpha: float):
    new = np.asarray(new, dtype=np.float64)
    if prev is None:
        return np.vectorize(wrap_pi)(new)
    prev = np.asarray(prev, dtype=np.float64)
    d = np.vectorize(wrap_pi)(new - prev)
    out = prev + alpha * d
    return np.vectorize(wrap_pi)(out)


def rate_limit_angles(prev, new, max_rate_rad_s: float, dt: float):
    new = np.asarray(new, dtype=np.float64)
    if prev is None:
        return new.copy()
    prev = np.asarray(prev, dtype=np.float64)
    max_step = float(max_rate_rad_s) * float(dt)
    d = np.vectorize(wrap_pi)(new - prev)
    d = np.clip(d, -max_step, max_step)
    out = prev + d
    return np.vectorize(wrap_pi)(out)


def dist(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b))


def norm(v) -> float:
    v = np.asarray(v, dtype=np.float64)
    return float(np.linalg.norm(v))


def normalize(v, eps: float = EPS):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def orthonormalize_cols(M):
    M = np.asarray(M, dtype=np.float64)
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        U[:, -1] *= -1.0
        Rm = U @ Vt
    return Rm


def quat_xyzw_from_rotm(Rm):
    q = R.from_matrix(np.asarray(Rm, dtype=np.float64)).as_quat()
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def rotm_from_quat_xyzw(q_xyzw):
    Rm = R.from_quat(np.asarray(q_xyzw, dtype=np.float64)).as_matrix()
    return np.asarray(Rm, dtype=np.float64)


def quat_mul(q1_xyzw, q2_xyzw):
    q = (R.from_quat(q1_xyzw) * R.from_quat(q2_xyzw)).as_quat()
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def quat_inv(q_xyzw):
    q = R.from_quat(np.asarray(q_xyzw, dtype=np.float64)).inv().as_quat()
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def euler_from_quat_xyzw(q_xyzw, order: str = "xyz"):
    e = R.from_quat(np.asarray(q_xyzw, dtype=np.float64)).as_euler(order, degrees=False)
    return (float(e[0]), float(e[1]), float(e[2]))


def quat_from_euler(rpy, order: str = "xyz"):
    q = R.from_euler(order, np.asarray(rpy, dtype=np.float64), degrees=False).as_quat()
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def mp_landmark_xyz(hand_lms, idx: int):
    lm = hand_lms.landmark[idx]
    return np.array([lm.x, lm.y, lm.z], dtype=np.float64)


def palm_frame_from_landmarks(hand_lms):
    wrist = mp_landmark_xyz(hand_lms, 0)
    idx_mcp = mp_landmark_xyz(hand_lms, 5)
    mid_mcp = mp_landmark_xyz(hand_lms, 9)
    pky_mcp = mp_landmark_xyz(hand_lms, 17)

    x_cam = normalize(pky_mcp - idx_mcp)      # across palm
    y_cam = normalize(mid_mcp - wrist)        # forward (wrist -> fingers)
    z_cam = normalize(np.cross(x_cam, y_cam)) # palm normal
    y_cam = normalize(np.cross(z_cam, x_cam))

    M = np.stack([x_cam, y_cam, z_cam], axis=1)
    Rm = orthonormalize_cols(M)
    return Rm


def cam_to_world_vec(v_cam):
    v_cam = np.asarray(v_cam, dtype=np.float64)
    return normalize(np.array([v_cam[0], 0.0, -v_cam[1]], dtype=np.float64))


def palm_quat_world_from_landmarks(hand_lms):
    R_cam = palm_frame_from_landmarks(hand_lms)
    Xw = cam_to_world_vec(R_cam[:, 0])
    Yw = cam_to_world_vec(R_cam[:, 1])
    Zw = normalize(np.cross(Xw, Yw))
    Yw = normalize(np.cross(Zw, Xw))
    Rw = orthonormalize_cols(np.stack([Xw, Yw, Zw], axis=1))
    return quat_xyzw_from_rotm(Rw)


def hand_center_xy(hand_lms):
    lm = hand_lms.landmark
    idxs = [0, 5, 9, 13, 17]
    x = float(sum(lm[i].x for i in idxs) / len(idxs))
    y = float(sum(lm[i].y for i in idxs) / len(idxs))
    return x, y


def hand_size_proxy(hand_lms):
    wrist = mp_landmark_xyz(hand_lms, 0)
    idx_mcp = mp_landmark_xyz(hand_lms, 5)
    d = np.linalg.norm((wrist - idx_mcp)[:2])
    return float(d)
