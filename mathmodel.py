from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation as R


def clamp(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


def clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def lerp(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * t)


def ema(prev, new: float, alpha: float) -> float:
    if prev is None:
        return float(new)
    return float((1.0 - alpha) * prev + alpha * new)


def normalize01(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return clamp((x - lo) / (hi - lo), 0.0, 1.0)


def dist(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b))


def angle_deg(p1, p2) -> float:
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    dx, dy = (p2 - p1)[:2]
    a = np.degrees(np.arctan2(dy, dx))
    # normalize to (-180, 180]
    if a <= -180.0:
        a += 360.0
    elif a > 180.0:
        a -= 360.0
    return float(a)



def v3(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return (b - a).astype(np.float64)


def dot(u, v) -> float:
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return float(np.dot(u, v))


def cross(u, v):
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return np.cross(u, v).astype(np.float64)


def norm(u) -> float:
    u = np.asarray(u, dtype=np.float64)
    return float(np.linalg.norm(u))


def normalize(u, eps: float = 1e-12):
    u = np.asarray(u, dtype=np.float64)
    n = np.linalg.norm(u)
    if n < eps:
        return np.zeros(3, dtype=np.float64)
    return (u / n).astype(np.float64)


def map01_to_limits(t01: float, lim) -> float:
    lo, hi = float(lim[0]), float(lim[1])
    return float(lo + (hi - lo) * clamp01(t01))



def quat_from_rotm(Rm):
    Rm = np.asarray(Rm, dtype=np.float64)
    q = R.from_matrix(Rm).as_quat()  # (x,y,z,w)
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def rotm_from_quat(quat_xyzw):
    q = np.asarray(quat_xyzw, dtype=np.float64)
    Rm = R.from_quat(q).as_matrix()
    return Rm.astype(np.float64)


def quat_multiply(q1_xyzw, q2_xyzw):
    q1 = R.from_quat(np.asarray(q1_xyzw, dtype=np.float64))
    q2 = R.from_quat(np.asarray(q2_xyzw, dtype=np.float64))
    q = (q1 * q2).as_quat()
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def quat_inverse(q_xyzw):
    q = R.from_quat(np.asarray(q_xyzw, dtype=np.float64)).inv().as_quat()
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def quat_to_euler(quat_xyzw, order: str = "xyz"):
    q = np.asarray(quat_xyzw, dtype=np.float64)
    e = R.from_quat(q).as_euler(order, degrees=False)
    return (float(e[0]), float(e[1]), float(e[2]))


def euler_to_quat(rpy, order: str = "xyz"):
    rpy = np.asarray(rpy, dtype=np.float64)
    q = R.from_euler(order, rpy, degrees=False).as_quat()
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))



def palm_basis_from_landmarks(hand_lms):
    lm = hand_lms.landmark

    wrist = np.array([lm[0].x, lm[0].y, lm[0].z], dtype=np.float64)
    idx_mcp = np.array([lm[5].x, lm[5].y, lm[5].z], dtype=np.float64)
    mid_mcp = np.array([lm[9].x, lm[9].y, lm[9].z], dtype=np.float64)
    pky_mcp = np.array([lm[17].x, lm[17].y, lm[17].z], dtype=np.float64)

    x_axis = normalize(pky_mcp - idx_mcp)
    y_axis = normalize(mid_mcp - wrist)
    z_axis = normalize(np.cross(x_axis, y_axis))

    y_axis = normalize(np.cross(z_axis, x_axis))

    return x_axis, y_axis, z_axis


def cam_to_world_axis(v):
    v = np.asarray(v, dtype=np.float64)
    return normalize(np.array([v[0], 0.0, -v[1]], dtype=np.float64))


def palm_quat_from_landmarks(hand_lms):
    x_axis, y_axis, _ = palm_basis_from_landmarks(hand_lms)

    Xw = cam_to_world_axis(x_axis)
    Yw = cam_to_world_axis(y_axis)
    Zw = normalize(np.cross(Xw, Yw))
    Yw = normalize(np.cross(Zw, Xw))

    Rm = np.stack([Xw, Yw, Zw], axis=1)  # columns are basis vectors
    q = R.from_matrix(Rm).as_quat()      # (x,y,z,w)
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def palm_euler_from_landmarks(hand_lms, pitch_limit_deg: float = 80.0):
    x_axis, y_axis, z_axis = palm_basis_from_landmarks(hand_lms)

    ax = float(x_axis[0])
    ay_up = float(-x_axis[1])
    fx = float(y_axis[0])
    fy_up = float(-y_axis[1])

    yaw = float(np.arctan2(fx, fy_up))
    roll = float(np.arctan2(ay_up, ax))

    pitch = float(clamp(float(z_axis[2]), -1.0, 1.0) * np.deg2rad(pitch_limit_deg))
    return roll, pitch, yaw
