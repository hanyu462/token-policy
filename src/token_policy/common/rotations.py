"""Quaternion helpers (wxyz, scalar first), matching the GEAR-SONIC deploy math."""
from __future__ import annotations

import numpy as np


def quat_unit(q):
    q = np.asarray(q, dtype=float)
    return q / max(np.linalg.norm(q), 1e-12)


def quat_conjugate(q):
    q = np.asarray(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_rotate(q, v):
    """Rotate vector v by quaternion q."""
    qw = q[0]
    qv = np.asarray(q[1:4], dtype=float)
    v = np.asarray(v, dtype=float)
    return v * (2.0 * qw * qw - 1.0) + np.cross(qv, v) * (2.0 * qw) + qv * (2.0 * np.dot(qv, v))


def quat_from_angle_axis(angle, axis):
    axis = np.asarray(axis, dtype=float)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    s = np.sin(angle / 2.0)
    return quat_unit(np.array([np.cos(angle / 2.0), *(axis * s)]))


def calc_heading(q):
    """Yaw of the body +x axis in the world frame."""
    d = quat_rotate(q, [1.0, 0.0, 0.0])
    return np.arctan2(d[1], d[0])


def calc_heading_quat(q):
    return quat_from_angle_axis(calc_heading(q), [0.0, 0.0, 1.0])


def calc_heading_quat_inv(q):
    return quat_from_angle_axis(-calc_heading(q), [0.0, 0.0, 1.0])


def quat_to_rotation_matrix(q):
    w, x, y, z = quat_unit(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def gravity_in_body_frame(base_quat):
    """World gravity direction (0,0,-1) expressed in the base frame."""
    return quat_rotate(quat_conjugate(base_quat), [0.0, 0.0, -1.0])


def quat_to_rpy(q):
    """ZYX Euler angles (roll, pitch, yaw) in radians from a wxyz quaternion."""
    w, x, y, z = quat_unit(q)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])
