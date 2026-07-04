"""Derivative utilities for mocap signals."""

from __future__ import annotations

import numpy as np


def compute_velocity(pos: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Compute central-difference velocity for a [T, D] position signal."""
    if pos.ndim != 2:
        raise ValueError(f"pos must be [T, D], got shape {pos.shape}")
    if len(pos) != len(time):
        raise ValueError("pos and time length mismatch")
    if len(time) < 2:
        return np.zeros_like(pos)

    vel = np.zeros_like(pos, dtype=float)
    dt_mid = time[2:] - time[:-2]
    valid_mid = dt_mid > 0
    vel[1:-1][valid_mid] = (pos[2:][valid_mid] - pos[:-2][valid_mid]) / dt_mid[valid_mid, None]

    dt0 = time[1] - time[0]
    dtn = time[-1] - time[-2]
    if dt0 > 0:
        vel[0] = (pos[1] - pos[0]) / dt0
    if dtn > 0:
        vel[-1] = (pos[-1] - pos[-2]) / dtn
    return vel


def _normalize_quat_xyzw(quat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    out = quat.astype(float, copy=True)
    valid = np.isfinite(out).all(axis=1, keepdims=True) & (norm > eps)
    out[valid[:, 0]] = out[valid[:, 0]] / norm[valid[:, 0]]
    out[~valid[:, 0]] = np.nan
    return out


def canonicalize_quat_signs(quat_xyzw: np.ndarray) -> np.ndarray:
    """Flip quaternion signs to avoid artificial frame-to-frame discontinuities."""
    if quat_xyzw.ndim != 2 or quat_xyzw.shape[1] != 4:
        raise ValueError(f"quat_xyzw must be [T, 4], got shape {quat_xyzw.shape}")
    quat = _normalize_quat_xyzw(quat_xyzw)
    for i in range(1, len(quat)):
        if not (np.isfinite(quat[i]).all() and np.isfinite(quat[i - 1]).all()):
            continue
        if np.dot(quat[i - 1], quat[i]) < 0:
            quat[i] *= -1.0
    return quat


def _quat_multiply_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def _quat_conjugate_xyzw(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., :3] *= -1.0
    return out


def _quat_delta_to_rotvec_xyzw(delta: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    delta = _normalize_quat_xyzw(delta)
    xyz = delta[:, :3]
    w = np.clip(delta[:, 3], -1.0, 1.0)
    sin_half = np.linalg.norm(xyz, axis=1)
    angle = 2.0 * np.arctan2(sin_half, w)
    angle = (angle + np.pi) % (2.0 * np.pi) - np.pi
    rotvec = np.zeros((len(delta), 3), dtype=float)
    valid = np.isfinite(delta).all(axis=1) & (sin_half > eps)
    rotvec[valid] = xyz[valid] / sin_half[valid, None] * angle[valid, None]
    small = np.isfinite(delta).all(axis=1) & ~valid
    rotvec[small] = 2.0 * xyz[small]
    rotvec[~np.isfinite(delta).all(axis=1)] = np.nan
    return rotvec


def compute_angular_velocity(quat_xyzw: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Compute body-frame angular velocity from xyzw quaternions in rad/s."""
    if quat_xyzw.ndim != 2 or quat_xyzw.shape[1] != 4:
        raise ValueError(f"quat_xyzw must be [T, 4], got shape {quat_xyzw.shape}")
    if len(quat_xyzw) != len(time):
        raise ValueError("quat_xyzw and time length mismatch")
    omega = np.full((len(time), 3), np.nan, dtype=float)
    if len(time) < 2:
        return np.zeros((len(time), 3), dtype=float)

    quat = canonicalize_quat_signs(quat_xyzw)
    q0 = quat[:-1]
    q1 = quat[1:]
    finite = np.isfinite(q0).all(axis=1) & np.isfinite(q1).all(axis=1)
    dt = time[1:] - time[:-1]
    valid = finite & (dt > 0)
    if np.any(valid):
        delta = _quat_multiply_xyzw(_quat_conjugate_xyzw(q0[valid]), q1[valid])
        rotvec = _quat_delta_to_rotvec_xyzw(delta)
        omega[:-1][valid] = rotvec / dt[valid, None]
    omega[-1] = omega[-2]
    return omega
