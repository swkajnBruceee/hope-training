"""Time resampling utilities for cleaned mocap samples."""

from __future__ import annotations

import numpy as np

from analysis.mocap_cleaning.derivative import canonicalize_quat_signs


def make_hit_centered_time(hit_time: float, pre_s: float, post_s: float, fps: float) -> np.ndarray:
    """Create a uniform time axis that includes hit_time exactly."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    dt = 1.0 / fps
    pre_frames = int(round(pre_s * fps))
    post_frames = int(round(post_s * fps))
    offsets = np.arange(-pre_frames, post_frames + 1, dtype=float) * dt
    return hit_time + offsets


def resample_array(time_old: np.ndarray, values: np.ndarray, time_new: np.ndarray) -> np.ndarray:
    """Linearly resample a [T, D] signal, preserving NaNs outside valid support."""
    if values.ndim != 2:
        raise ValueError(f"values must be [T, D], got shape {values.shape}")
    if len(time_old) != len(values):
        raise ValueError("time_old and values length mismatch")
    out = np.full((len(time_new), values.shape[1]), np.nan, dtype=float)
    for d in range(values.shape[1]):
        valid = np.isfinite(values[:, d]) & np.isfinite(time_old)
        if np.sum(valid) < 2:
            continue
        out[:, d] = np.interp(time_new, time_old[valid], values[valid, d], left=np.nan, right=np.nan)
    return out


def resample_quat_xyzw(time_old: np.ndarray, quat_xyzw: np.ndarray, time_new: np.ndarray) -> np.ndarray:
    """Resample quaternions by sign-canonicalized linear interpolation + normalization.

    For the small 360Hz -> 200Hz step used here this is adequate and avoids adding a
    SciPy dependency requirement to the cleaning package. The result remains xyzw.
    """
    quat = canonicalize_quat_signs(quat_xyzw)
    out = resample_array(time_old, quat, time_new)
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    valid = np.isfinite(out).all(axis=1, keepdims=True) & (norm > 1e-12)
    return np.divide(out, norm, out=np.full_like(out, np.nan), where=valid)


def resample_mask(time_old: np.ndarray, mask: np.ndarray, time_new: np.ndarray) -> np.ndarray:
    """Nearest-neighbor resample a boolean/int valid mask."""
    if len(time_old) != len(mask):
        raise ValueError("time_old and mask length mismatch")
    if len(time_old) == 0:
        return np.zeros(len(time_new), dtype=np.int8)
    indexes = np.searchsorted(time_old, time_new, side="left")
    indexes = np.clip(indexes, 0, len(time_old) - 1)
    prev = np.clip(indexes - 1, 0, len(time_old) - 1)
    choose_prev = np.abs(time_new - time_old[prev]) <= np.abs(time_new - time_old[indexes])
    nearest = np.where(choose_prev, prev, indexes)
    in_range = (time_new >= time_old[0]) & (time_new <= time_old[-1])
    out = np.asarray(mask, dtype=np.int8)[nearest]
    out[~in_range] = 0
    return out.astype(np.int8)
