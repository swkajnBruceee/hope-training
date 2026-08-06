"""Low-dimensional smooth deformation basis for strike trajectory adapters.

The module is deliberately NumPy-only so offline repair, unit tests and future
supervised adapter training share the exact same representation contract.
"""

from __future__ import annotations

import math

import numpy as np


CONTRACT_VERSION = "strike_trajectory_bernstein_adapter/v1"


def bernstein_basis(num_frames: int, degree: int) -> np.ndarray:
    """Return a partition-of-unity Bernstein basis with shape ``[T, K]``."""
    if num_frames < 2:
        raise ValueError("num_frames must be at least 2")
    if degree < 1:
        raise ValueError("degree must be positive")
    phase = np.linspace(0.0, 1.0, num_frames, dtype=np.float64)
    columns = [
        math.comb(degree, k) * phase**k * (1.0 - phase) ** (degree - k)
        for k in range(degree + 1)
    ]
    basis = np.stack(columns, axis=1)
    if not np.allclose(basis.sum(axis=1), 1.0, atol=1.0e-12):
        raise RuntimeError("Bernstein basis lost partition-of-unity")
    return basis


def fit_deformation_coefficients(
    deformation: np.ndarray,
    degree: int,
    *,
    ridge: float = 1.0e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit ``delta_q[T,J] = basis[T,K] @ coefficients[K,J]``.

    Returns the coefficients and reconstructed deformation.  The small ridge
    term gives deterministic behavior for future variants with masked frames.
    """
    values = np.asarray(deformation, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"deformation must have shape [T,J], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("deformation contains NaN or Inf")
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")
    basis = bernstein_basis(values.shape[0], degree)
    lhs = basis.T @ basis + ridge * np.eye(basis.shape[1], dtype=np.float64)
    rhs = basis.T @ values
    coefficients = np.linalg.solve(lhs, rhs)
    return coefficients, basis @ coefficients


def apply_deformation(
    nominal_joint_position: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Apply low-dimensional coefficients to a nominal ``[T,J]`` trajectory."""
    nominal = np.asarray(nominal_joint_position, dtype=np.float64)
    coeff = np.asarray(coefficients, dtype=np.float64)
    if nominal.ndim != 2 or coeff.ndim != 2:
        raise ValueError("nominal_joint_position and coefficients must be rank-2")
    degree = coeff.shape[0] - 1
    if degree < 1 or coeff.shape[1] != nominal.shape[1]:
        raise ValueError(
            f"coefficient shape {coeff.shape} is incompatible with nominal shape {nominal.shape}"
        )
    return nominal + bernstein_basis(nominal.shape[0], degree) @ coeff


def finite_difference_velocity(joint_position: np.ndarray, fps: float) -> np.ndarray:
    """Use the shared central/one-sided finite-difference velocity contract."""
    position = np.asarray(joint_position, dtype=np.float64)
    if position.ndim != 2 or position.shape[0] < 2:
        raise ValueError("joint_position must have shape [T>=2,J]")
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    velocity = np.empty_like(position)
    velocity[0] = (position[1] - position[0]) * fps
    velocity[-1] = (position[-1] - position[-2]) * fps
    if position.shape[0] > 2:
        velocity[1:-1] = (position[2:] - position[:-2]) * (0.5 * fps)
    return velocity
