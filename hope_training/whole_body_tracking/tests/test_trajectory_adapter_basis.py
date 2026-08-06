from __future__ import annotations

import numpy as np

from training.utils.trajectory_adapter_basis import (
    apply_deformation,
    bernstein_basis,
    finite_difference_velocity,
    fit_deformation_coefficients,
)


def test_bernstein_basis_is_smooth_partition_of_unity() -> None:
    basis = bernstein_basis(39, 8)
    assert basis.shape == (39, 9)
    np.testing.assert_allclose(basis.sum(axis=1), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(basis[0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(basis[-1], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def test_fit_and_apply_reconstructs_low_order_smooth_deformation() -> None:
    phase = np.linspace(0.0, 1.0, 39)
    deformation = np.stack((0.05 + 0.03 * phase, -0.02 * phase**2), axis=1)
    coefficients, reconstructed = fit_deformation_coefficients(deformation, degree=4)
    np.testing.assert_allclose(reconstructed, deformation, atol=1.0e-8)
    nominal = np.zeros_like(deformation)
    np.testing.assert_allclose(apply_deformation(nominal, coefficients), deformation, atol=1.0e-8)


def test_finite_difference_velocity_matches_linear_trajectory() -> None:
    fps = 50.0
    time = np.arange(39, dtype=np.float64) / fps
    position = np.stack((2.0 * time, -0.5 * time), axis=1)
    velocity = finite_difference_velocity(position, fps)
    np.testing.assert_allclose(velocity, np.tile(np.array([2.0, -0.5]), (39, 1)), atol=1.0e-12)
