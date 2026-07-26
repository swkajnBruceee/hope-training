"""Compatibility transforms for archived Stage-A actor checkpoints.

The archived Stage-A policy was trained before the corrected root-yaw contract.
This module adapts only its *inference observation*; it never changes the
simulator state, upper-body controller, manifest, or leg action coordinates.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


STAGE_A_OBSERVATION_DIM = 126
STAGE_A_TERM_NAMES = (
    "base_lin_vel",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
    "projected_gravity",
    "racket_target_pos_b",
    "racket_target_vel_b",
    "racket_target_normal_b",
    "racket_pos_b",
    "racket_lin_vel_b",
    "racket_normal_b",
    "time_to_strike",
    "swing_type",
    "strike_joint_pos",
    "strike_joint_vel",
    "strike_reference_joint_pos",
    "strike_reference_joint_vel",
    "strike_reference_joint_vel_8",
    "strike_reference_joint_vel_16",
    "strike_phase",
)
STAGE_A_TERM_DIMS = (
    3, 3, 14, 14, 14, 3, 3, 3, 3, 3, 3, 3, 1, 1, 9, 9, 9, 9, 9, 9, 1,
)

# The 126-D policy observation is a concatenation of named terms. The target
# is anchored in the world and then expressed in the root-yaw frame, so its XY
# components change under the legacy yaw convention. Base velocity and racket
# state are already body-frame quantities: rotating them again would invert
# genuine local proprioception.
_YAW_VECTOR_STARTS = (
    51,  # racket_target_pos_b
    54,  # racket_target_vel_b
    57,  # racket_target_normal_b
)


def adapt_stage_a_observation_legacy_yaw_pi(observation: torch.Tensor) -> torch.Tensor:
    """Express a current Stage-A observation in the archived actor's yaw frame.

    The transform is a 180 degree rotation around the vertical axis. It is
    intentionally limited to vector XY components in the frozen 126-D actor
    contract. Passing another layout is an error because silently rotating a
    different schema would produce unsafe leg commands.
    """
    if observation.ndim < 1 or observation.shape[-1] != STAGE_A_OBSERVATION_DIM:
        raise ValueError(
            "legacy Stage-A yaw adapter requires a trailing observation width "
            f"of {STAGE_A_OBSERVATION_DIM}, got {tuple(observation.shape)}"
        )
    transformed = observation.clone()
    for start in _YAW_VECTOR_STARTS:
        transformed[..., start : start + 2] *= -1.0
    return transformed


def validate_stage_a_legacy_layout(term_names: Sequence[str], term_dims: Sequence[int]) -> None:
    """Fail closed unless the runtime group is the archived 126-D layout."""
    names = tuple(term_names)
    dims = tuple(int(dim) for dim in term_dims)
    if (
        names != STAGE_A_TERM_NAMES
        or dims != STAGE_A_TERM_DIMS
        or sum(dims) != STAGE_A_OBSERVATION_DIM
    ):
        raise RuntimeError(
            "legacy Stage-A yaw adapter requires the frozen 126-D observation "
            f"layout; got names={list(names)}, dims={list(dims)}"
        )
