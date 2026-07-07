"""HOPE racket-target observation terms.

These wrap :class:`RacketTargetCommand`. The actor (policy) group should use only the *desired*
quantities the planner provides at deploy time (HITTER actor observation, Table I):

* :func:`racket_target_pos_b`  — desired racket position relative to base (3)
* :func:`racket_target_vel_w`  — desired racket velocity in world frame (3)
* :func:`time_to_strike`       — time remaining until strike (1)
* :func:`base_target_pos_b`    — desired base XY position relative to base (2)

The desired racket *normal* and the *actual* racket state are privileged/critic-only or used by
the reward; they are intentionally NOT in the HITTER actor observation (the racket is never sensed
on hardware). :func:`swing_type` is provided for a unified forehand+backhand policy variant; the
HOPE default trains separate policies and does not need it.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_rotate_inverse, yaw_quat

from training.tasks.tracking.mdp.hope_commands import RacketTargetCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _cmd(env: ManagerBasedRLEnv, command_name: str) -> RacketTargetCommand:
    return env.command_manager.get_term(command_name)


# --- actor (policy) observations: desired targets only ------------------------------------ #
def racket_target_pos_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return _cmd(env, command_name).racket_target_pos_b()


def racket_target_vel_w(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return _cmd(env, command_name).racket_target_vel_w


def time_to_strike(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return _cmd(env, command_name).time_to_strike.unsqueeze(-1)


def base_target_pos_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return _cmd(env, command_name).base_target_pos_b()


def swing_type(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Forehand (+1) / backhand (-1). Only needed for a unified (single) policy."""
    return _cmd(env, command_name).swing_sign.unsqueeze(-1)


# --- privileged (critic) observations: desired normal + actual racket state --------------- #
def racket_target_normal_w(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return _cmd(env, command_name).racket_target_normal_w


def racket_pos_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Actual racket position relative to base (FK). Privileged — not sensed on hardware."""
    cmd = _cmd(env, command_name)
    return quat_rotate_inverse(yaw_quat(cmd.base_quat_w), cmd.racket_pos_w - cmd.base_pos_w)


def racket_lin_vel_w(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Actual racket linear velocity (FK), world frame. Privileged."""
    return _cmd(env, command_name).racket_lin_vel_w


def racket_normal_w(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Actual racket face normal (FK), world frame. Privileged."""
    return _cmd(env, command_name).racket_normal_w


def episode_time_left(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Time remaining in the episode (seconds). HITTER critic privileged input."""
    episode_length_buf = getattr(env, "episode_length_buf", None)
    if episode_length_buf is None:
        # Isaac Lab calls observation terms once while constructing the manager,
        # before runtime episode buffers are attached. Return the reset-state
        # shape so the observation manager can infer dimensions.
        left = torch.full((env.num_envs,), float(env.max_episode_length) * float(env.step_dt), device=env.device)
    else:
        left = (env.max_episode_length - episode_length_buf).float() * env.step_dt
    return left.unsqueeze(-1)
