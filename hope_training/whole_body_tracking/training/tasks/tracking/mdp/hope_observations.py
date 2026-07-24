"""HOPE racket-target observation terms.

These wrap :class:`RacketTargetCommand`. The actor (policy) group should use only the *desired*
quantities the planner provides at deploy time (HITTER actor observation, Table I):

* :func:`racket_target_pos_b`  — desired racket position relative to base (3)
* :func:`racket_target_vel_b`  — desired racket velocity in base frame (3)
* :func:`racket_target_normal_b` — desired racket normal in base frame (3)
* :func:`time_to_strike`       — time remaining until strike (1)
* :func:`base_target_pos_b`    — desired base XY position relative to base (2)

The actual racket state can be actor-visible when it is computed from deployable proprioception
(joint state + fixed racket mount FK). :func:`swing_type` is provided for a unified forehand+backhand
policy.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_rotate_inverse, yaw_quat
from isaaclab.managers import SceneEntityCfg

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


def racket_target_vel_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return _cmd(env, command_name).racket_target_vel_b()


def racket_target_normal_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return _cmd(env, command_name).racket_target_normal_b()


def racket_target_error_pos_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Desired-minus-actual racket position in the current base yaw frame."""

    command = _cmd(env, command_name)
    base_yaw = yaw_quat(command.base_quat_w)
    actual = quat_rotate_inverse(base_yaw, command.racket_pos_w - command.base_pos_w)
    return command.racket_target_pos_b() - actual


def racket_target_error_vel_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Desired-minus-actual racket velocity in the current base yaw frame."""

    command = _cmd(env, command_name)
    base_yaw = yaw_quat(command.base_quat_w)
    actual = quat_rotate_inverse(base_yaw, command.racket_lin_vel_w)
    return command.racket_target_vel_b() - actual


def racket_target_error_normal_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Desired-minus-actual racket face normal in the current base yaw frame."""

    command = _cmd(env, command_name)
    base_yaw = yaw_quat(command.base_quat_w)
    actual = quat_rotate_inverse(base_yaw, command.racket_normal_w)
    return command.racket_target_normal_b() - actual


def time_to_strike(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return _cmd(env, command_name).time_to_strike.unsqueeze(-1)


def base_target_pos_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return _cmd(env, command_name).base_target_pos_b()


def swing_type(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Forehand (+1) / backhand (-1). Only needed for a unified (single) policy."""
    return _cmd(env, command_name).swing_sign.unsqueeze(-1)


def fixed_swing_type(env: ManagerBasedRLEnv, value: float = 1.0) -> torch.Tensor:
    """Return a semantic swing label fixed by the task contract.

    A single-family policy must not infer forehand/backhand from target-side
    geometry: a valid forehand target can cross the base-Y centerline after a
    stance or reference-frame change.  Use this term for a forehand-only (or
    backhand-only) task; unified policies should instead provide an explicit
    semantic label from the motion manifest/planner.
    """
    command = _cmd(env, "racket_target")
    return torch.full_like(command.swing_sign, float(value)).unsqueeze(-1)


def manifest_swing_type(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Return the explicit forehand/backhand label stored in the motion manifest.

    Target geometry is intentionally not used: a target can cross the base-Y
    centerline after a stance or reference-frame change while its stroke family
    remains unchanged.  The legacy forehand checkpoint used ``-1`` for this
    channel, so the unified contract retains forehand ``-1`` and assigns
    backhand ``+1``.  Unknown entries get ``0`` rather than silently being
    classified as either family.
    """
    command = _cmd(env, command_name)
    motion_command = env.command_manager.get_term(command.cfg.motion_command_name)
    stroke_ids = motion_command.motion.stroke_ids[motion_command.motion_ids]
    label = torch.zeros_like(stroke_ids, dtype=torch.float)
    label = torch.where(stroke_ids == 0, -torch.ones_like(label), label)
    label = torch.where(stroke_ids == 1, torch.ones_like(label), label)
    return label.unsqueeze(-1)


def motion_phase(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Normalized phase of the known strike reference, in ``[0, 1]``."""

    command = env.command_manager.get_term(command_name)
    lengths = command.motion.motion_lengths[command.motion_ids].clamp(min=2)
    phase = command.time_steps.float() / (lengths - 1).float()
    return phase.clamp(0.0, 1.0).unsqueeze(-1)


def motion_joint_pos(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    lookahead_steps: int = 0,
) -> torch.Tensor:
    """Phase-indexed joint-position reference for deployable strike feed-forward."""

    command = env.command_manager.get_term(command_name)
    if int(lookahead_steps) == 0:
        return command.joint_pos[:, asset_cfg.joint_ids]
    steps = command.time_steps + int(lookahead_steps)
    if command._use_motion_library:
        steps = torch.minimum(steps, command.motion.motion_lengths[command.motion_ids] - 1)
        reference = command.motion.joint_pos[command.motion_ids, steps]
    else:
        steps = torch.clamp(steps, max=command.motion.time_step_total - 1)
        reference = command.motion.joint_pos[steps]
    return (reference + command._joint_position_offset)[:, asset_cfg.joint_ids]


def motion_joint_vel(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    lookahead_steps: int = 0,
) -> torch.Tensor:
    """Phase-indexed joint-velocity reference for deployable strike feed-forward."""

    command = env.command_manager.get_term(command_name)
    if int(lookahead_steps) == 0:
        return command.joint_vel[:, asset_cfg.joint_ids]
    steps = command.time_steps + int(lookahead_steps)
    if command._use_motion_library:
        steps = torch.minimum(steps, command.motion.motion_lengths[command.motion_ids] - 1)
        return command.motion.joint_vel[command.motion_ids, steps][:, asset_cfg.joint_ids]
    steps = torch.clamp(steps, max=command.motion.time_step_total - 1)
    return command.motion.joint_vel[steps][:, asset_cfg.joint_ids]


# --- privileged (critic) observations: desired normal + actual racket state --------------- #
def racket_target_normal_w(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return _cmd(env, command_name).racket_target_normal_w


def racket_pos_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Actual racket position relative to base yaw frame (FK from robot joint state)."""
    cmd = _cmd(env, command_name)
    return quat_rotate_inverse(yaw_quat(cmd.base_quat_w), cmd.racket_pos_w - cmd.base_pos_w)


def racket_lin_vel_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Actual racket linear velocity in base yaw frame (FK from robot joint state)."""
    cmd = _cmd(env, command_name)
    return quat_rotate_inverse(yaw_quat(cmd.base_quat_w), cmd.racket_lin_vel_w)


def racket_normal_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Actual racket face normal in base yaw frame (FK from robot joint state)."""
    cmd = _cmd(env, command_name)
    return quat_rotate_inverse(yaw_quat(cmd.base_quat_w), cmd.racket_normal_w)


def racket_lin_vel_w(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Actual racket linear velocity (FK), world frame."""
    return _cmd(env, command_name).racket_lin_vel_w


def racket_normal_w(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Actual racket face normal (FK), world frame."""
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
