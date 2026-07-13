"""HOPE goal-tracking reward terms (HITTER r_goal).

These implement the racket/base target tracking rewards on top of the BeyondMimic imitation
reward (``r_imitation``, the ``motion_*`` terms already in ``rewards.py``) and the regularization
reward (``r_regularization``, ``action_rate_l2`` / ``joint_torques_l2`` / contact penalties).

Activation timing follows HITTER: the base-position reward is active **before** the strike; the
racket position/velocity/normal rewards are active only in a **short window around** the strike.
Because a ``RewardTermCfg`` weight is constant, the time gating is applied *inside* each term by
multiplying the exponential kernel by the command's ``pre_strike`` / ``strike_window`` mask.

The exponential kernel form (``exp(-error/std**2)``) mirrors the BeyondMimic motion-tracking
rewards. HITTER does not publish reward weights or kernel forms, so the weights in the env config
are HOPE choices to be tuned, not paper-sourced values.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from training.tasks.tracking.mdp.hope_commands import RacketTargetCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _cmd(env: ManagerBasedRLEnv, command_name: str) -> RacketTargetCommand:
    return env.command_manager.get_term(command_name)


def racket_position_tracking_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Track racket center position near strike.

    Manifest targets are impact states, so do not extrapolate them backward/forward with velocity.
    The legacy sampled modes keep the old swing-through target for compatibility.
    """
    cmd = _cmd(env, command_name)
    if cmd.cfg.target_mode == "manifest":
        target_pos_now = cmd.racket_target_pos_w
    else:
        target_pos_now = cmd.racket_target_pos_w - cmd.racket_target_vel_w * cmd.time_to_strike.unsqueeze(-1)
    error = torch.sum(torch.square(cmd.racket_pos_w - target_pos_now), dim=-1)
    return torch.exp(-error / std**2) * cmd.strike_temporal_weight()


def racket_position_axis_tracking_exp(env: ManagerBasedRLEnv, command_name: str, axis: int, std: float) -> torch.Tensor:
    """Track one world-frame racket position axis near strike.

    This is an auxiliary shaping term for early curricula. It prevents one
    stubborn axis from being hidden by a broad 3D position kernel.
    """
    cmd = _cmd(env, command_name)
    if cmd.cfg.target_mode == "manifest":
        target_pos_now = cmd.racket_target_pos_w
    else:
        target_pos_now = cmd.racket_target_pos_w - cmd.racket_target_vel_w * cmd.time_to_strike.unsqueeze(-1)
    axis_error = torch.square(cmd.racket_pos_w[:, axis] - target_pos_now[:, axis])
    return torch.exp(-axis_error / std**2) * cmd.strike_temporal_weight()


def racket_velocity_tracking_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Track racket linear velocity near the strike time (FK actual vs desired, world frame)."""
    cmd = _cmd(env, command_name)
    error = torch.sum(torch.square(cmd.racket_lin_vel_w - cmd.racket_target_vel_w), dim=-1)
    return torch.exp(-error / std**2) * cmd.strike_temporal_weight()


def racket_normal_tracking_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Track racket face-normal orientation near the strike time. ``std`` is in radians."""
    cmd = _cmd(env, command_name)
    cos_ang = torch.sum(cmd.racket_normal_w * cmd.racket_target_normal_w, dim=-1).clamp(-1.0, 1.0)
    angle = torch.acos(cos_ang)
    raw = torch.exp(-(angle**2) / std**2)
    reward = raw * cmd.strike_temporal_weight()
    if "racket_normal_reward_raw" in cmd.metrics:
        cmd.metrics["racket_normal_reward_raw"] = raw
        cmd.metrics["racket_normal_reward_temporal"] = reward
        cmd.metrics["racket_normal_reward_std_rad"][:] = float(std)
    return reward


def racket_hit_coupled_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    pos_std: float,
    vel_std: float,
    normal_std: float,
    base: float = 0.35,
    vel_coeff: float = 0.30,
    normal_coeff: float = 0.35,
) -> torch.Tensor:
    """Softly couple impact position with velocity and normal quality.

    This avoids a policy getting almost full hit reward by placing the racket
    center correctly while ignoring face normal or velocity. The term is still
    dense: bad velocity/normal reduce the position reward instead of creating a
    hard sparse gate.
    """
    cmd = _cmd(env, command_name)
    if cmd.cfg.target_mode == "manifest":
        target_pos_now = cmd.racket_target_pos_w
    else:
        target_pos_now = cmd.racket_target_pos_w - cmd.racket_target_vel_w * cmd.time_to_strike.unsqueeze(-1)

    pos_error = torch.sum(torch.square(cmd.racket_pos_w - target_pos_now), dim=-1)
    vel_error = torch.sum(torch.square(cmd.racket_lin_vel_w - cmd.racket_target_vel_w), dim=-1)
    cos_ang = torch.sum(cmd.racket_normal_w * cmd.racket_target_normal_w, dim=-1).clamp(-1.0, 1.0)
    angle = torch.acos(cos_ang)

    r_pos = torch.exp(-pos_error / pos_std**2)
    r_vel = torch.exp(-vel_error / vel_std**2)
    r_normal = torch.exp(-(angle**2) / normal_std**2)
    coupling = float(base) + float(vel_coeff) * r_vel + float(normal_coeff) * r_normal
    reward = r_pos * coupling * cmd.strike_temporal_weight()

    if "racket_hit_coupled_reward_raw" in cmd.metrics:
        cmd.metrics["racket_hit_coupled_reward_raw"] = r_pos * coupling
        cmd.metrics["racket_hit_coupled_pos_raw"] = r_pos
        cmd.metrics["racket_hit_coupled_vel_raw"] = r_vel
        cmd.metrics["racket_hit_coupled_normal_raw"] = r_normal
    return reward


def base_position_tracking_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Track desired base XY position before the strike (encourages repositioning footwork)."""
    cmd = _cmd(env, command_name)
    error = torch.sum(torch.square(cmd.base_pos_w[:, :2] - cmd.base_target_pos_w), dim=-1)
    return torch.exp(-error / std**2) * cmd.pre_strike.float()
