"""Opt-in wide reward terms for V1.3B CompletePriors Precision Rescue."""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv

from .hope_rewards import _cmd, racket_normal_tracking_exp, racket_velocity_tracking_exp


def _record(cmd, name: str, reward: torch.Tensor) -> torch.Tensor:
    record = getattr(cmd, "record_rescue_contribution", None)
    if record is not None:
        record(name, reward)
    return reward


def racket_normal_tracking_exact_audited(
    env: ManagerBasedRLEnv, command_name: str, std: float, audit_weight: float = 1.0
) -> torch.Tensor:
    """Bit-for-bit existing exact normal reward plus private audit counter."""
    reward = racket_normal_tracking_exp(env, command_name, std)
    _record(_cmd(env, command_name), "exact_normal", reward * float(audit_weight))
    return reward


def racket_velocity_tracking_exact_audited(
    env: ManagerBasedRLEnv, command_name: str, std: float, audit_weight: float = 1.0
) -> torch.Tensor:
    """Bit-for-bit existing exact velocity reward plus private audit counter."""
    reward = racket_velocity_tracking_exp(env, command_name, std)
    _record(_cmd(env, command_name), "exact_velocity", reward * float(audit_weight))
    return reward


def racket_normal_tracking_wide_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, audit_weight: float = 1.0
) -> torch.Tensor:
    """Same acos(clamp(dot)) normal semantics as the exact kernel, wider std."""
    cmd = _cmd(env, command_name)
    cos_ang = torch.sum(cmd.racket_normal_w * cmd.racket_target_normal_w, dim=-1).clamp(-1.0, 1.0)
    angle = torch.acos(cos_ang)
    raw = torch.exp(-(angle**2) / float(std) ** 2)
    reward = raw * cmd.strike_temporal_weight()
    if "v13b_rescue_wide_normal_raw" not in cmd.metrics:
        cmd.metrics["v13b_rescue_wide_normal_raw"] = torch.zeros_like(raw)
        cmd.metrics["v13b_rescue_wide_normal_temporal"] = torch.zeros_like(raw)
    cmd.metrics["v13b_rescue_wide_normal_raw"] = raw
    cmd.metrics["v13b_rescue_wide_normal_temporal"] = reward
    _record(cmd, "wide_normal", reward * float(audit_weight))
    return reward


def racket_velocity_tracking_position_gated_wide_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    velocity_std: float,
    position_threshold: float,
    position_excess_std: float,
    audit_weight: float = 1.0,
) -> torch.Tensor:
    """Same world-frame velocity error as exact reward, gated by current position.

    The gate is continuous: full through 2 cm, about 0.70 at 5 cm, about
    0.08 at 10 cm, and near zero at 15 cm for the audited 2/5 cm settings.
    """
    cmd = _cmd(env, command_name)
    target_pos = cmd.racket_target_pos_w
    position_error = torch.linalg.vector_norm(cmd.racket_pos_w - target_pos, dim=-1)
    velocity_error_sq = torch.sum(torch.square(cmd.racket_lin_vel_w - cmd.racket_target_vel_w), dim=-1)
    velocity_reward = torch.exp(-velocity_error_sq / float(velocity_std) ** 2)
    excess = torch.relu(position_error - float(position_threshold))
    position_gate = torch.exp(-torch.square(excess) / float(position_excess_std) ** 2)
    raw = velocity_reward * position_gate
    reward = raw * cmd.strike_temporal_weight()
    for key, value in {
        "v13b_rescue_wide_velocity_raw": raw,
        "v13b_rescue_wide_velocity_kernel": velocity_reward,
        "v13b_rescue_wide_velocity_position_gate": position_gate,
    }.items():
        if key not in cmd.metrics:
            cmd.metrics[key] = torch.zeros_like(value)
        cmd.metrics[key] = value
    _record(cmd, "wide_velocity", reward * float(audit_weight))
    return reward
