"""Opt-in wide reward terms for V1.3B CompletePriors Precision Rescue."""

from __future__ import annotations

import math

import torch
from isaaclab.envs import ManagerBasedRLEnv

from .hope_rewards import (
    _cmd,
    racket_normal_tracking_exp,
    racket_velocity_tracking_exp,
)


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


def racket_position_tracking_wide_recovery_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    audit_weight: float = 1.0,
    time_std_s: float = 0.05,
) -> torch.Tensor:
    """Broad position recovery with an independent pre-strike time kernel.

    Unlike the exact strike terms, this term supplies credit over the approach
    to the impact event.  It is intentionally zero after impact so it cannot
    reward a second swing or post-hit drift.  The spatial kernel and all other
    Rescue terms remain unchanged.
    """
    cmd = _cmd(env, command_name)
    if cmd.cfg.target_mode in ("manifest", "manifest_perturbed", "reference_free_global"):
        target_pos_now = cmd.racket_target_pos_w
    else:
        target_pos_now = cmd.racket_target_pos_w - cmd.racket_target_vel_w * cmd.time_to_strike.unsqueeze(-1)
    error_sq = torch.sum(torch.square(cmd.racket_pos_w - target_pos_now), dim=-1)
    spatial_raw = torch.exp(-error_sq / max(float(std), 1.0e-6) ** 2)
    tau = cmd.time_to_strike
    temporal = torch.exp(-torch.square(tau) / max(float(time_std_s), 1.0e-6) ** 2)
    temporal = torch.where(tau >= 0.0, temporal, torch.zeros_like(temporal))
    reward = spatial_raw * temporal
    for key, value in {
        "v13b_rescue_wide_position_raw": spatial_raw,
        "v13b_rescue_wide_position_temporal": temporal,
    }.items():
        if key not in cmd.metrics:
            cmd.metrics[key] = torch.zeros_like(value)
        cmd.metrics[key] = value
    cmd.metrics["v13b_rescue_wide_position_temporal_sum"] += temporal
    cmd.metrics["v13b_rescue_wide_position_frames_temporal_gt_0_1"] += (temporal > 0.1).float()
    cmd.metrics["v13b_rescue_wide_position_frames_temporal_gt_0_01"] += (temporal > 0.01).float()
    _record(cmd, "wide_position", reward * float(audit_weight))
    return reward


def racket_strike_position_recovery_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    audit_weight: float = 1.0,
) -> torch.Tensor:
    """Broad position recovery reward at the canonical strike frame.

    The existing wide-position term provides approach shaping before impact,
    while the inherited exact term is effectively zero at the current
    0.5--0.6 m error scale.  This term targets the same one-frame strike pulse
    used by the canonical evaluator, but keeps a broad spatial kernel so the
    policy receives useful recovery signal before it reaches centimetre-level
    precision.  The narrow exact reward remains active independently.
    """
    cmd = _cmd(env, command_name)
    position_error_sq = torch.sum(
        torch.square(cmd.racket_pos_w - cmd.racket_target_pos_w), dim=-1
    )
    raw = torch.exp(-position_error_sq / max(float(std), 1.0e-6) ** 2)
    reward = raw * cmd.strike_reward_mask().to(dtype=raw.dtype)
    cmd.metrics.setdefault("v13b_rescue_strike_position_raw", torch.zeros_like(raw))
    cmd.metrics["v13b_rescue_strike_position_raw"] = raw
    _record(cmd, "strike_position", reward * float(audit_weight))
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


def racket_joint_quality_recovery_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    position_std: float,
    velocity_std: float,
    normal_std: float,
    time_std_s: float = 0.18,
    softmax_beta: float = 4.0,
    progress_weight: float = 0.25,
    regression_weight: float = 0.10,
    progress_clip: float = 0.25,
    regression_clip: float = 0.25,
    audit_weight: float = 1.0,
) -> torch.Tensor:
    """Unified strike-state recovery for position, velocity, and normal.

    The normalized component errors are combined with a smooth maximum.  Thus
    the weakest strike component controls the joint quality, while clipped
    error-progress and per-component regression terms provide recovery signal
    without requiring every instantaneous frame to improve monotonically.
    """
    cmd = _cmd(env, command_name)
    if cmd.cfg.target_mode in (
        "manifest",
        "manifest_perturbed",
        "reference_free_global",
    ):
        target_pos_now = cmd.racket_target_pos_w
    else:
        target_pos_now = (
            cmd.racket_target_pos_w
            - cmd.racket_target_vel_w * cmd.time_to_strike.unsqueeze(-1)
        )

    eps = 1.0e-6

    position_error_sq = torch.sum(
        torch.square(cmd.racket_pos_w - target_pos_now), dim=-1
    )
    velocity_error_sq = torch.sum(
        torch.square(cmd.racket_lin_vel_w - cmd.racket_target_vel_w), dim=-1
    )
    normal_cos = torch.sum(
        cmd.racket_normal_w * cmd.racket_target_normal_w, dim=-1
    ).clamp(-1.0, 1.0)
    normal_error_rad = torch.acos(normal_cos)

    position_error_norm = position_error_sq / max(float(position_std), eps) ** 2
    velocity_error_norm = velocity_error_sq / max(float(velocity_std), eps) ** 2
    normal_error_norm = normal_error_rad**2 / max(float(normal_std), eps) ** 2
    component_error = torch.stack(
        (position_error_norm, velocity_error_norm, normal_error_norm), dim=-1
    )

    beta = max(float(softmax_beta), eps)
    joint_error = (
        torch.logsumexp(beta * component_error, dim=-1)
        - math.log(component_error.shape[-1])
    ) / beta
    joint_quality = torch.exp(-joint_error)

    prev_joint_error = cmd.metrics.setdefault(
        "v13b_rescue_joint_error_prev", torch.zeros_like(joint_error)
    )
    prev_component_error = cmd.metrics.setdefault(
        "v13b_rescue_joint_component_error_prev",
        torch.zeros_like(component_error),
    )
    prev_tau = cmd.metrics.setdefault(
        "v13b_rescue_joint_tau_prev", torch.zeros_like(cmd.time_to_strike)
    )
    initialized = cmd.metrics.setdefault(
        "v13b_rescue_joint_initialized", torch.zeros_like(joint_error)
    )

    tau = cmd.time_to_strike
    reset_mask = initialized < 0.5
    reset_mask = reset_mask | (tau > prev_tau + eps)
    episode_length_buf = getattr(env, "episode_length_buf", None)
    if episode_length_buf is not None:
        reset_mask = reset_mask | (episode_length_buf <= 1)

    progress = prev_joint_error.detach() - joint_error
    progress = torch.clamp(
        progress,
        min=-float(progress_clip),
        max=float(progress_clip),
    )
    progress = torch.where(reset_mask, torch.zeros_like(progress), progress)

    component_delta = component_error - prev_component_error.detach()
    regression = torch.relu(component_delta).amax(dim=-1)
    regression = torch.clamp(regression, max=float(regression_clip))
    regression = torch.where(reset_mask, torch.zeros_like(regression), regression)

    temporal = torch.exp(-torch.square(tau) / max(float(time_std_s), eps) ** 2)
    temporal = torch.where(tau >= 0.0, temporal, torch.zeros_like(temporal))
    reward = temporal * (
        joint_quality
        + float(progress_weight) * progress
        - float(regression_weight) * regression
    )

    prev_joint_error.copy_(joint_error.detach())
    prev_component_error.copy_(component_error.detach())
    prev_tau.copy_(tau.detach())
    initialized.fill_(1.0)

    for key, value in {
        "v13b_rescue_joint_position_error_norm": position_error_norm,
        "v13b_rescue_joint_velocity_error_norm": velocity_error_norm,
        "v13b_rescue_joint_normal_error_norm": normal_error_norm,
        "v13b_rescue_joint_error": joint_error,
        "v13b_rescue_joint_quality": joint_quality,
        "v13b_rescue_joint_progress": progress,
        "v13b_rescue_joint_regression": regression,
        "v13b_rescue_joint_temporal": temporal,
        "v13b_rescue_joint_worst_error": component_error.amax(dim=-1),
    }.items():
        if key not in cmd.metrics:
            cmd.metrics[key] = torch.zeros_like(value)
        cmd.metrics[key] = value
    _record(cmd, "joint_quality", reward * float(audit_weight))
    return reward
