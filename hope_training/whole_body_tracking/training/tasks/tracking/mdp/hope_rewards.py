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
    if cmd.cfg.target_mode in ("manifest", "manifest_perturbed", "reference_free_global"):
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
    if cmd.cfg.target_mode in ("manifest", "manifest_perturbed", "reference_free_global"):
        target_pos_now = cmd.racket_target_pos_w
    else:
        target_pos_now = cmd.racket_target_pos_w - cmd.racket_target_vel_w * cmd.time_to_strike.unsqueeze(-1)
    axis_error = torch.square(cmd.racket_pos_w[:, axis] - target_pos_now[:, axis])
    return torch.exp(-axis_error / std**2) * cmd.strike_temporal_weight()


def racket_incremental_position_tracking(
    env: ManagerBasedRLEnv, command_name: str, std: float = 0.03
) -> torch.Tensor:
    """Train the local adapter against the requested displacement.

    The residual contract is expressed relative to the frozen anchor target:
    ``(actual - anchor) - (external - anchor)``.  Paired +/- environments can
    replace this with a measured nominal baseline later without changing the
    observation or action interfaces.
    """
    cmd = _cmd(env, command_name)
    actual_delta = cmd.racket_pos_w - cmd.racket_anchor_target_pos_w
    target_delta = cmd.racket_target_pos_w - cmd.racket_anchor_target_pos_w
    error = torch.sum(torch.square(actual_delta - target_delta), dim=-1)
    reward = torch.exp(-error / max(float(std), 1.0e-6) ** 2) * cmd.strike_temporal_weight()
    cmd.metrics.setdefault("adapter_incremental_error", torch.zeros_like(error))
    cmd.metrics["adapter_incremental_error"] = torch.sqrt(error)
    return reward


def racket_target_progress(env: ManagerBasedRLEnv, command_name: str, scale_m: float = 0.10) -> torch.Tensor:
    """Dense pre-hit approach reward: previous distance minus current distance."""
    cmd = _cmd(env, command_name)
    actual = cmd.racket_pos_w
    target = cmd.racket_target_pos_w
    distance = torch.linalg.vector_norm(actual - target, dim=-1)
    previous = getattr(cmd, "_v13b_previous_distance", None)
    if previous is None:
        previous = distance.detach().clone()
        cmd._v13b_previous_distance = previous
    progress = (previous - distance) / max(float(scale_m), 1.0e-6)
    cmd._v13b_previous_distance = distance.detach()
    return torch.where(cmd.pre_strike, progress, torch.zeros_like(progress)).clamp(-1.0, 1.0)


def racket_paired_incremental_position_tracking(
    env: ManagerBasedRLEnv, command_name: str, std: float = 0.02
) -> torch.Tensor:
    """Paired local displacement reward used by P0.

    For each ``0,+/-axis`` group, subtract the simultaneously simulated
    zero-offset racket state before comparing it with the requested target
    difference.  This removes the frozen anchor's absolute impact bias from
    the primary adapter learning signal.
    """
    cmd = _cmd(env, command_name)
    baseline = cmd.adapter_pair_baseline_env
    origins = env.scene.env_origins
    # Parallel environments have distinct world origins. Compare each racket
    # in its own environment-relative world frame before forming the pair.
    actual_rel = cmd.racket_pos_w - origins
    target_rel = cmd.racket_target_pos_w - origins
    actual_delta = actual_rel - actual_rel[baseline]
    target_delta = target_rel - target_rel[baseline]
    error = torch.sum(torch.square(actual_delta - target_delta), dim=-1)
    active = cmd.adapter_pair_active & (baseline != torch.arange(cmd.num_envs, device=cmd.device))
    reward = torch.exp(-error / max(float(std), 1.0e-6) ** 2)
    reward = torch.where(active, reward, torch.zeros_like(reward)) * cmd.strike_temporal_weight()
    cmd.metrics.setdefault("adapter_paired_incremental_error", torch.zeros_like(error))
    cmd.metrics["adapter_paired_incremental_error"] = torch.sqrt(error)
    return reward


def racket_incremental_direction_gain(
    env: ManagerBasedRLEnv, command_name: str, min_norm: float = 1.0e-3
) -> torch.Tensor:
    """Directional/gain shaping for non-zero local target offsets."""
    cmd = _cmd(env, command_name)
    actual = cmd.racket_pos_w - cmd.racket_anchor_target_pos_w
    target = cmd.racket_target_pos_w - cmd.racket_anchor_target_pos_w
    target_norm = torch.linalg.vector_norm(target, dim=-1)
    actual_norm = torch.linalg.vector_norm(actual, dim=-1)
    dot = torch.sum(actual * target, dim=-1)
    cosine = dot / (actual_norm * target_norm).clamp_min(min_norm**2)
    gain = dot / torch.square(target_norm).clamp_min(min_norm**2)
    valid = target_norm > float(min_norm)
    score = torch.where(
        valid,
        0.5 * (cosine.clamp(-1.0, 1.0) + 1.0) * torch.exp(-torch.square(gain - 1.0)),
        torch.ones_like(target_norm),
    )
    return score * cmd.strike_temporal_weight()


def racket_paired_incremental_direction_gain(
    env: ManagerBasedRLEnv, command_name: str, min_norm: float = 1.0e-3
) -> torch.Tensor:
    """Paired direction/gain shaping consistent with P0's primary reward.

    The inherited absolute anchor miss is much larger than P0's centimetre
    command.  Compare every non-baseline environment with its simultaneously
    simulated nominal sibling so direction and gain describe the incremental
    response rather than that fixed miss.
    """
    cmd = _cmd(env, command_name)
    baseline = cmd.adapter_pair_baseline_env
    origins = env.scene.env_origins
    actual_rel = cmd.racket_pos_w - origins
    target_rel = cmd.racket_target_pos_w - origins
    actual = actual_rel - actual_rel[baseline]
    target = target_rel - target_rel[baseline]
    target_norm = torch.linalg.vector_norm(target, dim=-1)
    actual_norm = torch.linalg.vector_norm(actual, dim=-1)
    dot = torch.sum(actual * target, dim=-1)
    cosine = dot / (actual_norm * target_norm).clamp_min(min_norm**2)
    gain = dot / torch.square(target_norm).clamp_min(min_norm**2)
    valid = (target_norm > float(min_norm)) & cmd.adapter_pair_active
    score = 0.5 * (cosine.clamp(-1.0, 1.0) + 1.0) * torch.exp(-torch.square(gain - 1.0))
    return torch.where(valid, score, torch.zeros_like(score)) * cmd.strike_temporal_weight()


def _paired_incremental_state(env: ManagerBasedRLEnv, command_name: str):
    cmd = _cmd(env, command_name)
    baseline = cmd.adapter_pair_baseline_env
    origins = env.scene.env_origins
    actual = (cmd.racket_pos_w - origins) - (cmd.racket_pos_w - origins)[baseline]
    target = (cmd.racket_target_pos_w - origins) - (cmd.racket_target_pos_w - origins)[baseline]
    active = cmd.adapter_pair_active & (baseline != torch.arange(cmd.num_envs, device=cmd.device))

    # A paired displacement is meaningful only when both siblings are on the
    # same reference clip and control step.  Normal operation enforces this
    # at runner startup, but masking a group after an asynchronous reset
    # prevents a single early termination from creating a false Cartesian
    # reward signal.
    motion = env.command_manager.get_term(cmd.cfg.motion_command_name)
    phase_match = motion.time_steps == motion.time_steps[baseline]
    motion_ids = getattr(motion, "motion_ids", None)
    if motion_ids is not None:
        phase_match &= motion_ids == motion_ids[baseline]
    active &= phase_match
    cmd.metrics["adapter_pair_phase_synced"] = phase_match.to(dtype=actual.dtype)
    gate = getattr(env, "target_adapter_gate", None)
    if gate is None:
        gate = cmd.strike_temporal_weight().unsqueeze(-1)
    return cmd, actual, target, active, gate.squeeze(-1)


def racket_paired_incremental_dense_huber(
    env: ManagerBasedRLEnv, command_name: str, scale_m: float = 0.01, beta: float = 1.0
) -> torch.Tensor:
    """Dense paired Smooth-L1 tracking along the adapter's own phase gate."""
    cmd, actual, target, active, gate = _paired_incremental_state(env, command_name)
    error = (actual - gate.unsqueeze(-1) * target) / max(float(scale_m), 1.0e-6)
    abs_error = torch.abs(error)
    loss = torch.where(abs_error < beta, 0.5 * torch.square(error) / beta, abs_error - 0.5 * beta).sum(-1)
    cmd.metrics["adapter_dense_huber"] = loss
    return torch.where(active, -loss * gate, torch.zeros_like(loss))


def racket_paired_incremental_gain_loss(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    cmd, actual, target, active, gate = _paired_incremental_state(env, command_name)
    target_norm_sq = torch.sum(torch.square(target), dim=-1).clamp_min(1.0e-8)
    gain = torch.sum(actual * target, dim=-1) / target_norm_sq
    loss = torch.nn.functional.smooth_l1_loss(gain, torch.ones_like(gain), reduction="none")
    cmd.metrics["adapter_projected_gain"] = gain
    return torch.where(active, -loss * gate, torch.zeros_like(loss))


def racket_paired_incremental_cross_axis_loss(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    cmd, actual, target, active, gate = _paired_incremental_state(env, command_name)
    target_norm = torch.linalg.vector_norm(target, dim=-1).clamp_min(1.0e-4)
    unit = target / target_norm.unsqueeze(-1)
    parallel = torch.sum(actual * unit, dim=-1, keepdim=True) * unit
    loss = torch.linalg.vector_norm(actual - parallel, dim=-1) / target_norm
    cmd.metrics["adapter_cross_axis_ratio"] = loss
    return torch.where(active, -loss * gate, torch.zeros_like(loss))


def target_adapter_zero_action_hold(env: ManagerBasedRLEnv, command_name: str, threshold_m: float = 1.0e-5) -> torch.Tensor:
    """Keep the adapter silent for zero-offset (baseline) paired samples."""
    cmd = _cmd(env, command_name)
    delta = torch.linalg.vector_norm(cmd.external_target_delta_local_b(), dim=-1)
    action = getattr(env, "target_adapter_last_action", None)
    gate = getattr(env, "target_adapter_gate", None)
    if action is None or gate is None:
        return torch.zeros_like(delta)
    loss = torch.mean(torch.square(action), dim=-1)
    return torch.where(delta <= threshold_m, -loss * gate.squeeze(-1), torch.zeros_like(loss))


def racket_velocity_tracking_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Track racket linear velocity near the strike time (FK actual vs desired, world frame)."""
    cmd = _cmd(env, command_name)
    error = torch.sum(torch.square(cmd.racket_lin_vel_w - cmd.racket_target_vel_w), dim=-1)
    return torch.exp(-error / std**2) * cmd.strike_temporal_weight()


def _canonical_hit_window_weight(cmd: RacketTargetCommand, half_window_steps: int = 3) -> torch.Tensor:
    motion = cmd._motion_term
    if motion is None:
        motion = cmd._env.command_manager.get_term(cmd.cfg.motion_command_name)
    if motion._use_motion_library:
        hit = motion.motion.hit_frame[motion.motion_ids]
    else:
        hit = torch.full_like(motion.time_steps, int(motion.motion.hit_frame[0]))
    distance = torch.abs(motion.time_steps.to(torch.float32) - hit.to(torch.float32))
    inside = distance <= float(half_window_steps)
    weight = torch.exp(-0.5 * torch.square(distance / max(float(half_window_steps), 1.0)))
    return torch.where(inside, weight, torch.zeros_like(weight))


def racket_velocity_magnitude_tracking_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float = 1.0, half_window_steps: int = 3
) -> torch.Tensor:
    cmd = _cmd(env, command_name)
    error = torch.abs(torch.linalg.vector_norm(cmd.racket_lin_vel_w, dim=-1) - torch.linalg.vector_norm(cmd.racket_target_vel_w, dim=-1))
    return torch.exp(-torch.square(error) / max(float(std), 1.0e-6) ** 2) * _canonical_hit_window_weight(cmd, half_window_steps)


def racket_velocity_direction_tracking(
    env: ManagerBasedRLEnv, command_name: str, half_window_steps: int = 3
) -> torch.Tensor:
    cmd = _cmd(env, command_name)
    actual = cmd.racket_lin_vel_w
    target = cmd.racket_target_vel_w
    cosine = torch.sum(actual * target, dim=-1) / (
        torch.linalg.vector_norm(actual, dim=-1) * torch.linalg.vector_norm(target, dim=-1)
    ).clamp_min(1.0e-6)
    return 0.5 * (cosine.clamp(-1.0, 1.0) + 1.0) * _canonical_hit_window_weight(cmd, half_window_steps)


def racket_signed_velocity_tracking(
    env: ManagerBasedRLEnv, command_name: str, half_window_steps: int = 3
) -> torch.Tensor:
    cmd = _cmd(env, command_name)
    target_norm = torch.linalg.vector_norm(cmd.racket_target_vel_w, dim=-1).clamp_min(1.0e-6)
    signed = torch.sum(cmd.racket_lin_vel_w * cmd.racket_target_vel_w, dim=-1) / target_norm
    scale = target_norm.clamp_min(1.0)
    return (signed / scale).clamp(-1.0, 1.0) * _canonical_hit_window_weight(cmd, half_window_steps)


def racket_pass_through_reward(
    env: ManagerBasedRLEnv, command_name: str, position_gate: float = 0.10, minimum_speed: float = 0.5,
    half_window_steps: int = 3,
) -> torch.Tensor:
    cmd = _cmd(env, command_name)
    pos_error = torch.linalg.vector_norm(cmd.racket_pos_w - cmd.racket_target_pos_w, dim=-1)
    speed = torch.linalg.vector_norm(cmd.racket_lin_vel_w, dim=-1)
    target_norm = torch.linalg.vector_norm(cmd.racket_target_vel_w, dim=-1).clamp_min(1.0e-6)
    signed = torch.sum(cmd.racket_lin_vel_w * cmd.racket_target_vel_w, dim=-1) / target_norm
    gate = (pos_error < float(position_gate)) & (speed > float(minimum_speed)) & (signed > 0.0)
    return gate.to(torch.float32) * _canonical_hit_window_weight(cmd, half_window_steps)


def racket_stop_at_target_penalty(
    env: ManagerBasedRLEnv, command_name: str, position_gate: float = 0.10, minimum_speed: float = 0.5,
    half_window_steps: int = 3,
) -> torch.Tensor:
    cmd = _cmd(env, command_name)
    pos_error = torch.linalg.vector_norm(cmd.racket_pos_w - cmd.racket_target_pos_w, dim=-1)
    speed = torch.linalg.vector_norm(cmd.racket_lin_vel_w, dim=-1)
    stopped = (pos_error < float(position_gate)) & (speed < float(minimum_speed))
    return -stopped.to(torch.float32) * _canonical_hit_window_weight(cmd, half_window_steps)


def racket_reverse_motion_penalty(
    env: ManagerBasedRLEnv, command_name: str, half_window_steps: int = 3
) -> torch.Tensor:
    cmd = _cmd(env, command_name)
    target_norm = torch.linalg.vector_norm(cmd.racket_target_vel_w, dim=-1).clamp_min(1.0e-6)
    signed = torch.sum(cmd.racket_lin_vel_w * cmd.racket_target_vel_w, dim=-1) / target_norm
    return -torch.relu(-signed) * _canonical_hit_window_weight(cmd, half_window_steps)


def racket_hit_timing_kernel(env: ManagerBasedRLEnv, command_name: str, half_window_steps: int = 3) -> torch.Tensor:
    return _canonical_hit_window_weight(_cmd(env, command_name), half_window_steps)


def phase_magnitude_penalty(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    phase = getattr(env, "p5u_phase_effective", None)
    if phase is None or phase.shape[-1] == 0:
        return torch.zeros(env.num_envs, device=env.device)
    return -torch.mean(torch.abs(phase), dim=-1)


def phase_rate_penalty(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    rate = getattr(env, "p5u_phase_rate", None)
    if rate is None or rate.shape[-1] == 0:
        return torch.zeros(env.num_envs, device=env.device)
    return -torch.mean(torch.square(rate), dim=-1)


def phase_group_consistency_penalty(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    phase = getattr(env, "p5u_phase_effective", None)
    if phase is None or phase.shape[-1] < 4:
        return torch.zeros(env.num_envs, device=env.device)
    return -(torch.abs(phase[:, 1] - phase[:, 2]) + torch.abs(phase[:, 2] - phase[:, 3]))


def racket_velocity_tracking_position_gated_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    velocity_std: float,
    position_threshold: float,
    position_excess_std: float,
) -> torch.Tensor:
    """Improve impact speed only while the racket remains in the hit corridor.

    Velocity-only shaping let the V3 coordinator trade exact placement for a
    small speed improvement.  This term is fully active inside the accepted
    placement radius, then rapidly fades as placement moves outside it.
    """
    cmd = _cmd(env, command_name)
    if cmd.cfg.target_mode in ("manifest", "manifest_perturbed", "reference_free_global"):
        target_pos_now = cmd.racket_target_pos_w
    else:
        target_pos_now = cmd.racket_target_pos_w - cmd.racket_target_vel_w * cmd.time_to_strike.unsqueeze(-1)

    position_error = torch.linalg.vector_norm(cmd.racket_pos_w - target_pos_now, dim=-1)
    velocity_error_sq = torch.sum(torch.square(cmd.racket_lin_vel_w - cmd.racket_target_vel_w), dim=-1)
    velocity_reward = torch.exp(-velocity_error_sq / velocity_std**2)
    position_excess = torch.relu(position_error - position_threshold)
    position_gate = torch.exp(-torch.square(position_excess) / position_excess_std**2)
    raw_reward = velocity_reward * position_gate
    reward = raw_reward * cmd.strike_temporal_weight()

    if "racket_velocity_position_gated_reward_raw" in cmd.metrics:
        cmd.metrics["racket_velocity_position_gated_reward_raw"] = raw_reward
        cmd.metrics["racket_velocity_position_gated_velocity_raw"] = velocity_reward
        cmd.metrics["racket_velocity_position_gated_position_gate"] = position_gate
    return reward


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
    if cmd.cfg.target_mode in ("manifest", "manifest_perturbed", "reference_free_global"):
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


def racket_exact_hit_precision_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    pos_std: float,
    vel_std: float,
    normal_std: float,
    time_std: float,
    pos_coeff: float = 0.40,
    velocity_coeff: float = 0.30,
    normal_coeff: float = 0.30,
) -> torch.Tensor:
    """Prioritize the canonical TCP state at the actual strike frame.

    The original hit terms are intentionally active over the whole configured
    strike window.  That is useful for early shaping, but a residual tracker
    can then improve an average window score while moving the tagged canonical
    frame in the wrong direction.  This term keeps position primary and uses
    a narrow Gaussian in ``time_to_strike`` so PPO receives its strongest
    signal at the frame used by the canonical evaluator.
    """
    cmd = _cmd(env, command_name)
    target_pos = cmd.racket_target_pos_w
    pos_error_sq = torch.sum(torch.square(cmd.racket_pos_w - target_pos), dim=-1)
    vel_error_sq = torch.sum(torch.square(cmd.racket_lin_vel_w - cmd.racket_target_vel_w), dim=-1)
    cos_ang = torch.sum(cmd.racket_normal_w * cmd.racket_target_normal_w, dim=-1).clamp(-1.0, 1.0)
    normal_error = torch.acos(cos_ang)

    r_pos = torch.exp(-pos_error_sq / max(float(pos_std), 1.0e-6) ** 2)
    r_vel = torch.exp(-vel_error_sq / max(float(vel_std), 1.0e-6) ** 2)
    r_normal = torch.exp(-torch.square(normal_error) / max(float(normal_std), 1.0e-6) ** 2)
    time_gate = torch.exp(
        -0.5 * torch.square(cmd.time_to_strike / max(float(time_std), 1.0e-6))
    )
    # Position remains the primary task.  Velocity and normal are bounded
    # quality factors, so they cannot buy a better reward by sacrificing the
    # canonical strike location.  The coefficients are explicit so the
    # training contract can deliberately emphasize impact speed and face
    # orientation without changing the 26-D action contract.
    coeff_sum = max(float(pos_coeff) + float(velocity_coeff) + float(normal_coeff), 1.0e-6)
    score = r_pos * (
        float(pos_coeff) + float(velocity_coeff) * r_vel + float(normal_coeff) * r_normal
    ) / coeff_sum * time_gate
    return score


def base_position_tracking_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Track desired base XY position before the strike (encourages repositioning footwork)."""
    cmd = _cmd(env, command_name)
    error = torch.sum(torch.square(cmd.base_pos_w[:, :2] - cmd.base_target_pos_w), dim=-1)
    return torch.exp(-error / std**2) * cmd.pre_strike.float()
