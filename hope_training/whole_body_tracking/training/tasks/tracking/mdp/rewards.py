from __future__ import annotations

import json
from pathlib import Path
import torch
from typing import TYPE_CHECKING

import numpy as np

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from training.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_REALIZED_PHASE_REFERENCE_CACHE: dict[tuple[int, str, str], dict[str, torch.Tensor]] = {}


def _realized_phase_reference(
    env: ManagerBasedRLEnv, command_name: str, reference_bank: str
) -> dict[str, torch.Tensor]:
    """Load continuous zero-residual traces as reward-only phase references.

    These traces deliberately never reset physics.  They are safe to compare
    against during a continuous prefix rollout, but are not simulator states
    that may be teleported into the environment.
    """

    cache_key = (id(env), command_name, str(reference_bank))
    cached = _REALIZED_PHASE_REFERENCE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    bank = Path(reference_bank).expanduser()
    manifest = json.loads((bank / "rsi_capture_manifest.json").read_text(encoding="utf-8"))
    by_episode = {str(entry["episode_id"]): entry for entry in manifest["entries"]}
    command: MotionCommand = env.command_manager.get_term(command_name)
    episode_ids = [str(item) for item in command.motion.episode_ids]
    fields = {"body_pos_w": [], "body_quat_w": [], "body_ang_vel_w": [], "motion_step": []}
    for episode_id in episode_ids:
        entry = by_episode.get(episode_id)
        if entry is None:
            raise ValueError(f"Realized phase bank has no episode {episode_id!r}")
        with np.load(bank / entry["state_file"], allow_pickle=False) as data:
            for field in fields:
                fields[field].append(torch.as_tensor(data[field], dtype=torch.float32, device=env.device))
    lengths = {tensor.shape[0] for tensor in fields["motion_step"]}
    if len(lengths) != 1:
        raise ValueError("Stage-A realized phase references must have equal lengths")
    cached = {name: torch.stack(values, dim=0) for name, values in fields.items()}
    _REALIZED_PHASE_REFERENCE_CACHE[cache_key] = cached
    return cached


def _realized_phase_indices(
    env: ManagerBasedRLEnv, command_name: str, reference_bank: str
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    reference = _realized_phase_reference(env, command_name, reference_bank)
    command: MotionCommand = env.command_manager.get_term(command_name)
    # Capture frame zero is simulator motion step one.  Clamp the brief reset
    # interval to that first continuous state rather than inventing a teleport.
    indices = (command.time_steps - 1).clamp(min=0, max=reference["motion_step"].shape[1] - 1)
    if not getattr(command, "_use_motion_library", False):
        motion_ids = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    else:
        motion_ids = command.motion_ids
    if motion_ids.numel() != env.num_envs:
        raise RuntimeError(
            "Stage-A realized reference motion-id batch does not match the environment batch: "
            f"motion_ids={motion_ids.numel()} envs={env.num_envs}"
        )
    if int(motion_ids.max().item()) >= reference["motion_step"].shape[0]:
        raise RuntimeError(
            "Stage-A realized reference bank lacks a sampled motion: "
            f"max_motion_id={int(motion_ids.max().item())} bank_size={reference['motion_step'].shape[0]}"
        )
    return reference, indices, motion_ids


def realized_torso_orientation_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    reference_bank: str,
    torso_body_name: str,
    std: float,
) -> torch.Tensor:
    reference, indices, motion_ids = _realized_phase_indices(env, command_name, reference_bank)
    body_id = env.scene["robot"].body_names.index(torso_body_name)
    expected = reference["body_quat_w"][motion_ids, indices, body_id]
    actual = env.scene["robot"].data.body_quat_w[:, body_id]
    error = quat_error_magnitude(expected, actual) ** 2
    return torch.exp(-error / std**2)


def realized_torso_angular_velocity_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    reference_bank: str,
    torso_body_name: str,
    std: float,
) -> torch.Tensor:
    reference, indices, motion_ids = _realized_phase_indices(env, command_name, reference_bank)
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_id = env.scene["robot"].body_names.index(torso_body_name)
    expected = reference["body_ang_vel_w"][motion_ids, indices, body_id]
    # In a finite strike's settling tail the arm/waist target is held at the
    # last frame.  The desired torso velocity is therefore zero: the policy is
    # rewarded for dissipating residual swing momentum rather than reproducing
    # the last nonzero sampled motion velocity.
    tail_steps = getattr(command, "tail_steps", None)
    if tail_steps is not None:
        expected = torch.where(tail_steps.unsqueeze(-1) > 0, torch.zeros_like(expected), expected)
    actual = env.scene["robot"].data.body_ang_vel_w[:, body_id]
    error = torch.sum(torch.square(expected - actual), dim=-1)
    return torch.exp(-error / std**2)


def realized_root_height_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    reference_bank: str,
    std: float,
) -> torch.Tensor:
    reference, indices, motion_ids = _realized_phase_indices(env, command_name, reference_bank)
    expected_z = reference["body_pos_w"][motion_ids, indices, 0, 2]
    actual_z = env.scene["robot"].data.root_pos_w[:, 2]
    return torch.exp(-torch.square(expected_z - actual_z) / std**2)


def post_strike_root_linear_velocity_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
) -> torch.Tensor:
    """Reward translational settling only after the finite strike reaches its tail.

    During the swing the torso/COM may move deliberately.  Once the final
    arm/waist target is held, persistent root translation is instead evidence
    of an unresolved fall.  The term is zero before the tail so it cannot
    dominate the swing phase with a constant reward.
    """

    command: MotionCommand = env.command_manager.get_term(command_name)
    tail_steps = getattr(command, "tail_steps", None)
    if tail_steps is None:
        return torch.zeros(env.num_envs, device=env.device)
    error = torch.sum(torch.square(env.scene["robot"].data.root_lin_vel_b), dim=-1)
    settling_reward = torch.exp(-error / std**2)
    return torch.where(tail_steps > 0, settling_reward, torch.zeros_like(settling_reward))


def post_strike_ready_score(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    linear_velocity_std: float,
    angular_velocity_std: float,
    minimum_height: float,
    height_std: float,
    contact_threshold: float,
) -> torch.Tensor:
    """Score a task-level, post-strike ready state without prescribing joints.

    The score is enabled only after the final swing frame is held.  It rewards
    the *outcome* needed for a next strike: low body-frame translation and
    rotation, a non-collapsing root height, and both feet bearing load.  It
    deliberately contains no knee/hip/ankle target or directional action cue.
    """

    command: MotionCommand = env.command_manager.get_term(command_name)
    tail_steps = getattr(command, "tail_steps", None)
    if tail_steps is None:
        return torch.zeros(env.num_envs, device=env.device)

    robot = env.scene["robot"]
    linear_error = torch.sum(torch.square(robot.data.root_lin_vel_b), dim=-1)
    angular_error = torch.sum(torch.square(robot.data.root_ang_vel_b), dim=-1)
    height_deficit = torch.relu(minimum_height - robot.data.root_pos_w[:, 2])

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = torch.linalg.vector_norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
    both_feet = (forces > contact_threshold).all(dim=-1).to(dtype=torch.float32)

    score = (
        torch.exp(-linear_error / linear_velocity_std**2)
        * torch.exp(-angular_error / angular_velocity_std**2)
        * torch.exp(-torch.square(height_deficit) / height_std**2)
        * both_feet
    )
    return torch.where(tail_steps > 0, score, torch.zeros_like(score))


def post_strike_root_velocity_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    angular: bool = False,
) -> torch.Tensor:
    """Dense tail-only body-frame velocity cost for result-driven recovery.

    Unlike an exponential reward with a narrow standard deviation, this stays
    informative while the robot is still moving quickly.  It says only that
    the floating base must stop translating/rotating after the strike; it does
    not encode any preferred leg motion.
    """

    command: MotionCommand = env.command_manager.get_term(command_name)
    tail_steps = getattr(command, "tail_steps", None)
    if tail_steps is None:
        return torch.zeros(env.num_envs, device=env.device)
    robot = env.scene["robot"]
    velocity = robot.data.root_ang_vel_b if angular else robot.data.root_lin_vel_b
    cost = torch.sum(torch.square(velocity), dim=-1)
    return torch.where(tail_steps > 0, cost, torch.zeros_like(cost))


def post_strike_root_height_deficit_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    minimum_height: float,
) -> torch.Tensor:
    """Dense tail-only penalty for a base that keeps dropping after a swing."""

    command: MotionCommand = env.command_manager.get_term(command_name)
    tail_steps = getattr(command, "tail_steps", None)
    if tail_steps is None:
        return torch.zeros(env.num_envs, device=env.device)
    deficit = torch.relu(minimum_height - env.scene["robot"].data.root_pos_w[:, 2])
    cost = torch.square(deficit)
    return torch.where(tail_steps > 0, cost, torch.zeros_like(cost))


def post_strike_both_feet_contact(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Tail-only result reward for both feet carrying contact load."""

    command: MotionCommand = env.command_manager.get_term(command_name)
    tail_steps = getattr(command, "tail_steps", None)
    if tail_steps is None:
        return torch.zeros(env.num_envs, device=env.device)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = torch.linalg.vector_norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
    both_feet = (force > threshold).all(dim=-1).to(dtype=torch.float32)
    return torch.where(tail_steps > 0, both_feet, torch.zeros_like(both_feet))


def feet_contact_fraction(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Fraction of designated feet with current contact force above threshold."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = torch.linalg.vector_norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
    return (force > threshold).to(dtype=torch.float32).mean(dim=-1)


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_joint_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Track reference joint positions for a selected joint subset."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.square(command.robot_joint_pos[:, asset_cfg.joint_ids] - command.joint_pos[:, asset_cfg.joint_ids])
    return torch.exp(-error.mean(-1) / std**2)


def action_raw_l2(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    """Penalize raw policy residual magnitude for custom action terms."""
    action_term = env.action_manager.get_term(action_name)
    raw_actions = getattr(action_term, "_raw_actions", None)
    if raw_actions is None:
        raw_actions = env.action_manager.action
    return torch.mean(torch.square(raw_actions), dim=-1)


def action_unbounded_excess_l2(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
    raw_limit: float = 0.20,
    action_indices: tuple[int, ...] = (),
) -> torch.Tensor:
    """Penalize latent actor output beyond a pre-saturation trust band.

    The action term applies masks and phase gates first, then exposes the
    resulting unbounded residual.  Penalizing that value prevents PPO from
    treating the bounded actuator envelope as a flat, free control region.
    ``action_indices`` keeps structurally masked channels (e.g. waist during
    Stage-A) out of the leg-stabilizer objective.
    """
    action_term = env.action_manager.get_term(action_name)
    raw_actions = getattr(action_term, "unbounded_actions", None)
    if raw_actions is None:
        raw_actions = env.action_manager.action
    if action_indices:
        ids = torch.tensor(action_indices, dtype=torch.long, device=raw_actions.device)
        raw_actions = raw_actions.index_select(dim=-1, index=ids)
    excess = torch.relu(torch.abs(raw_actions) - raw_limit)
    return torch.mean(torch.square(excess), dim=-1)


def action_execution_gap_l2(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
    action_indices: tuple[int, ...] = (),
    deadband: float = 0.02,
) -> torch.Tensor:
    """Penalize only the meaningful latent-to-executed action mismatch.

    The action term may structurally mask channels or phase-gate authority.
    This uses the post-gate latent command, then ignores a small smooth-bound
    deadband so normal near-zero compression is free while persistent
    saturation dependence remains visible to PPO.
    """
    action_term = env.action_manager.get_term(action_name)
    latent = getattr(action_term, "unbounded_actions", None)
    executed = getattr(action_term, "raw_actions", None)
    if latent is None or executed is None:
        return torch.zeros(env.num_envs, device=env.device)
    if action_indices:
        ids = torch.tensor(action_indices, dtype=torch.long, device=latent.device)
        latent = latent.index_select(dim=-1, index=ids)
        executed = executed.index_select(dim=-1, index=ids)
    excess = torch.relu(torch.abs(latent - executed) - deadband)
    return torch.mean(torch.square(excess), dim=-1)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward
