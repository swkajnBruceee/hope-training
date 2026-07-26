from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from training.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def f0_upper_last_action(env: ManagerBasedEnv) -> torch.Tensor:
    """Return the frozen upper actor's 10-D action history for F0."""
    value = getattr(env, "f0_upper_last_action", None)
    if value is None:
        return torch.zeros((env.num_envs, 10), device=env.device)
    return value


def legacy_stage_a_last_action(env: ManagerBasedEnv) -> torch.Tensor:
    """Return the frozen model_3396 14-D action history, not PPO's upper correction."""
    value = getattr(env, "legacy_stage_a_last_action", None)
    if value is None:
        return torch.zeros((env.num_envs, 14), device=env.device)
    return value


def joint_coordinator_last_action(env: ManagerBasedEnv) -> torch.Tensor:
    """Return the previous 22-D joint-coordinator correction action."""
    value = getattr(env, "joint_coordinator_last_action", None)
    if value is None:
        return torch.zeros((env.num_envs, 22), device=env.device)
    return value


def _observation_group_tensor(env: ManagerBasedEnv, name: str) -> torch.Tensor:
    """Read a private observation group without duplicating its term contract."""
    value = env.observation_manager.compute_group(name)
    if isinstance(value, tuple):
        value = value[0]
    if isinstance(value, dict):
        value = value.get(name, next(iter(value.values())))
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Observation group {name!r} did not return a tensor")
    return value


def joint_coordinator_observation(env: ManagerBasedEnv) -> torch.Tensor:
    """Combine the immutable Stage-A and model_900 contracts for joint PPO.

    The frozen actors keep consuming their own normalized 126-D and 56-D
    groups.  The new coordinator sees both unnormalized runtime contracts plus
    its own previous correction, so it can coordinate support, waist and arm
    corrections without altering either checkpoint's input semantics.
    """
    # ObservationManager probes term shapes while it is still constructing
    # itself. The private frozen-policy groups are unavailable at that point,
    # but their reviewed widths are immutable: 126 + 56 + 22.
    if not hasattr(env, "observation_manager"):
        return torch.zeros((env.num_envs, 204), device=env.device)
    return torch.cat(
        (
            _observation_group_tensor(env, "stage_a"),
            _observation_group_tensor(env, "upper"),
            joint_coordinator_last_action(env),
        ),
        dim=-1,
    )


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)
