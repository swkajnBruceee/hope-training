"""Reset-time disturbance events for A3 Stand Recovery curricula."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp import reset_scene_to_default


def reset_scene_with_recovery_a_disturbance(
    env,
    env_ids: torch.Tensor,
    undisturbed_fraction: float,
    roll_pitch_range_rad: tuple[float, float],
    angular_velocity_range_rad_s: tuple[float, float],
    medium_fraction: float = 0.0,
    medium_roll_pitch_range_rad: tuple[float, float] | None = None,
    medium_angular_velocity_range_rad_s: tuple[float, float] | None = None,
):
    """Reset normally, then perturb only root roll/pitch and angular velocity.

    The disturbance mask is deliberately hidden from actor observations.  It
    is retained on the environment only for reward masking and audit metrics.
    """
    if not 0.0 <= undisturbed_fraction <= 1.0:
        raise ValueError("undisturbed_fraction must be inside [0, 1]")
    if not 0.0 <= medium_fraction <= 1.0:
        raise ValueError("medium_fraction must be inside [0, 1]")
    if medium_fraction and (medium_roll_pitch_range_rad is None or medium_angular_velocity_range_rad_s is None):
        raise ValueError("medium ranges are required when medium_fraction is non-zero")
    reset_scene_to_default(env, env_ids)
    robot = env.scene["robot"]
    count = len(env_ids)
    disturbed = torch.rand(count, device=env.device) >= undisturbed_fraction
    medium = disturbed & (torch.rand(count, device=env.device) < medium_fraction)

    pose_samples = torch.zeros((count, 2), device=env.device)
    velocity_samples = torch.zeros((count, 2), device=env.device)
    if disturbed.any():
        disturbed_count = int(disturbed.sum().item())
        pose_samples[disturbed] = math_utils.sample_uniform(
            roll_pitch_range_rad[0],
            roll_pitch_range_rad[1],
            (disturbed_count, 2),
            env.device,
        )
        velocity_samples[disturbed] = math_utils.sample_uniform(
            angular_velocity_range_rad_s[0],
            angular_velocity_range_rad_s[1],
            (disturbed_count, 2),
            env.device,
        )
        if medium.any():
            medium_count = int(medium.sum().item())
            pose_samples[medium] = math_utils.sample_uniform(
                medium_roll_pitch_range_rad[0],
                medium_roll_pitch_range_rad[1],
                (medium_count, 2),
                env.device,
            )
            velocity_samples[medium] = math_utils.sample_uniform(
                medium_angular_velocity_range_rad_s[0],
                medium_angular_velocity_range_rad_s[1],
                (medium_count, 2),
                env.device,
            )

    root_state = robot.data.default_root_state[env_ids].clone()
    root_state[:, :3] += env.scene.env_origins[env_ids]
    orientation_delta = math_utils.quat_from_euler_xyz(
        pose_samples[:, 0], pose_samples[:, 1], torch.zeros(count, device=env.device)
    )
    root_state[:, 3:7] = math_utils.quat_mul(root_state[:, 3:7], orientation_delta)
    root_state[:, 10] += velocity_samples[:, 0]
    root_state[:, 11] += velocity_samples[:, 1]
    robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids)

    if not hasattr(env, "recovery_disturbed_mask"):
        env.recovery_disturbed_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env.recovery_initial_roll_pitch_rad = torch.zeros((env.num_envs, 2), device=env.device)
        env.recovery_initial_angular_velocity_rad_s = torch.zeros((env.num_envs, 2), device=env.device)
    env.recovery_disturbed_mask[env_ids] = disturbed
    env.recovery_initial_roll_pitch_rad[env_ids] = pose_samples
    env.recovery_initial_angular_velocity_rad_s[env_ids] = velocity_samples
