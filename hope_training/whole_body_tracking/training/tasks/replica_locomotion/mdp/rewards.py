"""H1/G1 velocity-MDP reward terms retained locally for A3 comparison."""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat


def track_lin_vel_xy_yaw_frame_exp(env, command_name: str, std: float) -> torch.Tensor:
    """H1/G1 exponential xy velocity tracking in the gravity-aligned yaw frame."""
    asset = env.scene["robot"]
    velocity = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    error = torch.sum(torch.square(env.command_manager.get_command(command_name)[:, :2] - velocity[:, :2]), dim=1)
    return torch.exp(-error / std**2)


def track_ang_vel_z_world_exp(env, command_name: str, std: float) -> torch.Tensor:
    """H1/G1 exponential world-yaw velocity tracking."""
    asset = env.scene["robot"]
    error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-error / std**2)


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """H1/G1 contact-gated foot sliding penalty."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    foot_velocity = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(foot_velocity.norm(dim=-1) * contacts, dim=1)


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """H1/G1 alternating-single-support reward, gated off for a stand command."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward
