"""Reward terms for the table-tennis environment.

Only a small example ball-aware term lives here; the generic robot rewards (alive, action-rate, ...) are
reused from ``isaaclab.envs.mdp``. Add real match objectives (return success, ball-over-net, landing in
the opponent half, racket-to-ball tracking) here as the policy is developed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from .observations import predicted_hit_state_w
from .racket import racket_normal_w, racket_state_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ball_above_surface(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """1.0 while the ball is above the table surface (HOPE z > 0), else 0.0. Shape ``(N,)``.

    A placeholder "ball in play" signal demonstrating how to read ball state in the HOPE frame
    (subtract the per-environment origin) for reward shaping."""
    ball: RigidObject = env.scene[asset_cfg.name]
    z_hope = ball.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (z_hope > 0.0).float()


def racket_ball_proximity_exp(
    env: "ManagerBasedRLEnv",
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Exponential reward for bringing the racket center close to the ball."""
    ball: RigidObject = env.scene[ball_cfg.name]
    racket_pos_w, _, _ = racket_state_w(env, robot_cfg)
    distance_sq = torch.sum(torch.square(ball.data.root_pos_w - racket_pos_w), dim=-1)
    return torch.exp(-distance_sq / std**2)


def racket_closing_speed(
    env: "ManagerBasedRLEnv",
    max_speed: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Reward racket motion that closes the distance to the incoming ball."""
    ball: RigidObject = env.scene[ball_cfg.name]
    racket_pos_w, racket_vel_w, _ = racket_state_w(env, robot_cfg)
    rel_pos = ball.data.root_pos_w - racket_pos_w
    rel_dir = rel_pos / (torch.norm(rel_pos, dim=-1, keepdim=True) + 1.0e-6)
    closing = torch.sum((racket_vel_w - ball.data.root_lin_vel_w) * rel_dir, dim=-1)
    return torch.clamp(closing / max_speed, min=0.0, max=1.0)


def touch_ball_forward(
    env: "ManagerBasedRLEnv",
    distance_threshold: float,
    min_forward_velocity: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Sparse reward when the racket is near the ball and the ball leaves toward +X."""
    ball: RigidObject = env.scene[ball_cfg.name]
    racket_pos_w, _, _ = racket_state_w(env, robot_cfg)
    distance = torch.norm(ball.data.root_pos_w - racket_pos_w, dim=-1)
    close = distance < distance_threshold
    returned = ball.data.root_lin_vel_w[:, 0] > min_forward_velocity
    return (close & returned).float()


def racket_ball_close(
    env: "ManagerBasedRLEnv",
    distance_threshold: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Sparse reward when the racket center gets within a distance threshold of the ball."""
    ball: RigidObject = env.scene[ball_cfg.name]
    racket_pos_w, _, _ = racket_state_w(env, robot_cfg)
    distance = torch.norm(ball.data.root_pos_w - racket_pos_w, dim=-1)
    return (distance < distance_threshold).float()


def racket_face_alignment(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    normal_axis: int = 1,
    normal_sign: float = 1.0,
) -> torch.Tensor:
    """Reward the racket face normal for opposing the incoming ball velocity."""
    ball: RigidObject = env.scene[ball_cfg.name]
    normal_w = racket_normal_w(env, robot_cfg, normal_axis, normal_sign)
    incoming_dir = -ball.data.root_lin_vel_w / (torch.norm(ball.data.root_lin_vel_w, dim=-1, keepdim=True) + 1.0e-6)
    return torch.clamp(torch.sum(normal_w * incoming_dir, dim=-1), min=0.0, max=1.0)


def racket_ball_plane_alignment_exp(
    env: "ManagerBasedRLEnv",
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    normal_axis: int = 1,
    normal_sign: float = 1.0,
) -> torch.Tensor:
    """Reward lateral alignment of the ball with the racket face plane.

    This ignores distance along the face normal and focuses on whether the ball is aimed at the blade,
    which gives a cleaner learning signal than only minimizing center-to-center distance.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    racket_pos_w, _, _ = racket_state_w(env, robot_cfg)
    normal_w = racket_normal_w(env, robot_cfg, normal_axis, normal_sign)
    rel = ball.data.root_pos_w - racket_pos_w
    signed_normal_dist = torch.sum(rel * normal_w, dim=-1, keepdim=True)
    lateral = rel - signed_normal_dist * normal_w
    lateral_dist_sq = torch.sum(torch.square(lateral), dim=-1)
    return torch.exp(-lateral_dist_sq / std**2)


def racket_ball_face_close(
    env: "ManagerBasedRLEnv",
    lateral_threshold: float,
    normal_threshold: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    normal_axis: int = 1,
    normal_sign: float = 1.0,
) -> torch.Tensor:
    """Sparse reward when the ball is close to the racket face plane and inside the blade target area."""
    ball: RigidObject = env.scene[ball_cfg.name]
    racket_pos_w, _, _ = racket_state_w(env, robot_cfg)
    normal_w = racket_normal_w(env, robot_cfg, normal_axis, normal_sign)
    rel = ball.data.root_pos_w - racket_pos_w
    signed_normal_dist = torch.sum(rel * normal_w, dim=-1)
    lateral = rel - signed_normal_dist.unsqueeze(-1) * normal_w
    lateral_dist = torch.norm(lateral, dim=-1)
    return ((lateral_dist < lateral_threshold) & (torch.abs(signed_normal_dist) < normal_threshold)).float()


def racket_ball_face_contact_exp(
    env: "ManagerBasedRLEnv",
    lateral_std: float,
    normal_std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    normal_axis: int = 1,
    normal_sign: float = 1.0,
) -> torch.Tensor:
    """Dense reward for simultaneous lateral and normal closeness to the racket face."""
    ball: RigidObject = env.scene[ball_cfg.name]
    racket_pos_w, _, _ = racket_state_w(env, robot_cfg)
    normal_w = racket_normal_w(env, robot_cfg, normal_axis, normal_sign)
    rel = ball.data.root_pos_w - racket_pos_w
    signed_normal_dist = torch.sum(rel * normal_w, dim=-1)
    lateral = rel - signed_normal_dist.unsqueeze(-1) * normal_w
    lateral_dist_sq = torch.sum(torch.square(lateral), dim=-1)
    normal_dist_sq = torch.square(signed_normal_dist)
    return torch.exp(-(lateral_dist_sq / lateral_std**2 + normal_dist_sq / normal_std**2))


def _racket_face_close_mask(
    env: "ManagerBasedRLEnv",
    lateral_threshold: float,
    normal_threshold: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    normal_axis: int = 1,
    normal_sign: float = 1.0,
) -> torch.Tensor:
    ball: RigidObject = env.scene[ball_cfg.name]
    racket_pos_w, _, _ = racket_state_w(env, robot_cfg)
    normal_w = racket_normal_w(env, robot_cfg, normal_axis, normal_sign)
    rel = ball.data.root_pos_w - racket_pos_w
    signed_normal_dist = torch.sum(rel * normal_w, dim=-1)
    lateral = rel - signed_normal_dist.unsqueeze(-1) * normal_w
    lateral_dist = torch.norm(lateral, dim=-1)
    return (lateral_dist < lateral_threshold) & (torch.abs(signed_normal_dist) < normal_threshold)


def racket_predicted_hit_position_exp(
    env: "ManagerBasedRLEnv",
    std: float,
    hit_x: float = -0.10,
    min_height: float = 0.12,
    max_height: float = 0.40,
    max_time: float = 0.45,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Reward moving the racket toward the ball's short-horizon predicted hit-plane crossing."""
    pred_w, _, valid = predicted_hit_state_w(env, hit_x=hit_x, max_time=max_time, ball_cfg=ball_cfg)
    pred_l = pred_w - env.scene.env_origins
    valid = valid & (pred_l[:, 2] > min_height) & (pred_l[:, 2] < max_height)
    racket_pos_w, _, _ = racket_state_w(env, robot_cfg)
    error_sq = torch.sum(torch.square(racket_pos_w - pred_w), dim=-1)
    return torch.where(valid, torch.exp(-error_sq / std**2), torch.zeros_like(error_sq))


def racket_predicted_hit_face_exp(
    env: "ManagerBasedRLEnv",
    lateral_std: float,
    normal_std: float,
    target_normal_dist: float = 0.025,
    hit_x: float = -0.10,
    min_height: float = 0.12,
    max_height: float = 0.40,
    max_time: float = 0.45,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    normal_axis: int = 1,
    normal_sign: float = 1.0,
) -> torch.Tensor:
    """Reward placing the racket face, not only its center, at the predicted hit point.

    The ball center should be slightly in front of the blade plane at impact. A small positive
    ``target_normal_dist`` gives the policy a physical target instead of asking it to put the ball
    exactly on the mathematical racket plane.
    """
    pred_w, _, valid = predicted_hit_state_w(env, hit_x=hit_x, max_time=max_time, ball_cfg=ball_cfg)
    pred_l = pred_w - env.scene.env_origins
    valid = valid & (pred_l[:, 2] > min_height) & (pred_l[:, 2] < max_height)

    racket_pos_w, _, _ = racket_state_w(env, robot_cfg)
    normal_w = racket_normal_w(env, robot_cfg, normal_axis, normal_sign)
    rel = pred_w - racket_pos_w
    signed_normal_dist = torch.sum(rel * normal_w, dim=-1)
    lateral = rel - signed_normal_dist.unsqueeze(-1) * normal_w
    lateral_dist_sq = torch.sum(torch.square(lateral), dim=-1)
    normal_dist_sq = torch.square(signed_normal_dist - target_normal_dist)
    reward = torch.exp(-(lateral_dist_sq / lateral_std**2 + normal_dist_sq / normal_std**2))
    return torch.where(valid, reward, torch.zeros_like(reward))


def racket_predicted_hit_lateral_exp(
    env: "ManagerBasedRLEnv",
    std: float,
    hit_x: float = -0.10,
    min_height: float = 0.12,
    max_height: float = 0.40,
    max_time: float = 0.45,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    normal_axis: int = 1,
    normal_sign: float = 1.0,
) -> torch.Tensor:
    """Reward putting the predicted hit point inside the racket face area, ignoring normal depth."""
    pred_w, _, valid = predicted_hit_state_w(env, hit_x=hit_x, max_time=max_time, ball_cfg=ball_cfg)
    pred_l = pred_w - env.scene.env_origins
    valid = valid & (pred_l[:, 2] > min_height) & (pred_l[:, 2] < max_height)

    racket_pos_w, _, _ = racket_state_w(env, robot_cfg)
    normal_w = racket_normal_w(env, robot_cfg, normal_axis, normal_sign)
    rel = pred_w - racket_pos_w
    signed_normal_dist = torch.sum(rel * normal_w, dim=-1, keepdim=True)
    lateral = rel - signed_normal_dist * normal_w
    lateral_dist_sq = torch.sum(torch.square(lateral), dim=-1)
    reward = torch.exp(-lateral_dist_sq / std**2)
    return torch.where(valid, reward, torch.zeros_like(reward))


def racket_forward_swing(
    env: "ManagerBasedRLEnv",
    max_speed: float,
    hit_x: float = -0.10,
    max_time: float = 0.45,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Reward a forward racket swing near the predicted contact window."""
    _, _, valid = predicted_hit_state_w(env, hit_x=hit_x, max_time=max_time, ball_cfg=ball_cfg)
    _, racket_vel_w, _ = racket_state_w(env, robot_cfg)
    forward = torch.clamp(racket_vel_w[:, 0] / max_speed, min=0.0, max=1.0)
    return torch.where(valid, forward, torch.zeros_like(forward))


def racket_timed_forward_swing(
    env: "ManagerBasedRLEnv",
    max_speed: float,
    hit_x: float = -0.10,
    min_time: float = 0.02,
    max_time: float = 0.18,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Reward forward racket velocity only in the short contact window before interception."""
    _, time_to_hit, valid = predicted_hit_state_w(env, hit_x=hit_x, max_time=max_time, ball_cfg=ball_cfg)
    in_window = valid & (time_to_hit > min_time) & (time_to_hit < max_time)
    _, racket_vel_w, _ = racket_state_w(env, robot_cfg)
    forward = torch.clamp(racket_vel_w[:, 0] / max_speed, min=0.0, max=1.0)
    return torch.where(in_window, forward, torch.zeros_like(forward))


def racket_ball_contact_force(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("racket_ball_contact"),
) -> torch.Tensor:
    """Ball contact force norm.

    The ball sensor is intentionally unfiltered because GPU PhysX does not support filtered contact
    reporting for the nested racket collider. Racket contact terms combine this force with a face-close
    geometric gate to reject table/floor contacts.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    return torch.linalg.norm(forces, dim=-1).amax(dim=-1)


def racket_ball_first_contact(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("racket_ball_contact"),
    force_threshold: float = 0.05,
    lateral_threshold: float = 0.10,
    normal_threshold: float = 0.10,
    normal_axis: int = 1,
    normal_sign: float = 1.0,
) -> torch.Tensor:
    """Sparse reward on racket-ball contact, gated by force and racket-face geometry."""
    force_contact = racket_ball_contact_force(env, sensor_cfg) > force_threshold
    face_close = _racket_face_close_mask(
        env,
        lateral_threshold=lateral_threshold,
        normal_threshold=normal_threshold,
        normal_axis=normal_axis,
        normal_sign=normal_sign,
    )
    return (force_contact & face_close).float()


def contact_ball_forward(
    env: "ManagerBasedRLEnv",
    min_forward_velocity: float,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("racket_ball_contact"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Reward contact that redirects the ball toward +X."""
    ball: RigidObject = env.scene[ball_cfg.name]
    first_contact = racket_ball_first_contact(env, sensor_cfg) > 0.0
    returned = ball.data.root_lin_vel_w[:, 0] > min_forward_velocity
    return (first_contact & returned).float()


def racket_face_ball_forward(
    env: "ManagerBasedRLEnv",
    min_forward_velocity: float,
    lateral_threshold: float = 0.10,
    normal_threshold: float = 0.10,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    normal_axis: int = 1,
    normal_sign: float = 1.0,
) -> torch.Tensor:
    """Reward a return-like touch: ball near the racket face and moving back toward +X.

    This is deliberately kinematic. The Isaac contact sensor is useful as a diagnostic, but with the
    converted A3 paddle mesh it can miss the exact paddle-ball frame. A face-gated velocity reversal
    is the robust success signal for this curriculum stage.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    face_close = _racket_face_close_mask(
        env,
        lateral_threshold=lateral_threshold,
        normal_threshold=normal_threshold,
        normal_axis=normal_axis,
        normal_sign=normal_sign,
    )
    returned = ball.data.root_lin_vel_w[:, 0] > min_forward_velocity
    return (face_close & returned).float()
