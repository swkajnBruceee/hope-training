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
