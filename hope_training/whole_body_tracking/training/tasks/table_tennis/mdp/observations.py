"""Observation terms for the table-tennis environment (ball state in the robot base frame)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_rotate_inverse

from .racket import racket_normal_b, racket_position_b, racket_state_w, racket_velocity_b

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ball_position_b(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Ball position relative to the robot base, expressed in the robot base frame. Shape ``(N, 3)``."""
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    rel_w = ball.data.root_pos_w - robot.data.root_pos_w
    return quat_rotate_inverse(robot.data.root_quat_w, rel_w)


def ball_velocity_b(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Ball linear velocity expressed in the robot base frame. Shape ``(N, 3)``."""
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    return quat_rotate_inverse(robot.data.root_quat_w, ball.data.root_lin_vel_w)


def racket_to_ball_b(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Vector from racket center to ball, expressed in the robot base frame. Shape ``(N, 3)``."""
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    racket_pos_w, _, _ = racket_state_w(env, robot_cfg)
    return quat_rotate_inverse(robot.data.root_quat_w, ball.data.root_pos_w - racket_pos_w)


def predicted_hit_state_w(
    env: "ManagerBasedRLEnv",
    hit_x: float = -0.10,
    min_time: float = 0.0,
    max_time: float = 0.45,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict where the incoming ball crosses a fixed P1-side hit plane.

    The prediction is intentionally simple and local: constant velocity over a short time window. It is
    used as a curriculum signal for right-arm reaching, not as a long-horizon ballistic planner.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    env_origins = env.scene.env_origins
    pos_l = ball.data.root_pos_w - env_origins
    vel_w = ball.data.root_lin_vel_w
    approaching = vel_w[:, 0] < -1.0e-4
    t = (pos_l[:, 0] - hit_x) / torch.clamp(-vel_w[:, 0], min=1.0e-4)
    valid = approaching & (t > min_time) & (t < max_time)
    t_clamped = torch.clamp(t, min=min_time, max=max_time)
    pred_l = pos_l + vel_w * t_clamped.unsqueeze(-1)
    pred_w = pred_l + env_origins
    return pred_w, t_clamped, valid


def predicted_hit_position_b(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    hit_x: float = -0.10,
    max_time: float = 0.45,
) -> torch.Tensor:
    """Predicted ball position at the hit plane, relative to the robot base frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    pred_w, _, _ = predicted_hit_state_w(env, hit_x=hit_x, max_time=max_time, ball_cfg=ball_cfg)
    return quat_rotate_inverse(robot.data.root_quat_w, pred_w - robot.data.root_pos_w)


def time_to_hit(
    env: "ManagerBasedRLEnv",
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    hit_x: float = -0.10,
    max_time: float = 0.45,
) -> torch.Tensor:
    """Clamped time until the incoming ball crosses the nominal hit plane."""
    _, t, valid = predicted_hit_state_w(env, hit_x=hit_x, max_time=max_time, ball_cfg=ball_cfg)
    return torch.where(valid, t, torch.full_like(t, max_time)).unsqueeze(-1)
