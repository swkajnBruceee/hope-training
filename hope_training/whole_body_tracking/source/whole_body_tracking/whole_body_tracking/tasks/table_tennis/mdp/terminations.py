"""Termination terms for the table-tennis environment (ball out of play, robot fallen)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from .racket import racket_normal_w, racket_state_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ball_out_of_bounds(
    env: "ManagerBasedRLEnv",
    bounds: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """True when the ball leaves the axis-aligned play volume (``bounds`` in the HOPE frame). Shape ``(N,)``."""
    ball: RigidObject = env.scene[asset_cfg.name]
    p = ball.data.root_pos_w - env.scene.env_origins  # HOPE-frame position
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    bx, by, bz = bounds["x"], bounds["y"], bounds["z"]
    return (
        (x < bx[0]) | (x > bx[1]) | (y < by[0]) | (y > by[1]) | (z < bz[0]) | (z > bz[1])
    )


def robot_base_too_low(
    env: "ManagerBasedRLEnv",
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """True when the robot base (pelvis) drops below ``minimum_height`` in the HOPE frame. Shape ``(N,)``."""
    robot: Articulation = env.scene[asset_cfg.name]
    z_hope = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return z_hope < minimum_height


def ball_touched_by_racket(
    env: "ManagerBasedRLEnv",
    distance_threshold: float,
    min_forward_velocity: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """True when the racket is near the ball and the ball has been redirected toward +X."""
    ball: RigidObject = env.scene[ball_cfg.name]
    racket_pos_w, _, _ = racket_state_w(env, robot_cfg)
    distance = torch.norm(ball.data.root_pos_w - racket_pos_w, dim=-1)
    return (distance < distance_threshold) & (ball.data.root_lin_vel_w[:, 0] > min_forward_velocity)


def ball_close_to_racket(
    env: "ManagerBasedRLEnv",
    distance_threshold: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """True when the racket center is close enough to the ball, independent of return direction."""
    ball: RigidObject = env.scene[ball_cfg.name]
    racket_pos_w, _, _ = racket_state_w(env, robot_cfg)
    distance = torch.norm(ball.data.root_pos_w - racket_pos_w, dim=-1)
    return distance < distance_threshold


def ball_close_to_racket_face(
    env: "ManagerBasedRLEnv",
    lateral_threshold: float,
    normal_threshold: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    normal_axis: int = 1,
    normal_sign: float = 1.0,
) -> torch.Tensor:
    """True when the ball is close to the racket face plane and within the blade target area."""
    ball: RigidObject = env.scene[ball_cfg.name]
    racket_pos_w, _, _ = racket_state_w(env, robot_cfg)
    normal_w = racket_normal_w(env, robot_cfg, normal_axis, normal_sign)
    rel = ball.data.root_pos_w - racket_pos_w
    signed_normal_dist = torch.sum(rel * normal_w, dim=-1)
    lateral = rel - signed_normal_dist.unsqueeze(-1) * normal_w
    lateral_dist = torch.norm(lateral, dim=-1)
    return (lateral_dist < lateral_threshold) & (torch.abs(signed_normal_dist) < normal_threshold)


def ball_contacted_by_racket(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("racket_ball_contact"),
    force_threshold: float = 0.05,
    lateral_threshold: float = 0.10,
    normal_threshold: float = 0.10,
    normal_axis: int = 1,
    normal_sign: float = 1.0,
) -> torch.Tensor:
    """True when the ball is under contact force while close to the racket face."""
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    force_contact = torch.linalg.norm(forces, dim=-1).amax(dim=-1) > force_threshold

    ball: RigidObject = env.scene["ball"]
    racket_pos_w, _, _ = racket_state_w(env)
    normal_w = racket_normal_w(env, normal_axis=normal_axis, normal_sign=normal_sign)
    rel = ball.data.root_pos_w - racket_pos_w
    signed_normal_dist = torch.sum(rel * normal_w, dim=-1)
    lateral = rel - signed_normal_dist.unsqueeze(-1) * normal_w
    lateral_dist = torch.norm(lateral, dim=-1)
    face_close = (lateral_dist < lateral_threshold) & (torch.abs(signed_normal_dist) < normal_threshold)
    return force_contact & face_close


def ball_contacted_by_racket_forward(
    env: "ManagerBasedRLEnv",
    min_forward_velocity: float,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("racket_ball_contact"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """True when racket-ball contact is detected and the ball is moving toward +X."""
    ball: RigidObject = env.scene[ball_cfg.name]
    return ball_contacted_by_racket(env, sensor_cfg) & (ball.data.root_lin_vel_w[:, 0] > min_forward_velocity)


def ball_returned_from_racket_face(
    env: "ManagerBasedRLEnv",
    min_forward_velocity: float = 0.2,
    lateral_threshold: float = 0.10,
    normal_threshold: float = 0.10,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    normal_axis: int = 1,
    normal_sign: float = 1.0,
) -> torch.Tensor:
    """True when the ball is near the racket face and has been redirected toward +X."""
    ball: RigidObject = env.scene[ball_cfg.name]
    racket_pos_w, _, _ = racket_state_w(env, robot_cfg)
    normal_w = racket_normal_w(env, robot_cfg, normal_axis, normal_sign)
    rel = ball.data.root_pos_w - racket_pos_w
    signed_normal_dist = torch.sum(rel * normal_w, dim=-1)
    lateral = rel - signed_normal_dist.unsqueeze(-1) * normal_w
    lateral_dist = torch.norm(lateral, dim=-1)
    face_close = (lateral_dist < lateral_threshold) & (torch.abs(signed_normal_dist) < normal_threshold)
    return face_close & (ball.data.root_lin_vel_w[:, 0] > min_forward_velocity)
