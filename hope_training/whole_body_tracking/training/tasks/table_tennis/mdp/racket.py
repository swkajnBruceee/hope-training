"""Racket-state helpers for table-tennis MDP terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_mul, quat_rotate_inverse

from training.robots.agibot_a3 import A3_MOUNT_OFFSET, A3_RACKET_BODY, A3_WRIST_BODY

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _racket_cache_key(
    robot_cfg: SceneEntityCfg,
    racket_body_name: str,
    wrist_body_name: str,
    mount_offset: tuple[float, float, float],
    mount_quat: tuple[float, float, float, float],
) -> str:
    return (
        f"_hope_racket_fk_{robot_cfg.name}_{racket_body_name}_{wrist_body_name}_"
        f"{mount_offset}_{mount_quat}"
    )


def _resolve_racket_fk(
    env: "ManagerBasedRLEnv",
    robot: Articulation,
    robot_cfg: SceneEntityCfg,
    racket_body_name: str,
    wrist_body_name: str,
    mount_offset: tuple[float, float, float],
    mount_quat: tuple[float, float, float, float],
):
    key = _racket_cache_key(robot_cfg, racket_body_name, wrist_body_name, mount_offset, mount_quat)
    cached = getattr(env, key, None)
    if cached is not None:
        return cached

    if racket_body_name in robot.body_names:
        mode = "body"
        body_index = robot.find_bodies(racket_body_name, preserve_order=True)[0][0]
        wrist_index = -1
    else:
        mode = "wrist_offset"
        body_index = -1
        if wrist_body_name not in robot.body_names:
            raise RuntimeError(
                f"Could not resolve racket FK: neither '{racket_body_name}' nor '{wrist_body_name}' "
                f"exists on robot '{robot_cfg.name}'."
            )
        wrist_index = robot.find_bodies(wrist_body_name, preserve_order=True)[0][0]

    state = {
        "mode": mode,
        "body_index": body_index,
        "wrist_index": wrist_index,
        "mount_offset": torch.tensor(mount_offset, dtype=torch.float32, device=env.device).repeat(env.num_envs, 1),
        "mount_quat": torch.tensor(mount_quat, dtype=torch.float32, device=env.device).repeat(env.num_envs, 1),
    }
    setattr(env, key, state)
    return state


def racket_state_w(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    racket_body_name: str = A3_RACKET_BODY,
    wrist_body_name: str = A3_WRIST_BODY,
    mount_offset: tuple[float, float, float] = A3_MOUNT_OFFSET,
    mount_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return racket center position, linear velocity, and orientation in the world frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    fk = _resolve_racket_fk(env, robot, robot_cfg, racket_body_name, wrist_body_name, mount_offset, mount_quat)
    data = robot.data

    if fk["mode"] == "body":
        idx = fk["body_index"]
        return data.body_pos_w[:, idx], data.body_lin_vel_w[:, idx], data.body_quat_w[:, idx]

    widx = fk["wrist_index"]
    wpos = data.body_pos_w[:, widx]
    wquat = data.body_quat_w[:, widx]
    wlin = data.body_lin_vel_w[:, widx]
    wang = data.body_ang_vel_w[:, widx]
    offset_w = quat_apply(wquat, fk["mount_offset"])
    return wpos + offset_w, wlin + torch.cross(wang, offset_w, dim=-1), quat_mul(wquat, fk["mount_quat"])


def racket_position_b(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Racket center position relative to the robot base, expressed in the robot base frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    racket_pos_w, _, _ = racket_state_w(env, robot_cfg)
    return quat_rotate_inverse(robot.data.root_quat_w, racket_pos_w - robot.data.root_pos_w)


def racket_velocity_b(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Racket center linear velocity expressed in the robot base frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    _, racket_vel_w, _ = racket_state_w(env, robot_cfg)
    return quat_rotate_inverse(robot.data.root_quat_w, racket_vel_w)


def racket_normal_w(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    normal_axis: int = 1,
    normal_sign: float = 1.0,
) -> torch.Tensor:
    """Racket face normal in world frame.

    For the A3 ping-pong asset the blade is thin along local Y; +Y corresponds to the red face.
    """
    _, _, racket_quat_w = racket_state_w(env, robot_cfg)
    return matrix_from_quat(racket_quat_w)[:, :, normal_axis] * normal_sign


def racket_normal_b(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    normal_axis: int = 1,
    normal_sign: float = 1.0,
) -> torch.Tensor:
    """Racket face normal expressed in the robot base frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    normal_w = racket_normal_w(env, robot_cfg, normal_axis, normal_sign)
    return quat_rotate_inverse(robot.data.root_quat_w, normal_w)
