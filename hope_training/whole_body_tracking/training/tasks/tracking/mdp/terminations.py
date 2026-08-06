from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from training.tasks.tracking.mdp.commands import MotionCommand
from training.tasks.tracking.mdp.rewards import _get_body_indexes
from training.tasks.tracking.mdp.fall_state import reset_unified_fall_state, unified_fall_state


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


class SustainedRootTiltExceeded(ManagerTermBase):
    """Legacy root-only compatibility term; strict P5U uses unified state."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._env = env
        self.consecutive_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.consecutive_steps.zero_()
        else:
            self.consecutive_steps[env_ids] = 0

    def __call__(self, env: ManagerBasedRLEnv, max_tilt_rad: float, required_steps: int) -> torch.Tensor:
        gravity_b = env.scene["robot"].data.projected_gravity_b
        cos_tilt = torch.clamp(-gravity_b[:, 2], min=-1.0, max=1.0)
        tilt_rad = torch.acos(cos_tilt)
        exceeded = tilt_rad > max_tilt_rad
        self.consecutive_steps[:] = torch.where(
            exceeded, self.consecutive_steps + 1, torch.zeros_like(self.consecutive_steps)
        )
        return self.consecutive_steps >= int(required_steps)


class StrictRootFallExceeded(ManagerTermBase):
    """Expose the unified physical confirmed-fall state to termination manager."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._env = env
        self.consecutive_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.tilt_rad = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        self.root_height_m = torch.zeros_like(self.tilt_rad)
        self.torso_tilt_rad = torch.zeros_like(self.tilt_rad)
        self.torso_height_m = torch.zeros_like(self.tilt_rad)
        self.bad_state = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        robot = env.scene["robot"]
        try:
            torso_ids, torso_names = robot.find_bodies(["torso_Link"], preserve_order=True)
            self.torso_body_id = int(torso_ids[0]) if torso_names else 0
            self.has_torso_probe = bool(torso_names)
        except Exception:
            # Keep the root-only fallback for non-A3 tasks; P5U's contract
            # verifies the real torso link through the runtime audit.
            self.torso_body_id = 0
            self.has_torso_probe = False

    def reset(self, env_ids=None):
        if env_ids is None:
            self.consecutive_steps.zero_()
        else:
            self.consecutive_steps[env_ids] = 0
        reset_unified_fall_state(self._env, env_ids)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        max_tilt_rad: float = 0.785398,
        minimum_height: float = 0.82,
        max_torso_tilt_rad: float = 0.95,
        minimum_torso_height: float = 0.70,
        required_steps: int = 2,
    ) -> torch.Tensor:
        state = unified_fall_state(
            env,
            max_tilt_rad=max_tilt_rad,
            minimum_height=minimum_height,
            max_torso_tilt_rad=max_torso_tilt_rad,
            minimum_torso_height=minimum_torso_height,
            required_steps=required_steps,
        )
        self.tilt_rad[:] = torch.maximum(
            torch.abs(state.forward_tilt_rad), torch.abs(state.lateral_tilt_rad)
        )
        self.root_height_m[:] = state.relative_root_height_m
        self.torso_tilt_rad[:] = torch.maximum(
            torch.abs(state.torso_forward_tilt_rad), torch.abs(state.torso_lateral_tilt_rad)
        )
        self.torso_height_m[:] = state.relative_torso_height_m
        self.bad_state[:] = state.confirmed_fall
        self.consecutive_steps[:] = torch.where(
            state.confirmed_fall,
            self.consecutive_steps + 1,
            torch.zeros_like(self.consecutive_steps),
        )
        return state.confirmed_fall


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_rotate_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = math_utils.quat_rotate_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return torch.any(error > threshold, dim=-1)
