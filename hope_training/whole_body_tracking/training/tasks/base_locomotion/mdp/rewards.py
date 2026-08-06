"""Small, auditable Stand-v0 reward terms."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg


def joint_posture_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    normalize_by_dof: bool = False,
) -> torch.Tensor:
    """Squared deviation from the deterministic A3 reset posture."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    squared = torch.square(joint_pos_rel)
    return torch.mean(squared, dim=-1) if normalize_by_dof else torch.sum(squared, dim=-1)


def torso_upright_l2(env, torso_body_name: str) -> torch.Tensor:
    """Penalize torso roll/pitch through its projected gravity x/y components."""
    asset: Articulation = env.scene["robot"]
    body_ids, body_names = asset.find_bodies([torso_body_name], preserve_order=True)
    if body_names != [torso_body_name]:
        raise ValueError(f"Could not resolve torso body {torso_body_name!r}")
    from isaaclab.utils.math import quat_rotate_inverse

    gravity_b = quat_rotate_inverse(asset.data.body_quat_w[:, body_ids[0]], asset.data.GRAVITY_VEC_W)
    return torch.sum(torch.square(gravity_b[:, :2]), dim=-1)


def undisturbed_action_l2(env) -> torch.Tensor:
    """Penalize pre-clip recovery action only on the do-no-harm slice."""
    disturbed = getattr(env, "recovery_disturbed_mask", None)
    if disturbed is None:
        raise RuntimeError("Recovery disturbance mask is unavailable")
    action_l2 = torch.sum(torch.square(env.action_manager.action), dim=-1)
    return action_l2 * (~disturbed).to(action_l2.dtype)


def raw_action_excess_l2(env, raw_limit: float = 0.125) -> torch.Tensor:
    """Penalize actor output outside the v2 raw residual band before execution clipping."""
    raw = env.action_manager.action
    excess = torch.relu(torch.abs(raw) - raw_limit)
    return torch.sum(torch.square(excess), dim=-1)


def physical_residual_l2(env) -> torch.Tensor:
    """Penalize the actual per-joint position residual after action scaling."""
    action_term = env.action_manager.get_term("base")
    residual = action_term.raw_actions * action_term._scale
    return torch.sum(torch.square(residual), dim=-1)


def healthy_action_l2(
    env,
    tilt_scale_rad: float = 0.05,
    angular_velocity_scale_rad_s: float = 0.20,
    height_scale_m: float = 0.02,
) -> torch.Tensor:
    """Penalize residuals continuously as observable state approaches healthy standing."""
    robot = env.scene["robot"]
    gravity = robot.data.projected_gravity_b[:, :2]
    tilt_error = torch.sum(torch.square(gravity / tilt_scale_rad), dim=-1)
    angular_error = torch.sum(
        torch.square(robot.data.root_ang_vel_b[:, :2] / angular_velocity_scale_rad_s), dim=-1
    )
    height_error = torch.square(
        (robot.data.root_pos_w[:, 2] - robot.data.default_root_state[:, 2]) / height_scale_m
    )
    health_weight = torch.exp(-0.5 * (tilt_error + angular_error + height_error))
    return torch.sum(torch.square(env.action_manager.action), dim=-1) * health_weight


class RecoveryTiltProgress(ManagerTermBase):
    """One-step reduction in root roll/pitch error with reset-safe state."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._previous_error = torch.zeros(env.num_envs, device=env.device)
        self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self._initialized[:] = False
        else:
            self._initialized[env_ids] = False

    def __call__(self, env) -> torch.Tensor:
        gravity = env.scene["robot"].data.projected_gravity_b
        error = torch.sum(torch.square(gravity[:, :2]), dim=-1)
        # RewardManager multiplies weighted terms by policy_dt.  Return a
        # rate here so the integrated shaping reward equals the actual
        # decrease in squared tilt error instead of being scaled by dt twice.
        progress = (self._previous_error - error) / float(env.step_dt)
        progress = torch.where(self._initialized, progress, torch.zeros_like(progress))
        self._previous_error[:] = error
        self._initialized[:] = True
        disturbed = getattr(env, "recovery_disturbed_mask", None)
        if disturbed is None:
            raise RuntimeError("Recovery disturbance mask is unavailable")
        return progress * disturbed.to(progress.dtype)


class RecoveryPotentialProgress(ManagerTermBase):
    """Progress in observable tilt, angular-velocity, and height potential."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._previous = torch.zeros(env.num_envs, device=env.device)
        self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self._initialized[:] = False
        else:
            self._initialized[env_ids] = False

    def __call__(
        self,
        env,
        tilt_scale_rad: float = 0.05,
        angular_velocity_scale_rad_s: float = 0.20,
        height_scale_m: float = 0.02,
    ) -> torch.Tensor:
        robot = env.scene["robot"]
        gravity = robot.data.projected_gravity_b[:, :2]
        potential = torch.sum(torch.square(gravity / tilt_scale_rad), dim=-1)
        potential += torch.sum(
            torch.square(robot.data.root_ang_vel_b[:, :2] / angular_velocity_scale_rad_s), dim=-1
        )
        potential += torch.square(
            (robot.data.root_pos_w[:, 2] - robot.data.default_root_state[:, 2]) / height_scale_m
        )
        progress = (self._previous - potential) / float(env.step_dt)
        progress = torch.where(self._initialized, progress, torch.zeros_like(progress))
        self._previous[:] = potential
        self._initialized[:] = True
        return progress
