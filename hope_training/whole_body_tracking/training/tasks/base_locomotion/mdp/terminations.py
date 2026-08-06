"""Fail-fast safety terminations for deterministic A3 Base Stand smoke."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase


def nonfinite_robot_state(env) -> torch.Tensor:
    asset: Articulation = env.scene["robot"]
    finite = (
        torch.isfinite(asset.data.root_state_w).all(dim=-1)
        & torch.isfinite(asset.data.joint_pos).all(dim=-1)
        & torch.isfinite(asset.data.joint_vel).all(dim=-1)
    )
    return ~finite


def torso_tilt_exceeded(env, torso_body_name: str, max_tilt_rad: float) -> torch.Tensor:
    asset: Articulation = env.scene["robot"]
    body_ids, body_names = asset.find_bodies([torso_body_name], preserve_order=True)
    if body_names != [torso_body_name]:
        raise ValueError(f"Could not resolve torso body {torso_body_name!r}")
    from isaaclab.utils.math import quat_rotate_inverse

    gravity_b = quat_rotate_inverse(asset.data.body_quat_w[:, body_ids[0]], asset.data.GRAVITY_VEC_W)
    # For a normalized gravity vector, -z is cos(tilt) when upright.
    cos_tilt = torch.clamp(-gravity_b[:, 2], min=-1.0, max=1.0)
    return torch.acos(cos_tilt) > max_tilt_rad


class HardJointPositionLimitExceeded(ManagerTermBase):
    """Terminate if any A3 joint leaves the physical limit plus numerical tolerance.

    Soft limits remain target clamps and reward boundaries.  Treating a soft
    boundary crossing as a hard reset caused healthy zero-action Stand states
    to reset before learning could correct a small servo transient.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.violation_mask = torch.zeros(
            (env.num_envs, env.scene["robot"].num_joints), dtype=torch.bool, device=env.device
        )
        self.excess_rad = torch.zeros_like(self.violation_mask, dtype=torch.float)
        # Diagnostic snapshots are captured before Isaac Lab auto-resets a
        # terminated environment.  They do not participate in the predicate.
        self.joint_position_rad = torch.zeros_like(self.excess_rad)
        self.joint_velocity_rad_s = torch.zeros_like(self.excess_rad)
        self.applied_torque_nm = torch.zeros_like(self.excess_rad)

    def __call__(self, env, tolerance_rad: float = 1.0e-4) -> torch.Tensor:
        asset: Articulation = env.scene["robot"]
        self.joint_position_rad[:] = asset.data.joint_pos
        self.joint_velocity_rad_s[:] = asset.data.joint_vel
        self.applied_torque_nm[:] = asset.data.applied_torque
        limits = asset.data.joint_pos_limits
        lower_excess = limits[..., 0] - asset.data.joint_pos
        upper_excess = asset.data.joint_pos - limits[..., 1]
        self.excess_rad[:] = torch.maximum(lower_excess, upper_excess)
        self.violation_mask[:] = self.excess_rad > tolerance_rad
        return torch.any(self.violation_mask, dim=-1)


class RootHeightBelowMinimum(ManagerTermBase):
    """Equivalent root-height predicate with a pre-auto-reset snapshot."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.height_m = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

    def __call__(self, env, minimum_height: float) -> torch.Tensor:
        asset: Articulation = env.scene["robot"]
        self.height_m[:] = asset.data.root_pos_w[:, 2]
        return self.height_m < minimum_height


class SustainedTorsoTiltExceeded(ManagerTermBase):
    """Terminate after a conservative tilt envelope is exceeded repeatedly."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.consecutive_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.consecutive_steps[:] = 0
        else:
            self.consecutive_steps[env_ids] = 0

    def __call__(
        self,
        env,
        torso_body_name: str,
        max_tilt_rad: float,
        required_steps: int,
    ) -> torch.Tensor:
        exceeded = torso_tilt_exceeded(env, torso_body_name, max_tilt_rad)
        self.consecutive_steps[:] = torch.where(
            exceeded, self.consecutive_steps + 1, torch.zeros_like(self.consecutive_steps)
        )
        return self.consecutive_steps >= required_steps
