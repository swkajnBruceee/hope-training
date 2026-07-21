"""Leg-only action for the isolated A3 locomotion replica."""

from __future__ import annotations

from dataclasses import MISSING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


class A3ReplicaLegPositionAction(ActionTerm):
    """Apply a bounded 12-DOF leg residual while holding every other joint.

    This intentionally does not use the Base/Strike Composer.  The complete
    target is rebuilt from the reset posture each policy step, so waist and arm
    joints have no learned ownership in this diagnostic task.
    """

    cfg: "A3ReplicaLegPositionActionCfg"

    def __init__(self, cfg: "A3ReplicaLegPositionActionCfg", env):
        super().__init__(cfg, env)
        if not isinstance(self._asset, Articulation):
            raise TypeError(f"A3 replica requires Articulation, got {type(self._asset).__name__}")
        self._joint_ids, names = self._asset.find_joints(list(cfg.joint_names), preserve_order=True)
        if names != list(cfg.joint_names):
            raise ValueError(f"Replica leg joint order mismatch: expected={list(cfg.joint_names)}, got={names}")
        if len(cfg.action_scale_rad) != len(self._joint_ids):
            raise ValueError("Replica action scale count must equal the leg joint count")
        self._joint_ids_tensor = torch.tensor(self._joint_ids, dtype=torch.long, device=self.device)
        self._scale = torch.tensor(cfg.action_scale_rad, dtype=torch.float, device=self.device).unsqueeze(0)
        self._raw_actions = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._full_joint_targets = self._asset.data.default_joint_pos.clone()

    @property
    def action_dim(self) -> int:
        return len(self.cfg.joint_names)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape != self._raw_actions.shape:
            raise ValueError(f"Expected replica action shape {self._raw_actions.shape}, got {actions.shape}")
        if not torch.isfinite(actions).all():
            raise ValueError("A3 replica action contains NaN or infinity")
        self._raw_actions[:] = torch.clamp(actions, -self.cfg.raw_clip, self.cfg.raw_clip)
        default = self._asset.data.default_joint_pos[:, self._joint_ids_tensor]
        self._processed_actions[:] = default + self._raw_actions * self._scale
        self._full_joint_targets[:] = self._asset.data.default_joint_pos
        self._full_joint_targets[:, self._joint_ids_tensor] = self._processed_actions
        if self.cfg.clip_to_soft_joint_limits:
            limits = self._asset.data.soft_joint_pos_limits
            self._full_joint_targets[:] = torch.clamp(
                self._full_joint_targets, min=limits[..., 0], max=limits[..., 1]
            )
            self._processed_actions[:] = self._full_joint_targets[:, self._joint_ids_tensor]

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self._full_joint_targets)


@configclass
class A3ReplicaLegPositionActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = A3ReplicaLegPositionAction
    joint_names: tuple[str, ...] = MISSING
    action_scale_rad: tuple[float, ...] = MISSING
    raw_clip: float = 0.25
    clip_to_soft_joint_limits: bool = True
