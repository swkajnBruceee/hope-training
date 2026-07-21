"""Deployable A3 Base action semantics for the bounded Stand smoke task."""

from __future__ import annotations

from dataclasses import MISSING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


class A3BaseCompositePositionAction(ActionTerm):
    """Compose a bounded 14-DOF Base residual into one full articulation target.

    The residual is non-integrating and the resulting target is held for every
    physics substep.  In Stand v0 there is no Strike reference: all non-Base
    joints, including waist yaw and both arms, remain at their reset defaults.
    """

    cfg: "A3BaseCompositePositionActionCfg"

    def __init__(self, cfg: "A3BaseCompositePositionActionCfg", env):
        super().__init__(cfg, env)
        if not isinstance(self._asset, Articulation):
            raise TypeError(f"A3 Base action requires Articulation, got {type(self._asset).__name__}")

        self._base_joint_ids, resolved_base_names = self._asset.find_joints(
            list(cfg.base_joint_names), preserve_order=True
        )
        if resolved_base_names != list(cfg.base_joint_names):
            raise ValueError(
                "A3 Base action joint order mismatch: "
                f"expected={list(cfg.base_joint_names)}, resolved={resolved_base_names}"
            )
        self._backend_joint_ids, resolved_backend_names = self._asset.find_joints(
            list(cfg.backend_joint_names), preserve_order=True
        )
        if resolved_backend_names != list(cfg.backend_joint_names):
            raise ValueError(
                "A3 backend joint order mismatch: "
                f"expected={list(cfg.backend_joint_names)}, resolved={resolved_backend_names}"
            )
        if len(self._backend_joint_ids) != self._asset.num_joints:
            raise ValueError(
                f"A3 backend contract has {len(self._backend_joint_ids)} joints, "
                f"articulation has {self._asset.num_joints}"
            )
        if len(cfg.action_scale_rad) != len(self._base_joint_ids):
            raise ValueError(
                f"Expected {len(self._base_joint_ids)} action scales, got {len(cfg.action_scale_rad)}"
            )

        self._base_joint_ids_tensor = torch.tensor(self._base_joint_ids, dtype=torch.long, device=self.device)
        self._backend_joint_ids_tensor = torch.tensor(self._backend_joint_ids, dtype=torch.long, device=self.device)
        self._scale = torch.tensor(cfg.action_scale_rad, dtype=torch.float, device=self.device).unsqueeze(0)
        self._mask = torch.tensor(cfg.action_mask, dtype=torch.float, device=self.device).unsqueeze(0)
        self._raw_actions = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._full_joint_targets = self._asset.data.default_joint_pos.clone()

    @property
    def action_dim(self) -> int:
        return len(self.cfg.base_joint_names)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """The 14 absolute joint targets in Base action order."""
        return self._processed_actions

    @property
    def full_joint_targets(self) -> torch.Tensor:
        """The composed command in native articulation order (diagnostic only)."""
        return self._full_joint_targets

    def process_actions(self, actions: torch.Tensor):
        if actions.shape != self._raw_actions.shape:
            raise ValueError(f"Expected Base action shape {self._raw_actions.shape}, got {actions.shape}")
        if not torch.isfinite(actions).all():
            raise ValueError("A3 Base action contains NaN or infinity")

        masked = actions * self._mask
        self._raw_actions[:] = torch.clamp(masked, -self.cfg.raw_clip, self.cfg.raw_clip)
        default_base = self._asset.data.default_joint_pos[:, self._base_joint_ids_tensor]
        self._processed_actions[:] = default_base + self._raw_actions * self._scale

        # Rebuild from the immutable reset/default posture on every policy step.
        # This is intentionally non-integrating and supplies exactly one target
        # source for all 31 joints.
        self._full_joint_targets[:] = self._asset.data.default_joint_pos
        self._full_joint_targets[:, self._base_joint_ids_tensor] = self._processed_actions

        if self.cfg.clip_to_soft_joint_limits:
            limits = self._asset.data.soft_joint_pos_limits
            self._full_joint_targets[:] = torch.clamp(
                self._full_joint_targets, min=limits[..., 0], max=limits[..., 1]
            )
            self._processed_actions[:] = self._full_joint_targets[:, self._base_joint_ids_tensor]

    def apply_actions(self):
        # apply_actions is called once per physics substep.  Reusing the exact
        # target implements the approved causal 50 Hz -> 200 Hz ZOH transport.
        self._asset.set_joint_position_target(self._full_joint_targets)


@configclass
class A3BaseCompositePositionActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = A3BaseCompositePositionAction
    base_joint_names: tuple[str, ...] = MISSING
    backend_joint_names: tuple[str, ...] = MISSING
    action_scale_rad: tuple[float, ...] = MISSING
    action_mask: tuple[float, ...] = MISSING
    raw_clip: float = 0.25
    clip_to_soft_joint_limits: bool = True
