"""Deployable A3 Base action semantics for the bounded Stand smoke task."""

from __future__ import annotations

from dataclasses import MISSING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


# Diagnostic Base14 actuator authority selected for the PD_STAND plant.  Kept
# next to the action semantics so Stand and Strike-conditioned environments use
# one physical residual contract.
A3_PD_STAND_BASE_ACTION_SCALE_RAD = (
    0.03666666666666667,
    0.1375,
    0.18333333333333332,
    0.04,
    0.0591,
    0.027375,
    0.03666666666666667,
    0.1375,
    0.18333333333333332,
    0.04,
    0.0591,
    0.027375,
    0.023,
    0.059,
)


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
        self._phase_gate_base_indices = tuple(
            self._base_joint_ids.index(self._asset.find_joints([name], preserve_order=True)[0][0])
            for name in cfg.phase_gate_joint_names
        )
        self._raw_actions = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        # Latent policy action after structural masks/gates but before the
        # optional execution bound.  It lets rewards distinguish an actor
        # that uses the available authority from one that lives in saturation.
        self._unbounded_actions = torch.zeros_like(self._raw_actions)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._full_joint_targets = self._asset.data.default_joint_pos.clone()

    @property
    def action_dim(self) -> int:
        return len(self.cfg.base_joint_names)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def unbounded_actions(self) -> torch.Tensor:
        """Action after structural gates, before hard/smooth bounding."""
        return self._unbounded_actions

    def _bound_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Map latent residuals into the immutable execution envelope."""
        if self.cfg.smooth_raw_bound:
            # Unit slope around zero preserves the existing small-action
            # semantics while approaching +/-raw_clip continuously.
            return self.cfg.raw_clip * torch.tanh(actions / self.cfg.raw_clip)
        return torch.clamp(actions, -self.cfg.raw_clip, self.cfg.raw_clip)

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
        self._unbounded_actions[:] = masked
        self._raw_actions[:] = self._bound_actions(masked)
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
    smooth_raw_bound: bool = False
    clip_to_soft_joint_limits: bool = True


class A3StrikeConditionedBaseCompositePositionAction(A3BaseCompositePositionAction):
    """Compose Base14 residuals around a phase-indexed whole-body reference.

    Strike owns waist yaw, supplies the waist-pitch feed-forward reference,
    and owns the right arm.  Base owns both legs and waist roll; its bounded
    waist-pitch residual is added to the strike reference.  The actor therefore
    remains exactly 14 DOF while the action term emits one 31-DOF target.
    """

    cfg: "A3StrikeConditionedBaseCompositePositionActionCfg"

    def __init__(self, cfg: "A3StrikeConditionedBaseCompositePositionActionCfg", env):
        super().__init__(cfg, env)
        self._env = env
        self._strike_joint_ids, resolved_strike_names = self._asset.find_joints(
            list(cfg.strike_joint_names), preserve_order=True
        )
        if resolved_strike_names != list(cfg.strike_joint_names):
            raise ValueError(
                "A3 Strike reference joint order mismatch: "
                f"expected={list(cfg.strike_joint_names)}, resolved={resolved_strike_names}"
            )
        self._strike_joint_ids_tensor = torch.tensor(
            self._strike_joint_ids, dtype=torch.long, device=self.device
        )
        waist_pitch_ids, waist_pitch_names = self._asset.find_joints(
            [cfg.waist_pitch_joint_name], preserve_order=True
        )
        if waist_pitch_names != [cfg.waist_pitch_joint_name]:
            raise ValueError(f"A3 waist-pitch joint not resolved: {cfg.waist_pitch_joint_name}")
        self._waist_pitch_id = int(waist_pitch_ids[0])
        self._waist_pitch_base_index = self._base_joint_ids.index(self._waist_pitch_id)

    def process_actions(self, actions: torch.Tensor):
        if actions.shape != self._raw_actions.shape:
            raise ValueError(f"Expected Base action shape {self._raw_actions.shape}, got {actions.shape}")
        if not torch.isfinite(actions).all():
            raise ValueError("A3 Strike-conditioned Base action contains NaN or infinity")

        motion_cmd = self._env.command_manager.get_term(self.cfg.reference_command_name)
        masked = actions * self._mask
        if self._phase_gate_base_indices:
            if motion_cmd._use_motion_library:
                lengths = motion_cmd.motion.motion_lengths[motion_cmd.motion_ids].clamp(min=2)
                phase = motion_cmd.time_steps.float() / (lengths - 1).float()
            else:
                phase = motion_cmd.time_steps.float() / max(motion_cmd.motion.time_step_total - 1, 1)
            u = ((phase - self.cfg.phase_gate_start) / max(
                self.cfg.phase_gate_end - self.cfg.phase_gate_start, 1.0e-6
            )).clamp(0.0, 1.0)
            smooth = u * u * (3.0 - 2.0 * u)
            gate = self.cfg.phase_gate_min_scale + (1.0 - self.cfg.phase_gate_min_scale) * smooth
            if self.cfg.phase_gate_tail_release_steps > 0:
                tail = motion_cmd.tail_steps.to(dtype=gate.dtype)
                release_u = (tail / float(self.cfg.phase_gate_tail_release_steps)).clamp(0.0, 1.0)
                release_smooth = release_u * release_u * (3.0 - 2.0 * release_u)
                tail_gate = self.cfg.phase_gate_min_scale + (1.0 - self.cfg.phase_gate_min_scale) * (1.0 - release_smooth)
                gate = torch.where(tail > 0, tail_gate, gate)
            for index in self._phase_gate_base_indices:
                masked[:, index] *= gate
        handoff_steps = getattr(self._env, "strike_stabilizer_handoff_steps", None)
        if handoff_steps is not None:
            # Stage-A curriculum: run the prefix under zero residual, then
            # activate the same policy without a reset or target discontinuity.
            active = (motion_cmd.time_steps >= handoff_steps).to(masked.dtype).unsqueeze(-1)
            masked = masked * active
        # A finite strike ends with a return to the ready reference.  Once that
        # return is complete, the learned residual must not remain a permanent
        # hidden stand controller: smoothly hand authority back to the nominal
        # PD ready pose.  This is deliberately a whole-leg gate (rather than a
        # joint template) and is disabled by default for legacy tasks.
        release_steps = int(self.cfg.ready_hold_residual_release_steps)
        if release_steps > 0 and motion_cmd.return_to_default_steps > 0:
            ready_elapsed = (
                motion_cmd.tail_steps
                - int(motion_cmd.cfg.hold_last_frame_steps)
                - int(motion_cmd.return_to_default_steps)
            ).clamp(min=0).to(dtype=masked.dtype)
            u = (ready_elapsed / float(release_steps)).clamp(0.0, 1.0)
            smooth_u = u * u * (3.0 - 2.0 * u)
            ready_gate = (1.0 - smooth_u).unsqueeze(-1)
            masked = masked * ready_gate
        self._unbounded_actions[:] = masked
        self._raw_actions[:] = self._bound_actions(masked)
        reference_full = motion_cmd.joint_pos

        self._full_joint_targets[:] = self._asset.data.default_joint_pos
        self._full_joint_targets[:, self._strike_joint_ids_tensor] = reference_full[
            :, self._strike_joint_ids_tensor
        ]

        if self.cfg.base_reference_mode == "motion":
            base_reference = reference_full[:, self._base_joint_ids_tensor]
        elif self.cfg.base_reference_mode == "default":
            base_reference = self._asset.data.default_joint_pos[:, self._base_joint_ids_tensor]
        else:
            raise ValueError(f"Unsupported base_reference_mode={self.cfg.base_reference_mode!r}")
        self._processed_actions[:] = base_reference + self._raw_actions * self._scale
        self._full_joint_targets[:, self._base_joint_ids_tensor] = self._processed_actions

        # Waist pitch is the only intentional overlap: strike supplies the
        # phase reference and Base contributes only a bounded residual.
        waist_residual = (
            self._raw_actions[:, self._waist_pitch_base_index]
            * self._scale[:, self._waist_pitch_base_index]
        )
        self._full_joint_targets[:, self._waist_pitch_id] = (
            reference_full[:, self._waist_pitch_id] + waist_residual
        )
        self._processed_actions[:, self._waist_pitch_base_index] = self._full_joint_targets[
            :, self._waist_pitch_id
        ]

        if self.cfg.clip_to_soft_joint_limits:
            limits = self._asset.data.soft_joint_pos_limits
            self._full_joint_targets[:] = torch.clamp(
                self._full_joint_targets, min=limits[..., 0], max=limits[..., 1]
            )
            self._processed_actions[:] = self._full_joint_targets[:, self._base_joint_ids_tensor]


@configclass
class A3StrikeConditionedBaseCompositePositionActionCfg(A3BaseCompositePositionActionCfg):
    class_type: type[ActionTerm] = A3StrikeConditionedBaseCompositePositionAction
    strike_joint_names: tuple[str, ...] = MISSING
    reference_command_name: str = "motion"
    waist_pitch_joint_name: str = "waist_pitch_joint"
    base_reference_mode: str = "motion"
    # Optional continuous authority schedule for joints that should be quiet
    # in the ready state but available during task-relevant swing dynamics.
    phase_gate_joint_names: tuple[str, ...] = ()
    phase_gate_min_scale: float = 1.0
    phase_gate_start: float = 0.0
    phase_gate_end: float = 1.0
    phase_gate_tail_release_steps: int = 0
    # Beginning only after the upper-body reference has fully returned to the
    # ready pose, fade all learned leg residuals to zero over this many policy
    # steps.  Zero preserves legacy persistent-residual behavior.
    ready_hold_residual_release_steps: int = 0
