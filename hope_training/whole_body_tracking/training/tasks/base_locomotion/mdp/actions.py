"""Deployable A3 Base action semantics for the bounded Stand smoke task."""

from __future__ import annotations

from dataclasses import MISSING
from pathlib import Path

import torch

from stage_a_compat import adapt_stage_a_observation_legacy_yaw_pi

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
        # Implicit actuators accept a position and a velocity target.  The
        # legacy contract intentionally leaves this at zero; strike-specific
        # terms may opt in for a small set of joints.
        self._full_joint_velocity_targets = torch.zeros_like(self._full_joint_targets)

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

    @property
    def full_joint_velocity_targets(self) -> torch.Tensor:
        """The composed velocity command in articulation order (diagnostic only)."""
        return self._full_joint_velocity_targets

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
        self._asset.set_joint_velocity_target(self._full_joint_velocity_targets)


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
        self._full_joint_velocity_targets.zero_()
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


class A3F0UpperBaseCompositePositionAction(A3StrikeConditionedBaseCompositePositionAction):
    """F0 evaluator action: frozen upper policy plus optional Base14 residual.

    The upper policy is supplied externally because F0 loads two independent
    checkpoints. The public action remains Base14, preserving Stage-A's 14-D
    previous-action observation while only the twelve leg channels affect the
    floating base.
    """

    cfg: "A3F0UpperBaseCompositePositionActionCfg"

    def __init__(self, cfg: "A3F0UpperBaseCompositePositionActionCfg", env):
        super().__init__(cfg, env)
        self._upper_joint_ids, resolved_upper_names = self._asset.find_joints(
            list(cfg.upper_joint_names), preserve_order=True
        )
        if resolved_upper_names != list(cfg.upper_joint_names):
            raise ValueError(
                "F0 upper joint order mismatch: "
                f"expected={list(cfg.upper_joint_names)}, resolved={resolved_upper_names}"
            )
        overlap = sorted(set(cfg.upper_joint_names) & set(cfg.base_joint_names))
        if overlap != ["waist_pitch_joint", "waist_roll_joint"]:
            raise ValueError(f"Unexpected F0 upper/base overlap: {overlap}")
        self._upper_joint_ids_tensor = torch.tensor(
            self._upper_joint_ids, dtype=torch.long, device=self.device
        )
        self._upper_scale = torch.tensor(
            [float(cfg.scale[name]) for name in cfg.upper_joint_names],
            dtype=torch.float,
            device=self.device,
        ).unsqueeze(0)
        self._upper_lead = torch.zeros(
            len(self._upper_joint_ids), dtype=torch.float32, device=self.device
        )
        configured = getattr(cfg, "joint_reference_lookahead_steps", {}) or {}
        for i, name in enumerate(cfg.upper_joint_names):
            self._upper_lead[i] = float(getattr(cfg, "reference_lookahead_steps", 0))
            self._upper_lead[i] += float(configured.get(name, 0.0))
        self._upper_raw_actions = torch.zeros(
            (self.num_envs, len(self._upper_joint_ids)), device=self.device
        )
        self._upper_processed_actions = torch.zeros_like(self._upper_raw_actions)
        # Keep the upper command decomposition observable.  Full-cycle audits
        # need to distinguish an unsafe reference from a frozen-prior residual
        # or a learned coordinator correction.
        self._upper_reference_actions = torch.zeros_like(self._upper_raw_actions)
        self._upper_primary_contribution = torch.zeros_like(self._upper_raw_actions)
        self._upper_coordinator_contribution = torch.zeros_like(self._upper_raw_actions)
        self._upper_safety_override = torch.zeros_like(self._upper_raw_actions)
        self._upper_velocity_safety_override = torch.zeros_like(self._upper_raw_actions)
        self._upper_dynamic_safety_override = torch.zeros_like(self._upper_raw_actions)
        self._upper_dynamic_velocity_safety_override = torch.zeros_like(self._upper_raw_actions)
        waist_names = tuple(getattr(cfg, "upper_waist_joint_names", ()) or ())
        if waist_names:
            unknown_waist = sorted(set(waist_names) - set(cfg.upper_joint_names))
            if unknown_waist:
                raise ValueError(f"Configured waist joints are not upper joints: {unknown_waist}")
        self._upper_waist_indices = torch.tensor(
            [cfg.upper_joint_names.index(name) for name in waist_names], dtype=torch.long, device=self.device
        )
        self._upper_arm_indices = torch.tensor(
            [i for i, name in enumerate(cfg.upper_joint_names) if name not in set(waist_names)],
            dtype=torch.long,
            device=self.device,
        )
        dynamic_guard_names = tuple(
            getattr(cfg, "upper_dynamic_soft_limit_joint_names", ()) or ()
        )
        unknown_dynamic_guard = sorted(set(dynamic_guard_names) - set(cfg.upper_joint_names))
        if unknown_dynamic_guard:
            raise ValueError(
                "Dynamic soft-limit guard joints are not upper joints: "
                f"{unknown_dynamic_guard}"
            )
        dynamic_guard_margins = tuple(
            float(value)
            for value in (
                getattr(cfg, "upper_dynamic_soft_limit_margin_rad", ()) or ()
            )
        )
        if len(dynamic_guard_names) != len(dynamic_guard_margins):
            raise ValueError(
                "upper_dynamic_soft_limit_joint_names and "
                "upper_dynamic_soft_limit_margin_rad must have equal lengths"
            )
        if any(value <= 0.0 for value in dynamic_guard_margins):
            raise ValueError("Dynamic soft-limit margins must be positive")
        self._upper_dynamic_guard_indices = torch.tensor(
            [cfg.upper_joint_names.index(name) for name in dynamic_guard_names],
            dtype=torch.long,
            device=self.device,
        )
        self._upper_dynamic_guard_margins = torch.tensor(
            dynamic_guard_margins, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        recovery_offsets = dict(
            getattr(cfg, "upper_joint_recovery_ready_offset_rad", {}) or {}
        )
        unknown_recovery_offsets = sorted(
            set(recovery_offsets) - set(cfg.upper_joint_names)
        )
        if unknown_recovery_offsets:
            raise ValueError(
                "Recovery endpoint offsets are not upper joints: "
                f"{unknown_recovery_offsets}"
            )
        self._upper_recovery_ready_offsets = {
            cfg.upper_joint_names.index(name): float(value)
            for name, value in recovery_offsets.items()
        }
        recovery_offsets_by_motion = dict(
            getattr(
                cfg, "upper_joint_recovery_ready_offset_rad_by_motion", {}
            )
            or {}
        )
        unknown_recovery_motion_offsets = sorted(
            set(recovery_offsets_by_motion) - set(cfg.upper_joint_names)
        )
        if unknown_recovery_motion_offsets:
            raise ValueError(
                "Motion-conditioned recovery offsets are not upper joints: "
                f"{unknown_recovery_motion_offsets}"
            )
        self._upper_recovery_ready_offsets_by_motion = {
            cfg.upper_joint_names.index(name): torch.tensor(
                tuple(float(item) for item in values),
                dtype=torch.float32,
                device=self.device,
            )
            for name, values in recovery_offsets_by_motion.items()
        }
        passive_margins = dict(
            getattr(cfg, "passive_joint_soft_limit_margin_rad", {}) or {}
        )
        self._passive_guard_joint_ids = torch.empty(
            0, dtype=torch.long, device=self.device
        )
        self._passive_guard_margins = torch.empty(
            (1, 0), dtype=torch.float32, device=self.device
        )
        if passive_margins:
            passive_ids, passive_names = self._asset.find_joints(
                list(passive_margins), preserve_order=True
            )
            if passive_names != list(passive_margins):
                raise ValueError(
                    "Passive soft-limit guard joint order mismatch: "
                    f"expected={list(passive_margins)}, resolved={passive_names}"
                )
            values = tuple(float(passive_margins[name]) for name in passive_names)
            if any(value <= 0.0 for value in values):
                raise ValueError("Passive soft-limit margins must be positive")
            self._passive_guard_joint_ids = torch.tensor(
                passive_ids, dtype=torch.long, device=self.device
            )
            self._passive_guard_margins = torch.tensor(
                values, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
        self._upper_velocity_joint_indices = self._resolve_upper_velocity_indices()
        self._upper_velocity_targets = torch.zeros_like(self._upper_raw_actions)
        env.f0_upper_last_action = self._upper_raw_actions.clone()

    @property
    def upper_raw_actions(self) -> torch.Tensor:
        return self._upper_raw_actions

    @property
    def upper_processed_actions(self) -> torch.Tensor:
        return self._upper_processed_actions

    @property
    def upper_reference_actions(self) -> torch.Tensor:
        return self._upper_reference_actions

    @property
    def upper_primary_contribution(self) -> torch.Tensor:
        return self._upper_primary_contribution

    @property
    def upper_coordinator_contribution(self) -> torch.Tensor:
        return self._upper_coordinator_contribution

    @property
    def upper_safety_override(self) -> torch.Tensor:
        return self._upper_safety_override

    @property
    def upper_velocity_safety_override(self) -> torch.Tensor:
        return self._upper_velocity_safety_override

    @property
    def upper_dynamic_safety_override(self) -> torch.Tensor:
        return self._upper_dynamic_safety_override

    @property
    def upper_dynamic_velocity_safety_override(self) -> torch.Tensor:
        return self._upper_dynamic_velocity_safety_override

    def _apply_upper_dynamic_soft_limit_guard(self) -> None:
        """Predictively brake selected upper joints before actual soft-limit crossing.

        Unlike the phase-specific waist guard, this execution-contract guard
        depends only on measured q/qdot and the selected joints' soft limits.
        It is disabled unless explicit joint names and margins are configured.
        """
        self._upper_dynamic_safety_override.zero_()
        self._upper_dynamic_velocity_safety_override.zero_()
        if self._upper_dynamic_guard_indices.numel() == 0:
            return

        horizon_steps = int(
            getattr(self.cfg, "upper_dynamic_soft_limit_prediction_horizon_steps", 0)
        )
        velocity_gain = float(
            getattr(self.cfg, "upper_dynamic_soft_limit_velocity_brake_gain", 0.0)
        )
        position_gain = float(
            getattr(self.cfg, "upper_dynamic_soft_limit_position_brake_gain", 1.0)
        )
        max_position_correction = float(
            getattr(
                self.cfg,
                "upper_dynamic_soft_limit_max_position_correction_rad",
                0.05,
            )
        )
        max_velocity_correction = float(
            getattr(
                self.cfg,
                "upper_dynamic_soft_limit_max_velocity_correction_radps",
                0.5,
            )
        )
        if horizon_steps < 1:
            raise ValueError(
                "upper_dynamic_soft_limit_prediction_horizon_steps must be >= 1 "
                "when the dynamic guard is enabled"
            )
        if velocity_gain < 0.0 or position_gain < 0.0:
            raise ValueError("Dynamic soft-limit brake gains must be non-negative")
        if max_position_correction <= 0.0 or max_velocity_correction <= 0.0:
            raise ValueError("Dynamic soft-limit correction limits must be positive")

        control_dt = float(self._env.cfg.decimation * self._env.cfg.sim.dt)
        if control_dt <= 0.0:
            raise RuntimeError(f"Invalid control dt for dynamic soft-limit guard: {control_dt}")
        horizon_s = float(horizon_steps) * control_dt
        upper_indices = self._upper_dynamic_guard_indices
        joint_ids = self._upper_joint_ids_tensor[upper_indices]
        limits = self._asset.data.soft_joint_pos_limits[:, joint_ids]
        margin = self._upper_dynamic_guard_margins
        lower = limits[..., 0] + margin
        upper = limits[..., 1] - margin
        if torch.any(lower >= upper):
            raise ValueError("Dynamic soft-limit margin leaves no valid target interval")

        actual = self._asset.data.joint_pos[:, joint_ids]
        actual_velocity = self._asset.data.joint_vel[:, joint_ids]
        predicted = actual + horizon_s * actual_velocity
        upper_distance = torch.maximum(
            torch.relu(predicted - upper), torch.relu(actual - upper)
        )
        lower_distance = torch.maximum(
            torch.relu(lower - predicted), torch.relu(lower - actual)
        )
        risk = ((upper_distance + lower_distance) / margin).clamp(0.0, 1.0)

        before = self._full_joint_targets[:, joint_ids].clone()
        # First reserve the configured static inner margin, then use measured
        # velocity to pull the target opposite the predicted crossing.
        inner = torch.clamp(before, min=lower, max=upper)
        upper_brake = torch.minimum(
            inner, actual - position_gain * horizon_s * torch.relu(actual_velocity)
        )
        lower_brake = torch.maximum(
            inner, actual + position_gain * horizon_s * torch.relu(-actual_velocity)
        )
        braking = torch.where(upper_distance > 0.0, upper_brake, inner)
        braking = torch.where(lower_distance > 0.0, lower_brake, braking)
        braking = torch.clamp(braking, min=lower, max=upper)
        position_correction = torch.clamp(
            risk * (braking - inner),
            min=-max_position_correction,
            max=max_position_correction,
        )
        after = torch.clamp(inner + position_correction, min=lower, max=upper)
        self._full_joint_targets[:, joint_ids] = after
        self._upper_dynamic_safety_override[:, upper_indices] = after - before

        velocity_before = self._full_joint_velocity_targets[:, joint_ids].clone()
        upper_velocity = torch.minimum(
            velocity_before, -velocity_gain * upper_distance
        )
        lower_velocity = torch.maximum(
            velocity_before, velocity_gain * lower_distance
        )
        braking_velocity = torch.where(
            upper_distance > 0.0, upper_velocity, velocity_before
        )
        braking_velocity = torch.where(
            lower_distance > 0.0, lower_velocity, braking_velocity
        )
        velocity_correction = torch.clamp(
            risk * (braking_velocity - velocity_before),
            min=-max_velocity_correction,
            max=max_velocity_correction,
        )
        velocity_after = velocity_before + velocity_correction
        self._full_joint_velocity_targets[:, joint_ids] = velocity_after
        self._upper_dynamic_velocity_safety_override[:, upper_indices] = (
            velocity_after - velocity_before
        )

    def _apply_passive_soft_limit_guard(self) -> None:
        """Keep non-policy joints inside an explicit inner command interval."""
        if self._passive_guard_joint_ids.numel() == 0:
            return
        limits = self._asset.data.soft_joint_pos_limits[
            :, self._passive_guard_joint_ids
        ]
        lower = limits[..., 0] + self._passive_guard_margins
        upper = limits[..., 1] - self._passive_guard_margins
        if torch.any(lower >= upper):
            raise ValueError("Passive soft-limit margin leaves no valid interval")
        targets = self._full_joint_targets[:, self._passive_guard_joint_ids]
        self._full_joint_targets[:, self._passive_guard_joint_ids] = torch.clamp(
            targets, min=lower, max=upper
        )

    def _resolve_upper_velocity_indices(self) -> torch.Tensor:
        """Resolve the explicitly opt-in velocity-feedforward joints once."""
        configured = tuple(getattr(self.cfg, "joint_velocity_feedforward_joint_names", ()) or ())
        if not configured:
            return torch.empty(0, dtype=torch.long, device=self.device)
        unknown = sorted(set(configured) - set(self.cfg.upper_joint_names))
        if unknown:
            raise ValueError(f"Velocity-feedforward joints are not upper joints: {unknown}")
        return torch.tensor(
            [self.cfg.upper_joint_names.index(name) for name in configured],
            dtype=torch.long,
            device=self.device,
        )

    def _sample_upper_motion(
        self, motion_cmd, query: torch.Tensor
    ) -> torch.Tensor:
        """Sample raw upper joint positions without prelude or residual logic."""
        if motion_cmd._use_motion_library:
            lengths = motion_cmd.motion.motion_lengths[motion_cmd.motion_ids]
            full = motion_cmd.motion.joint_pos[motion_cmd.motion_ids]
            max_t = (lengths - 1).long().unsqueeze(-1).expand_as(query)
        else:
            full = motion_cmd.motion.joint_pos.unsqueeze(0).expand(query.shape[0], -1, -1)
            max_t = torch.full_like(query, full.shape[1] - 1, dtype=torch.long)
        query = query.clamp(min=0.0)
        query = torch.minimum(query, max_t.float())
        t0 = query.floor().long()
        t1 = torch.minimum(t0 + 1, max_t)
        alpha = (query - t0.float()).unsqueeze(-1)
        gather_shape = (*t0.shape, full.shape[-1])
        ref0 = torch.gather(full, 1, t0.unsqueeze(-1).expand(gather_shape))
        ref1 = torch.gather(full, 1, t1.unsqueeze(-1).expand(gather_shape))
        lead_reference = ref0 + alpha * (ref1 - ref0)
        joint_ids = self._upper_joint_ids_tensor.view(1, -1).expand(query.shape[0], -1)
        lead_reference = lead_reference.gather(2, joint_ids.unsqueeze(-1)).squeeze(-1)
        return lead_reference

    def _motion_final_steps(self, motion_cmd) -> torch.Tensor:
        if motion_cmd._use_motion_library:
            return motion_cmd.motion.motion_lengths[motion_cmd.motion_ids].long() - 1
        return torch.full_like(motion_cmd.time_steps, motion_cmd.motion.time_step_total - 1)

    def _motion_hit_steps(self, motion_cmd) -> torch.Tensor:
        if motion_cmd._use_motion_library:
            return motion_cmd.motion.hit_frame[motion_cmd.motion_ids].long()
        return torch.full_like(motion_cmd.time_steps, int(motion_cmd.motion.hit_frame[0]))

    def _post_hit_elapsed_steps(self, motion_cmd, time_steps: torch.Tensor) -> torch.Tensor:
        """Return non-wrapping elapsed control steps since impact."""
        hit = self._motion_hit_steps(motion_cmd)
        final = self._motion_final_steps(motion_cmd)
        active_elapsed = (time_steps - hit).clamp(min=0)
        tail_elapsed = (final - hit + motion_cmd.tail_steps).clamp(min=0)
        return torch.where(motion_cmd.tail_steps > 0, tail_elapsed, active_elapsed)

    def _minimum_jerk_blend(self, elapsed: torch.Tensor, settle_steps: int, return_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return position blend and d(blend)/dt for a finite return."""
        if settle_steps < 0 or return_steps < 1:
            raise ValueError("settle_steps must be >= 0 and return_steps must be >= 1")
        local = (elapsed - settle_steps).clamp(min=0).to(dtype=torch.float)
        u = (local / float(return_steps)).clamp(0.0, 1.0)
        blend = u * u * u * (10.0 - 15.0 * u + 6.0 * u * u)
        control_dt = float(self._env.cfg.decimation * self._env.cfg.sim.dt)
        if control_dt <= 0.0:
            raise RuntimeError(f"Invalid control dt for recovery reference: {control_dt}")
        rate = 30.0 * u * u * (1.0 - u) * (1.0 - u) / (float(return_steps) * control_dt)
        return blend, rate

    def _apply_split_post_hit_reference(self, motion_cmd, time_steps: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Use independent waist and arm recovery trajectories when enabled."""
        result = reference.clone()
        in_prelude = motion_cmd.prelude_elapsed_steps < int(motion_cmd.prelude_steps)
        hit = self._motion_hit_steps(motion_cmd)
        post_hit = (~in_prelude) & ((time_steps >= hit) | (motion_cmd.tail_steps > 0))

        waist_return_steps = int(getattr(self.cfg, "waist_post_hit_return_steps", 0))
        if waist_return_steps > 0 and self._upper_waist_indices.numel() > 0:
            settle_steps = int(getattr(self.cfg, "waist_post_hit_settle_steps", 0))
            elapsed = self._post_hit_elapsed_steps(motion_cmd, time_steps)
            blend, _ = self._minimum_jerk_blend(elapsed, settle_steps, waist_return_steps)
            hit_query = hit.float().unsqueeze(-1).expand(-1, len(self._upper_joint_ids))
            hit_reference = self._sample_upper_motion(motion_cmd, hit_query)
            ready = motion_cmd.ready_joint_pos[:, self._upper_joint_ids_tensor]
            waist_reference = hit_reference + blend.unsqueeze(-1) * (ready - hit_reference)
            result[:, self._upper_waist_indices] = torch.where(
                post_hit.unsqueeze(-1),
                waist_reference[:, self._upper_waist_indices],
                result[:, self._upper_waist_indices],
            )

        arm_return_steps = int(getattr(self.cfg, "arm_tail_return_steps", 0))
        if arm_return_steps > 0 and self._upper_arm_indices.numel() > 0:
            arm_hold_steps = int(getattr(self.cfg, "arm_tail_hold_steps", 0))
            tail = motion_cmd.tail_steps
            blend, _ = self._minimum_jerk_blend(tail, arm_hold_steps, arm_return_steps)
            final = self._motion_final_steps(motion_cmd)
            final_query = final.float().unsqueeze(-1).expand(-1, len(self._upper_joint_ids))
            final_reference = self._sample_upper_motion(motion_cmd, final_query)
            ready = motion_cmd.ready_joint_pos[:, self._upper_joint_ids_tensor]
            arm_reference = final_reference + blend.unsqueeze(-1) * (ready - final_reference)
            result[:, self._upper_arm_indices] = torch.where(
                (tail > 0).unsqueeze(-1),
                arm_reference[:, self._upper_arm_indices],
                result[:, self._upper_arm_indices],
            )
            # Joint-specific endpoint shaping retains the already qualified
            # return timing while reserving room for a joint whose measured
            # plant overshoots the nominal READY endpoint.
            shaped_indices = set(self._upper_recovery_ready_offsets) | set(
                self._upper_recovery_ready_offsets_by_motion
            )
            for upper_index in shaped_indices:
                if upper_index not in self._upper_arm_indices.tolist():
                    continue
                if upper_index in self._upper_recovery_ready_offsets_by_motion:
                    if not motion_cmd._use_motion_library:
                        raise ValueError(
                            "Motion-conditioned recovery endpoints require a motion library"
                        )
                    offsets = self._upper_recovery_ready_offsets_by_motion[upper_index]
                    if torch.any(motion_cmd.motion_ids >= offsets.shape[0]):
                        raise ValueError("motion_id has no recovery endpoint offset")
                    offset = offsets[motion_cmd.motion_ids]
                else:
                    offset = torch.full_like(
                        ready[:, upper_index],
                        self._upper_recovery_ready_offsets[upper_index],
                    )
                shaped_ready = ready[:, upper_index] + offset
                shaped_reference = final_reference[:, upper_index] + blend * (
                    shaped_ready - final_reference[:, upper_index]
                )
                result[:, upper_index] = torch.where(
                    tail > 0, shaped_reference, result[:, upper_index]
                )
        return result

    def _apply_waist_soft_limit_guard(self, motion_cmd, time_steps: torch.Tensor) -> None:
        """Brake toward an inner waist limit before an impact-pose overshoot.

        The normal soft-limit clamp only acts after the target has reached the
        boundary.  A fast floating-base swing can then overshoot physically
        even though the target is numerically clipped.  This optional guard
        starts a smooth blend a few steps before impact and reserves a small
        inner margin for PD braking.  It is disabled by default.
        """
        self._upper_safety_override.zero_()
        self._upper_velocity_safety_override.zero_()
        if self._upper_waist_indices.numel() == 0:
            return
        margins_by_motion = tuple(
            float(value)
            for value in (
                getattr(self.cfg, "waist_soft_limit_margin_rad_by_motion", ())
                or ()
            )
        )
        if margins_by_motion:
            if not motion_cmd._use_motion_library:
                raise ValueError("Motion-conditioned waist margins require a motion library")
            margin_values = torch.tensor(
                margins_by_motion,
                dtype=self._full_joint_targets.dtype,
                device=self.device,
            )
            if torch.any(motion_cmd.motion_ids >= margin_values.shape[0]):
                raise ValueError("motion_id has no waist soft-limit margin")
            margin = margin_values[motion_cmd.motion_ids].unsqueeze(-1)
        else:
            margin_value = float(
                getattr(self.cfg, "waist_soft_limit_margin_rad", 0.0)
            )
            if margin_value <= 0.0:
                return
            margin = torch.full(
                (self.num_envs, 1),
                margin_value,
                dtype=self._full_joint_targets.dtype,
                device=self.device,
            )
        if torch.any(margin <= 0.0):
            raise ValueError("waist soft-limit margins must be positive")
        lead_steps = int(getattr(self.cfg, "waist_soft_limit_brake_lead_steps", 0))
        if lead_steps < 1:
            raise ValueError("waist_soft_limit_brake_lead_steps must be >= 1 when the waist guard is enabled")
        hit = self._motion_hit_steps(motion_cmd)
        start = hit - lead_steps
        in_prelude = motion_cmd.prelude_elapsed_steps < int(motion_cmd.prelude_steps)
        allow_prelude = bool(getattr(self.cfg, "waist_soft_limit_guard_in_prelude", False))
        guard_allowed = (~in_prelude) | allow_prelude
        phase_active = (~in_prelude) & ((time_steps >= start) | (motion_cmd.tail_steps > 0))
        elapsed = torch.where(
            motion_cmd.tail_steps > 0,
            torch.full_like(time_steps, lead_steps),
            (time_steps - start).clamp(min=0),
        ).to(dtype=self._full_joint_targets.dtype)
        u = (elapsed / float(lead_steps)).clamp(0.0, 1.0)
        blend = (u * u * (3.0 - 2.0 * u)).unsqueeze(-1)
        waist_joint_ids = self._upper_joint_ids_tensor[self._upper_waist_indices]
        limits = self._asset.data.soft_joint_pos_limits[:, waist_joint_ids]
        lower = limits[..., 0] + margin
        upper = limits[..., 1] - margin
        if torch.any(lower >= upper):
            raise ValueError("waist_soft_limit_margin_rad leaves no valid waist target interval")
        before = self._full_joint_targets[:, waist_joint_ids].clone()
        guarded = torch.clamp(before, min=lower, max=upper)
        phase_after = before + blend * (guarded - before)

        # Position clipping alone reacts after a moving waist has already
        # crossed the inner bound.  Optional predictive braking uses current
        # measured q and qdot, not a motion-id-specific phase rule.  It starts
        # correcting both the position target and the PD velocity target when
        # the short-horizon state estimate would leave the inner interval.
        horizon_steps = int(getattr(self.cfg, "waist_soft_limit_prediction_horizon_steps", 0))
        velocity_gain = float(getattr(self.cfg, "waist_soft_limit_velocity_brake_gain", 0.0))
        if horizon_steps < 0:
            raise ValueError("waist_soft_limit_prediction_horizon_steps must be >= 0")
        if velocity_gain < 0.0:
            raise ValueError("waist_soft_limit_velocity_brake_gain must be >= 0")
        after = torch.where(phase_active.unsqueeze(-1), phase_after, before)
        if bool(getattr(self.cfg, "waist_soft_limit_enforce_inner_limit", False)):
            after = torch.where(guard_allowed.unsqueeze(-1), guarded, after)
        if horizon_steps > 0 and velocity_gain > 0.0:
            control_dt = float(self._env.cfg.decimation * self._env.cfg.sim.dt)
            if control_dt <= 0.0:
                raise RuntimeError(f"Invalid control dt for waist soft-limit guard: {control_dt}")
            actual = self._asset.data.joint_pos[:, waist_joint_ids]
            actual_velocity = self._asset.data.joint_vel[:, waist_joint_ids]
            predicted = actual + float(horizon_steps) * control_dt * actual_velocity
            upper_excess = torch.relu(predicted - upper)
            lower_excess = torch.relu(lower - predicted)
            risk_distance = upper_excess + lower_excess
            # A one-margin transition avoids discontinuous target changes
            # while fully braking states already outside the inner interval.
            risk = (risk_distance / margin).clamp(0.0, 1.0)
            risk = torch.maximum(risk, (actual > upper).to(risk.dtype))
            risk = torch.maximum(risk, (actual < lower).to(risk.dtype))
            predicted_after = after + risk * (guarded - after)
            after = torch.where(guard_allowed.unsqueeze(-1), predicted_after, after)

            velocity_before = self._full_joint_velocity_targets[:, waist_joint_ids].clone()
            brake_velocity = velocity_gain * (lower_excess - upper_excess)
            velocity_after = velocity_before + risk * (brake_velocity - velocity_before)
            velocity_after = torch.where(guard_allowed.unsqueeze(-1), velocity_after, velocity_before)
            self._full_joint_velocity_targets[:, waist_joint_ids] = velocity_after
            self._upper_velocity_safety_override[:, self._upper_waist_indices] = velocity_after - velocity_before
        self._full_joint_targets[:, waist_joint_ids] = after
        self._upper_safety_override[:, self._upper_waist_indices] = after - before

    def _upper_reference(self, motion_cmd, time_steps: torch.Tensor) -> torch.Tensor:
        """Gather the same lead-compensated raw motion reference as model_900."""
        query = time_steps.float().unsqueeze(-1) + self._upper_lead.unsqueeze(0)
        lead_reference = self._sample_upper_motion(motion_cmd, query)

        release_steps = int(getattr(self.cfg, "upper_prelude_release_steps", 0))
        if release_steps <= 0:
            return lead_reference
        # F0/F1 must not let an upper-body lookahead jump the robot out of the
        # validated flexed ready pose.  During the prelude MotionCommand owns
        # the physical blend to frame zero; after that, introduce the frozen
        # model_900 lead over a small number of swing frames.
        no_lead = motion_cmd.joint_pos[:, self._upper_joint_ids_tensor]
        in_prelude = motion_cmd.prelude_elapsed_steps < int(motion_cmd.prelude_steps)
        release = (time_steps.float() / float(release_steps)).clamp(0.0, 1.0).unsqueeze(-1)
        blended = no_lead + release * (lead_reference - no_lead)
        reference = torch.where(in_prelude.unsqueeze(-1), no_lead, blended)

        # ``MotionCommand.joint_pos`` owns the finite tail: final-pose hold,
        # minimum-jerk return to ready, then ready hold.  The raw lookahead
        # sampler above is valid only while the strike clip advances.  Keeping
        # it active after the final frame pins waist/arm joints at the strike
        # pose and defeats the configured recovery trajectory.
        in_tail = motion_cmd.tail_steps > 0
        reference = torch.where(in_tail.unsqueeze(-1), no_lead, reference)
        return self._apply_split_post_hit_reference(motion_cmd, time_steps, reference)

    def _upper_velocity_reference(self, motion_cmd, time_steps: torch.Tensor) -> torch.Tensor:
        """Return an optional, finite-difference upper joint velocity target.

        ``position_lead`` is contract A: sample velocity at the same phase as
        the lead-compensated position target.  ``task_phase`` is contract B:
        retain the task-phase velocity while position remains lead compensated.
        Both modes use the raw runtime joint trajectory, never the inconsistent
        NPZ velocity fields.  The finite tail is clamped, so this path cannot
        wrap a strike into its next repetition.
        """
        self._upper_velocity_targets.zero_()
        in_prelude = motion_cmd.prelude_elapsed_steps < int(motion_cmd.prelude_steps)
        launch_steps = int(getattr(motion_cmd, "prelude_launch_steps", 0))
        in_launch = (~in_prelude) & (motion_cmd.tail_steps == 0) & (time_steps < launch_steps)
        bridge_velocity_enabled = bool(getattr(motion_cmd.cfg, "prelude_continuous_velocity_reference", False))
        if bridge_velocity_enabled and torch.any(in_prelude | in_launch):
            # MotionCommand owns the bridge position and its analytic velocity.
            # Preserve the established position/velocity pair throughout the
            # bridge instead of forcing a zero-velocity PD target until the
            # first swing frame.
            bridge_velocity = motion_cmd.joint_vel[:, self._upper_joint_ids_tensor]
            self._upper_velocity_targets[:] = torch.where(
                (in_prelude | in_launch).unsqueeze(-1), bridge_velocity, self._upper_velocity_targets
            )
        mode = str(getattr(self.cfg, "joint_velocity_feedforward_mode", "none"))
        beta = float(getattr(self.cfg, "joint_velocity_feedforward_beta", 0.0))
        if mode == "none" or beta == 0.0 or self._upper_velocity_joint_indices.numel() == 0:
            return self._upper_velocity_targets
        if mode not in {"position_lead", "task_phase"}:
            raise ValueError(
                "joint_velocity_feedforward_mode must be one of "
                "'none', 'position_lead', or 'task_phase'"
            )

        if mode == "position_lead":
            phase = time_steps.float().unsqueeze(-1) + self._upper_lead.unsqueeze(0)
        else:
            phase = time_steps.float().unsqueeze(-1).expand(-1, len(self._upper_joint_ids))
        control_dt = float(self._env.cfg.decimation * self._env.cfg.sim.dt)
        if control_dt <= 0.0:
            raise RuntimeError(f"Invalid control dt for velocity feedforward: {control_dt}")
        before = self._sample_upper_motion(motion_cmd, phase - 1.0)
        after = self._sample_upper_motion(motion_cmd, phase + 1.0)
        velocity = (after - before) / (2.0 * control_dt)

        # Suppress the prelude, then smoothly remove the target velocity after
        # impact.  A hard hit->hit+1 zero would create a torque impulse.
        if motion_cmd._use_motion_library:
            hit = motion_cmd.motion.hit_frame[motion_cmd.motion_ids]
        else:
            hit = torch.full_like(time_steps, int(motion_cmd.motion.hit_frame[0]))
        decay_steps = int(getattr(self.cfg, "joint_velocity_feedforward_post_hit_decay_steps", 6))
        if decay_steps < 1:
            raise ValueError("joint_velocity_feedforward_post_hit_decay_steps must be >= 1")
        post_hit = (time_steps - hit).clamp(min=0).to(dtype=velocity.dtype)
        u = (post_hit / float(decay_steps)).clamp(0.0, 1.0)
        smooth = u * u * (3.0 - 2.0 * u)
        phase_gate = (1.0 - smooth).unsqueeze(-1)
        bridge_mask = in_prelude | in_launch
        velocity = torch.where(bridge_mask.unsqueeze(-1), torch.zeros_like(velocity), velocity * phase_gate)
        self._upper_velocity_targets[:, self._upper_velocity_joint_indices] = (
            beta * velocity[:, self._upper_velocity_joint_indices]
        )
        if bridge_velocity_enabled and torch.any(in_prelude | in_launch):
            self._upper_velocity_targets[:] = torch.where(
                (in_prelude | in_launch).unsqueeze(-1),
                motion_cmd.joint_vel[:, self._upper_joint_ids_tensor],
                self._upper_velocity_targets,
            )

        # During the finite return, use the command's continuous minimum-jerk
        # velocity for every upper joint.  The ready-hold velocity is already
        # zero, so this does not reintroduce residual tail velocity.
        in_tail = motion_cmd.tail_steps > 0
        tail_velocity = motion_cmd.joint_vel[:, self._upper_joint_ids_tensor]
        self._upper_velocity_targets[:] = torch.where(
            in_tail.unsqueeze(-1), tail_velocity, self._upper_velocity_targets
        )
        self._apply_split_post_hit_velocity(motion_cmd, time_steps)
        return self._upper_velocity_targets

    def _apply_split_post_hit_velocity(self, motion_cmd, time_steps: torch.Tensor) -> None:
        """Override velocity targets for enabled split waist/arm recoveries."""
        in_prelude = motion_cmd.prelude_elapsed_steps < int(motion_cmd.prelude_steps)
        hit = self._motion_hit_steps(motion_cmd)
        post_hit = (~in_prelude) & ((time_steps >= hit) | (motion_cmd.tail_steps > 0))
        ready = motion_cmd.ready_joint_pos[:, self._upper_joint_ids_tensor]

        waist_return_steps = int(getattr(self.cfg, "waist_post_hit_return_steps", 0))
        if waist_return_steps > 0 and self._upper_waist_indices.numel() > 0:
            settle_steps = int(getattr(self.cfg, "waist_post_hit_settle_steps", 0))
            elapsed = self._post_hit_elapsed_steps(motion_cmd, time_steps)
            _, rate = self._minimum_jerk_blend(elapsed, settle_steps, waist_return_steps)
            hit_query = hit.float().unsqueeze(-1).expand(-1, len(self._upper_joint_ids))
            hit_reference = self._sample_upper_motion(motion_cmd, hit_query)
            waist_velocity = rate.unsqueeze(-1) * (ready - hit_reference)
            self._upper_velocity_targets[:, self._upper_waist_indices] = torch.where(
                post_hit.unsqueeze(-1),
                waist_velocity[:, self._upper_waist_indices],
                self._upper_velocity_targets[:, self._upper_waist_indices],
            )

        arm_return_steps = int(getattr(self.cfg, "arm_tail_return_steps", 0))
        if arm_return_steps > 0 and self._upper_arm_indices.numel() > 0:
            arm_hold_steps = int(getattr(self.cfg, "arm_tail_hold_steps", 0))
            _, rate = self._minimum_jerk_blend(motion_cmd.tail_steps, arm_hold_steps, arm_return_steps)
            final = self._motion_final_steps(motion_cmd)
            final_query = final.float().unsqueeze(-1).expand(-1, len(self._upper_joint_ids))
            final_reference = self._sample_upper_motion(motion_cmd, final_query)
            arm_velocity = rate.unsqueeze(-1) * (ready - final_reference)
            in_tail = motion_cmd.tail_steps > 0
            self._upper_velocity_targets[:, self._upper_arm_indices] = torch.where(
                in_tail.unsqueeze(-1),
                arm_velocity[:, self._upper_arm_indices],
                self._upper_velocity_targets[:, self._upper_arm_indices],
            )

    def process_actions(self, actions: torch.Tensor):
        # Reuse the reviewed Stage-A/Base14 leg mask, gate, nominal reference,
        # and soft-limit handling without duplicating that contract here.
        super().process_actions(actions)
        upper = getattr(self._env, "f0_upper_raw_action", None)
        if upper is None:
            upper = torch.zeros_like(self._upper_raw_actions)
        if upper.shape != self._upper_raw_actions.shape:
            raise ValueError(
                f"Expected F0 upper action shape {self._upper_raw_actions.shape}, got {upper.shape}"
            )
        self._upper_raw_actions[:] = torch.clamp(upper, -self.cfg.upper_raw_clip, self.cfg.upper_raw_clip)
        motion_cmd = self._env.command_manager.get_term(self.cfg.reference_command_name)
        reference = self._upper_reference(motion_cmd, motion_cmd.time_steps)
        raw_gate = torch.ones((self.num_envs, 1), device=self.device)
        release_steps = int(getattr(self.cfg, "upper_prelude_release_steps", 0))
        if release_steps > 0:
            in_prelude = motion_cmd.prelude_elapsed_steps < int(motion_cmd.prelude_steps)
            release = (motion_cmd.time_steps.float() / float(release_steps)).clamp(0.0, 1.0).unsqueeze(-1)
            raw_gate = torch.where(in_prelude.unsqueeze(-1), torch.zeros_like(release), release)
        self._upper_processed_actions[:] = reference + raw_gate * self._upper_raw_actions * self._upper_scale
        self._upper_reference_actions[:] = reference
        self._upper_primary_contribution[:] = raw_gate * self._upper_raw_actions * self._upper_scale
        self._upper_coordinator_contribution.zero_()
        self._full_joint_targets[:, self._upper_joint_ids_tensor] = self._upper_processed_actions
        self._full_joint_velocity_targets[:, self._upper_joint_ids_tensor] = self._upper_velocity_reference(
            motion_cmd, motion_cmd.time_steps
        )
        self._apply_waist_soft_limit_guard(motion_cmd, motion_cmd.time_steps)
        self._apply_upper_dynamic_soft_limit_guard()
        self._apply_passive_soft_limit_guard()
        self._upper_processed_actions[:] = self._full_joint_targets[:, self._upper_joint_ids_tensor]
        if self.cfg.clip_to_soft_joint_limits:
            limits = self._asset.data.soft_joint_pos_limits
            self._full_joint_targets[:] = torch.clamp(
                self._full_joint_targets, min=limits[..., 0], max=limits[..., 1]
            )
            self._upper_processed_actions[:] = self._full_joint_targets[:, self._upper_joint_ids_tensor]
        self._env.f0_upper_last_action[:] = self._upper_raw_actions


@configclass
class A3F0UpperBaseCompositePositionActionCfg(A3StrikeConditionedBaseCompositePositionActionCfg):
    class_type: type[ActionTerm] = A3F0UpperBaseCompositePositionAction
    upper_joint_names: tuple[str, ...] = MISSING
    upper_raw_clip: float = 0.50
    upper_prelude_release_steps: int = 0
    upper_waist_joint_names: tuple[str, ...] = (
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"
    )
    # Disabled by default.  Full-cycle tasks can retire the waist from its
    # impact pose independently of arm completion and recovery.
    waist_post_hit_settle_steps: int = 0
    waist_post_hit_return_steps: int = 0
    arm_tail_hold_steps: int = 0
    arm_tail_return_steps: int = 0
    upper_joint_recovery_ready_offset_rad: dict[str, float] = {}
    upper_joint_recovery_ready_offset_rad_by_motion: dict[str, tuple[float, ...]] = {}
    waist_soft_limit_margin_rad: float = 0.0
    waist_soft_limit_margin_rad_by_motion: tuple[float, ...] = ()
    waist_soft_limit_brake_lead_steps: int = 0
    waist_soft_limit_prediction_horizon_steps: int = 0
    waist_soft_limit_velocity_brake_gain: float = 0.0
    waist_soft_limit_guard_in_prelude: bool = False
    waist_soft_limit_enforce_inner_limit: bool = False
    # Generic measured-state predictive guard.  Empty names keep the exact
    # historical command path; P4D enables only dynamically disqualified
    # joints after trace-based identification.
    upper_dynamic_soft_limit_joint_names: tuple[str, ...] = ()
    upper_dynamic_soft_limit_margin_rad: tuple[float, ...] = ()
    upper_dynamic_soft_limit_prediction_horizon_steps: int = 0
    upper_dynamic_soft_limit_velocity_brake_gain: float = 0.0
    upper_dynamic_soft_limit_position_brake_gain: float = 1.0
    upper_dynamic_soft_limit_max_position_correction_rad: float = 0.05
    upper_dynamic_soft_limit_max_velocity_correction_radps: float = 0.5
    passive_joint_soft_limit_margin_rad: dict[str, float] = {}
    # These fields let the common native-strike task override path configure
    # the frozen upper contract without knowing about the F0 composite term.
    joint_names: tuple[str, ...] = ()
    scale: dict[str, float] = MISSING
    preserve_order: bool = True
    reference_lookahead_steps: int = 0
    joint_reference_lookahead_steps: dict[str, float] = {}
    # Optional target-velocity feedforward for the implicit PD actuator.  The
    # default retains the historical position-only V2 execution contract.
    joint_velocity_feedforward_mode: str = "none"
    joint_velocity_feedforward_beta: float = 0.0
    joint_velocity_feedforward_joint_names: tuple[str, ...] = ()
    joint_velocity_feedforward_post_hit_decay_steps: int = 6


class _FrozenCheckpointActor:
    """Small inference-only actor loader for a frozen cross-task policy."""

    def __init__(self, path: str, device: torch.device):
        checkpoint = Path(path).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Frozen upper checkpoint does not exist: {checkpoint}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = state.get("model_state_dict", {})
        layer_ids = sorted(
            int(key.split(".")[1])
            for key in model
            if key.startswith("actor.") and key.endswith(".weight")
        )
        if not layer_ids:
            raise RuntimeError(f"Frozen upper checkpoint has no actor layers: {checkpoint}")
        layers: list[torch.nn.Module] = []
        for index, layer_id in enumerate(layer_ids):
            weight = model[f"actor.{layer_id}.weight"]
            bias = model[f"actor.{layer_id}.bias"]
            layer = torch.nn.Linear(weight.shape[1], weight.shape[0])
            layer.weight.data.copy_(weight)
            layer.bias.data.copy_(bias)
            layers.append(layer)
            if index + 1 < len(layer_ids):
                layers.append(torch.nn.ELU())
        self.actor = torch.nn.Sequential(*layers).to(device).eval()
        normalizer = state.get("obs_norm_state_dict")
        if normalizer is None:
            raise RuntimeError(f"Frozen upper checkpoint has no observation normalizer: {checkpoint}")
        self.mean = normalizer["_mean"].to(device)
        self.std = normalizer["_std"].to(device).clamp_min(1.0e-6)
        self.obs_dim = int(self.mean.shape[-1])
        self.action_dim = int(model["std"].shape[-1])
        self.path = str(checkpoint)

    @torch.inference_mode()
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[-1] != self.obs_dim:
            raise RuntimeError(
                f"Frozen upper observation width mismatch: checkpoint={self.obs_dim}, runtime={obs.shape[-1]}"
            )
        return self.actor(torch.clamp((obs - self.mean) / self.std, -100.0, 100.0))


class A3F1FrozenUpperBaseCompositePositionAction(A3F0UpperBaseCompositePositionAction):
    """F1 action: frozen model_900 upper actor plus trainable Stage-A Base14."""

    cfg: "A3F1FrozenUpperBaseCompositePositionActionCfg"

    def __init__(self, cfg: "A3F1FrozenUpperBaseCompositePositionActionCfg", env):
        super().__init__(cfg, env)
        self._upper_policy = _FrozenCheckpointActor(cfg.upper_checkpoint, self.device)
        if self._upper_policy.obs_dim != 56 or self._upper_policy.action_dim != len(self._upper_joint_ids):
            raise RuntimeError(
                "F1 frozen upper contract mismatch: "
                f"obs={self._upper_policy.obs_dim}, action={self._upper_policy.action_dim}, "
                f"expected=(56, {len(self._upper_joint_ids)})"
            )
        self._upper_observation_group = str(cfg.upper_observation_group)
        # Exact observation consumed by model_900 on the most recent control
        # step.  Evaluation tools read this buffer instead of recomputing the
        # observation group (which could draw a different noise sample).
        self._upper_last_observation = torch.zeros(
            (self.num_envs, self._upper_policy.obs_dim),
            dtype=torch.float,
            device=self.device,
        )

    @property
    def upper_last_observation(self) -> torch.Tensor:
        return self._upper_last_observation

    def _compute_observation_group(self, name: str) -> torch.Tensor:
        value = self._env.observation_manager.compute_group(name)
        if isinstance(value, tuple):
            value = value[0]
        if isinstance(value, dict):
            value = value.get(name, next(iter(value.values())))
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Observation group {name!r} did not return a tensor")
        return value

    def process_actions(self, actions: torch.Tensor):
        upper_obs = self._compute_observation_group(self._upper_observation_group)
        self._upper_last_observation[:] = upper_obs
        upper_action = self._upper_policy(upper_obs)
        self._env.f0_upper_raw_action = upper_action
        super().process_actions(actions)


@configclass
class A3F1FrozenUpperBaseCompositePositionActionCfg(A3F0UpperBaseCompositePositionActionCfg):
    class_type: type[ActionTerm] = A3F1FrozenUpperBaseCompositePositionAction
    upper_checkpoint: str = ""
    upper_observation_group: str = "upper"


class A3FrozenAnchorArmAdapterPositionAction(A3F1FrozenUpperBaseCompositePositionAction):
    """Fixed-base P0 executor: frozen anchor swing plus a 7-D arm adapter.

    The public PPO action is only the right-arm residual.  model_900 remains
    fully frozen and receives a separate observation group whose target terms
    are the nominal anchor, never the external target latch.
    """

    cfg: "A3FrozenAnchorArmAdapterPositionActionCfg"

    @property
    def action_dim(self) -> int:
        return len(self.cfg.adapter_joint_names)

    def __init__(self, cfg: "A3FrozenAnchorArmAdapterPositionActionCfg", env):
        super().__init__(cfg, env)
        if len(cfg.adapter_joint_names) != 7:
            raise ValueError("P0 target adapter requires exactly seven right-arm joints")
        unknown = sorted(set(cfg.adapter_joint_names) - set(cfg.upper_joint_names))
        if unknown:
            raise ValueError(f"Adapter joints must be in upper_joint_names: {unknown}")
        self._adapter_indices = torch.tensor(
            [cfg.upper_joint_names.index(name) for name in cfg.adapter_joint_names],
            dtype=torch.long,
            device=self.device,
        )
        self._adapter_scale = torch.tensor(
            cfg.adapter_scale_rad, dtype=torch.float, device=self.device
        ).unsqueeze(0)
        if self._adapter_scale.shape[-1] != self.action_dim:
            raise ValueError("adapter_scale_rad must contain one value per adapter joint")
        self._adapter_last = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._env.target_adapter_last_action = self._adapter_last
        self._adapter_policy_residual = torch.zeros_like(self._adapter_last)
        self._env.target_adapter_policy_residual = self._adapter_policy_residual
        self._adapter_feedforward = torch.zeros_like(self._adapter_last)
        self._env.target_adapter_feedforward = self._adapter_feedforward
        self._adapter_gate = torch.zeros((self.num_envs, 1), device=self.device)
        self._env.target_adapter_gate = self._adapter_gate
        matrix = torch.tensor(cfg.adapter_feedforward_pinv, dtype=torch.float, device=self.device)
        if matrix.shape != (self.action_dim, 3):
            raise ValueError("adapter_feedforward_pinv must have shape (7, 3)")
        self._adapter_feedforward_pinv = matrix
        matrix_by_motion = torch.tensor(
            cfg.adapter_feedforward_pinv_by_motion, dtype=torch.float, device=self.device
        )
        if matrix_by_motion.numel() == 0:
            matrix_by_motion = torch.empty((0, self.action_dim, 3), device=self.device)
        elif matrix_by_motion.ndim != 3 or matrix_by_motion.shape[1:] != (self.action_dim, 3):
            raise ValueError("adapter_feedforward_pinv_by_motion must have shape (M, 7, 3)")
        self._adapter_feedforward_pinv_by_motion = matrix_by_motion
        raw_clip_by_motion = torch.tensor(
            cfg.adapter_raw_clip_by_motion, dtype=torch.float, device=self.device
        )
        if raw_clip_by_motion.numel() > 0:
            if matrix_by_motion.shape[0] == 0 or raw_clip_by_motion.shape != (
                matrix_by_motion.shape[0],
            ):
                raise ValueError(
                    "adapter_raw_clip_by_motion must contain one value per feedforward matrix"
                )
            if torch.any(raw_clip_by_motion <= 0.0):
                raise ValueError("adapter_raw_clip_by_motion values must be positive")
        self._adapter_raw_clip_by_motion = raw_clip_by_motion
        transform = torch.tensor(cfg.adapter_feedforward_target_transform, dtype=torch.float, device=self.device)
        if transform.shape != (3, 3):
            raise ValueError("adapter_feedforward_target_transform must have shape (3, 3)")
        self._adapter_feedforward_target_transform = transform

    @property
    def adapter_last_action(self) -> torch.Tensor:
        return self._adapter_last

    @property
    def adapter_gate(self) -> torch.Tensor:
        return self._adapter_gate

    def _hit_steps(self, motion) -> torch.Tensor:
        if motion._use_motion_library:
            return motion.motion.hit_frame[motion.motion_ids].to(torch.long)
        return torch.full_like(motion.time_steps, int(motion.motion.hit_frame[0]))

    def _target_gate(self, motion) -> torch.Tensor:
        """Smooth hit-centered ramp in, active, and ramp out."""
        hit = self._hit_steps(motion)
        step = motion.time_steps.to(torch.float32)
        pre = float(self.cfg.adapter_ramp_in_steps)
        post = float(self.cfg.adapter_ramp_out_steps)
        ramp_in = ((step - (hit - pre)) / max(pre, 1.0)).clamp(0.0, 1.0)
        ramp_out = ((hit + post - step) / max(post, 1.0)).clamp(0.0, 1.0)
        smooth_in = ramp_in * ramp_in * (3.0 - 2.0 * ramp_in)
        smooth_out = ramp_out * ramp_out * (3.0 - 2.0 * ramp_out)
        return (smooth_in * smooth_out).unsqueeze(-1)

    def process_actions(self, actions: torch.Tensor):
        if actions.shape != self._raw_actions.shape or not torch.isfinite(actions).all():
            raise ValueError(f"Expected finite adapter action shape {self._raw_actions.shape}")
        motion = self._env.command_manager.get_term(self.cfg.reference_command_name)
        upper_obs = self._compute_observation_group(self._upper_observation_group)
        self._upper_last_observation.copy_(upper_obs)
        primary = self._upper_policy(upper_obs).clamp(-self.cfg.upper_raw_clip, self.cfg.upper_raw_clip)
        reference = self._upper_reference(motion, motion.time_steps)
        gate = self._target_gate(motion)
        target = self._env.command_manager.get_term("racket_target")
        delta_b = target.external_target_delta_local_b()
        delta_control = delta_b @ self._adapter_feedforward_target_transform.T
        if self._adapter_feedforward_pinv_by_motion.shape[0] > 0:
            motion_ids = motion.motion_ids.to(torch.long)
            if torch.any(motion_ids < 0) or torch.any(
                motion_ids >= self._adapter_feedforward_pinv_by_motion.shape[0]
            ):
                raise ValueError("motion_id has no target-adapter feedforward matrix")
            pinv = self._adapter_feedforward_pinv_by_motion[motion_ids]
            feedforward = torch.bmm(pinv, delta_control.unsqueeze(-1)).squeeze(-1)
        else:
            feedforward = delta_control @ self._adapter_feedforward_pinv.T
        feedforward = float(self.cfg.adapter_feedforward_gain) * feedforward
        policy_residual = float(self.cfg.adapter_policy_residual_gain) * actions
        unbounded = policy_residual + feedforward
        if self._adapter_raw_clip_by_motion.numel() > 0:
            raw_clip = self._adapter_raw_clip_by_motion[motion.motion_ids.to(torch.long)].unsqueeze(-1)
            bounded = torch.maximum(torch.minimum(unbounded, raw_clip), -raw_clip)
        else:
            bounded = torch.clamp(
                unbounded, -self.cfg.adapter_raw_clip, self.cfg.adapter_raw_clip
            )
        self._adapter_gate[:] = gate
        self._adapter_policy_residual[:] = policy_residual
        self._adapter_feedforward[:] = feedforward
        self._adapter_last[:] = bounded
        self._env.target_adapter_policy_residual[:] = policy_residual
        self._env.target_adapter_feedforward[:] = feedforward
        self._env.target_adapter_last_action[:] = bounded
        self._raw_actions[:] = bounded
        self._unbounded_actions[:] = unbounded

        # Fixed-base P0 owns no lower-body residual.  Build one complete target
        # source, then add adapter authority only to the seven arm joints.
        self._full_joint_targets[:] = self._asset.data.default_joint_pos
        self._full_joint_velocity_targets.zero_()
        primary_delta = primary * self._upper_scale
        upper = reference + primary_delta
        upper[:, self._adapter_indices] += gate * bounded * self._adapter_scale
        self._upper_reference_actions[:] = reference
        self._upper_primary_contribution[:] = primary_delta
        self._upper_coordinator_contribution.zero_()
        self._upper_processed_actions[:] = upper
        self._full_joint_targets[:, self._upper_joint_ids_tensor] = upper
        self._full_joint_velocity_targets[:, self._upper_joint_ids_tensor] = self._upper_velocity_reference(
            motion, motion.time_steps
        )
        limits = self._asset.data.soft_joint_pos_limits
        self._full_joint_targets[:] = torch.clamp(
            self._full_joint_targets, min=limits[..., 0], max=limits[..., 1]
        )
        self._upper_processed_actions[:] = self._full_joint_targets[:, self._upper_joint_ids_tensor]
        self._processed_actions[:] = gate * bounded * self._adapter_scale
        self._env.f0_upper_last_action[:] = primary


@configclass
class A3FrozenAnchorArmAdapterPositionActionCfg(A3F1FrozenUpperBaseCompositePositionActionCfg):
    class_type: type[ActionTerm] = A3FrozenAnchorArmAdapterPositionAction
    adapter_joint_names: tuple[str, ...] = ()
    adapter_scale_rad: tuple[float, ...] = (0.025,) * 7
    adapter_raw_clip: float = 1.0
    adapter_raw_clip_by_motion: tuple[float, ...] = ()
    adapter_ramp_in_steps: int = 10
    adapter_ramp_out_steps: int = 8
    # PPO is a low-authority correction around the calibrated Cartesian
    # feedforward.  A value below one prevents exploration or early updates
    # from destroying the clean motion-0 response matrix.
    adapter_policy_residual_gain: float = 0.1
    adapter_feedforward_gain: float = 1.0
    adapter_feedforward_pinv: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0),) * 7
    adapter_feedforward_pinv_by_motion: tuple[
        tuple[tuple[float, float, float], ...], ...
    ] = ()
    adapter_feedforward_target_transform: tuple[tuple[float, float, float], ...] = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


class A3FrozenStageAUpperCorrectionAction(A3F1FrozenUpperBaseCompositePositionAction):
    """Frozen model_3396 legs and model_900 swing, with PPO correction on upper joints only."""

    cfg: "A3FrozenStageAUpperCorrectionActionCfg"

    @property
    def action_dim(self) -> int:
        return len(self.cfg.upper_joint_names)

    def __init__(self, cfg: "A3FrozenStageAUpperCorrectionActionCfg", env):
        super().__init__(cfg, env)
        # Expose the controlled upper-joint indices using the same diagnostic
        # contract as the legacy composite actions.  Evaluators use this
        # tensor to report arm-specific limit margins; it does not alter the
        # command path or actor action dimension.
        self._joint_index_tensor = self._upper_joint_ids_tensor
        self._legacy_stage_a = _FrozenCheckpointActor(cfg.legacy_stage_a_checkpoint, self.device)
        if self._legacy_stage_a.obs_dim != 126 or self._legacy_stage_a.action_dim != 14:
            raise RuntimeError(
                "model_3396 contract mismatch: expected 126-D observation and 14-D action, "
                f"got ({self._legacy_stage_a.obs_dim}, {self._legacy_stage_a.action_dim})"
            )
        self._legacy_stage_a_group = str(cfg.legacy_stage_a_observation_group)
        self._correction_scale = torch.tensor(
            cfg.upper_correction_scale_rad, dtype=torch.float, device=self.device
        ).unsqueeze(0)
        if self._correction_scale.shape[-1] != self.action_dim:
            raise ValueError("upper_correction_scale_rad must contain one value per upper joint")
        self._legacy_raw = torch.zeros((self.num_envs, 14), device=self.device)
        self._legacy_bounded = torch.zeros_like(self._legacy_raw)
        # Replay-only diagnostic hook. Training keeps model_3396 unchanged;
        # scripts/play.py may set this runtime attribute for a visual replay.
        self._replay_lower_output_scale = 1.0
        env.legacy_stage_a_last_action = self._legacy_raw.clone()

    def process_actions(self, actions: torch.Tensor):
        if actions.shape != self._raw_actions.shape or not torch.isfinite(actions).all():
            raise ValueError(f"Expected finite upper correction shape {self._raw_actions.shape}")
        motion = self._env.command_manager.get_term(self.cfg.reference_command_name)
        stage_obs = self._compute_observation_group(self._legacy_stage_a_group)
        if self.cfg.legacy_stage_a_yaw_adapter:
            stage_obs = adapt_stage_a_observation_legacy_yaw_pi(stage_obs)
        self._legacy_raw[:] = self._legacy_stage_a(stage_obs)
        self._env.legacy_stage_a_last_action[:] = self._legacy_raw

        # Reproduce the evaluated leg-only Stage-A target path.  The two waist
        # channels are structurally masked, then the frozen upper owns all ten
        # native strike joints below.
        lower = self._legacy_raw * self._mask
        if self._phase_gate_base_indices:
            lengths = motion.motion.motion_lengths[motion.motion_ids].clamp(min=2)
            phase = motion.time_steps.float() / (lengths - 1).float()
            u = ((phase - self.cfg.phase_gate_start) / max(
                self.cfg.phase_gate_end - self.cfg.phase_gate_start, 1.0e-6
            )).clamp(0.0, 1.0)
            gate = self.cfg.phase_gate_min_scale + (1.0 - self.cfg.phase_gate_min_scale) * (u * u * (3.0 - 2.0 * u))
            for index in self._phase_gate_base_indices:
                lower[:, index] *= gate
        self._legacy_bounded[:] = self._bound_actions(lower)
        self._full_joint_targets[:] = self._asset.data.default_joint_pos
        self._full_joint_velocity_targets.zero_()
        base_default = self._asset.data.default_joint_pos[:, self._base_joint_ids_tensor]
        self._full_joint_targets[:, self._base_joint_ids_tensor] = base_default + self._legacy_bounded * self._scale

        upper_obs = self._compute_observation_group(self._upper_observation_group)
        self._upper_last_observation[:] = upper_obs
        primary = self._upper_policy(upper_obs).clamp(-self.cfg.upper_raw_clip, self.cfg.upper_raw_clip)
        self._unbounded_actions[:] = actions
        self._raw_actions[:] = self._bound_actions(actions)
        release = (motion.time_steps.float() / float(max(self.cfg.upper_prelude_release_steps, 1))).clamp(0.0, 1.0).unsqueeze(-1)
        in_prelude = motion.prelude_elapsed_steps < int(motion.prelude_steps)
        gate = torch.where(in_prelude.unsqueeze(-1), torch.zeros_like(release), release)
        reference = self._upper_reference(motion, motion.time_steps)
        self._upper_raw_actions[:] = primary
        primary_contribution = gate * primary * self._upper_scale
        tracker_contribution = gate * self._raw_actions * self._correction_scale
        unprojected_upper = reference + primary_contribution + tracker_contribution
        # Populate the common action-chain audit fields.  P5D uses the same
        # names as the older F0/F1 paths so a trace can prove exactly which
        # frozen prior, tracker residual and final projection reached PhysX.
        self._upper_reference_actions[:] = reference
        self._upper_primary_contribution[:] = primary_contribution
        self._upper_coordinator_contribution[:] = tracker_contribution
        self._upper_safety_override.zero_()
        self._upper_velocity_safety_override.zero_()
        self._upper_dynamic_safety_override.zero_()
        self._upper_dynamic_velocity_safety_override.zero_()
        self._upper_processed_actions[:] = unprojected_upper
        self._full_joint_targets[:, self._upper_joint_ids_tensor] = self._upper_processed_actions
        self._full_joint_velocity_targets[:, self._upper_joint_ids_tensor] = self._upper_velocity_reference(
            motion, motion.time_steps
        )
        limits = self._asset.data.soft_joint_pos_limits
        self._full_joint_targets[:] = torch.clamp(self._full_joint_targets, min=limits[..., 0], max=limits[..., 1])
        self._upper_processed_actions[:] = self._full_joint_targets[:, self._upper_joint_ids_tensor]
        self._upper_safety_override[:] = self._upper_processed_actions - unprojected_upper
        # The public P5 action contribution means the *applied*, phase-gated
        # residual.  This deliberately excludes frozen model_900 output and
        # makes a zero-residual replay auditable.
        self._processed_actions[:] = tracker_contribution
        self._env.f0_upper_last_action[:] = primary


@configclass
class A3FrozenStageAUpperCorrectionActionCfg(A3F1FrozenUpperBaseCompositePositionActionCfg):
    class_type: type[ActionTerm] = A3FrozenStageAUpperCorrectionAction
    legacy_stage_a_checkpoint: str = ""
    legacy_stage_a_observation_group: str = "stage_a"
    legacy_stage_a_yaw_adapter: bool = True
    upper_correction_scale_rad: tuple[float, ...] = (0.035,) * 10


class A3UnifiedUpperReferenceTrackerAction(A3F0UpperBaseCompositePositionAction):
    """Unified P5U tracker with a frozen lower prior and bounded residuals.

    This action deliberately does not instantiate or call ``model_900``.  The
    By default the historical Stage-A checkpoint owns the 12 lower-support
    channels (its two historical waist outputs remain masked), and the public
    actor owns ten upper residual channels.  The production P5U balance
    variant can additionally expose a bounded 12-D lower residual around that
    prior:

        q_cmd_lower = q_model_3396 + scale_lower * action_lower
        q_cmd_upper = q_ref_upper + scale * action

    The implementation keeps the common full-articulation target and safety
    bookkeeping used by the existing A3 composite actions so B0 can prove that
    the new contract is actually active before PPO is trained.
    """

    cfg: "A3UnifiedUpperReferenceTrackerActionCfg"

    @property
    def action_dim(self) -> int:
        phase_dims = {"A": 0, "B": 1, "C": 4}
        contract = str(getattr(self.cfg, "phase_contract", "A")).upper()
        if contract not in phase_dims:
            raise ValueError(f"Unsupported P5U phase contract {contract!r}; expected A, B, or C")
        lower_dims = 12 if bool(getattr(self.cfg, "lower_balance_residual_enabled", False)) else 0
        microstep_dims = 4 if bool(getattr(self.cfg, "microstep_enabled", False)) else 0
        return lower_dims + len(self.cfg.upper_joint_names) + phase_dims[contract] + microstep_dims

    def __init__(self, cfg: "A3UnifiedUpperReferenceTrackerActionCfg", env):
        # Calling the F0 base constructor initializes the reviewed Base14
        # lower-support composition and the upper reference/safety buffers,
        # without constructing the model_900 loader used by the F1 subclass.
        super().__init__(cfg, env)
        self._joint_index_tensor = self._upper_joint_ids_tensor
        self._legacy_stage_a = _FrozenCheckpointActor(cfg.legacy_stage_a_checkpoint, self.device)
        if self._legacy_stage_a.obs_dim != 126 or self._legacy_stage_a.action_dim != 14:
            raise RuntimeError(
                "model_3396 contract mismatch: expected 126-D observation and 14-D action, "
                f"got ({self._legacy_stage_a.obs_dim}, {self._legacy_stage_a.action_dim})"
            )
        self._legacy_stage_a_group = str(cfg.legacy_stage_a_observation_group)
        self._correction_scale = torch.tensor(
            cfg.upper_correction_scale_rad, dtype=torch.float, device=self.device
        ).unsqueeze(0)
        if self._correction_scale.shape[-1] != len(cfg.upper_joint_names):
            raise ValueError("upper_correction_scale_rad must contain one value per upper joint")
        self._lower_balance_enabled = bool(getattr(cfg, "lower_balance_residual_enabled", False))
        self._lower_balance_joint_ids_tensor = self._base_joint_ids_tensor[:12]
        self._lower_balance_scale = torch.tensor(
            tuple(float(value) for value in getattr(
                cfg,
                "lower_balance_correction_scale_rad",
                (0.048, 0.140, 0.184, 0.040, 0.060, 0.028) * 2,
            )),
            dtype=torch.float,
            device=self.device,
        ).unsqueeze(0)
        if self._lower_balance_enabled and self._lower_balance_scale.shape[-1] != 12:
            raise ValueError("lower_balance_correction_scale_rad must contain 12 values")
        self._lower_balance_prelude_ramp_steps = int(
            getattr(cfg, "lower_balance_prelude_ramp_steps", 50)
        )
        if self._lower_balance_prelude_ramp_steps < 0:
            raise ValueError("lower_balance_prelude_ramp_steps must be non-negative")
        self._lower_balance_contribution = torch.zeros(
            (self.num_envs, 12), dtype=torch.float, device=self.device
        )
        env.lower_balance_residual = self._lower_balance_contribution
        self._phase_contract = str(getattr(cfg, "phase_contract", "A")).upper()
        phase_limits = {
            "A": (),
            "B": (float(getattr(cfg, "global_phase_limit_steps", 4.0)),),
            "C": (
                float(getattr(cfg, "global_phase_limit_steps", 4.0)),
                float(getattr(cfg, "shoulder_phase_limit_steps", 2.0)),
                float(getattr(cfg, "elbow_phase_limit_steps", 2.0)),
                float(getattr(cfg, "wrist_phase_limit_steps", 2.0)),
            ),
        }
        if self._phase_contract not in phase_limits:
            raise ValueError(f"Unsupported P5U phase contract {self._phase_contract!r}")
        self._phase_limits = torch.tensor(phase_limits[self._phase_contract], dtype=torch.float, device=self.device).unsqueeze(0)
        self._phase_last = torch.zeros((self.num_envs, self._phase_limits.shape[-1]), device=self.device)
        self._phase_rate = torch.zeros_like(self._phase_last)
        self._phase_effective = torch.zeros_like(self._phase_last)
        self._phase_joint_offsets = torch.zeros((self.num_envs, len(cfg.upper_joint_names)), device=self.device)
        self._microstep_enabled = bool(getattr(cfg, "microstep_enabled", False))
        self._microstep_step_limit_m = float(getattr(cfg, "microstep_step_limit_m", 0.02))
        if self._microstep_enabled and not (0.005 <= self._microstep_step_limit_m <= 0.02):
            raise ValueError("microstep_step_limit_m must be in [0.005, 0.02] m")
        self._microstep_lowpass_alpha = float(getattr(cfg, "microstep_lowpass_alpha", 0.20))
        if self._microstep_enabled and not (0.0 < self._microstep_lowpass_alpha <= 1.0):
            raise ValueError("microstep_lowpass_alpha must be in (0, 1]")
        self._microstep_state = torch.zeros((self.num_envs, 4), device=self.device)
        self._microstep_joint_delta = torch.zeros((self.num_envs, 12), device=self.device)
        self._upper_effective_reference = torch.zeros((self.num_envs, len(cfg.upper_joint_names)), device=self.device)
        self._phase_joint_group = []
        for name in cfg.upper_joint_names:
            if name in {"right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint"}:
                self._phase_joint_group.append("shoulder")
            elif name == "right_elbow_joint":
                self._phase_joint_group.append("elbow")
            elif name in {"right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"}:
                self._phase_joint_group.append("wrist")
            else:
                self._phase_joint_group.append("waist")
        env.p5u_phase_action = self._phase_last
        env.p5u_phase_effective = self._phase_effective
        env.p5u_phase_rate = self._phase_rate
        env.p5u_phase_joint_offsets = self._phase_joint_offsets
        env.p5u_upper_effective_reference = self._upper_effective_reference
        env.p5u_microstep_state = self._microstep_state
        env.p5u_microstep_joint_delta = self._microstep_joint_delta
        from training.tasks.tracking.mdp.cycle_manager import CycleConfig, StrikeCycleManager
        self._fall_cycle_manager = StrikeCycleManager(
            self.num_envs,
            self.device,
            CycleConfig(post_hit_guard_steps=75, recovery_timeout_steps=250, ready_hold_steps=15),
        )
        env.fall_cycle_manager = self._fall_cycle_manager
        env.fall_cycle_phase = self._fall_cycle_manager.phase
        env.fall_cycle_guard_steps = self._fall_cycle_manager.guard_steps
        env.fall_cycle_recovery_steps = self._fall_cycle_manager.recovery_steps
        self._legacy_raw = torch.zeros((self.num_envs, 14), device=self.device)
        self._legacy_bounded = torch.zeros_like(self._legacy_raw)
        # Replay-only diagnostic hook; PPO training always uses 1.0.
        self._replay_lower_output_scale = 1.0
        env.legacy_stage_a_last_action = self._legacy_raw.clone()

    def _compute_observation_group(self, name: str) -> torch.Tensor:
        value = self._env.observation_manager.compute_group(name)
        if isinstance(value, tuple):
            value = value[0]
        if isinstance(value, dict):
            value = value.get(name, next(iter(value.values())))
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Observation group {name!r} did not return a tensor")
        return value

    def process_actions(self, actions: torch.Tensor):
        if actions.shape != self._raw_actions.shape or not torch.isfinite(actions).all():
            raise ValueError(f"Expected finite unified upper action shape {self._raw_actions.shape}")
        motion = self._env.command_manager.get_term(self.cfg.reference_command_name)
        from training.tasks.tracking.mdp.fall_state import unified_fall_state
        physical = unified_fall_state(self._env)
        if motion._use_motion_library:
            cycle_lengths = motion.motion.motion_lengths[motion.motion_ids]
            cycle_hit = motion.motion.hit_frame[motion.motion_ids]
        else:
            cycle_lengths = torch.full_like(motion.time_steps, motion.motion.time_step_total)
            cycle_hit = torch.full_like(motion.time_steps, int(motion.motion.hit_frame[0]))
        cycle_prelude = motion.prelude_elapsed_steps < int(motion.prelude_steps)
        cycle_done = (motion.tail_steps > 0) | (motion.time_steps >= cycle_lengths - 1)
        cycle_active = (~cycle_prelude) & (~cycle_done)
        cycle_hit_window = cycle_active & (torch.abs(motion.time_steps - cycle_hit) <= 3)
        cycle = self._fall_cycle_manager.update(
            prelude_active=cycle_prelude, strike_active=cycle_active,
            hit_window=cycle_hit_window, motion_done=cycle_done,
            recovery_ready=physical.recovery_ready,
            confirmed_fall=physical.confirmed_fall,
            predicted_unrecoverable=physical.predicted_unrecoverable,
            timeout=torch.zeros_like(physical.confirmed_fall),
            strike_pass=~physical.confirmed_fall,
            episode_step=self._env.episode_length_buf.to(dtype=torch.long),
        )
        self._env.fall_cycle_phase = cycle["cycle_phase"]
        self._env.fall_cycle_guard_steps = cycle["guard_steps"]
        self._env.fall_cycle_recovery_steps = cycle["recovery_steps"]
        bounded_actions = self._bound_actions(actions)
        lower_dim = 12 if self._lower_balance_enabled else 0
        upper_dim = len(self.cfg.upper_joint_names)
        phase_dim = self._phase_limits.shape[-1]
        lower_actions = bounded_actions[:, :lower_dim]
        upper_start = lower_dim
        residual_actions = bounded_actions[:, upper_start : upper_start + upper_dim]
        phase_start = upper_start + upper_dim
        phase_actions = bounded_actions[:, phase_start : phase_start + phase_dim]
        microstep_actions = (
            bounded_actions[:, phase_start + phase_dim : phase_start + phase_dim + 4]
            if self._microstep_enabled else None
        )
        if self._phase_limits.shape[-1] > 0:
            requested_phase = phase_actions * self._phase_limits
            previous_phase = self._phase_last.clone()
            alpha = float(getattr(self.cfg, "phase_lowpass_alpha", 0.25))
            alpha = max(0.0, min(1.0, alpha))
            phase = (1.0 - alpha) * self._phase_last + alpha * requested_phase
            in_prelude = motion.prelude_elapsed_steps < int(motion.prelude_steps)
            phase = torch.where(in_prelude.unsqueeze(-1), torch.zeros_like(phase), phase)
            self._phase_effective[:] = phase
            self._phase_rate[:] = phase - previous_phase
            self._phase_last[:] = phase
            self._phase_joint_offsets.zero_()
            self._phase_joint_offsets[:] = phase[:, :1]
            if self._phase_contract == "C":
                for i, group in enumerate(("shoulder", "elbow", "wrist"), start=1):
                    self._phase_joint_offsets[:, [j for j, g in enumerate(self._phase_joint_group) if g == group]] += phase[:, i : i + 1]
        else:
            self._phase_last.zero_()
            self._phase_effective.zero_()
            self._phase_joint_offsets.zero_()
            self._phase_rate.zero_()
        self._unbounded_actions[:] = actions
        self._raw_actions[:] = bounded_actions

        # Frozen model_3396 is the only historical policy called here.  Its
        # waist channels are masked by the reviewed Base14 split.
        stage_obs = self._compute_observation_group(self._legacy_stage_a_group)
        if self.cfg.legacy_stage_a_yaw_adapter:
            stage_obs = adapt_stage_a_observation_legacy_yaw_pi(stage_obs)
        self._legacy_raw[:] = self._legacy_stage_a(stage_obs)
        self._env.legacy_stage_a_last_action[:] = self._legacy_raw
        lower = self._legacy_raw * self._mask
        replay_lower_scale = float(getattr(self, "_replay_lower_output_scale", 1.0))
        if replay_lower_scale != 1.0:
            lower = lower * replay_lower_scale
        if self._phase_gate_base_indices:
            lengths = motion.motion.motion_lengths[motion.motion_ids].clamp(min=2)
            phase = motion.time_steps.float() / (lengths - 1).float()
            u = ((phase - self.cfg.phase_gate_start) / max(
                self.cfg.phase_gate_end - self.cfg.phase_gate_start, 1.0e-6
            )).clamp(0.0, 1.0)
            gate = self.cfg.phase_gate_min_scale + (1.0 - self.cfg.phase_gate_min_scale) * (
                u * u * (3.0 - 2.0 * u)
            )
            for index in self._phase_gate_base_indices:
                lower[:, index] *= gate
        self._legacy_bounded[:] = self._bound_actions(lower)

        # The learned balance residual is deliberately additive to the
        # frozen Stage-A leg target.  Ramp it in over the READY prelude so a
        # randomly initialized PPO policy cannot kick the feet at reset, while
        # still giving the learner authority before and after the strike.
        self._lower_balance_contribution.zero_()
        if self._lower_balance_enabled:
            if self._lower_balance_prelude_ramp_steps > 0:
                ramp = (
                    motion.prelude_elapsed_steps.to(dtype=torch.float32)
                    / float(self._lower_balance_prelude_ramp_steps)
                ).clamp(0.0, 1.0)
            else:
                ramp = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
            self._lower_balance_contribution[:] = (
                lower_actions * self._lower_balance_scale * ramp.unsqueeze(-1)
            )

        # Optional bounded foot micro-adjustment.  This is deliberately a
        # residual around model_3396, not a second leg policy: four channels
        # represent left/right sagittal and lateral intent, low-pass filtered
        # and projected to a coupled hip/knee/ankle pattern.  The 2 cm limit is
        # a conservative kinematic envelope; a later SE(3)+IK adapter can
        # replace this projection without changing the public contract.
        self._microstep_joint_delta.zero_()
        if self._microstep_enabled:
            new_episode = self._env.episode_length_buf <= 0
            self._microstep_state[new_episode] = 0.0
            alpha = self._microstep_lowpass_alpha
            self._microstep_state[:] = (1.0 - alpha) * self._microstep_state + alpha * microstep_actions
            # Scale linearly from the 2 cm reviewed envelope.  Values are
            # joint radians and keep the sole approximately level while the
            # foot makes a small corrective translation.
            s = self._microstep_step_limit_m / 0.02
            left_x, left_y, right_x, right_y = self._microstep_state.unbind(dim=-1)
            self._microstep_joint_delta[:, 0] = -0.055 * s * left_x
            self._microstep_joint_delta[:, 3] =  0.025 * s * left_x
            self._microstep_joint_delta[:, 4] =  0.030 * s * left_x
            self._microstep_joint_delta[:, 1] =  0.045 * s * left_y
            self._microstep_joint_delta[:, 5] = -0.040 * s * left_y
            self._microstep_joint_delta[:, 6] = -0.055 * s * right_x
            self._microstep_joint_delta[:, 9] =  0.025 * s * right_x
            self._microstep_joint_delta[:, 10] = 0.030 * s * right_x
            self._microstep_joint_delta[:, 7] = 0.045 * s * right_y
            self._microstep_joint_delta[:, 11] = -0.040 * s * right_y

        self._full_joint_targets[:] = self._asset.data.default_joint_pos
        self._full_joint_velocity_targets.zero_()
        base_default = self._asset.data.default_joint_pos[:, self._base_joint_ids_tensor]
        lower_targets = base_default + self._legacy_bounded * self._scale
        if self._lower_balance_enabled:
            lower_targets[:, :12] += self._lower_balance_contribution
        if self._microstep_enabled:
            lower_targets[:, :12] += self._microstep_joint_delta
        self._full_joint_targets[:, self._base_joint_ids_tensor] = lower_targets

        reference = self._upper_reference(motion, motion.time_steps)
        if self._phase_limits.shape[-1] > 0:
            reference = self._sample_upper_motion(
                motion,
                motion.time_steps.float().unsqueeze(-1) + self._phase_joint_offsets,
            )
        self._upper_effective_reference[:] = reference
        tracker_contribution = residual_actions * self._correction_scale
        unprojected_upper = reference + tracker_contribution
        self._upper_reference_actions[:] = reference
        self._upper_primary_contribution.zero_()
        self._upper_coordinator_contribution[:] = tracker_contribution
        self._upper_safety_override.zero_()
        self._upper_velocity_safety_override.zero_()
        self._upper_dynamic_safety_override.zero_()
        self._upper_dynamic_velocity_safety_override.zero_()
        self._upper_processed_actions[:] = unprojected_upper
        self._full_joint_targets[:, self._upper_joint_ids_tensor] = unprojected_upper
        self._full_joint_velocity_targets[:, self._upper_joint_ids_tensor] = self._upper_velocity_reference(
            motion, motion.time_steps
        )
        self._apply_waist_soft_limit_guard(motion, motion.time_steps)
        self._apply_upper_dynamic_soft_limit_guard()
        self._apply_passive_soft_limit_guard()
        limits = self._asset.data.soft_joint_pos_limits
        self._full_joint_targets[:] = torch.clamp(
            self._full_joint_targets, min=limits[..., 0], max=limits[..., 1]
        )
        self._upper_processed_actions[:] = self._full_joint_targets[:, self._upper_joint_ids_tensor]
        self._upper_safety_override[:] = self._upper_processed_actions - unprojected_upper
        self._processed_actions[:] = bounded_actions
        self._env.f0_upper_last_action[:] = torch.zeros_like(self._upper_raw_actions)


@configclass
class A3UnifiedUpperReferenceTrackerActionCfg(A3F0UpperBaseCompositePositionActionCfg):
    """Configuration for the model900-disabled P5U action contract."""

    class_type: type[ActionTerm] = A3UnifiedUpperReferenceTrackerAction
    legacy_stage_a_checkpoint: str = ""
    legacy_stage_a_observation_group: str = "stage_a"
    legacy_stage_a_yaw_adapter: bool = True
    upper_correction_scale_rad: tuple[float, ...] = (0.035,) * 10
    # Optional full lower-body balance residual around model_3396.  The
    # NoAssist production task enables this explicitly; legacy P5U variants
    # remain 10-D upper-only unless they opt in.
    lower_balance_residual_enabled: bool = False
    # Deliberately aggressive balance authority around model_3396.  The action
    # term still applies the immutable raw_clip=0.50 envelope, so the
    # effective peak is approximately half of each value below (about 4x the
    # original effective residual), followed by soft-joint-limit clipping.
    lower_balance_correction_scale_rad: tuple[float, ...] = (
        0.048, 0.140, 0.184, 0.040, 0.060, 0.028,
        0.048, 0.140, 0.184, 0.040, 0.060, 0.028,
    )
    lower_balance_prelude_ramp_steps: int = 50
    phase_contract: str = "A"
    global_phase_limit_steps: float = 4.0
    shoulder_phase_limit_steps: float = 2.0
    elbow_phase_limit_steps: float = 2.0
    wrist_phase_limit_steps: float = 2.0
    phase_lowpass_alpha: float = 0.25
    # Disabled for the original P5U contract.  The microstep task enables
    # this four-channel, 2 cm maximum lower-body residual explicitly.
    microstep_enabled: bool = False
    microstep_step_limit_m: float = 0.02
    microstep_lowpass_alpha: float = 0.20


class A3FrozenStageAJointCoordinatorAction(A3F1FrozenUpperBaseCompositePositionAction):
    """Freeze both parent actors and train one 12-leg/3-waist/7-arm coordinator.

    The historical Stage-A checkpoint keeps ownership of its twelve leg
    residuals only.  Its two legacy waist outputs are permanently masked.
    model_900 keeps supplying the lead-compensated waist/right-arm strike
    prior.  PPO publishes one small correction vector around those two
    contracts, so every final joint target has exactly one execution path.
    """

    cfg: "A3FrozenStageAJointCoordinatorActionCfg"

    @property
    def action_dim(self) -> int:
        return 22

    def __init__(self, cfg: "A3FrozenStageAJointCoordinatorActionCfg", env):
        super().__init__(cfg, env)
        self._legacy_stage_a = _FrozenCheckpointActor(cfg.legacy_stage_a_checkpoint, self.device)
        if self._legacy_stage_a.obs_dim != 126 or self._legacy_stage_a.action_dim != 14:
            raise RuntimeError(
                "model_3396 contract mismatch: expected 126-D observation and 14-D action, "
                f"got ({self._legacy_stage_a.obs_dim}, {self._legacy_stage_a.action_dim})"
            )
        if len(self._base_joint_ids) != 14 or len(self._upper_joint_ids) != 10:
            raise RuntimeError("Joint coordinator requires the reviewed Base14 and upper10 joint contracts")
        if len(cfg.leg_correction_scale_rad) != 12:
            raise ValueError("leg_correction_scale_rad must contain 12 values")
        if len(cfg.waist_correction_scale_rad) != 3:
            raise ValueError("waist_correction_scale_rad must contain 3 values")
        if len(cfg.arm_correction_scale_rad) != 7:
            raise ValueError("arm_correction_scale_rad must contain 7 values")

        self._legacy_stage_a_group = str(cfg.legacy_stage_a_observation_group)
        self._leg_correction_scale = torch.tensor(
            cfg.leg_correction_scale_rad, dtype=torch.float, device=self.device
        ).unsqueeze(0)
        self._upper_correction_scale = torch.tensor(
            (*cfg.waist_correction_scale_rad, *cfg.arm_correction_scale_rad),
            dtype=torch.float,
            device=self.device,
        ).unsqueeze(0)
        self._legacy_raw = torch.zeros((self.num_envs, 14), device=self.device)
        self._legacy_bounded = torch.zeros_like(self._legacy_raw)
        self._leg_joint_ids_tensor = self._base_joint_ids_tensor[:12]
        self._processed_actions = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        # Per-episode latch for retiring the frozen Stage-A sagittal bracing
        # residual after it has arrested the forward swing disturbance.
        self._stage_a_exit_state = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._stage_a_exit_positive_steps = torch.zeros_like(self._stage_a_exit_state)
        self._stage_a_exit_positive_confirmed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._stage_a_exit_neutral_steps = torch.zeros_like(self._stage_a_exit_state)
        self._stage_a_exit_decay_elapsed = torch.zeros_like(self._stage_a_exit_state)
        self._stage_a_exit_last_episode_step = torch.full_like(
            self._stage_a_exit_state, -1
        )
        self._stage_a_exit_scale = torch.ones(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self._stage_a_front_gain = torch.ones(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self._stage_a_exit_trigger_step = torch.full_like(
            self._stage_a_exit_state, -1
        )
        self._stage_a_rearm_stable_steps = torch.zeros_like(
            self._stage_a_exit_state
        )
        self._stage_a_rearm_ramp_elapsed = torch.zeros_like(
            self._stage_a_exit_state
        )
        self._stage_a_rearm_last_shot_cycle = torch.zeros_like(
            self._stage_a_exit_state
        )
        self._stage_a_rearm_rejected = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._stage_a_rearm_ready = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._stage_a_rearm_stable = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        from training.tasks.tracking.mdp.cycle_manager import CycleConfig, StrikeCycleManager
        self._fall_cycle_manager = StrikeCycleManager(
            self.num_envs,
            self.device,
            CycleConfig(
                post_hit_guard_steps=75,
                recovery_timeout_steps=250,
                ready_hold_steps=15,
            ),
        )
        env.fall_cycle_manager = self._fall_cycle_manager
        env.fall_cycle_phase = self._fall_cycle_manager.phase
        env.fall_cycle_guard_steps = self._fall_cycle_manager.guard_steps
        env.fall_cycle_recovery_steps = self._fall_cycle_manager.recovery_steps
        env.stage_a_sagittal_exit_scale = self._stage_a_exit_scale
        env.stage_a_sagittal_exit_state = self._stage_a_exit_state
        env.stage_a_sagittal_exit_trigger_step = self._stage_a_exit_trigger_step
        env.stage_a_sagittal_front_gain = self._stage_a_front_gain
        env.stage_a_sagittal_rearm_ready = self._stage_a_rearm_ready
        env.stage_a_sagittal_rearm_stable = self._stage_a_rearm_stable
        env.stage_a_sagittal_rearm_stable_steps = self._stage_a_rearm_stable_steps
        env.stage_a_sagittal_rearm_rejected = self._stage_a_rearm_rejected
        env.legacy_stage_a_last_action = self._legacy_raw.clone()
        env.joint_coordinator_last_action = torch.zeros((self.num_envs, self.action_dim), device=self.device)

    def export_v29_rsi_state(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Export V25/V26 latches, action history, and target caches for V29 RSI."""
        ids = env_ids.to(device=self.device, dtype=torch.long).flatten()
        fields = (
            "_raw_actions", "_unbounded_actions", "_processed_actions",
            "_full_joint_targets", "_full_joint_velocity_targets",
            "_legacy_raw", "_legacy_bounded", "_upper_raw_actions",
            "_upper_processed_actions", "_upper_reference_actions",
            "_upper_primary_contribution", "_upper_coordinator_contribution",
            "_upper_safety_override", "_upper_velocity_safety_override",
            "_upper_dynamic_safety_override",
            "_upper_dynamic_velocity_safety_override",
            "_upper_velocity_targets",
            "_stage_a_exit_state", "_stage_a_exit_positive_steps",
            "_stage_a_exit_positive_confirmed", "_stage_a_exit_neutral_steps",
            "_stage_a_exit_decay_elapsed", "_stage_a_exit_last_episode_step",
            "_stage_a_exit_scale", "_stage_a_front_gain",
            "_stage_a_exit_trigger_step", "_stage_a_rearm_stable_steps",
            "_stage_a_rearm_ramp_elapsed", "_stage_a_rearm_last_shot_cycle",
            "_stage_a_rearm_rejected", "_stage_a_rearm_ready", "_stage_a_rearm_stable",
        )
        payload = {
            "schema_version": torch.tensor(3, dtype=torch.int64),
            "snapshot_phase": "post_physics_pre_observation",
            **{name: getattr(self, name)[ids].detach().clone() for name in fields},
            "env_legacy_stage_a_last_action": self._env.legacy_stage_a_last_action[ids].detach().clone(),
            "env_joint_coordinator_last_action": self._env.joint_coordinator_last_action[ids].detach().clone(),
            "env_f0_upper_last_action": self._env.f0_upper_last_action[ids].detach().clone(),
        }
        return payload

    def restore_v29_rsi_state(self, state: dict[str, torch.Tensor], env_ids: torch.Tensor) -> None:
        """Restore an exact V29 action/controller snapshot without inferring latches."""
        ids = env_ids.to(device=self.device, dtype=torch.long).flatten()
        if (
            int(state.get("schema_version", torch.tensor(-1)).item()) != 3
            or state.get("snapshot_phase") != "post_physics_pre_observation"
        ):
            raise ValueError("Unsupported V29 action RSI snapshot schema")
        mapping = {
            "env_legacy_stage_a_last_action": self._env.legacy_stage_a_last_action,
            "env_joint_coordinator_last_action": self._env.joint_coordinator_last_action,
            "env_f0_upper_last_action": self._env.f0_upper_last_action,
        }
        for name, value in state.items():
            if name in {"schema_version", "snapshot_phase"}:
                continue
            target = mapping.get(name, getattr(self, name, None))
            if target is None or value.shape[0] != ids.numel():
                raise ValueError(f"Invalid V29 action RSI field {name!r}")
            target[ids] = value.to(device=self.device, dtype=target.dtype)

    def _stage_a_rearm_stability(self, motion, support) -> torch.Tensor:
        """Return the fail-closed settled condition for another strike."""
        from training.tasks.tracking.mdp.fall_state import unified_fall_state

        physical = unified_fall_state(self._env)
        center_half_width = float(
            getattr(self.cfg, "stage_a_sagittal_rearm_center_half_width_m", 0.05)
        )
        velocity_max = float(
            getattr(self.cfg, "stage_a_sagittal_rearm_velocity_max_mps", 0.06)
        )
        pitch_rate_max = float(
            getattr(self.cfg, "stage_a_sagittal_rearm_pitch_rate_max_radps", 0.10)
        )
        tilt_max = float(
            getattr(self.cfg, "stage_a_sagittal_rearm_tilt_max_rad", 0.10)
        )
        arm_error_max = float(
            getattr(self.cfg, "stage_a_sagittal_rearm_arm_error_max_rad", 0.15)
        )
        arm_velocity_max = float(
            getattr(
                self.cfg,
                "stage_a_sagittal_rearm_arm_velocity_max_radps",
                0.0,
            )
        )
        if min(
            center_half_width,
            velocity_max,
            pitch_rate_max,
            tilt_max,
            arm_error_max,
        ) <= 0.0:
            raise ValueError("Invalid Stage-A sagittal re-arm stability contract")
        if arm_velocity_max < 0.0:
            raise ValueError(
                "stage_a_sagittal_rearm_arm_velocity_max_radps must be non-negative"
            )

        gravity_b = self._asset.data.projected_gravity_b
        tilt = torch.acos(torch.clamp(-gravity_b[:, 2], -1.0, 1.0))
        arm_ids = self._upper_joint_ids_tensor[3:]
        arm_error = torch.max(
            torch.abs(
                self._asset.data.joint_pos[:, arm_ids]
                - motion.ready_joint_pos[:, arm_ids]
            ),
            dim=-1,
        ).values
        arm_velocity_ok = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        if arm_velocity_max > 0.0:
            arm_velocity = torch.max(
                torch.abs(self._asset.data.joint_vel[:, arm_ids]), dim=-1
            ).values
            arm_velocity_ok = arm_velocity <= arm_velocity_max
        ready_reference = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        if bool(
            getattr(
                self.cfg,
                "stage_a_sagittal_rearm_require_ready_reference",
                True,
            )
        ):
            ready_start = (
                int(motion.cfg.hold_last_frame_steps)
                + int(motion.return_to_default_steps)
            )
            ready_reference = motion.tail_steps >= ready_start
        return (
            ready_reference
            & (~physical.confirmed_fall)
            & (~physical.predicted_unrecoverable)
            & physical.foot_contact.all(dim=-1)
            & (~physical.illegal_body_contact)
            & (physical.support_margins.min(dim=-1).values > 0.0)
            & support["contacts"].all(dim=-1)
            & (
                torch.abs(support["capture_rel_support_x_b"])
                <= center_half_width
            )
            & (torch.abs(self._asset.data.root_lin_vel_b[:, 0]) <= velocity_max)
            & (
                torch.abs(self._asset.data.root_ang_vel_b[:, 1])
                <= pitch_rate_max
            )
            & (tilt <= tilt_max)
            & (arm_error <= arm_error_max)
            & arm_velocity_ok
        )

    def _stage_a_sagittal_exit_gate(self, motion) -> torch.Tensor:
        """Return a latched Stage-A sagittal exit gate for the current swing.

        Stage-A remains fully active while it counters the forward swing
        impulse. Once the body has crossed its velocity zero point near the
        support center, the same sagittal residual becomes a rearward bias;
        retire only those six leg channels with a finite smooth transition.
        """
        if not bool(getattr(self.cfg, "stage_a_sagittal_exit_enabled", False)):
            self._stage_a_exit_scale.fill_(1.0)
            return self._stage_a_exit_scale

        decay_steps = int(getattr(self.cfg, "stage_a_sagittal_exit_decay_steps", 5))
        positive_confirm_steps = int(
            getattr(self.cfg, "stage_a_sagittal_exit_positive_confirm_steps", 2)
        )
        neutral_confirm_steps = int(
            getattr(self.cfg, "stage_a_sagittal_exit_neutral_confirm_steps", 2)
        )
        center_half_width = float(
            getattr(self.cfg, "stage_a_sagittal_exit_center_half_width_m", 0.04)
        )
        velocity_deadband = float(
            getattr(self.cfg, "stage_a_sagittal_exit_velocity_deadband_mps", 0.03)
        )
        if (
            decay_steps < 1
            or positive_confirm_steps < 1
            or neutral_confirm_steps < 1
            or center_half_width <= 0.0
            or velocity_deadband < 0.0
        ):
            raise ValueError("Invalid Stage-A sagittal exit contract")

        episode_step = getattr(self._env, "episode_length_buf", None)
        if episode_step is None:
            episode_step = torch.zeros_like(self._stage_a_exit_state)
        reset = episode_step < self._stage_a_exit_last_episode_step
        if reset.any():
            self._fall_cycle_manager.reset(torch.where(reset)[0])
            self._stage_a_exit_state[reset] = 0
            self._stage_a_exit_positive_steps[reset] = 0
            self._stage_a_exit_positive_confirmed[reset] = False
            self._stage_a_exit_neutral_steps[reset] = 0
            self._stage_a_exit_decay_elapsed[reset] = 0
            self._stage_a_exit_trigger_step[reset] = -1
            self._stage_a_rearm_stable_steps[reset] = 0
            self._stage_a_rearm_ramp_elapsed[reset] = 0
            self._stage_a_rearm_rejected[reset] = False
            self._stage_a_rearm_ready[reset] = False
            self._stage_a_rearm_stable[reset] = False
            self._stage_a_rearm_last_shot_cycle[reset] = motion.shot_cycle[reset]
        self._stage_a_exit_last_episode_step[:] = episode_step

        from training.tasks.tracking.mdp.observations import stagger_support_state

        support = stagger_support_state(self._env)
        from training.tasks.tracking.mdp.fall_state import unified_fall_state
        physical = unified_fall_state(self._env)
        # A physical fall or predicted unrecoverable state revokes all
        # re-arm/exit latches immediately; a finite motion tail must not turn
        # it into a nominal next-action transition.
        failed_physical = physical.confirmed_fall | physical.predicted_unrecoverable
        self._stage_a_rearm_ready[failed_physical] = False
        self._stage_a_rearm_stable[failed_physical] = False
        self._stage_a_rearm_rejected[failed_physical] = True
        hit = self._motion_hit_steps(motion)
        if motion._use_motion_library:
            motion_length = motion.motion.motion_lengths[motion.motion_ids]
        else:
            motion_length = torch.full_like(motion.time_steps, motion.motion.time_step_total)
        prelude_active = motion.prelude_elapsed_steps < int(motion.prelude_steps)
        motion_done = (motion.tail_steps > 0) | (motion.time_steps >= motion_length - 1)
        strike_active = (~prelude_active) & (~motion_done)
        hit_window = strike_active & (torch.abs(motion.time_steps - hit) <= 3)
        cycle = self._fall_cycle_manager.update(
            prelude_active=prelude_active,
            strike_active=strike_active,
            hit_window=hit_window,
            motion_done=motion_done,
            recovery_ready=physical.recovery_ready,
            confirmed_fall=physical.confirmed_fall,
            predicted_unrecoverable=physical.predicted_unrecoverable,
            timeout=torch.zeros_like(physical.confirmed_fall),
            strike_pass=~physical.confirmed_fall,
            episode_step=self._env.episode_length_buf.to(dtype=torch.long),
        )
        self._env.fall_cycle_phase = cycle["cycle_phase"]
        self._env.fall_cycle_guard_steps = cycle["guard_steps"]
        self._env.fall_cycle_recovery_steps = cycle["recovery_steps"]
        rearm_enabled = bool(
            getattr(self.cfg, "stage_a_sagittal_rearm_enabled", False)
        )
        if rearm_enabled:
            stable_hold_steps = int(
                getattr(self.cfg, "stage_a_sagittal_rearm_stable_steps", 20)
            )
            ramp_steps = int(
                getattr(self.cfg, "stage_a_sagittal_rearm_ramp_steps", 8)
            )
            if stable_hold_steps < 1 or ramp_steps < 1:
                raise ValueError("Stage-A sagittal re-arm steps must be positive")
            # A command cycle change is the authoritative launch event.  Test
            # it against the READY latch before evaluating stability against
            # the new prelude reference, which is intentionally no longer in
            # the previous shot's ready-hold phase.
            new_shot = (
                motion.shot_cycle != self._stage_a_rearm_last_shot_cycle
            )
            accepted = new_shot & (self._stage_a_exit_state == 3)
            rejected = new_shot & (~accepted)
            self._stage_a_rearm_rejected |= rejected
            self._stage_a_exit_state = torch.where(
                accepted,
                torch.full_like(self._stage_a_exit_state, 4),
                self._stage_a_exit_state,
            )
            stable = self._stage_a_rearm_stability(motion, support)
            self._stage_a_rearm_stable[:] = stable
            settled = (self._stage_a_exit_state == 2) & (~new_shot)
            self._stage_a_rearm_stable_steps = torch.where(
                settled & stable,
                self._stage_a_rearm_stable_steps + 1,
                torch.where(
                    settled,
                    torch.zeros_like(self._stage_a_rearm_stable_steps),
                    self._stage_a_rearm_stable_steps,
                ),
            )
            become_ready = settled & (
                self._stage_a_rearm_stable_steps >= stable_hold_steps
            )
            self._stage_a_exit_state = torch.where(
                become_ready,
                torch.full_like(self._stage_a_exit_state, 3),
                self._stage_a_exit_state,
            )
            # READY is revocable until a new shot begins.  This prevents a
            # delayed disturbance from being hidden by an old stable window.
            lose_ready = (
                (self._stage_a_exit_state == 3) & (~stable) & (~new_shot)
            )
            self._stage_a_exit_state = torch.where(
                lose_ready,
                torch.full_like(self._stage_a_exit_state, 2),
                self._stage_a_exit_state,
            )
            self._stage_a_rearm_stable_steps = torch.where(
                lose_ready,
                torch.zeros_like(self._stage_a_rearm_stable_steps),
                self._stage_a_rearm_stable_steps,
            )
            self._stage_a_rearm_ramp_elapsed = torch.where(
                accepted,
                torch.zeros_like(self._stage_a_rearm_ramp_elapsed),
                self._stage_a_rearm_ramp_elapsed,
            )
            self._stage_a_exit_positive_steps = torch.where(
                accepted,
                torch.zeros_like(self._stage_a_exit_positive_steps),
                self._stage_a_exit_positive_steps,
            )
            self._stage_a_exit_positive_confirmed &= ~accepted
            self._stage_a_exit_neutral_steps = torch.where(
                accepted,
                torch.zeros_like(self._stage_a_exit_neutral_steps),
                self._stage_a_exit_neutral_steps,
            )
            self._stage_a_exit_decay_elapsed = torch.where(
                accepted,
                torch.zeros_like(self._stage_a_exit_decay_elapsed),
                self._stage_a_exit_decay_elapsed,
            )
            self._stage_a_exit_trigger_step = torch.where(
                accepted,
                torch.full_like(self._stage_a_exit_trigger_step, -1),
                self._stage_a_exit_trigger_step,
            )
            self._stage_a_rearm_last_shot_cycle[:] = motion.shot_cycle
            ramping = self._stage_a_exit_state == 4
            self._stage_a_rearm_ramp_elapsed = torch.where(
                ramping,
                self._stage_a_rearm_ramp_elapsed + 1,
                self._stage_a_rearm_ramp_elapsed,
            )
            ramp_u = (
                self._stage_a_rearm_ramp_elapsed.float() / float(ramp_steps)
            ).clamp(0.0, 1.0)
            ramp_scale = ramp_u * ramp_u * (3.0 - 2.0 * ramp_u)
            self._stage_a_exit_scale[:] = torch.where(
                ramping, ramp_scale, self._stage_a_exit_scale
            )
            ramp_complete = ramping & (
                self._stage_a_rearm_ramp_elapsed >= ramp_steps
            )
            self._stage_a_exit_state = torch.where(
                ramp_complete,
                torch.zeros_like(self._stage_a_exit_state),
                self._stage_a_exit_state,
            )
            self._stage_a_exit_scale[ramp_complete] = 1.0
            self._stage_a_rearm_ready[:] = self._stage_a_exit_state == 3
        else:
            self._stage_a_rearm_ready.zero_()
            self._stage_a_rearm_stable.zero_()

        hit = self._motion_hit_steps(motion)
        in_prelude = motion.prelude_elapsed_steps < int(motion.prelude_steps)
        post_hit = (~in_prelude) & (
            (motion.time_steps >= hit) | (motion.tail_steps > 0)
        )
        both_feet = support["contacts"].all(dim=-1)
        require_both_feet = bool(
            getattr(self.cfg, "stage_a_sagittal_exit_require_both_feet", True)
        )
        contact_ok = both_feet if require_both_feet else torch.ones_like(both_feet)
        forward_velocity = self._asset.data.root_lin_vel_b[:, 0]
        positive = post_hit & contact_ok & (forward_velocity > velocity_deadband)
        active = self._stage_a_exit_state == 0
        self._stage_a_exit_positive_steps = torch.where(
            active & positive,
            self._stage_a_exit_positive_steps + 1,
            torch.where(active, torch.zeros_like(self._stage_a_exit_positive_steps), self._stage_a_exit_positive_steps),
        )
        self._stage_a_exit_positive_confirmed |= (
            active
            & (
                self._stage_a_exit_positive_steps
                >= positive_confirm_steps
            )
        )
        centered = torch.abs(support["capture_rel_support_x_b"]) <= center_half_width
        neutral = (
            active
            & post_hit
            & contact_ok
            & self._stage_a_exit_positive_confirmed
            & centered
            & (forward_velocity <= velocity_deadband)
        )
        self._stage_a_exit_neutral_steps = torch.where(
            neutral,
            self._stage_a_exit_neutral_steps + 1,
            torch.where(active, torch.zeros_like(self._stage_a_exit_neutral_steps), self._stage_a_exit_neutral_steps),
        )
        trigger = active & (
            self._stage_a_exit_neutral_steps >= neutral_confirm_steps
        )
        self._stage_a_exit_state = torch.where(
            trigger,
            torch.ones_like(self._stage_a_exit_state),
            self._stage_a_exit_state,
        )
        self._stage_a_exit_trigger_step = torch.where(
            trigger,
            episode_step.to(dtype=torch.long),
            self._stage_a_exit_trigger_step,
        )

        decaying = self._stage_a_exit_state == 1
        self._stage_a_exit_decay_elapsed = torch.where(
            decaying,
            self._stage_a_exit_decay_elapsed + 1,
            self._stage_a_exit_decay_elapsed,
        )
        u = (
            self._stage_a_exit_decay_elapsed.to(dtype=torch.float)
            / float(decay_steps)
        ).clamp(0.0, 1.0)
        smooth = u * u * (3.0 - 2.0 * u)
        ordinary_exit = (
            (self._stage_a_exit_state == 1)
            | (self._stage_a_exit_state == 2)
            | (self._stage_a_exit_state == 3)
        )
        self._stage_a_exit_scale[:] = torch.where(
            ordinary_exit,
            1.0 - smooth,
            self._stage_a_exit_scale,
        )
        self._stage_a_exit_scale[:] = torch.where(
            self._stage_a_exit_state == 0,
            torch.ones_like(self._stage_a_exit_scale),
            self._stage_a_exit_scale,
        )
        complete = decaying & (self._stage_a_exit_decay_elapsed >= decay_steps)
        self._stage_a_exit_state = torch.where(
            complete,
            torch.full_like(self._stage_a_exit_state, 2),
            self._stage_a_exit_state,
        )
        self._stage_a_exit_scale[complete] = 0.0
        self._stage_a_rearm_ready[failed_physical] = False
        self._stage_a_rearm_stable[failed_physical] = False
        # ``torch.where`` above replaces the state tensors.  Refresh the
        # environment handles so audit traces observe the live latched state,
        # not the zero-filled tensors created during initialization.
        self._env.stage_a_sagittal_exit_scale = self._stage_a_exit_scale
        self._env.stage_a_sagittal_exit_state = self._stage_a_exit_state
        self._env.stage_a_sagittal_exit_trigger_step = self._stage_a_exit_trigger_step
        self._env.stage_a_sagittal_rearm_ready = self._stage_a_rearm_ready
        self._env.stage_a_sagittal_rearm_stable = self._stage_a_rearm_stable
        self._env.stage_a_sagittal_rearm_stable_steps = self._stage_a_rearm_stable_steps
        self._env.stage_a_sagittal_rearm_rejected = self._stage_a_rearm_rejected
        return self._stage_a_exit_scale

    def _stage_a_sagittal_front_gate(self, motion) -> torch.Tensor:
        """Boost sagittal Stage-A support only for observable front-side risk.

        The gain is applied before the frozen policy action is bounded, so it
        cannot exceed the established physical residual envelope.  It is not
        keyed to a motion ID or fixed recovery time: both the capture-point
        margin and body-frame forward velocity must indicate a live risk.
        """
        gain = float(getattr(self.cfg, "stage_a_sagittal_front_gain", 1.0))
        if gain < 1.0:
            raise ValueError("stage_a_sagittal_front_gain must be at least 1.0")
        if gain == 1.0:
            self._stage_a_front_gain.fill_(1.0)
            return self._stage_a_front_gain

        margin = float(getattr(self.cfg, "stage_a_sagittal_front_margin_m", 0.07))
        velocity = float(
            getattr(self.cfg, "stage_a_sagittal_front_velocity_mps", 0.02)
        )
        if margin <= 0.0 or velocity < 0.0:
            raise ValueError("Invalid Stage-A sagittal front-support contract")

        from training.tasks.tracking.mdp.observations import stagger_support_state

        support = stagger_support_state(self._env)
        hit = self._motion_hit_steps(motion)
        in_swing = motion.prelude_elapsed_steps >= int(motion.prelude_steps)
        at_or_after_hit = motion.time_steps >= hit
        both_feet = support["contacts"].all(dim=-1)
        if not bool(getattr(self.cfg, "stage_a_sagittal_exit_require_both_feet", True)):
            both_feet = torch.ones_like(both_feet)
        front_risk = (
            in_swing
            & at_or_after_hit
            & both_feet
            & (support["capture_front_margin"] <= margin)
            & (self._asset.data.root_lin_vel_b[:, 0] >= velocity)
        )
        self._stage_a_front_gain[:] = torch.where(
            front_risk,
            torch.full_like(self._stage_a_front_gain, gain),
            torch.ones_like(self._stage_a_front_gain),
        )
        self._env.stage_a_sagittal_front_gain = self._stage_a_front_gain
        return self._stage_a_front_gain

    def _legacy_leg_action(self, motion) -> torch.Tensor:
        stage_obs = self._compute_observation_group(self._legacy_stage_a_group)
        if self.cfg.legacy_stage_a_yaw_adapter:
            stage_obs = adapt_stage_a_observation_legacy_yaw_pi(stage_obs)
        self._legacy_raw[:] = self._legacy_stage_a(stage_obs)
        self._env.legacy_stage_a_last_action[:] = self._legacy_raw

        lower = self._legacy_raw * self._mask
        if self._phase_gate_base_indices:
            lengths = motion.motion.motion_lengths[motion.motion_ids].clamp(min=2)
            phase = motion.time_steps.float() / (lengths - 1).float()
            u = ((phase - self.cfg.phase_gate_start) / max(
                self.cfg.phase_gate_end - self.cfg.phase_gate_start, 1.0e-6
            )).clamp(0.0, 1.0)
            smooth = u * u * (3.0 - 2.0 * u)
            gate = self.cfg.phase_gate_min_scale + (1.0 - self.cfg.phase_gate_min_scale) * smooth
            # Reproduce model_3396's tail contract: hip yaw is useful while
            # bracing for the swing, but must relinquish authority while the
            # reference settles back to the ready stance.
            release_steps = int(self.cfg.phase_gate_tail_release_steps)
            if release_steps > 0:
                tail = motion.tail_steps.to(dtype=gate.dtype)
                release_u = (tail / float(release_steps)).clamp(0.0, 1.0)
                release_smooth = release_u * release_u * (3.0 - 2.0 * release_u)
                tail_gate = self.cfg.phase_gate_min_scale + (
                    1.0 - self.cfg.phase_gate_min_scale
                ) * (1.0 - release_smooth)
                gate = torch.where(tail > 0, tail_gate, gate)
            for index in self._phase_gate_base_indices:
                lower[:, index] *= gate

        # The original Stage-A checkpoint was not a permanent standing
        # controller.  Once the upper reference reaches ready, it smoothly
        # hands the legs back to nominal PD.  Without this gate the frozen
        # policy keeps emitting a swing-conditioned residual indefinitely.
        ready_release_steps = int(self.cfg.ready_hold_residual_release_steps)
        if ready_release_steps > 0 and motion.return_to_default_steps > 0:
            ready_elapsed = (
                motion.tail_steps
                - int(motion.cfg.hold_last_frame_steps)
                - int(motion.cfg.return_to_default_steps)
            ).clamp(min=0).to(dtype=lower.dtype)
            ready_u = (ready_elapsed / float(ready_release_steps)).clamp(0.0, 1.0)
            ready_smooth = ready_u * ready_u * (3.0 - 2.0 * ready_u)
            lower *= (1.0 - ready_smooth).unsqueeze(-1)

        runtime_sagittal_scale = self._stage_a_sagittal_exit_gate(motion)
        runtime_sagittal_front_gain = self._stage_a_sagittal_front_gate(motion)
        lower[:, (0, 3, 4, 6, 9, 10)] *= (
            runtime_sagittal_scale * runtime_sagittal_front_gain
        ).unsqueeze(-1)

        # Deterministic audit hook: isolate whether the frozen Stage-A
        # sagittal residual keeps pushing after capture-point recentering.
        # Training and deployment never set this attribute.
        sagittal_audit_scale = getattr(
            self._env, "stage_a_sagittal_audit_scale", None
        )
        if sagittal_audit_scale is not None:
            if sagittal_audit_scale.shape != (self.num_envs,):
                raise ValueError(
                    "stage_a_sagittal_audit_scale must have shape "
                    f"({self.num_envs},), got {tuple(sagittal_audit_scale.shape)}"
                )
            sagittal_indices = (0, 3, 4, 6, 9, 10)
            lower[:, sagittal_indices] *= sagittal_audit_scale.unsqueeze(-1)
        self._legacy_bounded[:] = self._bound_actions(lower)
        return self._legacy_bounded

    def _post_hit_release_gate(self, motion, release_steps: int) -> torch.Tensor:
        """Smoothly retire a strike-only residual during the finite recovery tail.

        The frozen upper actor was qualified only through impact.  Leaving its
        residual active while the reference returns to ready creates an
        unqualified second upper-body trajectory.  The coordinator keeps leg
        and waist authority for recovery; only strike-specific upper terms use
        this gate.
        """
        gate = torch.ones((self.num_envs, 1), dtype=torch.float, device=self.device)
        if release_steps <= 0:
            return gate
        tail = motion.tail_steps.to(dtype=gate.dtype)
        u = (tail / float(release_steps)).clamp(0.0, 1.0)
        smooth = u * u * (3.0 - 2.0 * u)
        return (1.0 - smooth).unsqueeze(-1)

    def _post_hit_phase_release_gate(self, motion, release_steps: int) -> torch.Tensor:
        """Fade a strike residual from impact, before the finite tail begins."""
        gate = torch.ones((self.num_envs, 1), dtype=torch.float, device=self.device)
        if release_steps <= 0:
            return gate
        elapsed = self._post_hit_elapsed_steps(motion, motion.time_steps).to(dtype=gate.dtype)
        hit = self._motion_hit_steps(motion)
        in_prelude = motion.prelude_elapsed_steps < int(motion.prelude_steps)
        active = (~in_prelude) & ((motion.time_steps >= hit) | (motion.tail_steps > 0))
        u = (elapsed / float(release_steps)).clamp(0.0, 1.0)
        smooth = u * u * (3.0 - 2.0 * u)
        return torch.where(active.unsqueeze(-1), (1.0 - smooth).unsqueeze(-1), gate)

    def process_actions(self, actions: torch.Tensor):
        if actions.shape != self._raw_actions.shape or not torch.isfinite(actions).all():
            raise ValueError(f"Expected finite joint-coordinator action shape {self._raw_actions.shape}")

        # Evaluation-only finite-difference hook for the public 22-D
        # coordinator.  It is an additive raw-action perturbation, so the
        # audited policy remains the nominal baseline and only the selected
        # coordinator direction changes.  Training and deployment never set
        # this attribute.
        coordinator_offset = getattr(self._env, "coordinator_action_offset_override", None)
        if coordinator_offset is not None:
            if coordinator_offset.shape != actions.shape:
                raise ValueError(
                    "coordinator_action_offset_override must match the "
                    f"{tuple(actions.shape)} coordinator action shape"
                )
            if not torch.isfinite(coordinator_offset).all():
                raise ValueError("coordinator_action_offset_override must be finite")
            actions = actions + coordinator_offset.to(device=actions.device, dtype=actions.dtype)

        # Natural-prefix recovery is gated in the recovery observation, not
        # here.  ``actions`` contains both the frozen V22 baseline and the
        # learned recovery residual; zeroing it here would accidentally turn
        # off the qualified coordinator during the physical prefix.  The
        # observation gate instead keeps only the adapter/exploration at zero
        # until the selected recovery window, while this full base action keeps
        # generating the natural contact history.
        natural_prefix_enabled = getattr(self._env, "natural_prefix_recovery_enabled", False)
        natural_action_mask = getattr(self._env, "natural_recovery_action_mask", None)
        if natural_prefix_enabled:
            if natural_action_mask is None or natural_action_mask.shape != (self.num_envs,):
                raise RuntimeError(
                    "natural_prefix_recovery_enabled requires a pre-action "
                    "natural_recovery_action_mask"
                )

        motion = self._env.command_manager.get_term(self.cfg.reference_command_name)
        legacy_leg = self._legacy_leg_action(motion)
        self._unbounded_actions[:] = actions
        self._raw_actions[:] = self._bound_actions(actions)

        # Audit-only phase-local ablation.  Upper corrections are already
        # gated during the prelude; this hook isolates whether the coordinator
        # leg residual is amplifying the ready-to-swing transition.
        prelude_mask_mode = str(getattr(self._env, "coordinator_prelude_audit_mode", "none"))
        if prelude_mask_mode not in {"none", "all", "leg", "waist", "arm"}:
            raise ValueError(f"Unknown coordinator_prelude_audit_mode={prelude_mask_mode!r}")
        in_prelude = motion.prelude_elapsed_steps < int(motion.prelude_steps)
        if prelude_mask_mode in {"all", "leg"}:
            self._raw_actions[:, :12] *= (~in_prelude).unsqueeze(-1)

        # Reproduce the validated legacy support target first, then let only
        # the new 12-D leg correction perturb its physical target.
        self._full_joint_targets[:] = self._asset.data.default_joint_pos
        self._full_joint_velocity_targets.zero_()
        base_default = self._asset.data.default_joint_pos[:, self._base_joint_ids_tensor]
        self._full_joint_targets[:, self._base_joint_ids_tensor] = base_default + legacy_leg * self._scale
        leg_delta = self._raw_actions[:, :12] * self._leg_correction_scale
        self._full_joint_targets[:, self._leg_joint_ids_tensor] += leg_delta

        upper_obs = self._compute_observation_group(self._upper_observation_group)
        self._upper_last_observation[:] = upper_obs
        primary = self._upper_policy(upper_obs).clamp(-self.cfg.upper_raw_clip, self.cfg.upper_raw_clip)
        release = (motion.time_steps.float() / float(max(self.cfg.upper_prelude_release_steps, 1))).clamp(0.0, 1.0)
        upper_gate = torch.where(in_prelude, torch.zeros_like(release), release).unsqueeze(-1)
        reference = self._upper_reference(motion, motion.time_steps)
        upper_delta = torch.cat((self._raw_actions[:, 12:15], self._raw_actions[:, 15:22]), dim=-1)
        # The frozen upper prior supplies a strike residual, not a recovery
        # controller.  Fade it out while the reference settles to ready.
        frozen_upper_gate = self._post_hit_release_gate(
            motion, int(getattr(self.cfg, "upper_policy_tail_release_steps", 0))
        )
        waist_primary_gate = self._post_hit_phase_release_gate(
            motion, int(getattr(self.cfg, "upper_policy_waist_post_hit_release_steps", 0))
        )
        arm_gate = self._post_hit_release_gate(
            motion, int(getattr(self.cfg, "coordinator_arm_tail_release_steps", 0))
        )
        gated_primary = primary * frozen_upper_gate
        gated_primary[:, self._upper_waist_indices] *= waist_primary_gate
        gated_upper_delta = upper_delta.clone()
        gated_upper_delta[:, 3:] *= arm_gate
        primary_contribution = upper_gate * gated_primary * self._upper_scale
        coordinator_contribution = upper_gate * gated_upper_delta * self._upper_correction_scale
        execution_mode = str(
            getattr(self._env, "p4c_upper_execution_mode", "policy")
        )
        if execution_mode not in {"policy", "reference_only"}:
            raise ValueError(
                "p4c_upper_execution_mode must be 'policy' or 'reference_only'"
            )
        if execution_mode == "reference_only":
            # Evaluation-only P4C responsibility ablation.  Preserve the
            # frozen Stage-A leg stabilizer, but remove every learned upper
            # correction so the repaired reference and low-level plant can be
            # assessed without actor/coordinator interference.
            primary_contribution = torch.zeros_like(primary_contribution)
            coordinator_contribution = torch.zeros_like(coordinator_contribution)
        self._upper_raw_actions[:] = primary
        self._upper_reference_actions[:] = reference
        self._upper_primary_contribution[:] = primary_contribution
        self._upper_coordinator_contribution[:] = coordinator_contribution
        self._upper_processed_actions[:] = reference + primary_contribution + coordinator_contribution
        self._full_joint_targets[:, self._upper_joint_ids_tensor] = self._upper_processed_actions
        self._full_joint_velocity_targets[:, self._upper_joint_ids_tensor] = self._upper_velocity_reference(
            motion, motion.time_steps
        )

        self._apply_waist_soft_limit_guard(motion, motion.time_steps)
        self._apply_upper_dynamic_soft_limit_guard()
        self._apply_passive_soft_limit_guard()

        limits = self._asset.data.soft_joint_pos_limits
        self._full_joint_targets[:] = torch.clamp(self._full_joint_targets, min=limits[..., 0], max=limits[..., 1])
        self._upper_processed_actions[:] = self._full_joint_targets[:, self._upper_joint_ids_tensor]
        self._processed_actions[:] = torch.cat((leg_delta, upper_delta * self._upper_correction_scale), dim=-1)
        self._env.f0_upper_last_action[:] = primary
        self._env.joint_coordinator_last_action[:] = self._raw_actions


@configclass
class A3FrozenStageAJointCoordinatorActionCfg(A3F1FrozenUpperBaseCompositePositionActionCfg):
    class_type: type[ActionTerm] = A3FrozenStageAJointCoordinatorAction
    legacy_stage_a_checkpoint: str = ""
    legacy_stage_a_observation_group: str = "stage_a"
    legacy_stage_a_yaw_adapter: bool = True
    # Corrections are physical radians, applied after the corresponding frozen
    # policy.  Waist starts deliberately smaller because it couples strike
    # precision directly into whole-body angular momentum.
    leg_correction_scale_rad: tuple[float, ...] = (0.012, 0.035, 0.046, 0.010, 0.015, 0.007) * 2
    waist_correction_scale_rad: tuple[float, ...] = (0.010, 0.010, 0.010)
    arm_correction_scale_rad: tuple[float, ...] = (0.025,) * 7
    # Disabled by default to preserve historical V2/V5 execution.  Full-cycle
    # tasks explicitly retire the unqualified strike residual after impact.
    upper_policy_tail_release_steps: int = 0
    # The waist can retire from the impact residual before the clip reaches its
    # final frame.  This prevents a strike-only model_900 waist command from
    # opposing the dedicated post-hit waist recovery reference.
    upper_policy_waist_post_hit_release_steps: int = 0
    coordinator_arm_tail_release_steps: int = 0
    # State-latched exit for the frozen Stage-A sagittal brace. It is off by
    # default so historical checkpoints remain exactly reproducible.
    stage_a_sagittal_exit_enabled: bool = False
    stage_a_sagittal_exit_center_half_width_m: float = 0.04
    stage_a_sagittal_exit_velocity_deadband_mps: float = 0.03
    stage_a_sagittal_exit_positive_confirm_steps: int = 2
    stage_a_sagittal_exit_neutral_confirm_steps: int = 2
    stage_a_sagittal_exit_decay_steps: int = 5
    stage_a_sagittal_exit_require_both_feet: bool = True
    stage_a_sagittal_front_gain: float = 1.0
    stage_a_sagittal_front_margin_m: float = 0.07
    stage_a_sagittal_front_velocity_mps: float = 0.02
    # Optional multi-shot re-arm.  V25 leaves this disabled; V26 enables it
    # only after a strict settled-state hold and ramps Stage-A back in at the
    # start of the next explicit shot cycle.
    stage_a_sagittal_rearm_enabled: bool = False
    stage_a_sagittal_rearm_stable_steps: int = 20
    stage_a_sagittal_rearm_ramp_steps: int = 8
    stage_a_sagittal_rearm_center_half_width_m: float = 0.05
    stage_a_sagittal_rearm_velocity_max_mps: float = 0.06
    stage_a_sagittal_rearm_pitch_rate_max_radps: float = 0.10
    stage_a_sagittal_rearm_tilt_max_rad: float = 0.10
    stage_a_sagittal_rearm_arm_error_max_rad: float = 0.15
    # Zero preserves V26. V27 enables a finite upper velocity READY gate.
    stage_a_sagittal_rearm_arm_velocity_max_radps: float = 0.0
    stage_a_sagittal_rearm_require_ready_reference: bool = True


class A3TargetConditionedJointCoordinatorAction(A3FrozenStageAJointCoordinatorAction):
    """V30 coordinator with an internal, zero-policy target-position adapter.

    The public action remains the reviewed 22-D coordinator contract.  The
    calibrated seven-joint Cartesian feedforward is injected after the frozen
    coordinator has composed the floating-base full-body target.  An optional
    calibrated 22-D coordinator feedforward is instead added at the public
    coordinator input before that composition.  Both paths retain the reviewed
    V30 policy dimension, and both are identically zero at the anchor target.
    """

    cfg: "A3TargetConditionedJointCoordinatorActionCfg"

    def __init__(self, cfg: "A3TargetConditionedJointCoordinatorActionCfg", env):
        super().__init__(cfg, env)
        self._upper_target_adapter_contribution = torch.zeros_like(
            self._upper_raw_actions
        )
        # Mark the runtime environment so the shared racket-target observation
        # terms return the immutable anchor for this replay action.  The
        # external offset is intentionally visible only to this adapter.
        self._env.target_conditioned_anchor_observation = bool(cfg.anchor_observation)
        self._env.target_conditioned_coordinator_external_observation = bool(
            cfg.coordinator_external_observation
        )
        if len(cfg.adapter_joint_names) != 7:
            raise ValueError("V30 target adapter requires exactly seven right-arm joints")
        unknown = sorted(set(cfg.adapter_joint_names) - set(cfg.upper_joint_names))
        if unknown:
            raise ValueError(f"V30 adapter joints must be in upper_joint_names: {unknown}")
        self._target_adapter_indices = torch.tensor(
            [cfg.upper_joint_names.index(name) for name in cfg.adapter_joint_names],
            dtype=torch.long,
            device=self.device,
        )
        self._target_adapter_scale = torch.tensor(
            cfg.adapter_scale_rad, dtype=torch.float, device=self.device
        ).unsqueeze(0)
        if self._target_adapter_scale.shape[-1] != 7:
            raise ValueError("V30 adapter_scale_rad must contain seven values")
        matrix = torch.tensor(cfg.adapter_feedforward_pinv, dtype=torch.float, device=self.device)
        if matrix.shape != (7, 3):
            raise ValueError("V30 adapter_feedforward_pinv must have shape (7, 3)")
        self._target_adapter_pinv = matrix
        transform = torch.tensor(
            cfg.adapter_feedforward_target_transform, dtype=torch.float, device=self.device
        )
        if transform.shape != (3, 3):
            raise ValueError("V30 adapter_feedforward_target_transform must have shape (3, 3)")
        self._target_adapter_transform = transform
        matrix_by_motion = torch.tensor(
            cfg.adapter_feedforward_pinv_by_motion, dtype=torch.float, device=self.device
        )
        if matrix_by_motion.numel() == 0:
            matrix_by_motion = torch.empty((0, 7, 3), device=self.device)
        elif matrix_by_motion.ndim != 3 or matrix_by_motion.shape[1:] != (7, 3):
            raise ValueError("V30 adapter_feedforward_pinv_by_motion must have shape (M, 7, 3)")
        self._target_adapter_pinv_by_motion = matrix_by_motion
        transform_by_motion = torch.tensor(
            cfg.adapter_feedforward_target_transform_by_motion,
            dtype=torch.float,
            device=self.device,
        )
        if transform_by_motion.numel() == 0:
            transform_by_motion = torch.empty((0, 3, 3), device=self.device)
        elif (
            transform_by_motion.ndim != 3
            or transform_by_motion.shape[1:] != (3, 3)
            or (
                matrix_by_motion.shape[0] > 0
                and transform_by_motion.shape[0] != matrix_by_motion.shape[0]
            )
        ):
            raise ValueError(
                "V30 adapter_feedforward_target_transform_by_motion must have shape (M, 3, 3) "
                "matching adapter_feedforward_pinv_by_motion"
            )
        self._target_adapter_transform_by_motion = transform_by_motion
        clip = torch.tensor(cfg.adapter_raw_clip_by_motion, dtype=torch.float, device=self.device)
        if clip.numel() > 0 and (matrix_by_motion.shape[0] == 0 or clip.shape != (matrix_by_motion.shape[0],)):
            raise ValueError("V30 adapter_raw_clip_by_motion must match feedforward matrices")
        self._target_adapter_clip_by_motion = clip

        # A manifest anchor is a reference-motion quantity, not necessarily
        # the position reached by the stabilized floating-base plant.  Keep a
        # separately calibrated *control* anchor for the Cartesian adapter so
        # an external physical target close to the measured hit centre still
        # becomes a centimetre-scale command.  The frozen model_900 path
        # continues to see the original manifest anchor.
        control_anchor_offset = torch.tensor(
            cfg.adapter_control_anchor_offset_b_by_motion,
            dtype=torch.float,
            device=self.device,
        )
        if control_anchor_offset.numel() == 0:
            control_anchor_offset = torch.empty((0, 3), device=self.device)
        elif (
            control_anchor_offset.ndim != 2
            or control_anchor_offset.shape[1] != 3
            or (
                matrix_by_motion.shape[0] > 0
                and control_anchor_offset.shape[0] != matrix_by_motion.shape[0]
            )
        ):
            raise ValueError(
                "V30 adapter_control_anchor_offset_b_by_motion must have shape (M, 3) "
                "matching adapter_feedforward_pinv_by_motion"
            )
        self._target_adapter_control_anchor_offset_b_by_motion = control_anchor_offset
        calibration_hit_steps = torch.tensor(
            cfg.adapter_control_anchor_calibration_hit_steps_by_motion,
            dtype=torch.long,
            device=self.device,
        )
        if calibration_hit_steps.numel() == 0:
            calibration_hit_steps = torch.empty((0,), dtype=torch.long, device=self.device)
        elif (
            control_anchor_offset.shape[0] == 0
            or calibration_hit_steps.shape != (control_anchor_offset.shape[0],)
        ):
            raise ValueError(
                "V30 adapter_control_anchor_calibration_hit_steps_by_motion must have one "
                "entry per calibrated control anchor"
            )
        self._target_adapter_control_anchor_calibration_hit_steps_by_motion = (
            calibration_hit_steps
        )
        calibration_prelude_steps = torch.tensor(
            cfg.adapter_control_anchor_calibration_prelude_steps_by_motion,
            dtype=torch.long,
            device=self.device,
        )
        if calibration_prelude_steps.numel() == 0:
            calibration_prelude_steps = torch.empty((0,), dtype=torch.long, device=self.device)
        elif (
            control_anchor_offset.shape[0] == 0
            or calibration_prelude_steps.shape != (control_anchor_offset.shape[0],)
        ):
            raise ValueError(
                "V30 adapter_control_anchor_calibration_prelude_steps_by_motion must have "
                "one entry per calibrated control anchor"
            )
        self._target_adapter_control_anchor_calibration_prelude_steps_by_motion = (
            calibration_prelude_steps
        )
        self._target_adapter_control_delta_b = torch.zeros((self.num_envs, 3), device=self.device)
        self._env.target_adapter_control_delta_b = self._target_adapter_control_delta_b
        self._target_adapter_gate = torch.zeros((self.num_envs, 1), device=self.device)
        self._env.target_adapter_gate = self._target_adapter_gate
        self._env.target_adapter_last_action = torch.zeros((self.num_envs, 7), device=self.device)

        coordinator_matrix_by_motion = torch.tensor(
            cfg.coordinator_target_feedforward_by_motion,
            dtype=torch.float,
            device=self.device,
        )
        if coordinator_matrix_by_motion.numel() == 0:
            coordinator_matrix_by_motion = torch.empty((0, 22, 3), device=self.device)
        elif coordinator_matrix_by_motion.ndim != 3 or coordinator_matrix_by_motion.shape[1:] != (22, 3):
            raise ValueError(
                "V30 coordinator_target_feedforward_by_motion must have shape (M, 22, 3)"
            )
        if cfg.coordinator_target_feedforward_enabled and coordinator_matrix_by_motion.shape[0] == 0:
            raise ValueError(
                "V30 coordinator_target_feedforward_enabled requires at least one motion matrix"
            )
        if float(cfg.coordinator_target_feedforward_raw_clip) <= 0.0:
            raise ValueError("V30 coordinator_target_feedforward_raw_clip must be positive")
        self._coordinator_target_feedforward_by_motion = coordinator_matrix_by_motion
        self._coordinator_target_feedforward_raw_clip = float(
            cfg.coordinator_target_feedforward_raw_clip
        )
        self.coordinator_target_feedforward_last_action = torch.zeros(
            (self.num_envs, 22), device=self.device
        )
        # Keep a shared alias for rewards and runtime diagnostics, while the
        # action-term attribute lets evaluation snapshots expose the exact
        # raw target feedforward that was physically composed this step.
        self._env.coordinator_target_feedforward_last_action = (
            self.coordinator_target_feedforward_last_action
        )

    @property
    def upper_target_adapter_contribution(self) -> torch.Tensor:
        """Post-clip physical target change introduced by the 7-D adapter."""
        return self._upper_target_adapter_contribution

    def _target_adapter_hit_steps(self, motion) -> torch.Tensor:
        if motion._use_motion_library:
            return motion.motion.hit_frame[motion.motion_ids].to(torch.long)
        return torch.full_like(motion.time_steps, int(motion.motion.hit_frame[0]))

    def _target_adapter_gate_for(self, motion) -> torch.Tensor:
        hit = self._target_adapter_hit_steps(motion)
        step = motion.time_steps.to(torch.float32)
        pre = float(getattr(self.cfg, "adapter_ramp_in_steps", 10))
        post = float(getattr(self.cfg, "adapter_ramp_out_steps", 8))
        ramp_in = ((step - (hit - pre)) / max(pre, 1.0)).clamp(0.0, 1.0)
        ramp_out = ((hit + post - step) / max(post, 1.0)).clamp(0.0, 1.0)
        smooth_in = ramp_in * ramp_in * (3.0 - 2.0 * ramp_in)
        smooth_out = ramp_out * ramp_out * (3.0 - 2.0 * ramp_out)
        return (smooth_in * smooth_out).unsqueeze(-1)

    def process_actions(self, actions: torch.Tensor):
        self._upper_target_adapter_contribution.zero_()
        execution_mode = str(
            getattr(self._env, "p4c_upper_execution_mode", "policy")
        )
        if execution_mode not in {"policy", "reference_only"}:
            raise ValueError(
                "p4c_upper_execution_mode must be 'policy' or 'reference_only'"
            )
        motion = self._env.command_manager.get_term(self.cfg.reference_command_name)
        target = self._env.command_manager.get_term("racket_target")
        delta_b = target.external_target_delta_local_b()
        if self._target_adapter_control_anchor_offset_b_by_motion.shape[0] > 0:
            motion_ids = motion.motion_ids.to(torch.long)
            if torch.any(motion_ids < 0) or torch.any(
                motion_ids >= self._target_adapter_control_anchor_offset_b_by_motion.shape[0]
            ):
                raise ValueError("motion_id has no calibrated target control anchor")
            expected_hit_steps = (
                self._target_adapter_control_anchor_calibration_hit_steps_by_motion[motion_ids]
            )
            expected_prelude_steps = (
                self._target_adapter_control_anchor_calibration_prelude_steps_by_motion[
                    motion_ids
                ]
            )
            actual_hit_steps = self._target_adapter_hit_steps(motion)
            external_active = getattr(target, "_external_target_position_active", None)
            if execution_mode != "reference_only" and external_active is not None and torch.any(
                external_active & ((expected_hit_steps < 0) | (expected_prelude_steps < 0))
            ):
                raise ValueError(
                    "external target requested for a motion without a calibrated "
                    "control-anchor contract"
                )
            if execution_mode != "reference_only" and external_active is not None and torch.any(
                external_active
                & (expected_hit_steps >= 0)
                & (actual_hit_steps != expected_hit_steps)
            ):
                raise ValueError(
                    "external target requested outside this motion's calibrated hit-time "
                    "contract; select a separately calibrated strike time"
                )
            if execution_mode != "reference_only" and external_active is not None and torch.any(
                external_active
                & (expected_prelude_steps >= 0)
                & (int(motion.prelude_steps) != expected_prelude_steps)
            ):
                raise ValueError(
                    "external target requested outside this motion's calibrated READY-hold "
                    "contract; select a separately calibrated strike time"
                )
            delta_b = delta_b - self._target_adapter_control_anchor_offset_b_by_motion[motion_ids]
        self._target_adapter_control_delta_b[:] = delta_b

        # This path is calibrated from the full 22-D finite-difference
        # controllability audit.  It starts at command receipt rather than in
        # the final arm-only hit window: leg and waist contributions need the
        # complete pre-hit phase to create a real relative-racket displacement
        # instead of a root-only late correction.
        coordinator_feedforward = torch.zeros_like(actions)
        if self.cfg.coordinator_target_feedforward_enabled:
            if actions.shape[-1] != 22:
                raise ValueError("V30 target-conditioned coordinator requires 22-D actions")
            motion_ids = motion.motion_ids.to(torch.long)
            if torch.any(motion_ids < 0) or torch.any(
                motion_ids >= self._coordinator_target_feedforward_by_motion.shape[0]
            ):
                raise ValueError("V30 motion_id has no coordinator target-feedforward matrix")
            matrix = self._coordinator_target_feedforward_by_motion[motion_ids]
            coordinator_feedforward = torch.bmm(matrix, delta_b.unsqueeze(-1)).squeeze(-1)
            coordinator_feedforward = torch.clamp(
                coordinator_feedforward,
                -self._coordinator_target_feedforward_raw_clip,
                self._coordinator_target_feedforward_raw_clip,
            )
        if execution_mode == "reference_only":
            coordinator_feedforward.zero_()
        self.coordinator_target_feedforward_last_action[:] = coordinator_feedforward
        super().process_actions(actions + coordinator_feedforward)

        if execution_mode == "reference_only":
            # The parent has already emitted the guarded/clipped safe
            # reference.  Do not reintroduce a legacy calibrated target
            # adapter after the upper actor/coordinator ablation.
            self._target_adapter_gate.zero_()
            self._env.target_adapter_last_action.zero_()
            return

        if self._target_adapter_transform_by_motion.shape[0] > 0:
            motion_ids = motion.motion_ids.to(torch.long)
            if torch.any(motion_ids < 0) or torch.any(
                motion_ids >= self._target_adapter_transform_by_motion.shape[0]
            ):
                raise ValueError("motion_id has no target-adapter control transform")
            transform = self._target_adapter_transform_by_motion[motion_ids]
            delta_b = torch.bmm(transform, delta_b.unsqueeze(-1)).squeeze(-1)
        else:
            delta_b = delta_b @ self._target_adapter_transform.T
        if self._target_adapter_pinv_by_motion.shape[0] > 0:
            motion_ids = motion.motion_ids.to(torch.long)
            if torch.any(motion_ids < 0) or torch.any(
                motion_ids >= self._target_adapter_pinv_by_motion.shape[0]
            ):
                raise ValueError("V30 motion_id has no target-adapter feedforward matrix")
            pinv = self._target_adapter_pinv_by_motion[motion_ids]
            raw = torch.bmm(pinv, delta_b.unsqueeze(-1)).squeeze(-1)
            clip = self._target_adapter_clip_by_motion[motion_ids].unsqueeze(-1)
            bounded = torch.maximum(torch.minimum(raw, clip), -clip)
        else:
            raw = delta_b @ self._target_adapter_pinv.T
            bounded = torch.clamp(raw, -float(self.cfg.adapter_raw_clip), float(self.cfg.adapter_raw_clip))
        # Evaluation-only hook for finite-difference controllability audits.
        # When present, the caller supplies the seven internal adapter values
        # directly, bypassing the Cartesian feedforward matrix while keeping
        # the same per-motion clip and phase gate.
        override = getattr(self._env, "target_adapter_override", None)
        if override is not None:
            if override.shape != bounded.shape:
                raise ValueError(
                    "target_adapter_override must match the (num_envs, 7) adapter shape"
                )
            bounded = override.to(device=bounded.device, dtype=bounded.dtype)
            if self._target_adapter_clip_by_motion.shape[0] > 0:
                bounded = torch.maximum(torch.minimum(bounded, clip), -clip)
            else:
                bounded = torch.clamp(
                    bounded, -float(self.cfg.adapter_raw_clip), float(self.cfg.adapter_raw_clip)
                )
        gate = self._target_adapter_gate_for(motion)
        self._target_adapter_gate[:] = gate
        self._env.target_adapter_last_action[:] = bounded

        upper_before_adapter = self._full_joint_targets[
            :, self._upper_joint_ids_tensor
        ].clone()
        upper = upper_before_adapter.clone()
        upper[:, self._target_adapter_indices] += gate * bounded * self._target_adapter_scale
        self._full_joint_targets[:, self._upper_joint_ids_tensor] = upper
        limits = self._asset.data.soft_joint_pos_limits
        self._full_joint_targets[:] = torch.clamp(
            self._full_joint_targets, min=limits[..., 0], max=limits[..., 1]
        )
        self._upper_processed_actions[:] = self._full_joint_targets[:, self._upper_joint_ids_tensor]
        self._upper_target_adapter_contribution[:] = (
            self._upper_processed_actions - upper_before_adapter
        )
        # Target-conditioned reshaping must not bypass the execution-level
        # measured-state safety contract.
        self._apply_upper_dynamic_soft_limit_guard()
        self._apply_passive_soft_limit_guard()
        self._upper_processed_actions[:] = self._full_joint_targets[
            :, self._upper_joint_ids_tensor
        ]


@configclass
class A3TargetConditionedJointCoordinatorActionCfg(A3FrozenStageAJointCoordinatorActionCfg):
    class_type: type[ActionTerm] = A3TargetConditionedJointCoordinatorAction
    # Replay defaults to the frozen anchor observation.  A continuation
    # experiment may explicitly disable it to fine-tune the 22-D coordinator
    # against the external target while retaining the same action contract.
    anchor_observation: bool = True
    coordinator_external_observation: bool = False
    # Optional motion-conditioned raw 22-D feedforward.  It is disabled in
    # replay by default.  The matrix maps the external target delta in the
    # command receipt heading frame [m] to coordinator raw action units.
    coordinator_target_feedforward_enabled: bool = False
    coordinator_target_feedforward_raw_clip: float = 0.20
    coordinator_target_feedforward_by_motion: tuple[
        tuple[tuple[float, float, float], ...], ...
    ] = ()
    adapter_joint_names: tuple[str, ...] = ()
    adapter_scale_rad: tuple[float, ...] = (0.10,) * 7
    adapter_raw_clip: float = 1.0
    adapter_raw_clip_by_motion: tuple[float, ...] = ()
    adapter_ramp_in_steps: int = 10
    adapter_ramp_out_steps: int = 8
    adapter_feedforward_pinv: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0),) * 7
    adapter_feedforward_pinv_by_motion: tuple[
        tuple[tuple[float, float, float], ...], ...
    ] = ()
    adapter_feedforward_target_transform: tuple[
        tuple[float, float, float], ...
    ] = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    # Optional per-motion Cartesian pre-compensation after control-anchor
    # subtraction.  Empty preserves the legacy shared transform exactly.
    adapter_feedforward_target_transform_by_motion: tuple[
        tuple[tuple[float, float, float], ...], ...
    ] = ()
    # Per-motion measured hit-centre offset from the manifest anchor in the
    # command-receipt base-yaw frame.  It is subtracted only from the target
    # adapter's external control delta; it never changes frozen actor inputs.
    adapter_control_anchor_offset_b_by_motion: tuple[tuple[float, float, float], ...] = ()
    # ``-1`` means no timing restriction.  A non-negative value locks an
    # external target to the exact measured manifest hit frame for its anchor.
    adapter_control_anchor_calibration_hit_steps_by_motion: tuple[int, ...] = ()
    # The optional external hit-time scheduler delays the same manifest hit
    # frame by changing READY/prelude duration.  Require that duration too so
    # a native-time position calibration cannot silently serve a delayed shot.
    adapter_control_anchor_calibration_prelude_steps_by_motion: tuple[int, ...] = ()
