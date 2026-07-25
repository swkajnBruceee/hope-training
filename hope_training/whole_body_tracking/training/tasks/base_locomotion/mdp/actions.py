"""Deployable A3 Base action semantics for the bounded Stand smoke task."""

from __future__ import annotations

from dataclasses import MISSING
from pathlib import Path

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
        env.f0_upper_last_action = self._upper_raw_actions.clone()

    @property
    def upper_raw_actions(self) -> torch.Tensor:
        return self._upper_raw_actions

    @property
    def upper_processed_actions(self) -> torch.Tensor:
        return self._upper_processed_actions

    def _upper_reference(self, motion_cmd, time_steps: torch.Tensor) -> torch.Tensor:
        """Gather the same lead-compensated raw motion reference as model_900."""
        query = time_steps.float().unsqueeze(-1) + self._upper_lead.unsqueeze(0)
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
        ref = ref0 + alpha * (ref1 - ref0)
        joint_ids = self._upper_joint_ids_tensor.view(1, -1).expand(query.shape[0], -1)
        return ref.gather(2, joint_ids.unsqueeze(-1)).squeeze(-1)

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
        self._upper_processed_actions[:] = reference + self._upper_raw_actions * self._upper_scale
        self._full_joint_targets[:, self._upper_joint_ids_tensor] = self._upper_processed_actions
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
    # These fields let the common native-strike task override path configure
    # the frozen upper contract without knowing about the F0 composite term.
    joint_names: tuple[str, ...] = ()
    scale: dict[str, float] = MISSING
    preserve_order: bool = True
    reference_lookahead_steps: int = 0
    joint_reference_lookahead_steps: dict[str, float] = {}


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
        upper_action = self._upper_policy(upper_obs)
        self._env.f0_upper_raw_action = upper_action
        super().process_actions(actions)


@configclass
class A3F1FrozenUpperBaseCompositePositionActionCfg(A3F0UpperBaseCompositePositionActionCfg):
    class_type: type[ActionTerm] = A3F1FrozenUpperBaseCompositePositionAction
    upper_checkpoint: str = ""
    upper_observation_group: str = "upper"
