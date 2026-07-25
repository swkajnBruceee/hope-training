from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.managers import ActionTerm
from isaaclab.utils import configclass


class ClampedJointPositionAction(JointPositionAction):
    """Joint position action with raw policy output clamped before scaling.

    IsaacLab's built-in ``JointPositionAction`` maps raw policy output directly
    to joint targets via ``target = raw * scale + offset``. For the A3 native
    strike policy we want a deployable command contract: action -1..1 means
    "within the configured per-joint command range". This keeps training from
    learning by issuing very large raw actions that would not be acceptable for
    the A3 arm command adapter.
    """

    cfg: "ClampedJointPositionActionCfg"

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = torch.clamp(actions, -self.cfg.raw_clip, self.cfg.raw_clip)
        self._processed_actions = self._raw_actions * self._scale + self._offset
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        self._apply_soft_limit_margin()

    def _apply_soft_limit_margin(self):
        margin_frac = float(getattr(self.cfg, "soft_limit_margin_frac", 0.0) or 0.0)
        if margin_frac <= 0.0:
            return
        if isinstance(self._joint_ids, slice):
            joint_ids = torch.arange(self._asset.num_joints, device=self.device)[self._joint_ids]
        else:
            joint_ids = torch.as_tensor(self._joint_ids, dtype=torch.long, device=self.device)
        limits = self._asset.data.soft_joint_pos_limits[:, joint_ids]
        span = torch.clamp(limits[..., 1] - limits[..., 0], min=1.0e-6)
        lower = limits[..., 0] + margin_frac * span
        upper = limits[..., 1] - margin_frac * span
        self._processed_actions = torch.clamp(self._processed_actions, min=lower, max=upper)


class ReferenceResidualJointPositionAction(ClampedJointPositionAction):
    """Joint position action around the current reference motion pose.

    This is useful for the native-strike calibration stage: the manifest motion
    supplies a feasible swing prior, and the policy learns bounded corrections
    for impact accuracy and robustness. It should be treated as a
    motion-library executor, not the final planner-only deployment contract.
    """

    cfg: "ReferenceResidualJointPositionActionCfg"

    def __init__(self, cfg: "ReferenceResidualJointPositionActionCfg", env):
        super().__init__(cfg, env)
        self._env = env
        if isinstance(self._joint_ids, slice):
            self._joint_index_tensor = torch.arange(self._asset.num_joints, device=self.device)[self._joint_ids]
        else:
            self._joint_index_tensor = torch.as_tensor(self._joint_ids, dtype=torch.long, device=self.device)
        joint_lead = torch.zeros(len(self._joint_index_tensor), dtype=torch.float32, device=self.device)
        configured = getattr(cfg, "joint_reference_lookahead_steps", {}) or {}
        for i, name in enumerate(self._joint_names):
            joint_lead[i] = float(getattr(cfg, "reference_lookahead_steps", 0)) + float(configured.get(name, 0.0))
        self._joint_reference_lookahead_steps = joint_lead

    def _reference_joint_pos(self, motion_cmd, time_steps: torch.Tensor) -> torch.Tensor:
        """Gather the full-motion reference in the action term's joint order."""
        if motion_cmd._use_motion_library:
            lengths = motion_cmd.motion.motion_lengths[motion_cmd.motion_ids]
            time_steps = torch.minimum(time_steps, lengths - 1)
            reference_joint_pos_full = motion_cmd.motion.joint_pos[motion_cmd.motion_ids, time_steps]
        else:
            time_steps = torch.clamp(time_steps, max=motion_cmd.motion.time_step_total - 1)
            reference_joint_pos_full = motion_cmd.motion.joint_pos[time_steps]
        return reference_joint_pos_full[:, self._joint_index_tensor]

    def _reference_joint_pos_with_joint_lead(self, motion_cmd, time_steps: torch.Tensor) -> torch.Tensor:
        """Gather reference positions with an independent fractional lead per action joint."""
        # A 1-D lead is the normal training configuration.  Diagnostics may
        # supply one lead vector per parallel environment to scan timing without
        # launching a separate simulator for every candidate.
        lead = self._joint_reference_lookahead_steps
        if lead.ndim == 1:
            lead = lead.unsqueeze(0)
        query = time_steps.to(dtype=torch.float32).unsqueeze(-1) + lead
        if motion_cmd._use_motion_library:
            lengths = motion_cmd.motion.motion_lengths[motion_cmd.motion_ids].unsqueeze(-1).to(torch.float32)
            query = torch.maximum(query, torch.zeros_like(query))
            query = torch.minimum(query, lengths - 1.0)
            t0 = query.floor().to(torch.long)
            max_t = (lengths - 1.0).to(torch.long).expand_as(t0)
            t1 = torch.minimum(t0 + 1, max_t)
            alpha = (query - t0.to(torch.float32)).unsqueeze(-1)
            full = motion_cmd.motion.joint_pos[motion_cmd.motion_ids]
        else:
            max_step = int(motion_cmd.motion.time_step_total) - 1
            query = query.clamp(min=0.0, max=float(max_step))
            t0 = query.floor().to(torch.long)
            t1 = (t0 + 1).clamp(max=max_step)
            alpha = (query - t0.to(torch.float32)).unsqueeze(-1)
            full = motion_cmd.motion.joint_pos.unsqueeze(0).expand(query.shape[0], -1, -1)
        gather_shape = (*t0.shape, full.shape[-1])
        ref0 = torch.gather(full, 1, t0.unsqueeze(-1).expand(gather_shape))
        ref1 = torch.gather(full, 1, t1.unsqueeze(-1).expand(gather_shape))
        ref = ref0 + alpha * (ref1 - ref0)
        joint_idx = self._joint_index_tensor.view(1, -1).expand(query.shape[0], -1)
        return ref.gather(2, joint_idx.unsqueeze(-1)).squeeze(-1)

    def _apply_target_limits(self, target: torch.Tensor) -> torch.Tensor:
        if self.cfg.clip is not None:
            target = torch.clamp(target, min=self._clip[:, :, 0], max=self._clip[:, :, 1])
        margin_frac = float(getattr(self.cfg, "soft_limit_margin_frac", 0.0) or 0.0)
        if margin_frac > 0.0:
            if isinstance(self._joint_ids, slice):
                joint_ids = torch.arange(self._asset.num_joints, device=self.device)[self._joint_ids]
            else:
                joint_ids = torch.as_tensor(self._joint_ids, dtype=torch.long, device=self.device)
            limits = self._asset.data.soft_joint_pos_limits[:, joint_ids]
            span = torch.clamp(limits[..., 1] - limits[..., 0], min=1.0e-6)
            target = torch.clamp(
                target,
                min=limits[..., 0] + margin_frac * span,
                max=limits[..., 1] - margin_frac * span,
            )
        return target

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = torch.clamp(actions, -self.cfg.raw_clip, self.cfg.raw_clip)
        motion_cmd = self._env.command_manager.get_term(self.cfg.reference_command_name)
        reference_joint_pos = self._reference_joint_pos_with_joint_lead(motion_cmd, motion_cmd.time_steps)
        self._processed_actions = reference_joint_pos + self._raw_actions * self._scale
        self._processed_actions = self._apply_target_limits(self._processed_actions)

    def apply_actions(self):
        """Apply the target, optionally interpolating reference frames per physics substep.

        The motion clock remains at the environment/control rate (50 Hz for the
        current 50 fps library). Only the target sent to the position servo is
        interpolated across the decimation substeps. This models a higher-rate
        command transport without pretending that the 50 fps reference contains
        new motion information at 500 Hz.
        """
        if not self.cfg.interpolate_reference:
            return super().apply_actions()

        motion_cmd = self._env.command_manager.get_term(self.cfg.reference_command_name)
        decimation = max(int(self._env.cfg.decimation), 1)
        sim_step_counter = int(getattr(self._env, "_sim_step_counter", 1))
        substep_index = (sim_step_counter - 1) % decimation
        alpha = float(substep_index) / float(decimation)
        # Keep the same lookahead for both endpoints.  Otherwise a positive
        # lookahead would interpolate from frame t+k back toward frame t+1
        # inside one control period, which is neither a lead-compensated
        # command nor a continuous reference trajectory.
        next_reference = self._reference_joint_pos_with_joint_lead(motion_cmd, motion_cmd.time_steps + 1)
        residual = self._raw_actions * self._scale
        next_target = next_reference + residual
        target = (1.0 - alpha) * self._processed_actions + alpha * next_target
        target = self._apply_target_limits(target)
        self._asset.set_joint_position_target(target, joint_ids=self._joint_ids)


@configclass
class ClampedJointPositionActionCfg(JointPositionActionCfg):
    class_type: type[ActionTerm] = ClampedJointPositionAction
    raw_clip: float = 1.0
    soft_limit_margin_frac: float = 0.0


@configclass
class ReferenceResidualJointPositionActionCfg(ClampedJointPositionActionCfg):
    class_type: type[ActionTerm] = ReferenceResidualJointPositionAction
    reference_command_name: str = "motion"
    reference_lookahead_steps: int = 0
    joint_reference_lookahead_steps: dict[str, float] = {}
    interpolate_reference: bool = False
