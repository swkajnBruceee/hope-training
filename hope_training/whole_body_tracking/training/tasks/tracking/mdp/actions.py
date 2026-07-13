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

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = torch.clamp(actions, -self.cfg.raw_clip, self.cfg.raw_clip)
        motion_cmd = self._env.command_manager.get_term(self.cfg.reference_command_name)
        if self.cfg.reference_lookahead_steps:
            time_steps = motion_cmd.time_steps + int(self.cfg.reference_lookahead_steps)
            if motion_cmd._use_motion_library:
                lengths = motion_cmd.motion.motion_lengths[motion_cmd.motion_ids]
                time_steps = torch.minimum(time_steps, lengths - 1)
                reference_joint_pos_full = motion_cmd.motion.joint_pos[motion_cmd.motion_ids, time_steps]
            else:
                time_steps = torch.clamp(time_steps, max=motion_cmd.motion.time_step_total - 1)
                reference_joint_pos_full = motion_cmd.motion.joint_pos[time_steps]
            reference_joint_pos = reference_joint_pos_full[:, self._joint_index_tensor]
        else:
            reference_joint_pos = motion_cmd.joint_pos[:, self._joint_index_tensor]
        self._processed_actions = reference_joint_pos + self._raw_actions * self._scale
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        self._apply_soft_limit_margin()


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
