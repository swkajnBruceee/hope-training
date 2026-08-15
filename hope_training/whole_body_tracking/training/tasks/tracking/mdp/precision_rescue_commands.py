"""Rescue-only command diagnostics; public target sampling is unchanged."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.managers import CommandTerm
from isaaclab.utils import configclass

from .hope_commands import ReferenceFreeRacketTargetCommand, ReferenceFreeRacketTargetCommandCfg


class PrecisionRescueRacketTargetCommand(ReferenceFreeRacketTargetCommand):
    """Current CompletePriors local sampler plus episode reward accounting.

    The subclass intentionally does not override target sampling, timing, or
    StrikeEvent logic.  It only records the authoritative temporal-gate
    support and Rescue reward contributions for audit.
    """

    _AUDIT_SUMS = (
        "v13b_rescue_sum_strike_temporal_weight",
        "v13b_rescue_frames_temporal_weight_gt_0_5",
        "v13b_rescue_frames_temporal_weight_gt_0_1",
        "v13b_rescue_frames_temporal_weight_gt_0_01",
        "v13b_rescue_exact_normal_episode_contribution",
        "v13b_rescue_wide_normal_episode_contribution",
        "v13b_rescue_exact_velocity_episode_contribution",
        "v13b_rescue_wide_velocity_episode_contribution",
        "v13b_rescue_wide_position_episode_contribution",
        "v13b_rescue_strike_position_episode_contribution",
        "v13b_rescue_joint_quality_episode_contribution",
        "v13b_rescue_wide_position_temporal_sum",
        "v13b_rescue_wide_position_frames_temporal_gt_0_1",
        "v13b_rescue_wide_position_frames_temporal_gt_0_01",
    )

    def __init__(self, cfg: "PrecisionRescueRacketTargetCommandCfg", env):
        super().__init__(cfg, env)
        for name in self._AUDIT_SUMS:
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["v13b_rescue_joint_error_prev"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["v13b_rescue_joint_component_error_prev"] = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self.metrics["v13b_rescue_joint_tau_prev"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["v13b_rescue_joint_initialized"] = torch.zeros(
            self.num_envs, device=self.device
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        ids = slice(None) if env_ids is None else torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        for name in self._AUDIT_SUMS:
            self.metrics[name][ids] = 0.0
        for name in (
            "v13b_rescue_joint_error_prev",
            "v13b_rescue_joint_tau_prev",
            "v13b_rescue_joint_initialized",
        ):
            self.metrics[name][ids] = 0.0
        self.metrics["v13b_rescue_joint_component_error_prev"][ids] = 0.0
        return super().reset(env_ids)

    def _update_metrics(self) -> None:
        super()._update_metrics()
        temporal = self.strike_temporal_weight()
        self.metrics["v13b_rescue_sum_strike_temporal_weight"] += temporal
        self.metrics["v13b_rescue_frames_temporal_weight_gt_0_5"] += (temporal > 0.5).float()
        self.metrics["v13b_rescue_frames_temporal_weight_gt_0_1"] += (temporal > 0.1).float()
        self.metrics["v13b_rescue_frames_temporal_weight_gt_0_01"] += (temporal > 0.01).float()

    def record_rescue_contribution(self, name: str, contribution: torch.Tensor) -> None:
        key = f"v13b_rescue_{name}_episode_contribution"
        if key not in self.metrics:
            raise KeyError(f"Unknown PrecisionRescue contribution {name!r}")
        self.metrics[key] += contribution.detach()


@configclass
class PrecisionRescueRacketTargetCommandCfg(ReferenceFreeRacketTargetCommandCfg):
    """Identical sampler config with Rescue-only audit term class."""

    class_type: type[CommandTerm] = PrecisionRescueRacketTargetCommand
