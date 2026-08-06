"""Full strike-cycle/recovery gate independent of motion-library IDs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch


class CyclePhase(IntEnum):
    READY_HOLD = 0
    STRIKE_PRELUDE = 1
    STRIKE_ACTIVE = 2
    HIT_WINDOW = 3
    FOLLOW_THROUGH = 4
    RECOVERY_MONITOR = 5
    NEXT_ACTION_READY = 6
    CYCLE_FAILED = 7


@dataclass(frozen=True)
class CycleConfig:
    post_hit_guard_steps: int = 75
    recovery_timeout_steps: int = 250
    ready_hold_steps: int = 15


class StrikeCycleManager:
    """Explicit cycle FSM; timeout never masks a confirmed physical fall."""

    def __init__(self, num_envs: int, device: torch.device | str, cfg: CycleConfig = CycleConfig()):
        if cfg.post_hit_guard_steps < 1 or cfg.ready_hold_steps < 1:
            raise ValueError("cycle guard/ready hold must be positive")
        self.cfg = cfg
        self.device = torch.device(device)
        self.phase = torch.full((num_envs,), int(CyclePhase.READY_HOLD), dtype=torch.long, device=self.device)
        self.phase_age = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.guard_steps = torch.zeros_like(self.phase_age)
        self.recovery_steps = torch.zeros_like(self.phase_age)
        self.ready_steps = torch.zeros_like(self.phase_age)
        self.confirmed_fall = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.recovery_timeout = torch.zeros_like(self.confirmed_fall)
        self.ordinary_timeout = torch.zeros_like(self.confirmed_fall)
        self.last_episode_step = torch.full_like(self.phase_age, -1)

    @property
    def num_envs(self) -> int:
        return self.phase.numel()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids.to(self.device, dtype=torch.long)
        self.phase[ids] = int(CyclePhase.READY_HOLD)
        self.phase_age[ids] = 0
        self.guard_steps[ids] = 0
        self.recovery_steps[ids] = 0
        self.ready_steps[ids] = 0
        self.confirmed_fall[ids] = False
        self.recovery_timeout[ids] = False
        self.ordinary_timeout[ids] = False
        self.last_episode_step[ids] = -1

    def update(
        self,
        *,
        prelude_active: torch.Tensor,
        strike_active: torch.Tensor,
        hit_window: torch.Tensor,
        motion_done: torch.Tensor,
        recovery_ready: torch.Tensor,
        confirmed_fall: torch.Tensor,
        predicted_unrecoverable: torch.Tensor,
        timeout: torch.Tensor,
        strike_pass: torch.Tensor,
        episode_step: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        fields = (prelude_active, strike_active, hit_window, motion_done, recovery_ready, confirmed_fall, predicted_unrecoverable, timeout, strike_pass)
        if any(field.shape != (self.num_envs,) for field in fields):
            raise ValueError("all cycle-manager inputs must have shape [num_envs]")
        if episode_step is not None:
            if episode_step.shape != (self.num_envs,):
                raise ValueError("episode_step must have shape [num_envs]")
            rewound = (self.last_episode_step >= 0) & (episode_step < self.last_episode_step)
            if rewound.any():
                self.reset(torch.where(rewound)[0])
            self.last_episode_step[:] = episode_step
        self.phase_age += 1
        self.confirmed_fall |= confirmed_fall
        failed = self.confirmed_fall.clone()
        phase = self.phase
        phase = torch.where((phase == int(CyclePhase.READY_HOLD)) & prelude_active, int(CyclePhase.STRIKE_PRELUDE), phase)
        phase = torch.where((phase == int(CyclePhase.READY_HOLD)) & strike_active, int(CyclePhase.STRIKE_ACTIVE), phase)
        phase = torch.where((phase == int(CyclePhase.STRIKE_PRELUDE)) & strike_active, int(CyclePhase.STRIKE_ACTIVE), phase)
        phase = torch.where((phase == int(CyclePhase.STRIKE_ACTIVE)) & hit_window, int(CyclePhase.HIT_WINDOW), phase)
        phase = torch.where((phase == int(CyclePhase.HIT_WINDOW)) & (~hit_window) & (~motion_done), int(CyclePhase.FOLLOW_THROUGH), phase)
        phase = torch.where((phase == int(CyclePhase.STRIKE_ACTIVE)) & motion_done, int(CyclePhase.FOLLOW_THROUGH), phase)
        phase = torch.where((phase == int(CyclePhase.HIT_WINDOW)) & motion_done, int(CyclePhase.FOLLOW_THROUGH), phase)
        phase = torch.where(predicted_unrecoverable & (~self.confirmed_fall), int(CyclePhase.RECOVERY_MONITOR), phase)
        phase = torch.where((phase == int(CyclePhase.FOLLOW_THROUGH)) & motion_done, int(CyclePhase.RECOVERY_MONITOR), phase)
        in_recovery = (phase == int(CyclePhase.RECOVERY_MONITOR)) | (phase == int(CyclePhase.FOLLOW_THROUGH))
        self.guard_steps = torch.where(in_recovery, self.guard_steps + 1, self.guard_steps)
        self.recovery_steps = torch.where(in_recovery, self.recovery_steps + 1, torch.zeros_like(self.recovery_steps))
        self.recovery_timeout |= in_recovery & (~self.confirmed_fall) & (timeout | (self.recovery_steps >= self.cfg.recovery_timeout_steps))
        self.ordinary_timeout |= (~in_recovery) & timeout & (~self.confirmed_fall)
        failed |= self.recovery_timeout | self.ordinary_timeout
        # A physical-looking pose is not enough to admit a new action while
        # the forward prediction says the remaining recovery envelope is
        # already insufficient.  Predicted-unrecoverable is therefore a hard
        # next-action veto, just like confirmed_fall.
        self.ready_steps = torch.where(
            in_recovery & recovery_ready & strike_pass & (~predicted_unrecoverable),
            self.ready_steps + 1,
            torch.zeros_like(self.ready_steps),
        )
        ready = (
            (self.ready_steps >= self.cfg.ready_hold_steps)
            & (self.guard_steps >= self.cfg.post_hit_guard_steps)
            & strike_pass
            & (~self.confirmed_fall)
            & (~predicted_unrecoverable)
        )
        phase = torch.where(ready, int(CyclePhase.NEXT_ACTION_READY), phase)
        phase = torch.where(failed, int(CyclePhase.CYCLE_FAILED), phase)
        self.phase = phase
        return {
            "cycle_phase": self.phase,
            "guard_steps": self.guard_steps,
            "recovery_steps": self.recovery_steps,
            "ready_steps": self.ready_steps,
            "next_action_allowed": self.phase == int(CyclePhase.NEXT_ACTION_READY),
            "cycle_failed": self.phase == int(CyclePhase.CYCLE_FAILED),
            "confirmed_fall": self.confirmed_fall,
            "predicted_unrecoverable": predicted_unrecoverable,
            "recovery_timeout": self.recovery_timeout,
            "ordinary_timeout": self.ordinary_timeout,
        }
