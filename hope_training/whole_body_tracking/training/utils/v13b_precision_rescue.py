"""Opt-in scheduling and numerical contracts for V1.3B Precision Rescue.

This module is deliberately pure Python.  It is not imported by the current
CompletePriors task, so adding the Rescue route cannot alter an in-flight or a
future unmodified CompletePriors run.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from training.utils.v13b_contract import lower_prior_alpha


def smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, float(value)))
    return value * value * (3.0 - 2.0 * value)


@dataclass
class PrecisionRescuePriorSchedule:
    """Continuation-only prior schedule.

    ``source_progress`` is the historical CompletePriors progress of the
    selected checkpoint.  It is never reset to zero.  The lower prior follows
    its existing monotone tail from that point.  The upper prior first holds
    exactly its historical value, then may reduce only after an external,
    evidence-backed readiness gate is opened.
    """

    source_progress: float
    source_lower_alpha: float
    source_upper_alpha: float
    total_chain_updates: int
    hold_updates: int = 300
    upper_step: float = 0.05
    # Optional controllability-recovery mode.  This is deliberately separate
    # from the historical source alpha: it records that a new rescue branch
    # intentionally starts with less upper-prior authority.
    controllability_recovery_enabled: bool = False
    controllability_start_alpha: float = -1.0
    controllability_min_alpha: float = 0.30
    # Safety deadline: regardless of an absent/stale external gate, the
    # upper teacher may not remain active beyond this continued curriculum
    # progress.  This guarantees a genuinely teacher-free tail phase.
    force_zero_start_progress: float = 0.60
    force_zero_progress: float = 0.70
    hard_zero_enabled: bool = True
    readiness_open: bool = False
    current_update: int = 0
    _upper_alpha: float | None = None
    _force_ramp_start_alpha: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.source_progress <= 1.0:
            raise ValueError("PrecisionRescue source_progress must be in [0, 1]")
        if not 0.0 <= self.source_lower_alpha <= 1.0:
            raise ValueError("PrecisionRescue source_lower_alpha must be in [0, 1]")
        if not 0.0 <= self.source_upper_alpha <= 1.0:
            raise ValueError("PrecisionRescue source_upper_alpha must be in [0, 1]")
        if self.total_chain_updates < 2:
            raise ValueError("PrecisionRescue total_chain_updates must be >= 2")
        if not 0.0 <= self.force_zero_progress <= 1.0:
            raise ValueError("force_zero_progress must be in [0, 1]")
        if not 0.0 <= self.force_zero_start_progress <= self.force_zero_progress:
            raise ValueError("force_zero_start_progress must be <= force_zero_progress")
        if self.hold_updates < 0:
            raise ValueError("PrecisionRescue hold_updates must be >= 0")
        if not 0.0 < self.upper_step <= 0.05:
            raise ValueError("PrecisionRescue upper_step must be in (0, 0.05]")
        if self.controllability_recovery_enabled:
            if not 0.0 <= self.controllability_start_alpha <= 1.0:
                raise ValueError("controllability_start_alpha must be in [0, 1]")
            if not 0.0 <= self.controllability_min_alpha <= self.controllability_start_alpha:
                raise ValueError("controllability_min_alpha must be <= controllability_start_alpha")
            if self.controllability_start_alpha > self.source_upper_alpha + 1.0e-9:
                raise ValueError("controllability start alpha cannot exceed source upper alpha")
            self._upper_alpha = float(self.controllability_start_alpha)
        else:
            self._upper_alpha = float(self.source_upper_alpha)

    @property
    def global_progress(self) -> float:
        # The source checkpoint already consumed source_progress of the
        # original 50k-update chain.  Continue the same clock rather than
        # treating Rescue update zero as complete-priors update zero.
        tail = max(1.0 - self.source_progress, 0.0)
        advanced = self.current_update / max(self.total_chain_updates - 1, 1)
        return min(1.0, self.source_progress + tail * advanced)

    def set_update(self, update: int) -> None:
        self.current_update = max(0, int(update))
        if self.hard_zero_enabled:
            progress = self.global_progress
            if progress >= self.force_zero_progress:
                # The deadline is monotone and irreversible.  It is deliberately
                # independent of the optional external readiness gate.
                self._upper_alpha = 0.0
                self.readiness_open = False
            elif progress >= self.force_zero_start_progress:
                # Smooth fallback withdrawal over the final curriculum band.
                # External gate withdrawals may already have made alpha lower;
                # never increase it during this safety ramp.
                if self._force_ramp_start_alpha is None:
                    self._force_ramp_start_alpha = max(float(self._upper_alpha), 0.0)
                span = max(self.force_zero_progress - self.force_zero_start_progress, 1.0e-8)
                phase = (progress - self.force_zero_start_progress) / span
                ramp_start = self._force_ramp_start_alpha
                self._upper_alpha = min(self._upper_alpha, ramp_start * (1.0 - phase))
                self.readiness_open = False

    def lower_alpha(self) -> float:
        # The historical schedule itself is monotone.  Clamp by source value
        # for numerical safety so this continuation can never increase it.
        return min(float(self.source_lower_alpha), float(lower_prior_alpha(self.global_progress)))

    def upper_alpha(self) -> float:
        assert self._upper_alpha is not None
        if self.hard_zero_enabled and self.global_progress >= self.force_zero_progress:
            return 0.0
        if self.current_update < self.hold_updates or not self.readiness_open:
            return float(self._upper_alpha)
        # A single call may happen many physics steps per PPO update; only
        # update this state from the runner.  The runner calls advance_upper
        # exactly once per update before rollout collection.
        return float(self._upper_alpha)

    def advance_upper_once(self) -> float:
        """Apply at most one readiness-gated, monotone upper withdrawal step."""
        assert self._upper_alpha is not None
        if self.current_update >= self.hold_updates and self.readiness_open:
            self._upper_alpha = max(0.0, self._upper_alpha - self.upper_step)
        return float(self._upper_alpha)

    def set_readiness(self, value: bool) -> None:
        # Gate can pause withdrawal but may never increase alpha.
        self.readiness_open = bool(value)

    def snapshot(self) -> dict[str, float | int | bool]:
        return {
            "source_progress": self.source_progress,
            "source_lower_alpha": self.source_lower_alpha,
            "source_upper_alpha": self.source_upper_alpha,
            "current_update": self.current_update,
            "global_progress": self.global_progress,
            "lower_alpha": self.lower_alpha(),
            "upper_alpha": self.upper_alpha(),
            "hold_updates": self.hold_updates,
            "upper_step": self.upper_step,
            "controllability_recovery_enabled": self.controllability_recovery_enabled,
            "controllability_start_alpha": self.controllability_start_alpha,
            "controllability_min_alpha": self.controllability_min_alpha,
            "force_zero_start_progress": self.force_zero_start_progress,
            "force_zero_progress": self.force_zero_progress,
            "hard_zero_enabled": self.hard_zero_enabled,
            "readiness_open": self.readiness_open,
        }


def normal_kernel(error_rad: float, std_rad: float) -> float:
    return math.exp(-(float(error_rad) ** 2) / (float(std_rad) ** 2))


def velocity_kernel(error_mps: float, std_mps: float) -> float:
    return math.exp(-(float(error_mps) ** 2) / (float(std_mps) ** 2))


def velocity_position_gate(position_error_m: float, threshold_m: float, excess_std_m: float) -> float:
    excess = max(float(position_error_m) - float(threshold_m), 0.0)
    return math.exp(-(excess**2) / (float(excess_std_m) ** 2))


def strike_temporal_weight(tau_s: float, std_s: float) -> float:
    return math.exp(-0.5 * (float(tau_s) / float(std_s)) ** 2)
