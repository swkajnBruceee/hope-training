"""Pure admission/scheduling arithmetic for a one-shot external strike."""

from __future__ import annotations

import math


def schedule_external_hit_time(
    *,
    requested_time_s: float,
    control_dt_s: float,
    initial_prelude_steps: int,
    motion_hit_frame: int,
    precommit_phase_steps: int,
    max_added_delay_s: float,
) -> dict[str, float | int]:
    """Schedule an impact no earlier than ``requested_time_s`` from COMMIT.

    The motion reference and hit frame are immutable.  The only allowed
    adjustment is a bounded extension of the READY/prelude hold.  Phase
    updates already performed before the external target latches are excluded
    so the reported schedule uses the caller-visible COMMIT boundary.
    """
    if control_dt_s <= 0.0 or not math.isfinite(control_dt_s):
        raise ValueError("control_dt_s must be a positive finite number")
    if requested_time_s <= 0.0 or not math.isfinite(requested_time_s):
        raise ValueError("requested_time_s must be a positive finite number")
    if initial_prelude_steps < 0 or motion_hit_frame < 0:
        raise ValueError("initial_prelude_steps and motion_hit_frame must be non-negative")
    if not 0 <= precommit_phase_steps <= initial_prelude_steps:
        raise ValueError("precommit_phase_steps must lie within the initial prelude")
    if max_added_delay_s < 0.0 or not math.isfinite(max_added_delay_s):
        raise ValueError("max_added_delay_s must be a non-negative finite number")

    native_hit_time_s = (
        initial_prelude_steps - precommit_phase_steps + motion_hit_frame
    ) * control_dt_s
    if requested_time_s < native_hit_time_s - 0.5 * control_dt_s:
        raise ValueError(
            "external_hit_time_s is earlier than the verified native swing: "
            f"requested_s={requested_time_s}, native_s={native_hit_time_s}"
        )
    added_delay_steps = max(
        0,
        math.ceil(
            (requested_time_s - native_hit_time_s) / control_dt_s - 1.0e-9
        ),
    )
    added_delay_s = added_delay_steps * control_dt_s
    if added_delay_s > max_added_delay_s + 1.0e-9:
        raise ValueError(
            "external_hit_time_s exceeds the verified READY-hold limit: "
            f"requested_added_delay_s={added_delay_s}, limit_s={max_added_delay_s}"
        )
    return {
        "request_time_from_commit_s": requested_time_s,
        "native_hit_time_s": native_hit_time_s,
        "motion_hit_frame": motion_hit_frame,
        "initial_prelude_steps": initial_prelude_steps,
        "precommit_phase_steps": precommit_phase_steps,
        "added_ready_hold_steps": added_delay_steps,
        "added_ready_hold_s": added_delay_s,
        "scheduled_hit_time_s": native_hit_time_s + added_delay_s,
        "max_added_ready_hold_s": max_added_delay_s,
    }
