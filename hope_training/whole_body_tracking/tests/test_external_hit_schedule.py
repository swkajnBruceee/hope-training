import pytest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.utils.external_hit_schedule import schedule_external_hit_time


def _schedule(requested_time_s: float):
    return schedule_external_hit_time(
        requested_time_s=requested_time_s,
        control_dt_s=0.02,
        initial_prelude_steps=50,
        motion_hit_frame=30,
        precommit_phase_steps=2,
        max_added_delay_s=0.50,
    )


def test_schedule_is_measured_from_target_commit_not_reference_rewind():
    result = _schedule(2.06)
    assert result["native_hit_time_s"] == pytest.approx(1.56)
    assert result["added_ready_hold_steps"] == 25
    assert result["added_ready_hold_s"] == pytest.approx(0.50)
    assert result["scheduled_hit_time_s"] == pytest.approx(2.06)


def test_schedule_rejects_earlier_than_native_swing():
    with pytest.raises(ValueError, match="earlier than the verified native swing"):
        _schedule(1.50)


def test_schedule_rejects_more_than_verified_ready_hold_limit():
    with pytest.raises(ValueError, match="exceeds the verified READY-hold limit"):
        _schedule(2.08)
