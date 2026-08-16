import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_short_transition import summarize


def _row(step, strike, clip, *, capture=True, legal=True, interval=None):
    return {
        "env_id": 0,
        "episode_id": 0,
        "global_step": step,
        "strike_index": strike,
        "clip_id": clip,
        "strike_interval_s": interval,
        "capture_gate": capture,
        "net_clear": legal,
        "landing_valid": legal,
        "on_opponent": legal,
    }


def test_transition_summary_uses_consecutive_completed_shots():
    payload = {
        "schema_version": 2,
        "rows": [
            _row(10, 0, 0, legal=True, interval=None),
            _row(40, 1, 1, legal=True, interval=0.60),
            _row(70, 2, 0, legal=False, interval=0.60),
            _row(100, 3, 1, capture=False, legal=False, interval=0.60),
        ],
    }
    result = summarize(payload)

    assert result["strike_interval_s"]["count"] == 3
    assert result["strike_interval_s"]["p50"] == 0.60
    assert result["transitions"]["FH->BH"] == {
        "attempts": 2,
        "next_legal": 1,
        "rate": 0.5,
    }
    assert result["transitions"]["BH->FH"] == {
        "attempts": 1,
        "next_legal": 0,
        "rate": 0.0,
    }
