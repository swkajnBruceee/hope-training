import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _profile(
    tmp_path: Path,
    name: str,
    settled_fraction: float,
    p95_time: float,
    *,
    clean: bool = False,
) -> dict:
    channels = {
        "abs_pelvis_roll_rad": {"p90": 0.006, "p95": 0.007, "p99": 0.008},
        "abs_pelvis_pitch_rad": {"p90": 0.007, "p95": 0.008, "p99": 0.009},
        "abs_root_angular_velocity_x_rad_s": {"p90": 0.03, "p95": 0.035, "p99": 0.04},
        "abs_root_angular_velocity_y_rad_s": {"p90": 0.035, "p95": 0.04, "p99": 0.045},
        "abs_root_linear_velocity_x_m_s": {"p90": 0.01, "p95": 0.015, "p99": 0.02},
        "abs_root_linear_velocity_y_m_s": {"p90": 0.015, "p95": 0.02, "p99": 0.025},
        "abs_base_height_error_m": {"p90": 0.004, "p95": 0.005, "p99": 0.006},
        "abs_joint_velocity_rad_s/waist_pitch_joint": {
            "p90": 0.02,
            "p95": 0.025,
            "p99": 0.03,
        },
    }
    trajectory = tmp_path / f"{name}.npz"
    arrays = {
        channel: np.zeros((20, 4), dtype=np.float32)
        for channel in channels
    }
    if not clean:
        for values in arrays.values():
            values[:3] = 0.2
    np.savez_compressed(
        trajectory,
        **arrays,
        active=np.ones((20, 4), dtype=np.bool_),
        disturbed=np.full(4, not clean, dtype=np.bool_),
        trace_index=np.arange(4, dtype=np.int32),
        policy_dt_s=np.asarray([0.02], dtype=np.float32),
    )
    return {
        "profile": name,
        "runtime_integrity_passed": True,
        "clean_timeout_fraction": 1.0 if clean else None,
        "disturbed_timeout_fraction": None if clean else 1.0,
        "termination_term_counts": {"time_out": 32, "joint_limit": 0},
        "disturbed_max_tilt_rad": {"max": 0.1},
        "clean_tail_statistics": channels,
        "settled_envelopes": {
            "strict": {
                "disturbed_settled_fraction": settled_fraction,
                "disturbed_settle_time_s": {"p95": p95_time},
            }
        },
        "trajectory_path": str(trajectory),
        "trajectory_sha256": hashlib.sha256(trajectory.read_bytes()).hexdigest(),
    }


def test_envelope_analysis_uses_p99_margin_and_remains_unapproved(tmp_path):
    source = tmp_path / "calibration.json"
    output = tmp_path / "envelope.json"
    source.write_text(
        json.dumps(
            {
                "calibration_measured": True,
                "disturbance_trace_sha256": "a" * 64,
                "runtime_contract_sha256": "b" * 64,
                "policy_dt_s": 0.02,
                "profiles": [
                    _profile(
                        tmp_path, "recovery_a_clean", 0.0, 0.0, clean=True
                    ),
                    _profile(tmp_path, "recovery_a_candidate", 0.90, 2.0),
                    _profile(tmp_path, "recovery_a_medium", 0.80, 3.0),
                    _profile(tmp_path, "recovery_a_upper_probe", 0.50, 5.0),
                ],
            }
        )
    )
    subprocess.run(
        [
            sys.executable,
            ROOT / "tools/analyze_a3_base_recovery_calibration.py",
            "--input",
            source,
            "--output",
            output,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.read_text())
    roll = result["envelope"]["abs_pelvis_roll_rad"]
    assert roll["enter_threshold"] == 0.01
    assert roll["exit_threshold"] == 0.0125
    assert result["clean_tail_source_profile"] == "recovery_a_clean"
    assert (
        result["recomputed_recovery"]["recovery_a_candidate"][
            "strict_recovery_fraction"
        ]
        == 1.0
    )
    assert (
        result["recomputed_recovery"]["recovery_a_candidate"][
            "post_recovery_exit_count"
        ]
        == 0
    )
    assert result["recomputed_recovery"]["recovery_a_candidate"][
        "maximum_overshoot_ratio"
    ]["count"] == 4
    assert result["provisional_first_smoke_mixture"]["supported_by_passive_calibration"] is True
    assert result["recovery_envelope_approved"] is False
    assert result["bounded_recovery_smoke_approved"] is False
