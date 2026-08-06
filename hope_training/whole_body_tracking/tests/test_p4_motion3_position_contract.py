from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "verify_p4_motion3_position_contract.py"
SPEC = importlib.util.spec_from_file_location("verify_p4_motion3_position_contract", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _trial(trial_id: int, offset: list[float]) -> dict:
    target = [
        center + delta
        for center, delta in zip(MODULE.CALIBRATED_CENTER_B_M, offset)
    ]
    return {
        "trial_id": trial_id,
        "control_step": 78,
        "position_error_m": 0.003,
        "first_physical_termination_control_step": None,
        "control_offset_from_calibrated_anchor_b_m": offset,
        "target_position_b_m": target,
    }


def _passing_report() -> dict:
    trials = [
        _trial(0, [0.0, 0.0, 0.0]),
        _trial(1, [-0.01, 0.0, 0.0]),
        _trial(2, [0.01, 0.0, 0.0]),
        _trial(3, [0.0, -0.01, 0.0]),
        _trial(4, [0.0, 0.01, 0.0]),
        _trial(5, [0.0, 0.0, -0.01]),
        _trial(6, [0.0, 0.0, 0.01]),
    ]
    return {
        "motion_id": 3,
        "physical_termination_count": 0,
        "trials": trials,
        "axis_pairs": [
            {"axis": "x", "position_jacobian_column": [0.90, -0.36, 0.11]},
            {"axis": "y", "position_jacobian_column": [0.10, 0.55, -0.01]},
            {"axis": "z", "position_jacobian_column": [-0.05, 0.03, 0.97]},
        ],
    }


def test_p4_motion3_position_contract_accepts_frozen_baseline():
    assert MODULE.validate(_passing_report()) == []


def test_p4_motion3_position_contract_rejects_position_regression():
    report = _passing_report()
    report["trials"][2]["position_error_m"] = 0.02
    assert any("exceeds" in failure for failure in MODULE.validate(report))
