"""Plain-Python tests for official TA mapping and stance-offset semantics."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "tools" / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ta = _load("official_ta_mapping", "official_ta_mapping.py")
stance = _load("stance_offset_adapter", "stance_offset_adapter.py")
waist_probe = _load("official_waist_reference_probe", "official_waist_reference_probe.py")
_stance_contract_spec = importlib.util.spec_from_file_location(
    "stance_contract",
    _ROOT / "training" / "tasks" / "tracking" / "mdp" / "stance_contract.py",
)
assert _stance_contract_spec is not None and _stance_contract_spec.loader is not None
stance_contract = importlib.util.module_from_spec(_stance_contract_spec)
sys.modules["stance_contract"] = stance_contract
_stance_contract_spec.loader.exec_module(stance_contract)


def _command(velocities=()):
    return ta.TaWholeBodyCommand(
        pelvis_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        leg_angles_rad=tuple(200.0 + i for i in range(12)),
        waist_angles_rad=tuple(10.0 + i for i in range(3)),
        arm_angles_rad=tuple(100.0 + i for i in range(14)),
        joint_velocities_rad_s=tuple(velocities),
        stamp_ns=12_000_000_034,
    )


def test_official_position_mapping_is_waist_arm_leg():
    view = ta.map_ta_command(_command())
    assert view.q_mujoco[:3] == (10.0, 11.0, 12.0)
    assert view.q_mujoco[3] == 100.0
    assert view.q_mujoco[17] == 200.0
    assert view.stamp_ns == 12_000_000_034


def test_waist_probe_resampling_preserves_duration():
    import numpy as np

    source = np.arange(5, dtype=np.float64).reshape(-1, 1)
    result = waist_probe.resample_reference(source, source_fps=50.0, target_fps=100.0)
    assert result.shape == (9, 1)
    assert result[0, 0] == 0.0
    assert result[-1, 0] == 4.0
    assert np.isclose((result.shape[0] - 1) / 100.0, 4 / 50.0)


@pytest.mark.parametrize(
    ("size", "waist_start", "arm_start", "leg_start"),
    [(29, 0, 3, 17), (30, 12, 16, 0), (31, 12, 17, 0)],
)
def test_official_velocity_layouts(size, waist_start, arm_start, leg_start):
    view = ta.map_ta_command(_command(velocities=range(size)))
    assert view.dq_mujoco[0] == waist_start
    assert view.dq_mujoco[3] == arm_start
    assert view.dq_mujoco[17] == leg_start


def test_missing_velocity_is_zero_and_head_is_not_in_policy_view():
    view = ta.map_ta_command(_command())
    assert view.dq_mujoco == (0.0,) * 29


@pytest.mark.parametrize(
    "field, value",
    [
        ("leg_angles_rad", (0.0,) * 11),
        ("waist_angles_rad", (0.0,) * 2),
        ("arm_angles_rad", (0.0,) * 13),
    ],
)
def test_official_mapping_rejects_wrong_position_lengths(field, value):
    command = _command().__dict__.copy()
    command[field] = value
    with pytest.raises(ValueError):
        ta.map_ta_command(ta.TaWholeBodyCommand(**command))


def test_stance_adapter_preserves_world_hit_and_reaches_calibrated_offset():
    result = stance.adapt_world_hit_point(
        hit_point_w=(1.0, 0.5, 0.9),
        current_base_w=stance.BasePose(0.0, 0.0, 0.0, 0.0),
        canonical_reach_offset_b=(0.6, 0.2, 0.9),
        comfort_region_b=stance.HorizontalRegion(0.4, 0.8, 0.1, 0.35),
    )
    assert result.status == "requires_stance_offset"
    assert result.required_horizontal_offset == (0.4, 0.3)
    assert result.relocated_target_b == (0.6, 0.2, 0.9)
    assert result.hit_point_w == (1.0, 0.5, 0.9)


def test_stance_adapter_rotates_reach_with_base_yaw():
    result = stance.adapt_world_hit_point(
        hit_point_w=(0.0, 1.0, 0.9),
        current_base_w=stance.BasePose(0.0, 0.0, 0.0, 0.0),
        canonical_reach_offset_b=(1.0, 0.0, 0.9),
        target_yaw=3.141592653589793 / 2.0,
    )
    assert abs(result.target_base_pose_w.x - 0.0) < 1e-9
    assert abs(result.target_base_pose_w.y - 0.0) < 1e-9
    assert abs(result.relocated_target_b[0] - 1.0) < 1e-9
    assert abs(result.relocated_target_b[1]) < 1e-9


def test_stance_manifest_contract_validates_real_prepositioned_bank():
    import json

    path = (
        _ROOT
        / "sample_motions"
        / "p2_fixed_forehand_heldout_stance_20260716"
        / "native_zero_residual_manifest.json"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    result = stance_contract.validate_stance_manifest(
        manifest,
        manifest_file=str(path),
        check_motion_paths=True,
    )
    assert result["motion_count"] == 4
    assert result["max_offset_error_m"] < 1e-6
    assert result["max_world_hit_reconstruction_error_m"] < 1e-6
    assert result["max_npz_root_at_hit_error_m"] < 1e-6


def test_stance_manifest_contract_rejects_mismatched_offset():
    entry = {
        "episode_id": "bad",
        "strike_target": {"racket_position_m": [0.0, 0.0, 0.0]},
        "stance_metadata": {
            "stance_mode": "prepositioned",
            "original_hit_position_w_m": [1.0, 0.0, 0.0],
            "base_pose_before_w": {"position_m": [0.0, 0.0, 0.0], "yaw_rad": 0.0},
            "base_pose_target_w": {"position_m": [0.5, 0.0, 0.0], "yaw_rad": 0.0},
            "stance_offset_xy_w_m": [0.25, 0.0],
            "strike_target_base_m": [0.5, 0.0, 0.0],
        },
    }
    with pytest.raises(ValueError, match="stance offset mismatch"):
        stance_contract.validate_stance_entry(entry)


def test_stance_manifest_contract_accepts_fixed_mode_when_requested():
    entry = {
        "episode_id": "fixed",
        "strike_target": {"racket_position_m": [1.0, 0.0, 0.5]},
        "stance_metadata": {
            "stance_mode": "fixed",
            "original_hit_position_w_m": [1.0, 0.0, 0.5],
            "base_pose_before_w": {"position_m": [0.0, 0.0, 0.0], "yaw_rad": 0.0},
            "base_pose_target_w": {"position_m": [0.0, 0.0, 0.0], "yaw_rad": 0.0},
            "stance_offset_xy_w_m": [0.0, 0.0],
            "strike_target_base_m": [1.0, 0.0, 0.5],
        },
    }
    result = stance_contract.validate_stance_entry(entry, expected_mode=None)
    assert result["stance_mode"] == "fixed"


def test_stance_contract_accepts_legacy_fixed_entry_without_metadata():
    entry = {
        "episode_id": "legacy-fixed",
        "strike_target": {"racket_position_m": [1.0, 0.0, 0.5]},
    }
    result = stance_contract.validate_stance_entry(entry, expected_mode="fixed")
    assert result["stance_mode"] == "fixed"
    assert result["metadata_present"] is False


def test_stance_train_and_heldout_manifests_are_disjoint_and_valid():
    import json

    train_path = _ROOT / "sample_motions" / "p2_stance_train_k8_v1_20260716" / "manifest.json"
    heldout_path = _ROOT / "sample_motions" / "p2_stance_heldout_k4_v1_20260716" / "manifest.json"
    train = json.loads(train_path.read_text(encoding="utf-8"))
    heldout = json.loads(heldout_path.read_text(encoding="utf-8"))
    train_ids = {str(item["episode_id"]) for item in train["motions"]}
    heldout_ids = {str(item["episode_id"]) for item in heldout["motions"]}
    assert train_ids.isdisjoint(heldout_ids)
    assert train["stance_contract"]["mode"] == "mixed"
    assert heldout["stance_contract"]["mode"] == "mixed"
    assert train["stance_contract"]["walking_enabled"] is False
    assert heldout["stance_contract"]["walking_enabled"] is False
    train_result = stance_contract.validate_stance_manifest(
        train, manifest_file=str(train_path), expected_mode="mixed", check_motion_paths=True
    )
    heldout_result = stance_contract.validate_stance_manifest(
        heldout, manifest_file=str(heldout_path), expected_mode="mixed", check_motion_paths=True
    )
    assert train_result["motion_count"] == 8
    assert heldout_result["motion_count"] == 4
    assert train_result["max_npz_root_at_hit_error_m"] < 1e-6
    assert heldout_result["max_npz_root_at_hit_error_m"] < 1e-6
