"""Dependency-free tests for the A3 Base Locomotion Phase 0 contracts."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import a3_base_contract as contract  # noqa: E402
import a3_base_calibration as calibration  # noqa: E402
import build_a3_base_calibration_command as command_builder  # noqa: E402
import a3_base_command_trace as command_trace  # noqa: E402
import a3_base_fixture_comparison as fixture_comparison  # noqa: E402
import a3_base_zero_baseline_comparison as zero_comparison  # noqa: E402
import build_a3_base_fixture_trace_bundle as fixture_bundle  # noqa: E402


CONTRACT_DIR = ROOT / "contracts" / "a3_base_locomotion_v1"


def _contracts() -> dict:
    return contract.load_contracts(CONTRACT_DIR)


def test_contracts_freeze_31_29_14_and_observation_dimensions():
    summary = contract.validate_contracts(_contracts())
    assert summary["backend_dof"] == 31
    assert summary["policy_view_dof"] == 29
    assert summary["base_action_dof"] == 14
    assert summary["strike_reference_dof"] == 9
    assert summary["actor_observation_dimension"] == 925
    assert summary["critic_observation_dimension"] == 970
    assert summary["training_approved"] is False
    assert summary["fixture_runner_qualified"] is True
    assert summary["fixture_matrix_approved"] is True
    assert summary["stand_task_approved"] is True
    assert summary["stand_smoke_approved"] is True
    assert summary["stand_long_training_approved"] is False


def test_joint_order_hash_is_order_sensitive_and_matches_contract():
    contracts = _contracts()
    composer = contracts["command_composer_contract.json"]
    names = composer["backend_joint_names"]
    assert contract.ordered_name_sha256(names) == composer["joint_order_sha256"]
    swapped = list(names)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    assert contract.ordered_name_sha256(swapped) != composer["joint_order_sha256"]


def test_composer_assigns_every_joint_and_matches_golden_vectors():
    contracts = _contracts()
    assert contract.validate_golden_vectors(contracts) == ["nominal_hold", "mixed_base_and_strike"]
    composer = contracts["command_composer_contract.json"]
    command = contract.compose_command(
        composer,
        [0.0] * 14,
        [0.0, 0.0, 0.3, -0.12, 0.0, 0.8, 0.0, 0.0, 0.0],
    )
    assert len(command["joint_names"]) == 31
    assert all(len(command[field]) == 31 for field in ("q_des", "dq_des", "tau_ff", "kp", "kd"))
    assert not any(command["debug"]["joint_limit_hit"])


def test_waist_pitch_is_strike_reference_plus_bounded_base_residual():
    composer = _contracts()["command_composer_contract.json"]
    strike = [0.0, 0.1, 0.3, -0.12, 0.0, 0.8, 0.0, 0.0, 0.0]
    action = [0.0] * 14
    action[-1] = 100.0
    command = contract.compose_command(composer, action, strike)
    assert command["debug"]["clipped_base_action"][-1] == 1.0
    assert command["debug"]["waist_pitch_residual_rad"] == pytest.approx(0.12)
    assert command["q_des"][2] == pytest.approx(0.22)


def test_contract_validation_rejects_waist_ownership_collision():
    contracts = copy.deepcopy(_contracts())
    contracts["command_composer_contract.json"]["ownership_by_backend_joint"][1] = "strike"
    with pytest.raises(ValueError, match="waist roll"):
        contract.validate_contracts(contracts)


def test_actor_schema_excludes_privileged_contact_and_true_velocity():
    actor = _contracts()["actor_observation_schema.json"]
    names = {field["name"] for field in actor["history_fields"] + actor["current_fields"]}
    assert "true_base_linear_velocity_mps" not in names
    assert "foot_contact_force_xyz_n" not in names
    assert "future_reference_valid_mask" in names


def test_source_urdf_prepared_urdf_and_mujoco_semantics_match():
    composer = _contracts()["command_composer_contract.json"]
    result = contract.validate_source_assets(composer, REPO_ROOT)
    assert result["joint_semantics_checked"] == 31
    assert result["official_mujoco_physics_hz"] == pytest.approx(1000.0)
    physical = result["physical_model_audit"]
    assert physical["active_body_count"] == 32
    assert physical["urdf_total_mass_kg"] - physical["mujoco_total_mass_kg"] == pytest.approx(0.01995932)
    assert any(item["body"] == "pelvis_link" for item in physical["mismatches"])
    assert result["foot_collision_audit"]["requires_contact_calibration"] is True


def test_calibration_matrix_is_deterministic_complete_and_hashed():
    contracts = _contracts()
    first = calibration.build_matrix(contracts)
    second = calibration.build_matrix(contracts)
    assert first == second
    summary = calibration.validate_matrix(first, contracts)
    assert summary["case_count"] == 339
    assert summary["logical_case_count"] == 113
    assert summary["categories"] == [
        "base_action_step",
        "command_basis",
        "joint_zero_baseline",
        "target_transport",
        "waist_pitch_residual",
    ]
    native_categories = {
        "base_action_step",
        "waist_pitch_residual",
        "target_transport",
    }
    assert all(
        case["inputs"].get("plant_constraint") == "single_joint_fixture_v1"
        for case in first["cases"]
        if case["category"] in native_categories
    )
    first["cases"][0]["inputs"]["duration_s"] += 0.1
    with pytest.raises(ValueError, match="hash mismatch"):
        calibration.validate_matrix(first, contracts)


def test_fixture_bundle_scopes_select_the_same_bounded_cases_for_each_repeat():
    matrix = calibration.build_matrix(_contracts())
    expected_counts = {
        "low_zoh": 42,
        "representative_medium": 24,
        "waist_working_point": 9,
        "transport_200hz": 14,
        "friction_diagnostic": 15,
        "stand_fixture_approval": 89,
    }
    for scope, expected_count in expected_counts.items():
        selected_by_repeat = [
            fixture_bundle._select(matrix, scope, repeat_number)
            for repeat_number in (1, 2, 3)
        ]
        assert all(len(selected) == expected_count for selected in selected_by_repeat)
        assert [
            case["case_id"].rsplit("__r", 1)[0]
            for case in selected_by_repeat[0]
        ] == [
            case["case_id"].rsplit("__r", 1)[0]
            for case in selected_by_repeat[2]
        ]


def test_calibration_result_gate_requires_complete_safe_evidence():
    contracts = _contracts()
    matrix = calibration.build_matrix(contracts)
    category_extra = {
        "command_basis": {
            "displacement_heading_xyz_m": [0.1, 0.0, 0.0],
            "yaw_delta_rad": 0.0,
            "observed_command_axis_sign": 1,
        },
        "base_action_step": {
            "commanded_joint_delta_rad": 0.01,
            "selected_joint_peak_torque_nm": 1.0,
            "selected_joint_effort_rms_nm": 0.5,
            "selected_joint_saturation_duration_s": 0.0,
            "constraint_reaction_available": False,
            "peak_joint_delta_rad": 0.01,
            "overshoot_rad": 0.0,
            "target_band_entry_time_s": 0.2,
            "target_band_reached_and_held": True,
            "end_window_joint_delta_rad": 0.009,
            "end_window_response_ratio": 0.9,
            "end_window_error_rad": 0.001,
            "end_window_slope_radps": 0.0,
        },
        "joint_zero_baseline": {
            "selected_joint_peak_torque_nm": 0.1,
            "selected_joint_effort_rms_nm": 0.05,
            "selected_joint_saturation_duration_s": 0.0,
            "constraint_reaction_available": False,
            "end_window_mean_q_rad": 0.0,
            "end_window_drift_from_baseline_rad": 0.0,
            "peak_abs_drift_from_baseline_rad": 0.0,
            "end_window_slope_radps": 0.0,
        },
        "waist_pitch_residual": {
            "commanded_joint_delta_rad": 0.01,
            "selected_joint_peak_torque_nm": 1.0,
            "selected_joint_effort_rms_nm": 0.5,
            "selected_joint_saturation_duration_s": 0.0,
            "constraint_reaction_available": False,
            "peak_joint_delta_rad": 0.01,
            "overshoot_rad": 0.0,
            "target_band_entry_time_s": 0.2,
            "target_band_reached_and_held": True,
            "end_window_joint_delta_rad": 0.009,
            "end_window_response_ratio": 0.9,
            "end_window_error_rad": 0.001,
            "end_window_slope_radps": 0.0,
            "composer_residual_clip_hit": False,
        },
        "target_transport": {
            "transport_mode": "zero_order_hold",
            "commanded_joint_delta_rad": 0.01,
            "selected_joint_peak_torque_nm": 1.0,
            "selected_joint_effort_rms_nm": 0.5,
            "selected_joint_saturation_duration_s": 0.0,
            "constraint_reaction_available": False,
            "tracking_rmse_rad": 0.01,
            "peak_tracking_error_rad": 0.02,
            "peak_joint_acceleration_radps2": 1.0,
        },
    }
    results = []
    for case in matrix["cases"]:
        metrics = {
            "nonfinite_count": 0,
            "safety_stop": False,
            "forbidden_contact_count": 0,
            "joint_limit_hit_count": 0,
            "max_tilt_deg": 2.0,
            "min_pelvis_height_m": 1.05,
            "max_abs_joint_velocity_radps": 1.0,
            "max_abs_torque_nm": 10.0,
        }
        metrics.update(category_extra[case["category"]])
        results.append({"case_id": case["case_id"], "metrics": metrics})
    artifact = {"matrix_sha256": matrix["matrix_sha256"], "results": results}
    summary = calibration.validate_result_artifact(artifact, matrix, contracts)
    assert summary["safety_envelope_passed"] is True
    assert summary["automatic_promotion"] is False
    artifact["results"][0]["metrics"]["max_tilt_deg"] = 7.0
    summary = calibration.validate_result_artifact(artifact, matrix, contracts)
    assert summary["safety_envelope_passed"] is False
    assert summary["violations"] == [f"{matrix['cases'][0]['case_id']}: tilt"]


def test_single_case_result_validation_never_implies_matrix_promotion():
    contracts = _contracts()
    result = {
        "case_id": "pilot",
        "metrics": {
            "nonfinite_count": 0,
            "safety_stop": False,
            "forbidden_contact_count": 0,
            "joint_limit_hit_count": 0,
            "max_tilt_deg": 1.0,
            "min_pelvis_height_m": 1.05,
            "max_abs_joint_velocity_radps": 1.0,
            "max_abs_torque_nm": 1.0,
            "commanded_joint_delta_rad": 0.01,
            "selected_joint_peak_torque_nm": 1.0,
            "selected_joint_effort_rms_nm": 0.5,
            "selected_joint_saturation_duration_s": 0.0,
            "constraint_reaction_available": False,
            "peak_joint_delta_rad": 0.01,
            "overshoot_rad": 0.0,
            "target_band_entry_time_s": 0.2,
            "target_band_reached_and_held": True,
            "end_window_joint_delta_rad": 0.009,
            "end_window_response_ratio": 0.9,
            "end_window_error_rad": 0.001,
            "end_window_slope_radps": 0.0,
        },
    }
    summary = calibration.validate_case_result(
        result, "base_action_step", contracts
    )
    assert summary["safety_envelope_passed"] is True
    assert summary["matrix_coverage_complete"] is False
    assert summary["automatic_promotion"] is False
    result["metrics"]["end_window_response_ratio"] = float("nan")
    with pytest.raises(ValueError, match="non-finite step metric"):
        calibration.validate_case_result(result, "base_action_step", contracts)


def test_shared_trace_uses_four_causal_200hz_substeps_without_future_target():
    contracts = _contracts()
    matrix = calibration.build_matrix(contracts)
    case = next(
        item
        for item in matrix["cases"]
        if item["case_id"].startswith(
            "transport__200hz__linear_substep_interpolation__left_hip_roll_joint__r01"
        )
    )
    trace, metadata = command_trace.build_trace(case, contracts)
    summary = command_trace.validate_trace(trace, metadata, contracts)
    assert summary["physics_rate_hz"] == pytest.approx(200.0)
    assert metadata["substeps_per_policy_command"] == 4
    joint_index = trace["joint_names"].tolist().index("left_hip_roll_joint")
    first = 200
    baseline = trace["composed_policy_target_rad"][first - 1, joint_index]
    current = trace["composed_policy_target_rad"][first, joint_index]
    delta = current - baseline
    fractions = [
        (trace["composed_target_rad"][first + offset, joint_index] - baseline)
        / delta
        for offset in range(4)
    ]
    assert fractions == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert trace["command_publish_time_s"][first] == pytest.approx(1.0)
    assert trace["first_effective_physics_step_time_s"][first] == pytest.approx(1.0)
    assert trace["state_sample_time_s"][first] == pytest.approx(1.005)
    assert metadata["future_policy_target_accessed"] is False


def test_zero_baseline_trace_is_constant_and_selects_joint_explicitly():
    contracts = _contracts()
    matrix = calibration.build_matrix(contracts)
    case = next(
        item for item in matrix["cases"] if item["case_id"] == "zero__left_hip_roll_joint__r01"
    )
    trace, metadata = command_trace.build_trace(case, contracts, 200.0)
    command_trace.validate_trace(trace, metadata, contracts)
    index = trace["joint_names"].tolist().index("left_hip_roll_joint")
    assert metadata["category"] == "joint_zero_baseline"
    assert metadata["selected_joint_name"] == "left_hip_roll_joint"
    assert np.all(trace["base_action"] == 0.0)
    assert np.all(trace["composed_target_rad"][:, index] == trace["composed_target_rad"][0, index])


def test_trace_and_instance_hashes_separate_shared_command_from_model_instance():
    contracts = _contracts()
    matrix = calibration.build_matrix(contracts)
    cases = [
        item
        for item in matrix["cases"]
        if item["case_id"].startswith(
            "step__a0.10__left_hip_pitch_joint__pos__r"
        )
    ]
    first_trace, first_meta = command_trace.build_trace(cases[0], contracts, 200.0)
    second_trace, second_meta = command_trace.build_trace(cases[1], contracts, 200.0)
    assert command_trace.trace_sha256(first_trace) == command_trace.trace_sha256(
        second_trace
    )
    assert first_meta["logical_case_definition_sha256"] == second_meta[
        "logical_case_definition_sha256"
    ]
    kwargs = {
        "trace_metadata": first_meta,
        "fixture_contract": contracts["calibration_contract.json"][
            "native_mujoco_runner"
        ],
        "initial_q_rad": [0.0] * 31,
        "kp": [1.0] * 31,
        "kd": [0.1] * 31,
    }
    first_instance = command_trace.case_instance_sha256(
        model_sha256="a" * 64, **kwargs
    )
    second_instance = command_trace.case_instance_sha256(
        model_sha256="b" * 64, **kwargs
    )
    assert first_instance != second_instance


def test_replayable_calibration_command_has_frozen_segments_and_composer_output():
    contracts = _contracts()
    matrix = calibration.build_matrix(contracts)
    case = next(
        item
        for item in matrix["cases"]
        if item["category"] == "base_action_step"
        and item["inputs"]["base_action"][0] == 0.10
    )
    payload, facts = command_builder.build_case_payload(case, contracts)
    assert payload["q_des"].shape == (150, 31)
    assert payload["timestamps_s"][-1] == pytest.approx(2.98)
    assert facts["pre_hold_ticks"] == 50
    assert facts["step_hold_ticks"] == 50
    assert facts["post_hold_ticks"] == 50
    assert facts["changed_joint_names"] == ["left_hip_pitch_joint"]
    left_hip_pitch = payload["joint_names"].tolist().index("left_hip_pitch_joint")
    assert payload["q_des"][49, left_hip_pitch] == pytest.approx(-0.1311)
    assert payload["q_des"][50, left_hip_pitch] == pytest.approx(-0.06235)
    assert payload["q_des"][100, left_hip_pitch] == pytest.approx(-0.1311)


def test_calibration_command_refuses_policy_and_native_substep_cases():
    contracts = _contracts()
    matrix = calibration.build_matrix(contracts)
    basis = next(
        item for item in matrix["cases"] if item["category"] == "command_basis"
    )
    with pytest.raises(ValueError, match="requires_a_trained_base_policy"):
        command_builder.build_case_payload(basis, contracts)
    transport = next(
        item for item in matrix["cases"] if item["category"] == "target_transport"
    )
    with pytest.raises(ValueError, match="simulator_native_substep"):
        command_builder.build_case_payload(transport, contracts)


def test_training_gate_fails_closed_while_candidate_values_are_unapproved():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "validate_a3_base_contract.py"),
            "--require-training-approved",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["structural_validation_passed"] is True
    assert payload["training_gate_passed"] is False


def test_fixture_comparison_requires_aligned_identity_and_explicit_classification():
    time = np.arange(6, dtype=np.float64) * 0.1
    target = np.asarray([0.0, 0.0, 0.0, 0.1, 0.1, 0.1])
    evidence = {
        "time_s": time,
        "joint_q_rad": target * 0.9,
        "joint_target_rad": target,
        "joint_dq_radps": np.zeros(6),
        "joint_torque_nm": np.ones(6),
        "selected_joint_saturated": np.zeros(6, dtype=np.bool_),
    }
    result = {
        "case_id": "case",
        "trace_sha256": "a" * 64,
        "matrix_sha256": "b" * 64,
        "metrics": {
            "commanded_joint_delta_rad": 0.1,
            "end_window_joint_delta_rad": 0.09,
            "end_window_response_ratio": 0.9,
            "selected_joint_effort_rms_nm": 1.0,
            "selected_joint_peak_torque_nm": 1.0,
            "selected_joint_saturation_duration_s": 0.0,
        },
        "runner_facts": {
            "selected_joint_name": "left_hip_roll_joint",
            "ground_contact_enabled": False,
        },
        "case_validation": {"safety_envelope_passed": True},
    }
    metadata = {
        "trace_sha256": "a" * 64,
        "transport_mode": "zero_order_hold",
        "metric_window": {
            "baseline_end_s": 0.2,
            "active_start_s": 0.2,
            "active_end_s": 0.5,
        },
    }
    summary = fixture_comparison.compare_pair(
        isaac_result=result,
        isaac_evidence=evidence,
        mujoco_result=copy.deepcopy(result),
        mujoco_evidence={key: value.copy() for key, value in evidence.items()},
        trace_metadata=metadata,
        difference_labels=["expected_integrator_difference"],
        rationale="identical synthetic evidence",
    )
    assert summary["identity_and_time_alignment_pass"] is True
    assert summary["active_delta_trajectory_rmse_rad"] == 0.0
    assert summary["unexplained_blocks_stand"] is False
    with pytest.raises(ValueError, match="labels"):
        fixture_comparison.compare_pair(
            isaac_result=result,
            isaac_evidence=evidence,
            mujoco_result=result,
            mujoco_evidence=evidence,
            trace_metadata=metadata,
            difference_labels=[],
            rationale="missing classification",
        )


def test_zero_baseline_comparison_reports_symmetric_gain_difference():
    time = np.arange(11, dtype=np.float64) * 0.05
    zero_q = np.zeros(11)
    isaac_step_q = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.025, 0.05, 0.07, 0.085, 0.09])
    mujoco_step_q = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03, 0.06, 0.08, 0.10, 0.11])

    def evidence(q):
        return {
            "time_s": time,
            "joint_q_rad": q,
            "joint_target_rad": np.zeros(11),
            "joint_dq_radps": np.zeros(11),
            "joint_torque_nm": np.ones(11),
            "selected_joint_saturated": np.zeros(11, dtype=np.bool_),
        }

    def result(case_id, trace_hash, category, gain=None):
        metrics = {
            "selected_joint_effort_rms_nm": 1.0,
            "end_window_slope_radps": 0.0,
        }
        if gain is not None:
            metrics["commanded_joint_delta_rad"] = 0.1
        return {
            "case_id": case_id,
            "trace_sha256": trace_hash,
            "matrix_sha256": "b" * 64,
            "metrics": metrics,
            "runner_facts": {
                "selected_joint_name": "left_hip_roll_joint",
                "ground_contact_enabled": False,
            },
            "case_validation": {
                "category": category,
                "safety_envelope_passed": True,
            },
        }

    step_meta = {
        "trace_sha256": "s" * 64,
        "metric_window": {
            "baseline_end_s": 0.2,
            "active_start_s": 0.2,
            "active_end_s": 0.5,
        },
    }
    zero_meta = {"trace_sha256": "z" * 64}
    summary = zero_comparison.compare_step_with_zero_baselines(
        isaac_step_result=result("step", "s" * 64, "base_action_step", 0.9),
        isaac_step_evidence=evidence(isaac_step_q),
        isaac_zero_result=result("zero", "z" * 64, "joint_zero_baseline"),
        isaac_zero_evidence=evidence(zero_q),
        mujoco_step_result=result("step", "s" * 64, "base_action_step", 1.1),
        mujoco_step_evidence=evidence(mujoco_step_q),
        mujoco_zero_result=result("zero", "z" * 64, "joint_zero_baseline"),
        mujoco_zero_evidence=evidence(zero_q),
        step_trace_metadata=step_meta,
        zero_trace_metadata=zero_meta,
        classification_color="yellow",
        difference_labels=["expected_actuator_difference"],
        rationale="synthetic test",
    )
    expected = abs(0.875 - 1.05) / ((0.875 + 1.05) / 2.0)
    assert summary["gain_symmetric_difference"] == pytest.approx(expected)
    assert summary["classification_frozen"] is False


def test_reference_composer_cli_emits_full_command():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "compose_a3_command_reference.py"),
            "--base-action",
            json.dumps([0.0] * 14),
            "--strike-q-reference",
            json.dumps([0.0, 0.0, 0.3, -0.12, 0.0, 0.8, 0.0, 0.0, 0.0]),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    command = json.loads(result.stdout)
    assert command["joint_names"][0] == "waist_yaw_joint"
    assert command["joint_names"][-1] == "right_ankle_roll_joint"
    assert len(command["q_des"]) == 31
