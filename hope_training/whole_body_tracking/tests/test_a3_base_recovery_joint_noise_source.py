import ast
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/run_a3_base_recovery_joint_noise_audit.py"


def _source() -> str:
    return TOOL.read_text(encoding="utf-8")


def _load_pure_functions(*names: str) -> dict:
    tree = ast.parse(_source())
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {"np": np}
    exec(compile(module, str(TOOL), "exec"), namespace)
    return namespace


def test_joint_noise_audit_uses_real_recovery_env_without_training_runner():
    source = _source()
    required = (
        '"A3BaseStandRecoveryA-v0"',
        'get_term("base")',
        "env.step(action)",
        "action_term.raw_actions",
        "action_term.processed_actions",
        "robot.data.applied_torque",
        "A3_BASE_ACTION_JOINTS",
        "zero_residual_baseline",
        "diagnostic_only",
    )
    for token in required:
        assert token in source
    for forbidden in (
        "RslRlVecEnvWrapper",
        "MyOnPolicyRunner",
        "OnPolicyRunner",
        "runner.learn",
        "runner.load",
        "get_inference_policy",
    ):
        assert forbidden not in source


def test_joint_noise_audit_is_fail_closed_on_trace_and_b_core_decision():
    source = _source()
    required = (
        "--trace",
        "--expected-trace-sha256",
        "--envelope-decision",
        "--expected-envelope-decision-sha256",
        "--group-size",
        "default=16",
        "--steps",
        "default=500",
        "--noise-std",
        "default=0.15",
        "AppLauncher.add_app_launcher_args(parser)",
        "Trace SHA-256 mismatch",
        "Envelope decision SHA-256 mismatch",
        'approved.get("name") != "B_core_only"',
        'decision.get("authorizes_ppo") is not False',
        "selected_clean_trace_index_sha256",
        "same_clean_trace_index_reused_by_all_groups",
    )
    for token in required:
        assert token in source


def test_joint_noise_audit_has_synchronized_group_masks_and_redundancy_label():
    source = _source()
    required = (
        '"passive_zero"',
        '"all_base14"',
        '"hip_knee_only"',
        '"ankle_only"',
        '"waist_only"',
        '"ankle_waist_frozen"',
        "base_sample.unsqueeze(0) * group_masks.unsqueeze(1)",
        "shared_base_normal_sample_across_groups",
        "independent_sample_each_policy_step",
        "redundant_control_group",
        "ankle_waist_frozen_equals_hip_knee_only",
    )
    for token in required:
        assert token in source


def test_joint_noise_audit_reports_joint_physics_distribution_and_association():
    source = _source()
    required = (
        "sampled_raw_action",
        "sampled_raw_clip_fraction",
        "effective_clipped_action_rms",
        "effective_residual_rms_rad",
        "action_rate_rms_per_policy_step",
        "q_target_residual_rms_rad",
        "q_actual_deviation_from_default_rms_rad",
        "tracking_error_rms_rad",
        "applied_torque_rms_nm",
        "applied_torque_peak_abs_nm",
        "effective_saturation_fraction",
        "b_core_outside_vs_inside_action",
        "association_not_causation",
        "sensitivity_ranking",
        "math.erfc",
        "theoretical_two_sided_raw_clip_probability",
        "action_scale_rad",
        "pd_and_physical_contract",
    )
    for token in required:
        assert token in source


def test_joint_noise_audit_uses_b_state_machine_and_safe_output_lifecycle():
    source = _source()
    required = (
        "episode_events(",
        "summarize_events(",
        "transient_recovery_rate",
        "durable_recovery_rate",
        "final_1s_stable_rate",
        "confirmed_recovery_time_s",
        "exit_cycle_count",
        "termination_counts",
        "max_pelvis_root_core_state",
        "completed_steps % 25 == 0",
        "runtime_finite",
        "allow_nan=False",
        'temporary = args_cli.output.with_name(f".{args_cli.output.name}.tmp")',
        "temporary.replace(args_cli.output)",
        "recovery_training_approved",
        "ppo_approved",
        "deployment_approved",
        "env.close()",
        "simulation_app.close()",
    )
    for token in required:
        assert token in source


def test_group_masks_are_exact_and_frozen_group_is_redundant():
    functions = _load_pure_functions("_build_group_masks")
    names = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    ]
    masks = functions["_build_group_masks"](names)
    assert list(masks) == [
        "passive_zero",
        "all_base14",
        "hip_knee_only",
        "ankle_only",
        "waist_only",
        "ankle_waist_frozen",
    ]
    assert masks["passive_zero"].sum() == 0
    assert masks["all_base14"].sum() == 14
    assert masks["hip_knee_only"].sum() == 8
    assert masks["ankle_only"].sum() == 4
    assert masks["waist_only"].sum() == 2
    np.testing.assert_array_equal(
        masks["ankle_waist_frozen"], masks["hip_knee_only"]
    )


def test_clip_and_outside_inside_helpers_on_synthetic_data():
    functions = _load_pure_functions(
        "_signal_metrics", "_signed_clip_fractions", "_outside_inside_action"
    )
    summary = functions["_signal_metrics"](np.asarray([-2.0, 0.0, 2.0]))
    assert summary["mean"] == 0.0
    np.testing.assert_allclose(summary["rms"], np.sqrt(8.0 / 3.0))
    assert summary["peak_abs"] == 2.0

    clipped = functions["_signed_clip_fractions"](
        np.asarray([-0.30, -0.10, 0.10, 0.30]), 0.25
    )
    assert clipped["positive"] == 0.25
    assert clipped["negative"] == 0.25
    assert clipped["two_sided"] == 0.5

    association = functions["_outside_inside_action"](
        np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        np.asarray([[False, True], [False, True]]),
        np.ones((2, 2), dtype=bool),
    )
    assert association["outside_mean_abs_action"] == 3.0
    assert association["inside_mean_abs_action"] == 2.0
    assert association["outside_inside_mean_ratio"] == 1.5

    zero_denominator = functions["_outside_inside_action"](
        np.asarray([[0.0, 1.0]]),
        np.asarray([[False, True]]),
        np.ones((1, 2), dtype=bool),
    )
    assert zero_denominator["outside_inside_mean_ratio"] is None
