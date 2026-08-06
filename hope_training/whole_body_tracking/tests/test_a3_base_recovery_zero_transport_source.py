import ast
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/run_a3_base_recovery_zero_transport_audit.py"


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


def test_zero_transport_audit_has_required_runtime_chain_and_three_modes():
    source = _source()
    required = (
        '"A3BaseStandRecoveryA-v0"',
        "RslRlVecEnvWrapper",
        "MyOnPolicyRunner",
        "A3BaseStandRecoveryAPPORunnerCfg",
        "initialize_zero_residual_actor_mean",
        "runner.get_inference_policy",
        'GROUP_NAMES = ("passive_zero", "deterministic_actor_mean", "forced_zero_after_actor")',
        "obs[actor_slice]",
        "action[b_slice] = actor_mean",
        "action[c_slice] = 0.0",
        "vec_env.step(action)",
    )
    for token in required:
        assert token in source
    assert "runner.load" not in source
    assert "runner.learn" not in source


def test_zero_transport_audit_is_fail_closed_on_trace_and_b_envelope():
    source = _source()
    required = (
        "--expected-trace-sha256",
        "--expected-envelope-decision-sha256",
        "Trace SHA-256 mismatch",
        "Envelope decision SHA-256 mismatch",
        'approved.get("name") != "B_core_only"',
        'decision.get("authorizes_ppo") is not False',
        "selected_clean_trace_index_sha256",
        "same_clean_trace_index_reused_by_all_groups",
    )
    for token in required:
        assert token in source


def test_zero_transport_audit_checks_action_transport_contract():
    source = _source()
    required = (
        "A3_BASE_ACTION_JOINTS",
        "joint_order_passed",
        "action_scale_rad",
        "raw_clip_abs",
        "default_target_by_joint",
        "raw_zero_to_scaled_residual_zero_to_target_default",
        "scale_applied_once",
        "non_integrating_passed",
        "default_added_once",
        "non_base_default_passed",
        "processed - default_base",
        "default_base + expected_raw * scale",
    )
    for token in required:
        assert token in source


def test_zero_transport_audit_has_real_zoh_and_post_rollout_reset_probes():
    source = _source()
    required = (
        "types.MethodType",
        "action_term.apply_actions",
        'probe["apply_count"] += 1',
        "zoh_apply_count == int(env_cfg.decimation)",
        "zoh_targets_identical",
        "# This probe intentionally runs after all main comparison statistics.",
        "vec_env.step(nonzero_action)",
        "vec_env.reset()",
        "immediate_buffer_clear_is_informational_only",
        "first_zero_step",
    )
    for token in required:
        assert token in source


def test_zero_transport_audit_reports_metrics_hashes_and_atomic_output():
    source = _source()
    required = (
        "base14_by_joint",
        "full_joint_target_by_joint",
        "paired_trajectory_max_abs_difference",
        "trajectory_sha256",
        "runtime_finite",
        "zero_action_command_transport_verified",
        "parallel_physics_exact_replication_verified",
        "transport_audit_passed",
        "completed_steps % 25 == 0",
        'temporary = args_cli.output.with_name(f".{args_cli.output.name}.tmp")',
        "temporary.replace(args_cli.output)",
        "gym_env.close()",
        "simulation_app.close()",
    )
    for token in required:
        assert token in source


def test_summary_and_paired_difference_helpers_on_synthetic_data():
    functions = _load_pure_functions("_summary", "_paired_max_differences")
    summary = functions["_summary"](np.asarray([-2.0, 0.0, 2.0], dtype=np.float32))
    assert summary["mean"] == 0.0
    assert summary["max_abs"] == 2.0
    np.testing.assert_allclose(summary["rms"], np.sqrt(8.0 / 3.0))

    values = np.zeros((2, 3, 2, 1), dtype=np.float32)
    values[:, 1] = 0.25
    values[:, 2] = -0.5
    differences = functions["_paired_max_differences"](values)
    assert differences["passive_zero_vs_deterministic_actor_mean"] == 0.25
    assert differences["passive_zero_vs_forced_zero_after_actor"] == 0.5
    assert differences["deterministic_actor_mean_vs_forced_zero_after_actor"] == 0.75
