import ast
import math
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/run_a3_base_recovery_noise_std_sweep.py"


def _source() -> str:
    return TOOL.read_text(encoding="utf-8")


def _load_pure_functions(*names: str) -> dict:
    tree = ast.parse(_source())
    selected = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {"np": np, "math": math}
    exec(compile(module, str(TOOL), "exec"), namespace)
    return namespace


def test_sweep_cli_and_fail_closed_b_no_ppo_contract():
    source = _source()
    required = (
        "--trace",
        "--expected-trace-sha256",
        "--envelope-decision",
        "--expected-envelope-decision-sha256",
        "--output",
        "--group-size",
        "default=16",
        "--steps",
        "default=500",
        "--stds",
        'default="0.15,0.10,0.075,0.05,0.025,0.0"',
        "--seed",
        "Trace SHA-256 mismatch",
        "Envelope decision SHA-256 mismatch",
        'approved.get("name") != "B_core_only"',
        'decision.get("authorizes_ppo") is not False',
        '"approval": False',
    )
    for token in required:
        assert token in source


def test_sweep_uses_real_manager_and_synchronized_fresh_base14_samples():
    source = _source()
    required = (
        '"A3BaseStandRecoveryA-v0"',
        'get_term("base")',
        "base_standard_normal_sample = torch.randn",
        "base_standard_normal_sample.unsqueeze(0) * std_tensor[:, None, None]",
        "sampled[std0_index].zero_()",
        "env.step(action)",
        "action_term.raw_actions",
        "action_term.processed_actions",
        "robot.data.applied_torque",
        "shared_standard_normal_sample_across_std_groups",
        "independent_sample_each_policy_step",
        "real_action_manager_clip_scale_pd",
        "same_clean_trace_index_reused_by_all_std_groups",
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


def test_sweep_reports_required_recovery_physics_and_clip_metrics():
    source = _source()
    required = (
        "episode_events(",
        "summarize_events(",
        "transient_recovery_rate",
        "durable_recovery_rate",
        "final_1s_stable_rate",
        "confirmed_recovery_time_s",
        "durable_recovery_time_s",
        "exit_cycle_count",
        "survival_rate",
        "termination_counts",
        "max_pelvis_roll_rad",
        "max_pelvis_pitch_rad",
        "root_linear_velocity_rms_m_s",
        "root_angular_velocity_rms_rad_s",
        '"sampled"',
        '"effective"',
        '"residual"',
        '"action_rate"',
        '"qtarget"',
        '"qactual"',
        '"tracking_error"',
        '"torque_nm"',
        "sampled_clip_fraction",
        "effective_saturation_fraction",
        "math.erfc",
        "theoretical_two_sided_probability_erfc",
        "measured_overall",
        "measured_by_joint",
        "std0_passive_like_transport",
    )
    for token in required:
        assert token in source


def test_sweep_has_non_gating_heuristic_deltas_trends_and_safe_output():
    source = _source()
    required = (
        "delta_vs_std0_passive_like",
        "diagnostic_clean_compatibility_heuristic",
        "highest_compatible_nonzero_std",
        "does_not_approve_or_modify_training_std",
        "transient_drop_max_pp",
        "durable_drop_max_pp",
        "final_1s_drop_max_pp",
        "confirmed_recovery_p90_increase_max_s",
        "new_non_timeout_terminations_allowed",
        "monotonic_trend_review",
        "violations",
        '"is_gate": False',
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


def test_parse_stds_requires_six_unique_finite_values_and_zero():
    parse = _load_pure_functions("_parse_stds")["_parse_stds"]
    assert parse("0.15,0.10,0.075,0.05,0.025,0.0") == (
        0.15, 0.10, 0.075, 0.05, 0.025, 0.0
    )
    for invalid in (
        "0.1,0.0",
        "0.1,0.1,0.08,0.06,0.04,0.0",
        "0.1,0.09,0.08,0.07,0.06,0.05",
        "0.1,0.09,0.08,0.07,-0.01,0.0",
        "0.1,0.09,0.08,0.07,nan,0.0",
    ):
        with pytest.raises(ValueError):
            parse(invalid)


def test_clip_and_percentile_helpers_are_numerically_explicit():
    functions = _load_pure_functions("_rms_peak", "_signed_fractions", "_percentiles")
    summary = functions["_rms_peak"](np.asarray([-2.0, 0.0, 2.0]))
    np.testing.assert_allclose(summary["rms"], np.sqrt(8.0 / 3.0))
    assert summary["peak_abs"] == 2.0
    fractions = functions["_signed_fractions"](
        np.asarray([-0.3, -0.1, 0.1, 0.3]), 0.25
    )
    assert fractions == {
        "count": 4, "positive": 0.25, "negative": 0.25, "two_sided": 0.5
    }
    assert functions["_percentiles"]([]) == {"p50": None, "p90": None, "p95": None}
    percentiles = functions["_percentiles"]([0.0, 1.0, 2.0, 3.0, 4.0])
    assert percentiles == {"p50": 2.0, "p90": 3.6, "p95": 3.8}


def _comparison_row(std=0.0, transient=0.90, durable=0.85, final=0.80, p90=2.0,
                    terminations=0, action=0.1, torque=1.0, finite=True):
    return {
        "std": std,
        "transient_recovery_rate": transient,
        "durable_recovery_rate": durable,
        "final_1s_stable_rate": final,
        "confirmed_recovery_p90_s": p90,
        "non_timeout_terminations": terminations,
        "overall_effective_action_rms": action,
        "overall_torque_rms_nm": torque,
        "runtime_finite": finite,
    }


def test_diagnostic_compatibility_heuristic_thresholds_and_fail_closed_cases():
    heuristic = _load_pure_functions("_compatibility_heuristic")[
        "_compatibility_heuristic"
    ]
    baseline = _comparison_row()
    passing = _comparison_row(
        std=0.05, transient=0.88, durable=0.80, final=0.75, p90=3.0
    )
    assert heuristic(passing, baseline) == {"compatible": True, "reasons": []}

    failing = _comparison_row(
        std=0.10, transient=0.879, durable=0.799, final=0.749,
        p90=3.01, terminations=1, finite=False,
    )
    report = heuristic(failing, baseline)
    assert report["compatible"] is False
    assert len(report["reasons"]) == 6

    unavailable = _comparison_row(std=0.15, p90=None)
    assert "confirmed_recovery_p90_unavailable" in heuristic(unavailable, baseline)["reasons"]


def test_monotonic_review_returns_violations_without_gate_decision():
    review = _load_pure_functions("_monotonic_violations")["_monotonic_violations"]
    rows = [
        _comparison_row(std=0.15, transient=0.8, action=0.15, torque=2.0),
        _comparison_row(std=0.10, transient=0.7, action=0.17, torque=2.2),
        _comparison_row(std=0.0, transient=0.9, action=0.0, torque=1.0),
    ]
    violations = review(rows)
    assert {item["metric"] for item in violations} == {
        "transient_recovery_rate",
        "overall_effective_action_rms",
        "overall_torque_rms_nm",
    }
