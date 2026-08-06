"""Dependency-free checks for the one-shot external target audit contract."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from summarize_target_response_audits import summarize_group  # noqa: E402


def _method(source_path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def test_external_target_is_a_latched_position_only_override():
    path = ROOT / "training" / "tasks" / "tracking" / "mdp" / "hope_commands.py"
    setter = _method(path, "RacketTargetCommand", "set_external_target_position_b")
    assigned_attributes = {
        node.targets[0].value.attr
        for node in ast.walk(setter)
        if isinstance(node, ast.Assign)
        and node.targets
        and isinstance(node.targets[0], ast.Subscript)
        and isinstance(node.targets[0].value, ast.Attribute)
    }
    assert "_external_target_position_b" in assigned_attributes
    assert "_external_target_position_active" in assigned_attributes
    assert "racket_target_vel_w" not in assigned_attributes
    assert "racket_target_normal_w" not in assigned_attributes


def test_manifest_resampling_reapplies_external_position_before_base_logic():
    path = ROOT / "training" / "tasks" / "tracking" / "mdp" / "hope_commands.py"
    source = ast.get_source_segment(
        path.read_text(encoding="utf-8"),
        _method(path, "RacketTargetCommand", "_resample_command"),
    )
    assert source is not None
    assert source.index("self._apply_external_target_position(env_ids)") < source.index(
        "if self.cfg.target_mode == \"manifest\":",
        source.index("self._apply_external_target_position(env_ids)"),
    )


def test_model900_exposes_the_exact_consumed_observation():
    path = (
        ROOT
        / "training"
        / "tasks"
        / "base_locomotion"
        / "mdp"
        / "actions.py"
    )
    source = path.read_text(encoding="utf-8")
    assert "def upper_last_observation(self)" in source
    assert source.count("self._upper_last_observation[:] = upper_obs") == 3


def test_play_audit_records_paired_actor_and_racket_responses():
    play = (ROOT / "scripts" / "play.py").read_text(encoding="utf-8")
    config = (ROOT / "cfg" / "play.yaml").read_text(encoding="utf-8")
    schedule = (ROOT / "training" / "utils" / "external_hit_schedule.py").read_text(
        encoding="utf-8"
    )
    assert "target_offset_grid_cm" in config
    assert "external_target_position_b" in config
    assert "external_target_offset_b" in config
    assert "external_strike_request_path" in config
    assert "external_hit_time_s" in config
    assert "external_hit_max_added_delay_s" in config
    assert "initial_upper_observation_normalized" in play
    assert "initial_upper_actor_output" in play
    assert "actual_response_from_nominal_b_m" in play
    assert "position_jacobian_column" in play
    assert "root_position_jacobian_column" in play
    assert "racket_relative_root_jacobian_column" in play
    assert "external hit schedule" in play
    assert "schedule_external_hit_time" in play
    assert "load_external_strike_request" in play
    assert "external_hit_time_s is earlier than the verified native swing" in schedule
    assert "precommit_phase_steps" in play


def test_summary_distinguishes_actor_change_from_useful_racket_response(
    tmp_path: Path,
):
    def trial(trial_id, offset, actor_delta):
        return {
            "trial_id": trial_id,
            "requested_offset_b_m": offset,
            "position_error_m": 0.05,
            "hit_upper_actor_output": [actor_delta],
            "nominal_paired_response": {
                "directional_cosine": 0.0,
                "along_command_gain": 0.0,
                "initial_actor_response_l2": abs(actor_delta),
            },
        }

    report = {
        "audit": "external_racket_position_conditioning",
        "motion_id": 0,
        "complete": True,
        "physical_termination_count": 0,
        "trials": [
            trial(0, [0.0, 0.0, 0.0], 0.0),
            trial(1, [-0.01, 0.0, 0.0], -0.1),
            trial(2, [0.01, 0.0, 0.0], 0.1),
        ],
        "axis_pairs": [
            {
                "axis": "x",
                "radius_m": 0.01,
                "position_jacobian_column": [0.0, 0.0, 0.0],
            }
        ],
    }
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    summary = summarize_group([path])
    assert summary["actor_response"]["initial_l2_median"] == 0.1
    assert summary["target_response"]["useful_direction_and_gain_count"] == 0
    assert (
        summary["classification"]
        == "actor_changes_but_racket_mapping_is_unreliable"
    )
