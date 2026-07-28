"""Dependency-free regression checks for the frozen V25/V26 contracts."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v25_preserved_five_seed_evidence():
    rows = []
    for seed in range(5):
        path = (
            ROOT
            / "eval_outputs"
            / "joint_coordinator_v24"
            / f"model1499_seed{seed}_front_gain125_margin07.json"
        )
        report = json.loads(path.read_text(encoding="utf-8"))
        assert report["safety_pass_count"] == 6
        rows.extend(report["results"])

    assert len(rows) == 30
    assert sum(bool(row["safety_pass"]) for row in rows) == 30
    assert sum(float(row["position_error_m"]) < 0.10 for row in rows) == 25
    mean_position = sum(float(row["position_error_m"]) for row in rows) / len(rows)
    assert abs(mean_position - 0.08245516419410706) < 1.0e-12


def test_begin_next_shot_never_writes_physical_state():
    source_path = ROOT / "training" / "tasks" / "tracking" / "mdp" / "commands.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "begin_next_shot"
    )
    forbidden = {
        "write_joint_state_to_sim",
        "write_root_state_to_sim",
        "write_root_pose_to_sim",
        "write_root_velocity_to_sim",
    }
    called = {
        node.func.attr
        for node in ast.walk(method)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert called.isdisjoint(forbidden)


def test_v25_is_single_shot_and_v26_explicitly_enables_rearm():
    v25 = (
        ROOT
        / "cfg"
        / "task"
        / "HOPEA3JointCoordinatorV25AdaptiveStageASupport.yaml"
    ).read_text(encoding="utf-8")
    v26 = (
        ROOT
        / "cfg"
        / "task"
        / "HOPEA3JointCoordinatorV26MultiShotRearm.yaml"
    ).read_text(encoding="utf-8")

    assert "stage_a_sagittal_rearm_enabled" not in v25
    assert "stage_a_sagittal_front_gain: 1.25" in v25
    assert "HOPEA3JointCoordinatorV25AdaptiveStageASupport" in v26
    assert "stage_a_sagittal_rearm_enabled: true" in v26
    assert "stage_a_sagittal_rearm_stable_steps: 20" in v26
    assert "stage_a_sagittal_rearm_ramp_steps: 8" in v26


def test_rearm_accepts_cycle_before_new_prelude_revokes_ready():
    source = (
        ROOT / "training" / "tasks" / "base_locomotion" / "mdp" / "actions.py"
    ).read_text(encoding="utf-8")
    start = source.index("def _stage_a_sagittal_exit_gate")
    end = source.index("def _stage_a_sagittal_front_gate", start)
    method = source[start:end]

    assert method.index("accepted = new_shot") < method.index(
        "stable = self._stage_a_rearm_stability"
    )
    assert "& (~new_shot)" in method
