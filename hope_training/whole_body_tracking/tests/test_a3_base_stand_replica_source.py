"""Dependency-free invariants for the isolated A3 locomotion MDP replica."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "training" / "tasks" / "replica_locomotion"


def test_replica_sources_parse() -> None:
    sources = list(TASK_ROOT.rglob("*.py"))
    assert sources
    for source in sources:
        ast.parse(source.read_text(encoding="utf-8"), filename=str(source))


def test_replica_has_an_independent_task_and_experiment_namespace() -> None:
    registry = (TASK_ROOT / "config" / "a3" / "__init__.py").read_text(encoding="utf-8")
    yaml = (ROOT / "cfg" / "task" / "A3BaseStandReplica.yaml").read_text(encoding="utf-8")
    runner = (TASK_ROOT / "config" / "a3" / "agents" / "ppo.py").read_text(encoding="utf-8")
    assert 'id="A3BaseStandReplica-v0"' in registry
    assert "gym_task: A3BaseStandReplica-v0" in yaml
    assert 'experiment_name = "a3_base_stand_replica_h1_flat"' in runner
    assert "base_locomotion" not in "\n".join((TASK_ROOT / "a3_replica_env_cfg.py").read_text().splitlines())
    algo = (ROOT / "cfg" / "algo" / "a3_base_stand_replica_ppo.yaml").read_text(encoding="utf-8")
    assert "name: a3_base_stand_replica_ppo" in algo
    assert "empirical_normalization: false" in algo
    assert "entropy_coef: 0.01" in algo


def test_replica_is_leg_only_zero_command_and_h1_flat_shaped() -> None:
    env = (TASK_ROOT / "a3_replica_env_cfg.py").read_text(encoding="utf-8")
    action = (TASK_ROOT / "mdp" / "actions.py").read_text(encoding="utf-8")
    assert "A3_REPLICA_LEG_JOINTS" in env
    assert "A3_BASE_ACTION_JOINTS" not in env
    assert "lin_vel_x=(0.0, 0.0)" in env
    assert "lin_vel_y=(0.0, 0.0)" in env
    assert "ang_vel_z=(0.0, 0.0)" in env
    assert "termination_penalty" in env
    assert "feet_slide" in env
    assert "flat_orientation_l2" in env
    assert "A3ReplicaLegPositionAction" in action
    assert "BaseComposite" not in action
