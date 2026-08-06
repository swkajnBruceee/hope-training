"""Dependency-free source/contract checks for the bounded A3 Base Stand task."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "training" / "tasks" / "base_locomotion"
CONTRACT_ROOT = ROOT / "contracts" / "a3_base_locomotion_v1"


def _json(name: str) -> dict:
    return json.loads((CONTRACT_ROOT / name).read_text(encoding="utf-8"))


def test_all_base_stand_python_sources_parse():
    sources = list(TASK_ROOT.rglob("*.py")) + [
        ROOT / "tools" / "run_a3_base_stand_audit.py",
        ROOT / "tools" / "run_a3_base_stand_waist_scan.py",
        ROOT / "tools" / "run_a3_base_stand_support_audit.py",
        ROOT / "tools" / "run_a3_base_stand_causal_audit.py",
    ]
    assert sources
    for source in sources:
        ast.parse(source.read_text(encoding="utf-8"), filename=str(source))


def test_stand_task_matches_bounded_gate():
    gate = _json("stand_fixture_gate_v1.json")
    execution = gate["stand_smoke_execution_contract"]
    status = gate["qualification_status"]
    task_yaml = (ROOT / "cfg" / "task" / "A3BaseStand.yaml").read_text(encoding="utf-8")
    env_source = (TASK_ROOT / "base_env_cfg.py").read_text(encoding="utf-8")
    assert execution["task"] == "A3BaseStand-v0"
    assert "gym_task: A3BaseStand-v0" in task_yaml
    assert status["stand_smoke_approved"] is True
    assert status["stand_long_training_approved"] is False
    assert status["locomotion_command_approved"] is False
    assert status["deployment_approved"] is False
    assert "raw_clip=0.25" in env_source
    assert "self.decimation = 4" in env_source
    assert "self.sim.dt = 0.005" in env_source


def test_action_scale_and_dimensions_match_contract():
    action = _json("action_schema.json")
    actor = _json("actor_observation_schema.json")
    critic = _json("critic_observation_schema.json")
    source = (TASK_ROOT / "base_env_cfg.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "A3_BASE_ACTION_SCALE_RAD" for target in node.targets)
    )
    scales = list(ast.literal_eval(assignment.value))
    assert scales == action["candidate_action_scale_rad"]
    assert len(scales) == action["action_joint_names"].__len__() == 14
    assert actor["total_dimension"] == 925
    assert critic["total_dimension"] == 970


def test_actor_source_does_not_read_privileged_contact_or_linear_velocity():
    source = (TASK_ROOT / "mdp" / "observations.py").read_text(encoding="utf-8")
    actor_section = source.split("class A3BaseActorObservation", 1)[0]
    # The shared actor builder may read root angular velocity and deployable
    # joint/IMU state, but never simulator true linear velocity or contact data.
    assert "root_lin_vel" not in actor_section
    assert "net_forces" not in actor_section
    critic_section = source.split("class A3BaseCriticObservation", 1)[1]
    assert "root_lin_vel_b" in critic_section
    assert "net_forces_w" in critic_section


def test_stand_source_contains_no_command_sampler_or_randomization_event():
    source = (TASK_ROOT / "base_env_cfg.py").read_text(encoding="utf-8")
    commands = source.split("class CommandsCfg", 1)[1].split("class ActionsCfg", 1)[0]
    events = source.split("class EventCfg", 1)[1].split("class RewardsCfg", 1)[0]
    assert "pass" in commands
    assert "random" not in events.lower()
    assert "push" not in events.lower()
    assert "strike" not in events.lower()


def test_waist_scan_is_bounded_and_cannot_promote_gates():
    source = (ROOT / "tools" / "run_a3_base_stand_waist_scan.py").read_text(encoding="utf-8")
    assert "abs(value) > 0.25" in source
    assert '"changes_authorized_by_this_report": []' in source
    assert '"stand_phase1_qualified": False' in source
    assert '"stand_long_training_approved": False' in source
    assert '"deployment_approved": False' in source


def test_authority_candidate_is_one_variable_and_fail_closed():
    env_source = (
        TASK_ROOT / "config" / "a3" / "stand_env_cfg.py"
    ).read_text(encoding="utf-8")
    registry_source = (
        TASK_ROOT / "config" / "a3" / "__init__.py"
    ).read_text(encoding="utf-8")
    train_source = (ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    gate = _json("stand_authority_candidate_gate_v1.json")
    assert '"waist_pitch_joint": 350.0' in env_source
    assert '"waist_pitch_joint": 7.0' in env_source
    assert "A3BaseStandAuthorityCandidate-v0" in registry_source
    assert "int(max_iterations) != 100" in train_source
    assert "int(num_envs) <= 64" in train_source
    assert gate["qualification_status"]["stand_authority_candidate_smoke_approved"] is True
    assert gate["qualification_status"]["candidate_gain_contract_approved"] is False
    assert gate["qualification_status"]["stand_long_training_approved"] is False
    assert gate["qualification_status"]["deployment_approved"] is False


def test_clip_candidate_is_one_variable_and_fail_closed():
    env_source = (
        TASK_ROOT / "config" / "a3" / "stand_env_cfg.py"
    ).read_text(encoding="utf-8")
    registry_source = (
        TASK_ROOT / "config" / "a3" / "__init__.py"
    ).read_text(encoding="utf-8")
    train_source = (ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    gate = _json("stand_clip_candidate_gate_v1.json")
    assert "self.actions.base.raw_clip = 0.5" in env_source
    assert "A3BaseStandClipCandidate-v0" in registry_source
    assert "A3BaseStandClipCandidate-v0 is approved for exactly one" in train_source
    assert gate["qualification_status"]["stand_clip_candidate_smoke_approved"] is True
    assert gate["qualification_status"]["candidate_action_contract_approved"] is False
    assert gate["qualification_status"]["stand_long_training_approved"] is False
    assert gate["qualification_status"]["deployment_approved"] is False


def test_authority_clip_candidate_closes_factorial_without_promotion():
    registry_source = (
        TASK_ROOT / "config" / "a3" / "__init__.py"
    ).read_text(encoding="utf-8")
    gate = _json("stand_authority_clip_candidate_gate_v1.json")
    assert "A3BaseStandAuthorityClipCandidate-v0" in registry_source
    assert set(gate["factorial_cells"]) == {
        "baseline",
        "gain_only",
        "clip_only",
        "gain_and_clip",
    }
    assert gate["qualification_status"]["factorial_final_smoke_approved"] is True
    assert gate["qualification_status"]["candidate_action_contract_approved"] is False
    assert gate["qualification_status"]["candidate_gain_contract_approved"] is False
    assert gate["qualification_status"]["stand_long_training_approved"] is False
    assert gate["qualification_status"]["deployment_approved"] is False


def test_causal_decision_closes_all_additional_ppo_and_freezes_low_noise_candidate():
    decision = _json("stand_causal_audit_decision_v1.json")
    status = decision["qualification_status"]
    train_source = (ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    candidate = (
        ROOT / "cfg" / "algo" / "ppo_a3_base_stand_low_noise_candidate.yaml"
    ).read_text(encoding="utf-8")
    assert status["causal_timeline_audit_complete"] is True
    assert status["reward_accounting_audit_complete"] is True
    assert status["static_working_point_approved"] is False
    assert status["additional_ppo_smoke_approved"] is False
    assert status["extended_smoke_approved"] is False
    assert status["stand_long_training_approved"] is False
    assert status["deployment_approved"] is False
    assert "stand_causal_audit_decision_v1.json" in train_source
    assert "All additional A3 Base Stand PPO is closed" in train_source
    assert "init_noise_std: 0.15" in candidate
    assert "raw_clip" not in candidate
