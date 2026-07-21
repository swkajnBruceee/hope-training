import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_passive_stable_candidate_contract_is_fail_closed_after_v3():
    decision = json.loads(
        (ROOT / "contracts/a3_base_locomotion_v1/stand_passive_stable_decision_v1.json").read_text()
    )
    status = decision["qualification_status"]
    assert status["passive_stand_plant_approved"] is True
    assert status["learned_stand_policy_approved"] is False
    assert status["additional_zero_command_ppo_approved"] is False
    assert status["stand_recovery_task_development_approved"] is True
    assert status["stand_recovery_training_approved"] is False
    assert status["deployment_approved"] is False


def test_candidate_uses_base14_gains_matched_scales_and_reward_v2():
    source = (ROOT / "training/tasks/base_locomotion/config/a3/stand_env_cfg.py").read_text()
    assert "A3BaseStandPassiveStableCandidateEnvCfg" in source
    assert "A3_PD_STAND_BASE_ACTION_SCALE_RAD" in source
    assert '"waist_roll_joint": 500.0' in source
    assert '"waist_pitch_joint": 500.0' in source
    assert "termination_penalty = RewTerm" in source
    assert "action_magnitude = RewTerm" in source
    assert "normalize_by_dof\": True" in source


def test_training_entry_zero_initializes_then_gate_closes_consumed_smoke():
    train = (ROOT / "scripts/train.py").read_text()
    assert "stand_passive_stable_candidate_gate_v3.json" in train
    assert "initialize_zero_residual_actor_mean(runner, action_dim=14)" in train
    helper = (ROOT / "training/utils/a3_base_actor_init.py").read_text()
    assert "torch.nn.init.zeros_(actor_output.weight)" in helper
    gate = json.loads(
        (ROOT / "contracts/a3_base_locomotion_v1/stand_passive_stable_candidate_gate_v3.json").read_text()
    )
    assert gate["qualification_status"]["bounded_100_iteration_smoke_consumed"] is True
    assert gate["qualification_status"]["bounded_100_iteration_smoke_approved"] is False
    assert gate["qualification_status"]["v3_checkpoint_rejected_for_action_clipping"] is True


def test_reward_and_working_point_audit_tools_are_checked_in():
    assert (ROOT / "tools/run_a3_base_static_working_point_calibration.py").is_file()
    assert (ROOT / "tools/run_a3_base_stand_reward_v2_audit.py").is_file()
