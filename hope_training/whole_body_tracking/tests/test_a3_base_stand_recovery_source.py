import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_recovery_a_task_is_registered_with_its_own_runner():
    registry = (ROOT / "training/tasks/base_locomotion/config/a3/__init__.py").read_text()
    runner = (ROOT / "training/tasks/base_locomotion/config/a3/agents/ppo.py").read_text()
    assert 'id="A3BaseStandRecoveryA-v0"' in registry
    assert "A3BaseStandRecoveryAPPORunnerCfg" in registry
    assert 'experiment_name = "a3_base_stand_recovery_a"' in runner


def test_recovery_a_contract_keeps_clean_slice_and_small_disturbance():
    source = (ROOT / "training/tasks/base_locomotion/config/a3/stand_env_cfg.py").read_text()
    assert '"undisturbed_fraction": 0.50' in source
    assert '"roll_pitch_range_rad": (-0.035, 0.035)' in source
    assert '"angular_velocity_range_rad_s": (-0.20, 0.20)' in source
    assert '"medium_fraction": 0.30' in source
    assert '"medium_roll_pitch_range_rad": (-0.05, 0.05)' in source
    assert '"medium_angular_velocity_range_rad_s": (-0.30, 0.30)' in source
    assert "undisturbed_action_magnitude" in source
    assert "recovery_tilt_progress" in source


def test_recovery_mask_is_not_added_to_actor_observation():
    observations = (ROOT / "training/tasks/base_locomotion/mdp/observations.py").read_text()
    events = (ROOT / "training/tasks/base_locomotion/mdp/events.py").read_text()
    assert "recovery_disturbed_mask" not in observations
    assert "env.recovery_disturbed_mask" in events


def test_recovery_training_remains_closed_and_zero_mean_is_initialized():
    decision = json.loads(
        (ROOT / "contracts/a3_base_locomotion_v1/stand_passive_stable_decision_v1.json").read_text()
    )
    train = (ROOT / "scripts/train.py").read_text()
    assert decision["qualification_status"]["stand_recovery_training_approved"] is False
    assert '"A3BaseStandRecoveryA-v0"' in train
    assert "zero_residual_tasks" in train
    assert "initialize_zero_residual_actor_mean(runner, action_dim=14)" in train
    gate = json.loads(
        (ROOT / "contracts/a3_base_locomotion_v1/stand_recovery_a_gate_v1.json").read_text()
    )
    status = gate["qualification_status"]
    assert status["recovery_a_environment_runtime_qualified"] is True
    assert status["recovery_reward_v3_semantics_approved"] is True
    assert status["recovery_disturbance_contract_approved"] is True
    assert status["recovery_envelope_approved"] is True
    assert status["zero_actor_initialization_runtime_verified"] is True
    assert status["untrained_stochastic_survival_safety_verified"] is True
    assert status["untrained_stochastic_recovery_compatibility_verified"] is False
    assert status["untrained_stochastic_policy_safety_verified"] is True
    assert status["bounded_recovery_smoke_approved"] is True
    assert status["deployment_approved"] is False
    assert gate["bounded_smoke_budget"]["max_iterations"] == 100
    assert gate["bounded_smoke_budget"]["max_num_envs"] == 64
    approved = gate["approved_recovery_envelope"]
    assert approved["name"] == "B_core_only"
    assert approved["dwell_s"] == 0.30
    assert approved["hysteresis_ratio"] == 1.25
    assert approved["raw_waist_ankle_velocity_role"] == "quality_only"
    assert approved["rms_200ms_waist_ankle_velocity_role"] == "quality_only"
    assert approved["upper_profile_role"] == "diagnostic_only"
    decision = json.loads(
        (
            ROOT
            / "contracts/a3_base_locomotion_v1/stand_recovery_envelope_decision_v1.json"
        ).read_text()
    )
    assert decision["recovery_envelope_approved"] is True
    assert decision["approved_envelope"]["name"] == "B_core_only"
    assert decision["authorizes_untrained_stochastic_policy_safety_audit"] is True
    assert decision["authorizes_ppo"] is False


def test_recovery_a_calibration_is_diagnostic_only():
    tool = (ROOT / "tools/run_a3_base_stand_recovery_a_calibration.py").read_text()
    assert '"recovery_a_clean", 0, 0.0, 0.0' in tool
    assert "recovery_a_candidate" in tool
    assert "formal recovery calibration requires --runtime-contract" in tool
    assert "formal recovery calibration requires exactly 500 policy steps" in tool
    assert '"completed_policy_steps": completed_policy_steps' in tool
    assert "runtime_passed and not args_cli.runtime_smoke" in tool
    assert '"recovery_training_approved": False' in tool
    assert '"deployment_approved": False' in tool


def test_recovery_gate_evidence_hashes_are_pinned():
    gate = json.loads(
        (ROOT / "contracts/a3_base_locomotion_v1/stand_recovery_a_gate_v1.json").read_text()
    )
    for evidence in gate["evidence"].values():
        artifact = ROOT / evidence["path"]
        assert artifact.is_file()
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == evidence["sha256"]


def test_recovery_progress_is_masked_and_reward_audit_is_checked_in():
    rewards = (ROOT / "training/tasks/base_locomotion/mdp/rewards.py").read_text()
    assert "progress * disturbed.to(progress.dtype)" in rewards
    assert "/ float(env.step_dt)" in rewards
    audit = ROOT / "tools/run_a3_base_stand_recovery_reward_v3_audit.py"
    assert audit.is_file()
    assert '"recovery_training_approved": False' in audit.read_text()


def test_runner_initialization_audit_checks_normalized_full_chain():
    helper = (ROOT / "training/utils/a3_base_actor_init.py").read_text()
    audit = (ROOT / "tools/run_a3_base_recovery_runner_initialization_audit.py").read_text()
    assert "torch.nn.init.zeros_(actor_output.weight)" in helper
    assert "normalized_obs = runner.obs_normalizer(obs)" in audit
    assert "full_chain_mean = inference_policy(obs)" in audit
    assert '"untrained_stochastic_policy_safety_verified": False' in audit


def test_untrained_safety_audit_is_paired_trace_and_fail_closed():
    audit = (ROOT / "tools/run_a3_base_recovery_untrained_safety_audit.py").read_text()
    assert "np.concatenate((pose_np, pose_np))" in audit
    assert '"--envelope-decision"' in audit
    assert 'approved.get("name") != "B_core_only"' in audit
    assert "episode_events(" in audit
    assert '"durable_recovery_rate"' in audit
    assert '"final_1s_stable_rate"' in audit
    assert '"recovery_time_p90_s"' in audit
    assert "formal untrained safety audit requires exactly 500 policy steps" in audit
    assert '"sampled_action_rate_rms_per_policy_step"' in audit
    assert '"effective_action_rate_rms_per_policy_step"' in audit
    assert '"sampled_clip_fraction_by_joint"' in audit
    assert '"effective_clip_fraction_by_joint"' in audit
    assert '"bounded_recovery_smoke_approved": False' in audit
