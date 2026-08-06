from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_external_target_is_sampled_after_anchor_snapshot():
    source = (ROOT / "training/tasks/tracking/mdp/hope_commands.py").read_text(encoding="utf-8")
    anchor = "self.racket_anchor_target_pos_w[env_ids] = self.racket_target_pos_w[env_ids]"
    external = "self._sample_adapter_external_offset(env_ids)"
    assert anchor in source
    assert external in source
    assert source.index(anchor) < source.index(external)
    assert "adapter_external_paired requires num_envs divisible by 7" in source


def test_adapter_observation_is_the_reviewed_25d_contract():
    source = (ROOT / "training/tasks/tracking/mdp/hope_observations.py").read_text(encoding="utf-8")
    assert "def target_adapter_observation(" in source
    for channel in ("delta,", "error,", "velocity,", "phase_sc,", "one_hot,", "previous,"):
        assert channel in source
    assert "num_classes=6" in source


def test_frozen_anchor_arm_adapter_has_no_waist_output():
    source = (ROOT / "training/tasks/base_locomotion/mdp/actions.py").read_text(encoding="utf-8")
    assert "class A3FrozenAnchorArmAdapterPositionAction" in source
    assert "return len(self.cfg.adapter_joint_names)" in source
    assert "if len(cfg.adapter_joint_names) != 7:" in source
    assert "upper[:, self._adapter_indices] += gate * bounded * self._adapter_scale" in source
    assert "policy_residual = float(self.cfg.adapter_policy_residual_gain) * actions" in source
    assert "unbounded = policy_residual + feedforward" in source
    assert "self._adapter_feedforward_pinv_by_motion[motion_ids]" in source
    assert "self._adapter_raw_clip_by_motion[motion.motion_ids.to(torch.long)]" in source


def test_p0_task_uses_zero_init_and_anchor_group():
    train = (ROOT / "scripts/train.py").read_text(encoding="utf-8")
    cfg = (ROOT / "training/tasks/tracking/config/agibot_a3/native_strike_env_cfg.py").read_text(
        encoding="utf-8"
    )
    assert '"HOPE-FixedBaseTargetAdapter-AgibotA3-v0"' in train
    assert 'else 7 if task_id == "HOPE-FixedBaseTargetAdapter-AgibotA3-v0"' in train
    assert 'upper_observation_group="anchor"' in cfg
    assert "adapter_external_offset_half_range = (0.01, 0.01, 0.01)" in cfg
    assert "racket_paired_incremental_position_tracking" in cfg


def test_floating_target_training_keeps_frozen_upper_on_anchor():
    action_source = (
        ROOT / "training/tasks/base_locomotion/mdp/actions.py"
    ).read_text(encoding="utf-8")
    observation_source = (
        ROOT / "training/tasks/tracking/mdp/observations.py"
    ).read_text(encoding="utf-8")
    hope_observation_source = (
        ROOT / "training/tasks/tracking/mdp/hope_observations.py"
    ).read_text(encoding="utf-8")
    task = (
        ROOT / "cfg/task/HOPEA3FloatingTargetConditionedCoordinatorTrain.yaml"
    ).read_text(encoding="utf-8")

    assert "target_conditioned_anchor_observation = bool(cfg.anchor_observation)" in action_source
    assert "target_conditioned_coordinator_external_observation" in action_source
    assert "def joint_coordinator_target_conditioned_observation(" in observation_source
    assert '"coordinator_upper"' in observation_source
    assert "def coordinator_racket_target_pos_b(" in hope_observation_source
    assert "anchor_observation: true" in task
    assert "coordinator_external_observation: true" in task


def test_p4d_y_precompensation_is_motion_scoped_and_keeps_p4c_available():
    actions = (ROOT / "training/tasks/base_locomotion/mdp/actions.py").read_text(
        encoding="utf-8"
    )
    cfg = (ROOT / "training/tasks/tracking/config/agibot_a3/native_strike_env_cfg.py").read_text(
        encoding="utf-8"
    )
    task = (ROOT / "cfg/task/HOPEA3FloatingTargetConditionedP4DYComp.yaml").read_text(
        encoding="utf-8"
    )
    assert "adapter_feedforward_target_transform_by_motion" in actions
    assert "A3FloatingTargetConditionedRecoveryYCompEnvCfg" in cfg
    assert "0.75 I + 0.25 J^{-1}" in cfg
    assert "HOPE-FloatingTargetConditionedRecoveryYComp-AgibotA3-v0" in task


def test_p5_motion0_uses_a_separate_measured_control_anchor():
    cfg = (ROOT / "training/tasks/tracking/config/agibot_a3/native_strike_env_cfg.py").read_text(
        encoding="utf-8"
    )
    task = (ROOT / "cfg/task/HOPEA3FloatingTargetConditionedP5Motion0.yaml").read_text(
        encoding="utf-8"
    )
    assert "A3FloatingTargetConditionedRecoveryMotion0CalibratedEnvCfg" in cfg
    assert "(-0.0343194, 0.0407395, -0.0581275)" in cfg
    assert "HOPE-FloatingTargetConditionedRecoveryMotion0Calibrated-AgibotA3-v0" in task


def test_p7_motion2_adds_its_own_control_anchor_without_replacing_p5():
    cfg = (ROOT / "training/tasks/tracking/config/agibot_a3/native_strike_env_cfg.py").read_text(
        encoding="utf-8"
    )
    task = (ROOT / "cfg/task/HOPEA3FloatingTargetConditionedP7Motion2.yaml").read_text(
        encoding="utf-8"
    )
    assert "A3FloatingTargetConditionedRecoveryMotion2CalibratedEnvCfg" in cfg
    assert "(-0.0168705, 0.0431306, -0.0826364)" in cfg
    assert "HOPE-FloatingTargetConditionedRecoveryMotion2Calibrated-AgibotA3-v0" in task


def test_p8_motion4_adds_its_own_control_anchor_without_replacing_p7():
    cfg = (ROOT / "training/tasks/tracking/config/agibot_a3/native_strike_env_cfg.py").read_text(
        encoding="utf-8"
    )
    task = (ROOT / "cfg/task/HOPEA3FloatingTargetConditionedP8Motion4.yaml").read_text(
        encoding="utf-8"
    )
    assert "A3FloatingTargetConditionedRecoveryMotion4CalibratedEnvCfg" in cfg
    assert "(-0.0312324, 0.0621604, -0.0230464)" in cfg
    assert "HOPE-FloatingTargetConditionedRecoveryMotion4Calibrated-AgibotA3-v0" in task


def test_p9_motion5_adds_its_own_control_anchor_without_replacing_p8():
    cfg = (ROOT / "training/tasks/tracking/config/agibot_a3/native_strike_env_cfg.py").read_text(
        encoding="utf-8"
    )
    task = (ROOT / "cfg/task/HOPEA3FloatingTargetConditionedP9Motion5.yaml").read_text(
        encoding="utf-8"
    )
    assert "A3FloatingTargetConditionedRecoveryMotion5CalibratedEnvCfg" in cfg
    assert "(-0.0744438, 0.0363935, -0.0106046)" in cfg
    assert "HOPE-FloatingTargetConditionedRecoveryMotion5Calibrated-AgibotA3-v0" in task


def test_p10_auto_selection_uses_admitted_measured_control_centres():
    motion_command = (ROOT / "training/tasks/tracking/mdp/commands.py").read_text(
        encoding="utf-8"
    )
    action = (ROOT / "training/tasks/base_locomotion/mdp/actions.py").read_text(
        encoding="utf-8"
    )
    cfg = (ROOT / "training/tasks/tracking/config/agibot_a3/native_strike_env_cfg.py").read_text(
        encoding="utf-8"
    )
    assert "external_control_anchor_offset_b_by_motion" in motion_command
    assert "external_control_anchor_enabled_by_motion" in motion_command
    assert "external_control_local_half_range_by_motion" in motion_command
    assert 'distances[:, ~enabled] = float("inf")' in motion_command
    assert "control-anchor contract" in action
    assert "Motion 1 is deliberately" in cfg
    assert "True,\n            False,\n            True" in cfg
    assert "local +/-1 cm cube" in cfg


def test_floating_target_training_is_a_short_incremental_episode():
    task = (
        ROOT / "cfg/task/HOPEA3FloatingTargetConditionedCoordinatorTrain.yaml"
    ).read_text(encoding="utf-8")
    train = (ROOT / "scripts/train.py").read_text(encoding="utf-8")

    assert "episode_length_s: 2.2" in task
    assert "adapter_external_offset_half_range: [0.01, 0.01, 0.01]" in task
    assert "racket_incremental_dense_huber_weight: 5.0" in task
    assert "post_hit_capture_point_barrier_l2_weight: 0.0" in task
    assert '"racket_incremental_dense_huber_weight"' in train


def test_paired_target_runner_uses_common_exploration_noise():
    runner = (ROOT / "training/utils/my_on_policy_runner.py").read_text(
        encoding="utf-8"
    )
    cfg = (
        ROOT / "training/tasks/tracking/config/agibot_a3/native_strike_env_cfg.py"
    ).read_text(encoding="utf-8")

    assert "def _install_paired_common_exploration_noise(" in runner
    assert "standardized_noise[baseline]" in runner
    assert "transition.actions = paired_actions.detach()" in runner
    assert "get_actions_log_prob(" in runner
    assert '("upper", "stage_a", "coordinator_upper")' in cfg


def test_paired_target_training_disables_random_episode_phase_offsets():
    train = (ROOT / "scripts/train.py").read_text(encoding="utf-8")
    assert "paired_target_identification" in train
    assert "fixed_motion_recovery" in train
    assert "and not fixed_motion_recovery" in train


def test_paired_incremental_reward_masks_out_of_phase_siblings():
    reward = (ROOT / "training/tasks/tracking/mdp/hope_rewards.py").read_text(
        encoding="utf-8"
    )
    assert 'cmd.metrics["adapter_pair_phase_synced"]' in reward
    assert "phase_match = motion.time_steps == motion.time_steps[baseline]" in reward
    assert "active &= phase_match" in reward


def test_coordinator_controllability_audit_is_additive_and_22d():
    actions = (ROOT / "training/tasks/base_locomotion/mdp/actions.py").read_text(
        encoding="utf-8"
    )
    play = (ROOT / "scripts/play.py").read_text(encoding="utf-8")
    assert "coordinator_action_offset_override" in actions
    assert "actions = actions + coordinator_offset" in actions
    assert "coordinator_jacobian_step" in play
    assert "range(22)" in play
    assert '"coordinator_action_index"' in play


def test_full_body_target_feedforward_is_pre_composition_and_anchor_safe():
    actions = (ROOT / "training/tasks/base_locomotion/mdp/actions.py").read_text(
        encoding="utf-8"
    )
    task = (
        ROOT / "cfg/task/HOPEA3FloatingTargetConditionedCoordinatorTrain.yaml"
    ).read_text(encoding="utf-8")
    assert "coordinator_target_feedforward_by_motion" in actions
    assert "super().process_actions(actions + coordinator_feedforward)" in actions
    # Deployment is enabled only after the sibling-synchronized six-motion
    # calibration has supplied one 22x3 matrix per manifest anchor.
    assert "coordinator_target_feedforward_enabled: true" in task
    assert "manifest_subset_size: 6" in task
    assert task.count("# motion ") == 6
    assert "coordinator_target_feedforward_raw_clip: 0.15" in task
    assert "coordinator_target_feedforward_last_action" in (
        ROOT / "scripts/play.py"
    ).read_text(encoding="utf-8")


def test_target_audits_synchronize_physical_siblings_before_latching_commands():
    play = (ROOT / "scripts/play.py").read_text(encoding="utf-8")
    assert "def _synchronize_target_audit_siblings" in play
    assert "target_audit_synchronize_siblings" in play
    assert "siblings_share_physical_strike_ready_state" in play
    assert "startup_physics_domain_randomization_disabled" in play
    assert 'env_cfg.events.physics_material = None' in play


def test_external_target_executor_can_select_and_bound_nearest_anchor():
    motion_command = (ROOT / "training/tasks/tracking/mdp/commands.py").read_text(
        encoding="utf-8"
    )
    play = (ROOT / "scripts/play.py").read_text(encoding="utf-8")
    config = (ROOT / "cfg/play.yaml").read_text(encoding="utf-8")
    assert "def select_nearest_strike_motion_ids" in motion_command
    assert "nearest-anchor selection requires a motion manifest" in motion_command
    assert "auto_select_motion: false" in config
    assert "auto_select_max_anchor_distance_m: 0.04" in config
    assert "auto_select_local_range_tolerance_m: 1.0e-6" in config
    assert "auto_select_local_half_range_by_motion:" in config
    assert "- [0.02, 0.01, 0.02]" in config
    assert "select_nearest_strike_motion_ids" in play
    assert "external target lies outside every verified local anchor range" in play
    assert "external target exceeds the verified per-motion local range" in play
    assert "half_ranges + auto_select_local_range_tolerance_m" in play
    assert "<= 1.0e-6" in play


def test_p11_motion1_recovery_training_is_fixed_and_not_deployment_admitted():
    commands = (ROOT / "training/tasks/tracking/mdp/commands.py").read_text(
        encoding="utf-8"
    )
    cfg = (
        ROOT / "training/tasks/tracking/config/agibot_a3/native_strike_env_cfg.py"
    ).read_text(encoding="utf-8")
    task = (
        ROOT / "cfg/task/HOPEA3FloatingTargetConditionedP11Motion1RecoveryTrain.yaml"
    ).read_text(encoding="utf-8")

    assert "fixed_motion_id: int | None = None" in commands
    assert "A3FloatingTargetConditionedRecoveryMotion1TrainEnvCfg" in cfg
    assert "self.commands.motion.fixed_motion_id = 1" in cfg
    assert "external_control_anchor_enabled_by_motion" in cfg
    assert "True,\n            False,\n            True," in cfg
    assert "HOPE-FloatingTargetConditionedRecoveryMotion1Train-AgibotA3-v0" in task
    assert "adapter_external_paired: false" in task
    assert "elif motion.cfg.fixed_motion_id is not None:" in (
        ROOT / "scripts/train.py"
    ).read_text(encoding="utf-8")
