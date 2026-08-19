"""Unit tests for the isolated first-milestone AMP reward sidecar."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml


_AMP_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "utils"
    / "amp.py"
)
_SPEC = importlib.util.spec_from_file_location("hope_amp_test_module", _AMP_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_AMP = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_AMP)
AMPDiscriminator = _AMP.AMPDiscriminator
AMPMotionFeature = _AMP.AMPMotionFeature
amp_transition_valid_mask = _AMP.amp_transition_valid_mask


def test_amp_discriminator_reward_and_update_contract():
    # Some host-only contract tests install a minimal omegaconf stub without a module spec;
    # torch.optim's optional torch._dynamo import scans specs when the full suite is collected.
    # Restore a harmless spec here so this independent test remains order-insensitive.
    omegaconf = sys.modules.get("omegaconf")
    if omegaconf is not None and getattr(omegaconf, "__spec__", None) is None:
        omegaconf.__spec__ = importlib.util.spec_from_loader("omegaconf", loader=None)
    torch.manual_seed(3)
    discriminator = AMPDiscriminator(
        state_dim=6,
        hidden_dims=(16, 8),
        batch_size=8,
        updates_per_rollout=2,
    )
    policy = torch.randn(32, 6)
    expert = policy * 0.25
    reward = discriminator.reward(policy)
    assert reward.shape == (32,)
    assert torch.isfinite(reward).all()
    assert float(reward.min()) >= 0.0
    stats = discriminator.update(policy, expert, lambda_amp=0.1)
    assert set(stats) == {
        "disc_loss",
        "expert_prob",
        "policy_prob",
        "reward_mean",
        "reward_std",
        "weighted_reward_mean",
        "sample_count",
        "expert_logit_mean",
        "policy_logit_mean",
        "valid_transition_fraction",
        "gradient_penalty",
    }
    assert stats["sample_count"] == 32.0
    assert stats["weighted_reward_mean"] == pytest.approx(stats["reward_mean"] * 0.1)


def test_amp_state_is_private_and_reference_aligned():
    class Data:
        joint_pos = torch.zeros(2, 3)
        joint_vel = torch.ones(2, 3)
        body_pos_w = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
             [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
        )
        body_quat_w = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
             [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]
        )
        body_lin_vel_w = torch.zeros(2, 2, 3)
        body_ang_vel_w = torch.zeros(2, 2, 3)

    class Robot:
        data = Data()

    class Motion:
        cfg = SimpleNamespace(body_names=("pelvis_link", "torso_Link"))
        joint_pos = torch.full((2, 3), 2.0)
        joint_vel = torch.full((2, 3), 4.0)
        body_pos_w = Data.body_pos_w.clone()
        body_quat_w = Data.body_quat_w.clone()
        body_lin_vel_w = Data.body_lin_vel_w.clone()
        body_ang_vel_w = Data.body_ang_vel_w.clone()
        robot_joint_pos = Data.joint_pos
        robot_joint_vel = Data.joint_vel
        robot_body_pos_w = Data.body_pos_w
        robot_body_quat_w = Data.body_quat_w
        robot_body_lin_vel_w = Data.body_lin_vel_w
        robot_body_ang_vel_w = Data.body_ang_vel_w

    env = SimpleNamespace(
        scene={"robot": Robot()},
        command_manager=SimpleNamespace(get_term=lambda name: Motion()),
    )
    feature = AMPMotionFeature(joint_velocity_scale=0.5)
    live = feature.robot(env)
    expert = feature.expert(env)
    assert live.shape == expert.shape == (2, 36)
    torch.testing.assert_close(live[:, :3], torch.zeros(2, 3))
    torch.testing.assert_close(live[:, 3:6], torch.full((2, 3), 0.5))
    torch.testing.assert_close(expert[:, :3], torch.full((2, 3), 2.0))
    torch.testing.assert_close(expert[:, 3:6], torch.full((2, 3), 2.0))


def test_amp_named_layout_excludes_passive_head_and_scales_lower_body():
    class Data:
        joint_pos = torch.zeros(1, 4)
        joint_vel = torch.zeros(1, 4)
        body_pos_w = torch.zeros(1, 2, 3)
        body_quat_w = torch.zeros(1, 2, 4)
        body_quat_w[..., 0] = 1.0
        body_lin_vel_w = torch.zeros(1, 2, 3)
        body_ang_vel_w = torch.zeros(1, 2, 3)

    class Robot:
        data = Data()
        joint_names = ("pelvis_joint", "left_knee_joint", "head_yaw_joint", "head_pitch_joint")

    class Motion:
        cfg = SimpleNamespace(body_names=("pelvis_link", "left_knee_Link"))
        robot = Robot()
        joint_pos = Data.joint_pos
        joint_vel = Data.joint_vel
        body_pos_w = Data.body_pos_w
        body_quat_w = Data.body_quat_w
        body_lin_vel_w = Data.body_lin_vel_w
        body_ang_vel_w = Data.body_ang_vel_w
        robot_joint_pos = Data.joint_pos
        robot_joint_vel = Data.joint_vel
        robot_body_pos_w = Data.body_pos_w
        robot_body_quat_w = Data.body_quat_w
        robot_body_lin_vel_w = Data.body_lin_vel_w
        robot_body_ang_vel_w = Data.body_ang_vel_w

    class Env:
        command_manager = SimpleNamespace(get_term=lambda name: Motion())

    feature = AMPMotionFeature(lower_body_feature_scale=0.3)
    state = feature.robot(Env())
    assert state.shape == (1, 2 * 2 + 2 * 15)
    signature = feature.signature()
    assert signature["joint_names"] == ["pelvis_joint", "left_knee_joint"]
    assert "head_yaw_joint" not in signature["joint_names"]
    assert signature["lower_body_feature_scale"] == pytest.approx(0.3)
    assert signature["root_frame"] == "pelvis_heading_yaw_only_world_up"


def test_amp_transition_and_quaternion_sign_invariance():
    feature = AMPMotionFeature()
    q = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    q_neg = -q
    matrix_a = feature._rotation_6d(q)
    matrix_b = feature._rotation_6d(q_neg)
    torch.testing.assert_close(matrix_a, matrix_b)
    previous = torch.zeros(4, 8)
    current = torch.ones(4, 8)
    transition = AMPDiscriminator.transition(previous, current)
    assert transition.shape == (4, 16)


def test_amp_heading_frame_preserves_pelvis_pitch():
    identity = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    half_pitch = torch.tensor(0.25)
    pitched = torch.tensor([[[torch.cos(half_pitch), 0.0, torch.sin(half_pitch), 0.0]]])
    body_pos = torch.zeros(1, 1, 3)
    body_lin = torch.zeros(1, 1, 3)
    body_ang = torch.zeros(1, 1, 3)
    _, identity_local, _, _ = AMPMotionFeature._root_relative(
        body_pos, identity, body_lin, body_ang, 0
    )
    _, pitched_local, _, _ = AMPMotionFeature._root_relative(
        body_pos, pitched, body_lin, body_ang, 0
    )
    assert not torch.allclose(identity_local, pitched_local)


def test_amp_transition_mask_blocks_hold_exit_wrap_and_terminal_steps():
    done = torch.tensor([False, True, False, False, False])
    current_hold = torch.tensor([False, False, True, False, False])
    previous_hold = torch.tensor([True, False, False, False, False])
    resampled = torch.tensor([False, False, False, True, False])
    valid = amp_transition_valid_mask(done, current_hold, previous_hold, resampled)
    torch.testing.assert_close(valid, torch.tensor([False, False, False, False, True]))


def test_amp_schedule_has_locked_final_contribution_floor():
    config_path = Path(__file__).resolve().parents[1] / "cfg" / "train.yaml"
    with config_path.open() as handle:
        amp = yaml.safe_load(handle)["amp"]
    schedule = amp["contribution_schedule"]
    assert schedule[0] == [0.0, 0.35]
    assert schedule[-1] == [1.0, 0.075]
    assert schedule[1] == [0.1, 0.35]
    assert schedule[2] == [0.35, 0.2]
    assert amp["schedule_iterations"] == 10000
    assert amp["warmup_rollouts"] == 2
    assert amp["lambda_min"] == 0.0
    assert amp["lambda_max"] == 2.0


def test_scratch_contract_rejects_residual_and_unmarked_checkpoints():
    contract_path = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "whole_body_tracking"
        / "whole_body_tracking"
        / "utils"
        / "scratch_contract.py"
    )
    spec = importlib.util.spec_from_file_location("hope_scratch_contract_test_module", contract_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.validate_scratch_algorithm({"policy": {}}, enabled=True)
    with pytest.raises(ValueError, match="standard randomly initialized ActorCritic"):
        module.validate_scratch_algorithm(
            {"policy": {"class_name": "ResidualMeanActorCritic"}}, enabled=True
        )
    with pytest.raises(ValueError, match="without hope_scratch_contract"):
        module.validate_scratch_checkpoint({}, amp_enabled=False, path="old.pt")

    checkpoint = {"hope_scratch_contract": module.build_scratch_contract(amp_enabled=True)}
    module.validate_scratch_checkpoint(checkpoint, amp_enabled=True, path="own.pt")
    with pytest.raises(ValueError, match="contract mismatch"):
        module.validate_scratch_checkpoint(checkpoint, amp_enabled=False, path="own.pt")


def test_current_playback_contract_defaults_to_task_overrides_and_scratch():
    root = Path(__file__).resolve().parents[1]
    with (root / "cfg/play.yaml").open() as handle:
        play = yaml.safe_load(handle)
    assert play["apply_task_overrides"] is True
    assert play["scratch_training"] is True
