from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "training/utils/target_conditioned_recovery_actor_critic.py"
)
SPEC = importlib.util.spec_from_file_location("target_conditioned_recovery_actor_critic", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
TargetConditionedRecoveryActorCritic = MODULE.TargetConditionedRecoveryActorCritic


def _policy() -> TargetConditionedRecoveryActorCritic:
    return TargetConditionedRecoveryActorCritic(
        num_actor_obs=213,
        num_critic_obs=213,
        num_actions=22,
        actor_hidden_dims=[16, 8],
        critic_hidden_dims=[16, 8],
        init_noise_std=0.03,
    )


def test_zero_initialized_recovery_adapter_preserves_p3_action_exactly():
    policy = _policy()
    observation = torch.linspace(-1.0, 1.0, steps=4 * 213).reshape(4, 213)
    # The new suffix is physical motion id + gate. Keep this regression in a
    # non-motion-3 row so zero learned residual still exactly preserves P3.
    observation[:, -2] = 0.0
    expected = policy.base_action_mean(observation[:, :204])
    actual = policy.act_inference(observation)

    assert torch.allclose(actual, expected, atol=1.0e-7, rtol=0.0)
    assert all(not parameter.requires_grad for parameter in policy.actor.parameters())
    assert all(
        torch.count_nonzero(value).item() == 0
        for value in policy.recovery_adapter.state_dict().values()
    )


def test_recovery_residual_is_gated_and_limited_to_lower_body_joints():
    policy = _policy()
    with torch.no_grad():
        policy.recovery_adapter.bias.fill_(1.0)

    observation = torch.zeros((1, 213))
    base = policy.base_action_mean(observation[:, :204])
    no_gate = policy.act_inference(observation)
    observation[:, -1] = 1.0
    gated = policy.act_inference(observation)

    assert torch.allclose(no_gate, base, atol=1.0e-7, rtol=0.0)
    changed = torch.nonzero((gated - base).abs()[0] > 1.0e-7).flatten().tolist()
    assert changed == list(policy.RECOVERY_ACTION_INDICES)


def test_motion1_bootstrap_brace_is_gated_and_motion_scoped():
    policy = _policy()
    observation = torch.zeros((3, 213))
    observation[:, -1] = 1.0
    observation[:, -2] = torch.tensor((0.0, 1.0, 3.0))
    base = policy.base_action_mean(observation[:, :204])
    actual = policy.act_inference(observation)
    brace = policy.audited_brace_action

    assert torch.allclose(actual[0], base[0], atol=1.0e-7, rtol=0.0)
    assert torch.allclose(actual[1] - base[1], brace, atol=1.0e-7, rtol=0.0)
    assert torch.allclose(actual[2] - base[2], brace, atol=1.0e-7, rtol=0.0)
