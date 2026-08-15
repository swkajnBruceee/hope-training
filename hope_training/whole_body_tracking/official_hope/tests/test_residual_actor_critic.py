"""Contract tests for Frozen HOPE + Zero-init Residual Mean Actor."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source" / "whole_body_tracking" / "whole_body_tracking"
CHECKPOINT = ROOT / "checkpoints" / "model_21800.pt"
sys.path.insert(0, str(SOURCE.parent))


def _load_residual_module():
    pytest.importorskip("rsl_rl")
    if importlib.util.find_spec("isaaclab") is None:
        pytest.skip("Isaac Lab is not installed in this test environment")
    try:
        import omni.kit.app  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("Isaac Sim omni.kit.app is not available in this test environment")
    from whole_body_tracking.utils import residual_actor_critic

    return residual_actor_critic


def test_residual_module_zero_init_matches_official_mean():
    residual = _load_residual_module()
    model = residual.ResidualMeanActorCritic(
        num_actor_obs=110,
        num_critic_obs=328,
        num_actions=31,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        residual_hidden_dims=[32, 16],
        residual_delta_q_max_rad=0.05,
        residual_time_scale=1.0,
        residual_train_std=False,
    )
    model.bind_residual_action_contract(
        torch.full((31,), 0.2),
        torch.tensor([True] * 29 + [False, False]),
    )
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_official_model_state(payload["model_state_dict"])

    obs = torch.randn(8, 110)
    obs[:, 109] = torch.tensor([0.8, 0.3, 0.0, -0.2, -0.8, 1.4, -1.4, 0.1])
    with torch.no_grad():
        official_mean = model.actor(obs)
        residual_mean = model.act_inference(obs)

    torch.testing.assert_close(residual_mean, official_mean, rtol=0.0, atol=0.0)
    assert model.residual_active_count == 29
    assert torch.count_nonzero(model.residual_bound_raw[-2:]) == 0
    assert model.std.requires_grad is False
    assert all(not parameter.requires_grad for parameter in model.actor.parameters())
    assert all(parameter.requires_grad for parameter in model.residual_actor.parameters())
    assert all(parameter.requires_grad for parameter in model.critic.parameters())
    diagnostics = model.residual_diagnostics(obs)
    assert {
        "residual_mean_l2_active",
        "residual_mean_abs_active",
        "residual_mean_max_abs_active",
        "residual_q_nom_abs_active",
        "residual_q_nom_max_abs_active",
        "residual_q_raw_clip_estimate_abs_active",
        "residual_mean_saturation_rate_active",
    } == set(diagnostics)
    assert all(tuple(value.shape) == (8,) for value in diagnostics.values())
    metadata = model.get_model_metadata()
    assert metadata["residual_time_scale"] == 1.0
    assert metadata["std_trainable"] is False
    assert metadata["observation_contract"] == "hitter_pure"


def test_residual_distribution_and_active_bound_are_well_formed():
    residual = _load_residual_module()
    model = residual.ResidualMeanActorCritic(
        num_actor_obs=110,
        num_critic_obs=328,
        num_actions=31,
        actor_hidden_dims=[16, 16, 8],
        critic_hidden_dims=[16, 16, 8],
        activation="elu",
        init_noise_std=1.0,
        residual_hidden_dims=[8, 8],
    )
    mask = torch.tensor([True] * 29 + [False, False])
    model.bind_residual_action_contract(torch.full((31,), 0.25), mask)
    obs = torch.randn(4, 110)
    model.update_distribution(obs)
    action = model.distribution.sample()
    log_prob = model.get_actions_log_prob(action)
    assert tuple(action.shape) == (4, 31)
    assert tuple(log_prob.shape) == (4,)
    assert torch.allclose(model.residual_bound_raw[-2:], torch.zeros(2))


def test_structured_residual_zero_init_preserves_hope_and_records_contract():
    residual = _load_residual_module()
    model = residual.ResidualMeanActorCritic(
        num_actor_obs=110,
        num_critic_obs=328,
        num_actions=31,
        actor_hidden_dims=[16, 16, 8],
        critic_hidden_dims=[16, 16, 8],
        activation="elu",
        residual_hidden_dims=[8, 8],
        residual_architecture="structured",
        structured_proprio_hidden_dims=[16, 8],
        structured_goal_hidden_dims=[8, 4],
        structured_time_hidden_dims=[4, 2],
        structured_fusion_hidden_dims=[16, 8],
    )
    model.bind_residual_action_contract(torch.full((31,), 0.25), torch.tensor([True] * 29 + [False, False]))
    obs = torch.randn(4, 110)
    with torch.no_grad():
        torch.testing.assert_close(model.act_inference(obs), model.actor(obs), rtol=0.0, atol=0.0)
    metadata = model.get_model_metadata()
    assert metadata["residual_architecture"] == "structured"
    assert metadata["structured_split_indices"]["time_to_strike"] == [109, 110]
    assert metadata["structured_proprio_terms"]
    assert metadata["structured_goal_terms"]
    assert metadata["residual_actor_parameter_count"] > 0


def test_structured_film_zero_init_preserves_hope_and_records_film_contract():
    residual = _load_residual_module()
    model = residual.ResidualMeanActorCritic(
        num_actor_obs=110,
        num_critic_obs=328,
        num_actions=31,
        actor_hidden_dims=[16, 16, 8],
        critic_hidden_dims=[16, 16, 8],
        activation="elu",
        init_noise_std=1.0,
        residual_architecture="structured_film",
        structured_proprio_hidden_dims=[16, 8],
        structured_goal_hidden_dims=[8, 4],
        structured_time_hidden_dims=[4, 2],
        structured_fusion_hidden_dims=[16, 8],
        structured_film_hidden_dims=[8, 8],
    )
    model.bind_residual_action_contract(
        torch.full((31,), 0.25),
        torch.tensor([True] * 29 + [False, False]),
    )
    obs = torch.randn(4, 110)
    with torch.no_grad():
        torch.testing.assert_close(model.act_inference(obs), model.actor(obs), rtol=0.0, atol=0.0)
    assert torch.count_nonzero(model.residual_actor.residual_head.weight) == 0
    assert torch.count_nonzero(model.residual_actor.film_generator[-1].weight) == 0
    metadata = model.get_model_metadata()
    assert metadata["residual_architecture"] == "structured_film"
    assert metadata["film_enabled"] is True
    assert metadata["structured_film_hidden_dims"] == [8, 8]


def test_residual_contract_defaults_fail_closed():
    residual = _load_residual_module()
    model = residual.ResidualMeanActorCritic(
        num_actor_obs=110,
        num_critic_obs=328,
        num_actions=31,
        actor_hidden_dims=[8, 8, 8],
        critic_hidden_dims=[8, 8, 8],
        activation="elu",
        init_noise_std=1.0,
        residual_hidden_dims=[8, 8],
    )
    assert model.residual_active_count == 0
    assert torch.count_nonzero(model.residual_bound_raw) == 0
    with pytest.raises(RuntimeError, match="action contract is not bound"):
        model.act_inference(torch.randn(1, 110))


def test_residual_optimizer_step_keeps_hope_actor_bitwise_fixed():
    residual = _load_residual_module()
    model = residual.ResidualMeanActorCritic(
        num_actor_obs=110,
        num_critic_obs=328,
        num_actions=31,
        actor_hidden_dims=[16, 16, 8],
        critic_hidden_dims=[16, 16, 8],
        activation="elu",
        init_noise_std=1.0,
        residual_hidden_dims=[8, 8],
    )
    model.bind_residual_action_contract(torch.full((31,), 0.25), torch.tensor([True] * 29 + [False, False]))
    actor_before = {name: value.detach().clone() for name, value in model.actor.state_dict().items()}
    residual_before = {name: value.detach().clone() for name, value in model.residual_actor.state_dict().items()}
    optimizer = torch.optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad), lr=1.0e-2)
    obs = torch.randn(8, 110)
    critic_obs = torch.randn(8, 328)
    loss = model.act_inference(obs).mean() + model.critic(critic_obs).mean()
    loss.backward()
    optimizer.step()
    for name, value in model.actor.state_dict().items():
        torch.testing.assert_close(value, actor_before[name], rtol=0.0, atol=0.0)
    assert any(
        not torch.equal(value, residual_before[name])
        for name, value in model.residual_actor.state_dict().items()
    )


def test_export_wrapper_uses_combined_act_inference_for_nonzero_residual(tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    export_path = ROOT / "scripts" / "export_onnx.py"
    spec = importlib.util.spec_from_file_location("residual_export_script", export_path)
    export_script = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(export_script)

    class FakePolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = torch.nn.Sequential(torch.nn.Linear(110, 31, bias=False))
            torch.nn.init.constant_(self.actor[0].weight, 0.01)

        def act_inference(self, obs):
            # Represents a non-zero ResidualMeanActorCritic correction.
            return self.actor(obs) + 0.1

        def get_model_metadata(self):
            return {"model_variant": "test-nonzero-residual"}

    contract = SimpleNamespace(
        name="hitter_pure",
        total_dim=110,
        layout=(("observation", 110),),
    )
    policy = FakePolicy().eval()
    export_script.export_deploy_policy(
        policy,
        contract,
        [f"j{i}" for i in range(31)],
        [f"j{i}" for i in range(31)],
        str(tmp_path),
        "policy.onnx",
        50,
    )
    model = onnx.load(str(tmp_path / "policy.onnx"))
    assert any(item.key == "model_variant" and item.value == "test-nonzero-residual" for item in model.metadata_props)
    session = ort.InferenceSession(str(tmp_path / "policy.onnx"), providers=["CPUExecutionProvider"])
    obs = torch.randn(1, 110)
    exported = torch.from_numpy(session.run(["raw_action"], {"observation": obs.numpy()})[0])
    expected = policy.act_inference(obs)
    base = policy.actor(obs)
    torch.testing.assert_close(exported, expected, rtol=1.0e-5, atol=1.0e-6)
    assert not torch.allclose(exported, base)


def test_periodic_exporter_has_residual_combined_policy_path():
    exporter_source = (
        ROOT / "source" / "whole_body_tracking" / "whole_body_tracking" / "utils" / "exporter.py"
    ).read_text(encoding="utf-8")
    assert "self._combined_policy.act_inference(self.normalizer(x))" in exporter_source
    assert "hasattr(\n            actor_critic, \"residual_actor\"\n        )" in exporter_source


def test_residual_policy_is_registered_before_runner_factory_resolution():
    cfg_source = (
        ROOT / "source" / "whole_body_tracking" / "whole_body_tracking" / "utils" / "ppo_cfg.py"
    ).read_text(encoding="utf-8")
    assert "register_with_rsl_rl_runner as register_residual_actor_critic" in cfg_source
    assert "register_residual_actor_critic()" in cfg_source
