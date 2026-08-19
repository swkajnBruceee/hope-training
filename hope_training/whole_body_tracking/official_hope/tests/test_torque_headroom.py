"""Host-side tests for the torque-headroom shaping and YAML override contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load_helper():
    path = ROOT / "source/whole_body_tracking/whole_body_tracking/utils/torque_headroom.py"
    spec = importlib.util.spec_from_file_location("torque_headroom_test_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_headroom_starts_at_safe_fraction_and_emphasizes_topk():
    helper = _load_helper()
    utilization = torch.tensor([[0.80, 0.95, 1.00, 1.10]])
    penalty, metrics = helper.torque_headroom_penalty(
        utilization, safe_fraction=0.9, topk=2, topk_blend=0.7
    )
    # debts are [0, .25, 1, 3] after the bounded linear tail, so top2=2 and all=1.0625.
    torch.testing.assert_close(penalty, torch.tensor([1.71875]))
    torch.testing.assert_close(metrics["saturation_fraction"], torch.tensor([0.25]))
    torch.testing.assert_close(metrics["utilization_p95"], torch.tensor([1.085]))


def test_headroom_parameters_fail_closed():
    helper = _load_helper()
    util = torch.ones(2, 3)
    with pytest.raises(ValueError):
        helper.torque_headroom_penalty(util, safe_fraction=1.0)
    with pytest.raises(ValueError):
        helper.torque_headroom_penalty(util, topk=0)
    with pytest.raises(ValueError):
        helper.torque_headroom_penalty(util, topk_blend=1.1)
    with pytest.raises(ValueError):
        helper.torque_headroom_penalty(util, penalty_cap=0.0)


def test_task_yaml_and_override_registry_expose_headroom_controls():
    with (ROOT / "cfg/task/HOPEPingPong.yaml").open() as handle:
        values = yaml.safe_load(handle)["rewards"]
    assert values["torque_headroom_weight"] == -0.2
    assert values["torque_headroom_safe_fraction"] == 0.9
    assert values["torque_headroom_topk"] == 2
    assert values["torque_headroom_topk_blend"] == 0.7
    assert values["torque_headroom_penalty_cap"] == 9.0

    source = (ROOT / "source/whole_body_tracking/whole_body_tracking/utils/task_reward_overrides.py").read_text()
    assert '"torque_headroom_weight": "torque_headroom"' in source
    assert '"torque_headroom_topk"' in source
    assert '"penalty_cap": "penalty_cap"' in source

    reward_source = (
        ROOT / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_rewards.py"
    ).read_text()
    assert '"all_active_torque"' in reward_source
    assert '"racket_side_torque"' in reward_source
