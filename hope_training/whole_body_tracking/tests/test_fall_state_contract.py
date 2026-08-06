"""Dependency-light contracts for the unified fall-state geometry."""

from pathlib import Path
import importlib.util
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "a3_fall_state", ROOT / "training/tasks/tracking/mdp/fall_state.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_signed_tilt_uses_immutable_forward_and_left_axes():
    forward = torch.tensor([[1.0, 0.0, 0.0]])
    upright = torch.tensor([[0.0, 0.0, 1.0]])
    f, l = MODULE.signed_tilt_from_up(upright, forward)
    assert torch.allclose(f, torch.zeros(1), atol=1e-6)
    assert torch.allclose(l, torch.zeros(1), atol=1e-6)

    # Up vector leaning +forward and +left must preserve signs even if the
    # current robot yaw later changes; the helper only accepts the frozen axis.
    tilted = torch.tensor([[0.30, 0.20, 0.9327379]])
    f, l = MODULE.signed_tilt_from_up(tilted, forward)
    assert f.item() > 0.0
    assert l.item() > 0.0


def test_fall_state_enums_keep_prediction_separate_from_physical_confirmation():
    assert MODULE.FallLevel.PREDICTED_UNRECOVERABLE != MODULE.FallLevel.FALL_CONFIRMED
    assert MODULE.FallReason.FORWARD_FALL != MODULE.FallReason.BACKWARD_FALL


def test_debounce_and_recovery_gate_are_not_single_frame_ready_or_fall():
    counter = torch.zeros(1, dtype=torch.long)
    counter, confirmed = MODULE.debounce_counter(counter, torch.ones(1, dtype=torch.bool), 3)
    assert not bool(confirmed[0])
    counter, confirmed = MODULE.debounce_counter(counter, torch.ones(1, dtype=torch.bool), 3)
    assert not bool(confirmed[0])
    counter, confirmed = MODULE.debounce_counter(counter, torch.ones(1, dtype=torch.bool), 3)
    assert bool(confirmed[0])
    ready = MODULE.recovery_ready_gate(
        double_foot_contact=torch.ones(1, dtype=torch.bool),
        illegal_contact=torch.zeros(1, dtype=torch.bool),
        relative_height=torch.tensor([0.95]), tilt_rad=torch.tensor([0.05]),
        rate_radps=torch.tensor([0.05]), com_speed_mps=torch.tensor([0.05]),
        capture_min_margin_m=torch.tensor([0.03]), foot_slip_mps=torch.tensor([0.01]),
        height_min_m=0.88, tilt_max_rad=0.20, rate_max_radps=0.45,
        com_speed_max_mps=0.18, margin_min_m=0.015, slip_max_mps=0.08,
    )
    assert bool(ready[0])
    ready = MODULE.recovery_ready_gate(
        double_foot_contact=torch.ones(1, dtype=torch.bool),
        illegal_contact=torch.zeros(1, dtype=torch.bool),
        relative_height=torch.tensor([0.95]), tilt_rad=torch.tensor([0.25]),
        rate_radps=torch.tensor([0.05]), com_speed_mps=torch.tensor([0.05]),
        capture_min_margin_m=torch.tensor([0.03]), foot_slip_mps=torch.tensor([0.01]),
        height_min_m=0.88, tilt_max_rad=0.20, rate_max_radps=0.45,
        com_speed_max_mps=0.18, margin_min_m=0.015, slip_max_mps=0.08,
    )
    assert not bool(ready[0])


def test_cycle_guard_requires_recovery_hold_and_never_hides_confirmed_fall():
    spec = importlib.util.spec_from_file_location(
        "a3_cycle_manager", ROOT / "training/tasks/tracking/mdp/cycle_manager.py"
    )
    assert spec is not None and spec.loader is not None
    cycle = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cycle
    spec.loader.exec_module(cycle)
    manager = cycle.StrikeCycleManager(1, "cpu", cycle.CycleConfig(post_hit_guard_steps=2, ready_hold_steps=2))
    false = torch.zeros(1, dtype=torch.bool)
    true = torch.ones(1, dtype=torch.bool)
    manager.update(
        prelude_active=false, strike_active=true, hit_window=false,
        motion_done=false, recovery_ready=false, confirmed_fall=false,
        predicted_unrecoverable=false, timeout=false, strike_pass=false,
    )
    for _ in range(3):
        result = manager.update(
            prelude_active=false, strike_active=false, hit_window=false,
            motion_done=true, recovery_ready=true, confirmed_fall=false,
            predicted_unrecoverable=false, timeout=false, strike_pass=true,
        )
    assert bool(result["next_action_allowed"][0])
    result = manager.update(
        prelude_active=false, strike_active=false, hit_window=false,
        motion_done=true, recovery_ready=false, confirmed_fall=true,
        predicted_unrecoverable=true, timeout=false, strike_pass=true,
    )
    assert int(result["cycle_phase"][0]) == int(cycle.CyclePhase.CYCLE_FAILED)
    assert not bool(result["next_action_allowed"][0])


def test_cycle_guard_predicted_unrecoverable_blocks_next_action_even_if_pose_is_ready():
    spec = importlib.util.spec_from_file_location(
        "a3_cycle_manager_predicted", ROOT / "training/tasks/tracking/mdp/cycle_manager.py"
    )
    assert spec is not None and spec.loader is not None
    cycle = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cycle
    spec.loader.exec_module(cycle)
    manager = cycle.StrikeCycleManager(1, "cpu", cycle.CycleConfig(post_hit_guard_steps=1, ready_hold_steps=1))
    false = torch.zeros(1, dtype=torch.bool)
    true = torch.ones(1, dtype=torch.bool)
    result = manager.update(
        prelude_active=false, strike_active=false, hit_window=false,
        motion_done=true, recovery_ready=true, confirmed_fall=false,
        predicted_unrecoverable=true, timeout=false, strike_pass=true,
    )
    assert not bool(result["next_action_allowed"][0])
