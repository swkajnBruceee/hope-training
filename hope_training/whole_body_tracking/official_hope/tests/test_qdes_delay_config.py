"""Host-only checks for the formal Hitter q_des execution-delay recipe."""

from __future__ import annotations

from pathlib import Path

import yaml


_ROOT = Path(__file__).resolve().parents[1]


def test_hitter_pingpong_uses_episode_fixed_zero_to_two_tick_delay():
    with (_ROOT / "cfg" / "task" / "HOPEPingPong.yaml").open() as handle:
        action = yaml.safe_load(handle)["action"]

    assert action["qdes_delay_min_steps"] == 0
    assert action["qdes_delay_max_steps"] == 2
    assert action["qdes_delay_nominal_fraction"] == 0.0


def test_delay_is_integer_control_ticks_not_per_step_jitter():
    """The action term's existing contract samples once on reset and stores an integer lag."""
    source = (
        _ROOT
        / "source"
        / "whole_body_tracking"
        / "whole_body_tracking"
        / "tasks"
        / "tracking"
        / "mdp"
        / "hope_actions.py"
    ).read_text()
    assert "def _resample_qdes_delay(self, env_ids: torch.Tensor)" in source
    assert "torch.randint(" in source
    assert "self._qdes_delay_steps[env_ids] = sampled" in source


def test_target_stream_uses_measured_defects_with_ability_gated_ramp_and_small_x_window():
    with (_ROOT / "cfg" / "task" / "HOPEPingPong.yaml").open() as handle:
        task = yaml.safe_load(handle)
    racket = task["racket"]
    assert racket["target_delay_steps"] == 1
    assert racket["target_noise_white"] == 0.0019
    assert racket["target_noise_ar1_sigma"] == 0.0052
    assert racket["target_noise_ar1_rho"] == 0.717
    assert racket["target_dropout_prob"] == 0.02
    assert racket["target_post_strike_dropout_s"] == 0.03
    assert racket["target_robustness_curriculum_by_ability_gate"] is True
    assert racket["target_robustness_ability_ramp_steps"] == 8000
    assert racket["base_target_x_range"] == [-0.02, 0.02]
    assert racket["pos_range_per_clip"]["forehand"]["x"] == [0.56, 0.60]
    assert racket["pos_range_per_clip"]["backhand"]["x"] == [0.56, 0.60]


def test_motor_capacity_randomization_is_in_the_formal_recipe():
    with (_ROOT / "cfg" / "task" / "HOPEPingPong.yaml").open() as handle:
        domain_rand = yaml.safe_load(handle)["domain_rand"]
    assert domain_rand["motor_strength_range"] == [0.9, 1.05]
    assert domain_rand["motor_capacity_nominal_fraction"] == 0.25
    source = (
        _ROOT / "source" / "whole_body_tracking" / "whole_body_tracking" /
        "tasks" / "tracking" / "mdp" / "events.py"
    ).read_text()
    assert "class randomize_a3_torque_capacity" in source
    assert "write_joint_effort_limit_to_sim" in source
