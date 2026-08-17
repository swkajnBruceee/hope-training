from pathlib import Path
import importlib.util

import pytest

ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = ROOT / "source/whole_body_tracking/whole_body_tracking/utils/stance_curriculum.py"
_SPEC = importlib.util.spec_from_file_location("stance_curriculum_pure", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
lerp = _MODULE.lerp
smoothstep_stance_alpha = _MODULE.smoothstep_stance_alpha
W_OLD = 0.264713494
W_NEW = 0.500000000
H_OLD = 1.068390000
H_NEW = 1.026223554


@pytest.mark.parametrize(
    ("iteration", "expected"),
    ((0, 0.0), (300, 0.0), (600, 0.0740740741), (900, 0.2592592593),
     (1200, 0.5), (1500, 0.7407407407), (1800, 0.9259259259),
     (2100, 1.0), (2400, 1.0), (3000, 1.0)),
)
def test_fixed_smoothstep_schedule(iteration, expected):
    assert smoothstep_stance_alpha(iteration) == pytest.approx(expected, abs=1.0e-9)


def test_action_reset_reward_geometry_share_targets():
    for iteration in (0, 300, 600, 1200, 1800, 2100, 3000):
        alpha = smoothstep_stance_alpha(iteration)
        width_target = lerp(W_OLD, W_NEW, alpha)
        reward_lo = lerp(0.25, 0.45, alpha)
        reward_hi = lerp(0.35, 0.55, alpha)
        root_target = lerp(H_OLD, H_NEW, alpha)
        # Action/reset geometry and reward reference use the same alpha; the reward band remains
        # a tolerance around the measured nominal geometry at both schedule endpoints.
        if iteration == 0:
            assert width_target == pytest.approx(W_OLD)
            assert root_target == pytest.approx(H_OLD)
            assert reward_lo == pytest.approx(0.25)
            assert reward_hi == pytest.approx(0.35)
        if iteration >= 2100:
            assert width_target == pytest.approx(W_NEW)
            assert root_target == pytest.approx(H_NEW)
            assert reward_lo == pytest.approx(0.45)
            assert reward_hi == pytest.approx(0.55)


def test_training_contract_is_3000_iterations():
    algo = (ROOT / "cfg/algo/ppo_residual_a5_stance_curriculum.yaml").read_text()
    task = (ROOT / "cfg/task/HOPEPingPongStanceCurriculum.yaml").read_text()
    assert "max_iterations: 3000" in algo
    assert "stance_curriculum_steps: 3000" in task
    assert "stance_curriculum_ramp_start_iteration: 300" in task
    assert "stance_curriculum_ramp_end_iteration: 2100" in task


def test_friction_curriculum_is_independent_and_late():
    task = (ROOT / "cfg/task/HOPEPingPongStanceCurriculum.yaml").read_text()
    assert "curriculum_start_iteration: 2100" in task
    assert "curriculum_end_iteration: 2700" in task
    beta = [smoothstep_stance_alpha(i, ramp_start_iteration=2100, ramp_end_iteration=2700)
            for i in (0, 2100, 2400, 2700, 3000)]
    assert beta == pytest.approx([0.0, 0.0, 0.5, 1.0, 1.0])
    assert 1.0 - 0.7 * beta[0] == pytest.approx(1.0)
    assert 1.0 + 0.5 * beta[0] == pytest.approx(1.0)
    assert 1.0 - 0.7 * beta[2] == pytest.approx(0.65)
    assert 1.0 + 0.5 * beta[2] == pytest.approx(1.25)
    assert 1.0 - 0.7 * beta[3] == pytest.approx(0.3)
    assert 1.0 + 0.5 * beta[3] == pytest.approx(1.5)
