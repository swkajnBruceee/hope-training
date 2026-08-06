from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.utils.strike_goal import (  # noqa: E402
    LatchedStrikeGoal,
    StrikeGoal10D,
    StrikeGoalFrameTransform,
    StrikeGoalValidationError,
    isaac_diagnostic_proxy_contact_calibration,
)
from training.utils.strike_goal_shadow import (  # noqa: E402
    RacketFaceState,
    StrikeGoalShadowPipeline,
)


IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _pipeline() -> StrikeGoalShadowPipeline:
    goal = StrikeGoal10D(
        position=(1.0, 2.0, 3.0),
        normal=(1.0, 0.0, 0.0),
        linear_velocity=(1.0, 0.0, 0.0),
        time_to_hit_s=0.5,
        frame_id="world",
        source="replay",
    )
    latched = LatchedStrikeGoal(
        goal,
        received_control_time_s=10.0,
        control_clock_domain="isaac_sim",
        verified_pre_receipt_delay_s=0.05,
    )
    transform = StrikeGoalFrameTransform(
        source_frame="world",
        target_frame="base_heading_receipt/v1",
        rotation=IDENTITY,
        translation=(-0.5, 0.0, 0.0),
    )
    return StrikeGoalShadowPipeline(
        latched_goal=latched,
        source_to_policy_transform=transform,
        contact_calibration=isaac_diagnostic_proxy_contact_calibration(),
    )


def test_shadow_pipeline_resolves_countdown_frame_and_contact_without_action_effect():
    pipeline = _pipeline()
    sample = pipeline.capture(control_step=0, current_control_time_s=10.0)
    assert sample.source_goal.time_to_hit_s == pytest.approx(0.45)
    assert sample.policy_goal.position == pytest.approx((0.5, 2.0, 3.0))
    assert sample.target.face_contact_position == pytest.approx((0.52, 2.0, 3.0))
    assert sample.target.link_origin_position == pytest.approx((0.517, 2.0, 3.0))
    assert sample.to_mapping()["action_effect"] is False
    later = pipeline.capture(control_step=1, current_control_time_s=10.1)
    assert later.policy_goal.time_to_hit_s == pytest.approx(0.35)
    report = pipeline.to_mapping()
    assert report["action_effect"] is False
    assert report["sample_count"] == 2


def test_face_state_adds_omega_cross_r_before_shadow_error_measurement():
    calibration = isaac_diagnostic_proxy_contact_calibration()
    actual = RacketFaceState.from_link_state(
        link_origin_position=(0.517, 2.0, 3.0),
        link_origin_linear_velocity=(1.0, -0.003, 0.0),
        link_angular_velocity=(0.0, 0.0, 1.0),
        face_normal=(1.0, 0.0, 0.0),
        frame_id="base_heading_receipt/v1",
        calibration=calibration,
    )
    assert actual.face_contact_position == pytest.approx((0.520, 2.0, 3.0))
    assert actual.face_linear_velocity == pytest.approx((1.0, 0.0, 0.0))
    sample = _pipeline().capture(
        control_step=0,
        current_control_time_s=10.0,
        actual=actual,
    )
    assert sample.errors == pytest.approx(
        {
            "face_position_error_m": 0.0,
            "link_origin_position_error_m": 0.0,
            "face_velocity_error_mps": 0.0,
            "normal_angle_error_deg": 0.0,
        }
    )


def test_shadow_pipeline_fails_closed_on_non_monotonic_steps_or_frame_mismatch():
    pipeline = _pipeline()
    pipeline.capture(control_step=0, current_control_time_s=10.0)
    with pytest.raises(StrikeGoalValidationError, match="control_step"):
        pipeline.capture(control_step=0, current_control_time_s=10.1)

    mismatched = RacketFaceState.from_link_state(
        link_origin_position=(0.0, 0.0, 0.0),
        link_origin_linear_velocity=(0.0, 0.0, 0.0),
        link_angular_velocity=(0.0, 0.0, 0.0),
        face_normal=(1.0, 0.0, 0.0),
        frame_id="wrong",
        calibration=isaac_diagnostic_proxy_contact_calibration(),
    )
    with pytest.raises(StrikeGoalValidationError, match="frame"):
        _pipeline().capture(
            control_step=0, current_control_time_s=10.0, actual=mismatched
        )
