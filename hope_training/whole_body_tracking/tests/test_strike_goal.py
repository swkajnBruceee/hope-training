from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.utils.strike_goal import (
    AxialRacketContactCalibration,
    HOPE_WORLD_FRAME,
    LatchedStrikeGoal,
    PlannerRacketCommand,
    STRIKE_GOAL_CONTRACT_VERSION,
    STRIKE_GOAL_LINEAR_VELOCITY_SEMANTICS,
    STRIKE_GOAL_NORMAL_SEMANTICS,
    STRIKE_GOAL_POSITION_SEMANTICS,
    StrikeGoal10D,
    StrikeGoalFrameTransform,
    StrikeGoalNormalizer,
    StrikeGoalTrace,
    StrikeGoalValidationError,
    StrikeGoalValidator,
    isaac_diagnostic_proxy_contact_calibration,
)


IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _goal(**overrides) -> StrikeGoal10D:
    values = {
        "position": (0.3, -0.2, 0.8),
        "normal": (0.0, 1.0, 0.0),
        "linear_velocity": (1.2, 0.1, -0.4),
        "time_to_hit_s": 0.6,
        "frame_id": "base_heading_receipt/v1",
        "source": "synthetic",
        "receipt_time_s": 42.0,
    }
    values.update(overrides)
    return StrikeGoal10D(**values)


def test_vector_and_mapping_round_trip_preserve_the_ten_dimensional_contract():
    goal = _goal()
    assert goal.to_vector() == pytest.approx((0.3, -0.2, 0.8, 0.0, 1.0, 0.0, 1.2, 0.1, -0.4, 0.6))
    restored = StrikeGoal10D.from_mapping(goal.to_mapping())
    assert restored == goal
    assert restored.contract_version == STRIKE_GOAL_CONTRACT_VERSION


def test_contract_version_exposes_the_planners_mixed_physical_semantics():
    assert STRIKE_GOAL_CONTRACT_VERSION == "strike_goal_10d/ball_center_impact_v1"
    assert STRIKE_GOAL_POSITION_SEMANTICS == "predicted_ball_center_at_strike"
    assert STRIKE_GOAL_NORMAL_SEMANTICS == "desired_ideal_racket_face_normal"
    assert (
        STRIKE_GOAL_LINEAR_VELOCITY_SEMANTICS
        == "desired_ideal_racket_impact_velocity"
    )


def test_proxy_contact_calibration_keeps_ball_face_and_link_points_distinct():
    calibration = isaac_diagnostic_proxy_contact_calibration()
    target = calibration.resolve(
        _goal(
            position=(1.0, 2.0, 3.0),
            normal=(1.0, 0.0, 0.0),
            linear_velocity=(1.2, 0.0, 0.4),
        )
    )
    assert target.ball_center_position == pytest.approx((1.0, 2.0, 3.0))
    assert target.face_contact_position == pytest.approx((1.020, 2.0, 3.0))
    assert target.link_origin_position == pytest.approx((1.017, 2.0, 3.0))
    assert target.face_linear_velocity == pytest.approx((1.2, 0.0, 0.4))
    assert calibration.ball_center_to_link_origin_along_normal_m == pytest.approx(0.017)
    assert target.qualified_domain == "isaac_diagnostic_proxy_only"


def test_contact_calibration_rejects_invalid_geometry_and_non_unit_goal_normal():
    with pytest.raises(StrikeGoalValidationError, match="ball_radius"):
        AxialRacketContactCalibration(0.0, 0.0, "cal/v1", "sim")
    calibration = AxialRacketContactCalibration(0.02, 0.0, "cal/v1", "sim")
    with pytest.raises(StrikeGoalValidationError, match="unit length"):
        calibration.resolve(_goal(normal=(2.0, 0.0, 0.0)))


def test_validator_rejects_non_unit_normal_expired_execution_and_frame_mismatch():
    validator = StrikeGoalValidator(accepted_frames=("base_heading_receipt/v1",))
    with pytest.raises(StrikeGoalValidationError, match="unit length"):
        validator.validate(_goal(normal=(0.0, 2.0, 0.0)))
    with pytest.raises(StrikeGoalValidationError, match="expired"):
        validator.validate(_goal(time_to_hit_s=0.0), require_future_hit=True)
    with pytest.raises(StrikeGoalValidationError, match="unaccepted goal frame"):
        validator.validate(_goal(frame_id="world/v1"))


def test_rigid_transform_rotates_normal_and_velocity_but_translates_only_position():
    transform = StrikeGoalFrameTransform(
        source_frame="base_heading_receipt/v1",
        target_frame="world/v1",
        rotation=((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        translation=(3.0, 4.0, 0.5),
    )
    actual = transform.apply(_goal())
    assert actual.position == pytest.approx((3.2, 4.3, 1.3))
    assert actual.normal == pytest.approx((-1.0, 0.0, 0.0))
    assert actual.linear_velocity == pytest.approx((-0.1, 1.2, -0.4))
    assert actual.time_to_hit_s == pytest.approx(0.6)
    StrikeGoalValidator().validate(actual)


def test_rigid_transform_composition_keeps_world_to_sim_and_receipt_frames_explicit():
    world_to_sim = StrikeGoalFrameTransform(
        source_frame="world",
        target_frame="isaac_tracking_world/v1",
        rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        translation=(0.0, 0.0, 0.76),
    )
    sim_to_receipt = StrikeGoalFrameTransform(
        source_frame="isaac_tracking_world/v1",
        target_frame="base_heading_receipt/v1",
        rotation=((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        translation=(0.35, 3.15, -1.04),
    )
    composed = world_to_sim.followed_by(sim_to_receipt)
    goal = StrikeGoal10D(
        position=(0.0, -0.7625, 0.3),
        normal=(1.0, 0.0, 0.0),
        linear_velocity=(1.2, 0.0, 0.4),
        time_to_hit_s=0.5,
        frame_id="world",
        source="replay",
    )
    composed_goal = composed.apply(goal)
    sequential_goal = sim_to_receipt.apply(world_to_sim.apply(goal))
    assert composed_goal.to_vector() == pytest.approx(sequential_goal.to_vector())
    assert composed_goal.frame_id == sequential_goal.frame_id
    with pytest.raises(StrikeGoalValidationError, match="middle frames"):
        world_to_sim.followed_by(
            StrikeGoalFrameTransform("wrong", "target", IDENTITY, (0.0, 0.0, 0.0))
        )


def test_normalizer_round_trip_and_time_advance_are_explicit_and_lossless():
    goal = _goal()
    normalizer = StrikeGoalNormalizer((0.5, 0.5, 1.0), (2.0, 2.0, 2.0), 1.0)
    normalized = normalizer.normalize(goal)
    assert normalized == pytest.approx((0.6, -0.4, 0.8, 0.0, 1.0, 0.0, 0.6, 0.05, -0.2, 0.6))
    restored = normalizer.denormalize(
        normalized,
        frame_id=goal.frame_id,
        source=goal.source,
        receipt_time_s=goal.receipt_time_s,
    )
    assert restored == goal
    assert goal.advance(0.15).time_to_hit_s == pytest.approx(0.45)
    assert goal.advance(1.0).time_to_hit_s == 0.0


def test_trace_retains_raw_and_normalized_goal_without_changing_its_source_semantics():
    goal = _goal(source="planner")
    trace = StrikeGoalTrace.capture(
        17,
        goal,
        StrikeGoalNormalizer((1.0, 1.0, 1.0), (1.0, 1.0, 1.0), 1.0),
    )
    payload = trace.to_mapping()
    assert payload["policy_step"] == 17
    assert payload["goal"]["source"] == "planner"
    assert payload["goal"]["frame_id"] == "base_heading_receipt/v1"
    assert payload["normalized_goal"] == pytest.approx(goal.to_vector())


def test_existing_planner_racket_command_maps_exactly_to_the_ten_dimensional_goal():
    message = {
        "header": {"frame_id": "world", "stamp": {"sec": 102, "nanosec": 250000000}},
        "position": {"x": 1.0, "y": -0.4, "z": 0.9},
        "velocity": {"x": 1.2, "y": 0.1, "z": 0.3},
        "normal": {"x": 0.0, "y": 1.0, "z": 0.0},
        "strike_time": 102.5,
        "time_to_strike": 0.55,
        "ball_velocity_incoming": {"x": -3.0, "y": 0.2, "z": -0.5},
        "ball_velocity_outgoing": {"x": 4.0, "y": -0.3, "z": 1.0},
        "valid": True,
        "clears_net": True,
        "bypasses_net_posts": False,
        "predicted_bounces": 1,
    }
    command = PlannerRacketCommand.from_ros_message(message)
    assert command.goal.frame_id == HOPE_WORLD_FRAME
    assert command.goal.to_vector() == pytest.approx((1.0, -0.4, 0.9, 0.0, 1.0, 0.0, 1.2, 0.1, 0.3, 0.55))
    assert command.strike_time_s == pytest.approx(102.5)
    assert command.header_stamp_s == pytest.approx(102.25)
    assert command.ball_velocity_incoming == pytest.approx((-3.0, 0.2, -0.5))


def test_planner_adapter_rejects_invalid_or_wrong_frame_messages():
    invalid = {
        "header": {"frame_id": "base_heading_receipt/v1"},
        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
        "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
        "normal": {"x": 1.0, "y": 0.0, "z": 0.0},
        "strike_time": 1.0,
        "time_to_strike": 0.1,
        "ball_velocity_incoming": {"x": 0.0, "y": 0.0, "z": 0.0},
        "ball_velocity_outgoing": {"x": 0.0, "y": 0.0, "z": 0.0},
        "valid": True,
        "clears_net": True,
        "bypasses_net_posts": False,
        "predicted_bounces": 0,
    }
    with pytest.raises(StrikeGoalValidationError, match="frame"):
        PlannerRacketCommand.from_ros_message(invalid)
    invalid["header"] = {"frame_id": "world"}
    invalid["valid"] = False
    with pytest.raises(StrikeGoalValidationError, match="invalid"):
        PlannerRacketCommand.from_ros_message(invalid)


def test_latched_goal_removes_verified_receive_delay_and_counts_down_on_control_clock():
    message = {
        "header": {"frame_id": "world", "stamp": {"sec": 100, "nanosec": 0}},
        "position": {"x": 0.0, "y": -0.7625, "z": 0.3},
        "velocity": {"x": 1.2, "y": 0.0, "z": 0.4},
        "normal": {"x": 1.0, "y": 0.0, "z": 0.0},
        "strike_time": 100.5,
        "time_to_strike": 0.5,
        "ball_velocity_incoming": {"x": -3.0, "y": 0.0, "z": -0.5},
        "ball_velocity_outgoing": {"x": 4.5, "y": 0.0, "z": 1.8},
        "valid": True,
        "clears_net": True,
        "bypasses_net_posts": False,
        "predicted_bounces": 1,
    }
    command = PlannerRacketCommand.from_ros_message(message)
    latched = LatchedStrikeGoal.from_planner_command(
        command,
        received_control_time_s=20.0,
        control_clock_domain="isaac_sim",
        verified_pre_receipt_delay_s=0.15,
    )
    assert latched.goal_at_receipt.time_to_hit_s == pytest.approx(0.35)
    assert latched.goal_at(20.1).time_to_hit_s == pytest.approx(0.25)
    assert latched.goal_at(20.6).time_to_hit_s == 0.0
    assert latched.is_expired(20.6)
    with pytest.raises(StrikeGoalValidationError, match="backwards"):
        latched.goal_at(19.9)
