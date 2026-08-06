from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.utils.strike_goal_contract_probe import (  # noqa: E402
    TcpProbeSample,
    TimeProbeSample,
    analyze_tcp_samples,
    analyze_time_samples,
)


IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def test_tcp_audit_identifies_a_consistent_local_offset_and_normal_alignment():
    samples = [
        TcpProbeSample(
            sample_id="ready", timestamp_s=1.0, pose_label="ready",
            planner_command_position_world=(1.1, 2.2, 3.3), planner_normal_world=(0.0, 1.0, 0.0),
            policy_tcp_position_world=(1.0, 2.0, 3.0), policy_tcp_normal_world=(0.0, 1.0, 0.0),
            racket_link_origin_world=(1.0, 2.0, 3.0), world_from_racket_rotation=IDENTITY,
            policy_tcp_name="pingpang_red_Link origin",
        ),
        TcpProbeSample(
            sample_id="hit", timestamp_s=2.0, pose_label="motion_2_hit",
            planner_command_position_world=(0.6, 0.8, 1.0), planner_normal_world=(0.0, 1.0, 0.0),
            policy_tcp_position_world=(0.5, 0.6, 0.7), policy_tcp_normal_world=(0.0, 1.0, 0.0),
            racket_link_origin_world=(0.5, 0.6, 0.7), world_from_racket_rotation=IDENTITY,
            policy_tcp_name="pingpang_red_Link origin",
        ),
    ]
    report = analyze_tcp_samples(samples)
    assert report["planner_minus_policy_tcp_racket_m"]["mean"] == pytest.approx([0.1, 0.2, 0.3])
    assert report["planner_minus_policy_tcp_racket_m"]["std"] == pytest.approx([0.0, 0.0, 0.0])
    assert report["normal_angle_deg"]["max"] == pytest.approx(0.0)


def test_time_audit_checks_only_known_same_clock_relations_and_policy_countdown():
    samples = [
        TimeProbeSample(
            command_id="goal-7", source_clock_domain="mocap", control_clock_domain="sim",
            header_stamp_s=10.0, strike_time_s=10.5, message_time_to_strike_s=0.5,
            received_control_time_s=4.0, current_control_time_s=4.1, policy_time_to_strike_s=0.4,
            simulation_time_s=4.1, control_step=41,
        ),
        TimeProbeSample(
            command_id="goal-7", source_clock_domain="mocap", control_clock_domain="sim",
            header_stamp_s=10.1, strike_time_s=10.5, message_time_to_strike_s=0.4,
            received_control_time_s=4.1, current_control_time_s=4.2, policy_time_to_strike_s=0.3,
            simulation_time_s=4.2, control_step=42,
        ),
    ]
    report = analyze_time_samples(samples)
    assert report["strike_minus_header_minus_message_tts_s"]["max_abs"] == pytest.approx(0.0)
    assert report["policy_countdown"]["residual_max_abs_s"] == pytest.approx(0.0)
    assert report["clock_mapping_status"] == "not_provided"
    assert report["mapped_remaining_time_at_current_control_s"] is None

    mapped = analyze_time_samples(samples, source_to_control_offset_s=-6.0)
    assert mapped["clock_mapping_status"] == "provided_by_caller"
    assert mapped["mapped_remaining_time_at_current_control_s"]["max"] == pytest.approx(0.4)
