#!/usr/bin/env python3
"""Probe the live ROS Planner message and its pre-receipt time staleness.

Run a ``solver_node`` from the current ``hope_ws`` overlay first.  This probe
publishes valid ``PredictedStrike`` messages with controlled publication
delays, receives ``RacketCommand``, then checks the repository's Python
adapter against the real generated ROS message class.

The source and receiver both use the node's ROS clock in this experiment.
That is deliberate: it makes communication-delay staleness measurable without
inventing a cross-clock mapping.  It does not claim that mocap and robot clocks
are synchronized on hardware.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rclpy  # noqa: E402
from msgs.msg import PredictedStrike, RacketCommand  # noqa: E402
from rclpy.node import Node  # noqa: E402

from training.utils.strike_goal import PlannerRacketCommand  # noqa: E402


def _set_stamp(message, time_s: float) -> None:
    sec = int(time_s)
    message.header.stamp.sec = sec
    message.header.stamp.nanosec = int(round((time_s - sec) * 1.0e9))


def _stamp_seconds(message) -> float:
    return float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1.0e-9


class PlannerCommandProbe(Node):
    def __init__(self) -> None:
        super().__init__("p1_planner_contract_probe")
        self.publisher = self.create_publisher(PredictedStrike, "/ball/predicted_strike", 10)
        self.latest_command: RacketCommand | None = None
        self.subscription = self.create_subscription(
            RacketCommand, "/racket/command", self._command_callback, 10
        )

    def _command_callback(self, message: RacketCommand) -> None:
        self.latest_command = message

    def wait_for_solver(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while self.publisher.get_subscription_count() < 1:
            if time.monotonic() >= deadline:
                raise TimeoutError("no solver subscription on /ball/predicted_strike")
            rclpy.spin_once(self, timeout_sec=0.02)

    def run_case(self, requested_delay_s: float, timeout_s: float) -> dict[str, object]:
        message = PredictedStrike()
        source_time_s = self.get_clock().now().nanoseconds * 1.0e-9
        _set_stamp(message, source_time_s)
        message.header.frame_id = "world"
        message.state = "p1_delay_probe"
        message.reason = "controlled_publication_delay"
        message.strike_position.x = 0.0
        message.strike_position.y = -0.7625
        message.strike_position.z = 0.3
        message.strike_velocity.x = -3.0
        message.strike_velocity.y = 0.0
        message.strike_velocity.z = -0.5
        message.strike_time = source_time_s + 0.5
        message.time_to_strike = 0.5
        message.predicted_bounces = 1
        message.valid = True

        time.sleep(requested_delay_s)
        self.latest_command = None
        self.publisher.publish(message)
        deadline = time.monotonic() + timeout_s
        while self.latest_command is None:
            if time.monotonic() >= deadline:
                raise TimeoutError("solver did not publish /racket/command")
            rclpy.spin_once(self, timeout_sec=0.02)

        received_time_s = self.get_clock().now().nanoseconds * 1.0e-9
        command = self.latest_command
        assert command is not None
        adapted = PlannerRacketCommand.from_ros_message(command)
        source_stamp_s = _stamp_seconds(command)
        actual_pre_receipt_delay_s = received_time_s - source_stamp_s
        actual_remaining_at_receive_s = command.strike_time - received_time_s
        message_staleness_s = command.time_to_strike - actual_remaining_at_receive_s
        return {
            "requested_publication_delay_s": requested_delay_s,
            "actual_pre_receipt_delay_s": actual_pre_receipt_delay_s,
            "source_stamp_s": source_stamp_s,
            "received_ros_time_s": received_time_s,
            "strike_time_s": command.strike_time,
            "message_time_to_strike_s": command.time_to_strike,
            "actual_remaining_at_receive_s": actual_remaining_at_receive_s,
            "message_staleness_s": message_staleness_s,
            "header_preserved": abs(adapted.header_stamp_s - source_stamp_s) < 1.0e-9,
            "position_equals_predicted_ball_position": all(
                abs(left - right) < 1.0e-12
                for left, right in zip(adapted.goal.position, (0.0, -0.7625, 0.3), strict=True)
            ),
            "goal_10d_raw": list(adapted.goal.to_vector()),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delays-s", nargs="+", type=float, default=(0.0, 0.05, 0.15))
    parser.add_argument("--timeout-s", type=float, default=3.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if any(delay < 0.0 for delay in args.delays_s):
        parser.error("--delays-s values must be non-negative")

    rclpy.init()
    node = PlannerCommandProbe()
    try:
        node.wait_for_solver(args.timeout_s)
        cases = [node.run_case(delay, args.timeout_s) for delay in args.delays_s]
    finally:
        node.destroy_node()
        rclpy.shutdown()

    report = {
        "schema_version": 1,
        "probe": "planner_racket_command_ros_delay",
        "clock_contract": "source and receive use the same ROS clock for this controlled experiment",
        "cases": cases,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
