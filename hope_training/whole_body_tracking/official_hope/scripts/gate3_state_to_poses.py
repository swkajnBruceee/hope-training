#!/usr/bin/env python3
"""Bridge the Gate3 physical-ball state into the HOPE planner ``/poses`` topic.

The Gate3 MuJoCo simulator publishes ``mujoco_sim_msgs/Gate3BallState`` in the
floor-origin world frame.  The HOPE planner consumes a one-pose ``PoseArray``
whose z origin is the table surface.  This bridge is deliberately kept outside
the planner and native runner so the physical-ball transport remains observable
and replaceable.
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import Pose, PoseArray
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from mujoco_sim_msgs.msg import Gate3BallState


class Gate3StateToPoses(Node):
    def __init__(self) -> None:
        super().__init__("gate3_state_to_poses")
        self.declare_parameter("input_topic", "/sim/gate3/ball_state")
        self.declare_parameter("output_topic", "/poses")
        self.declare_parameter("table_surface_z", 0.76)
        # The planner consumes a continuous mocap-like stream.  Gate3 keeps
        # the shot id and contact counters in every state sample, while the
        # active bit is a command/lifecycle flag and is not guaranteed to stay
        # true for the whole physical flight in every simulator build.
        # Publishing all finite state samples preserves the estimator's
        # velocity/history window; the planner itself decides whether a state
        # is incoming/usable.
        self.declare_parameter("active_only", False)

        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        self._table_z = float(self.get_parameter("table_surface_z").value)
        self._active_only = bool(self.get_parameter("active_only").value)
        self._last_state_key = None

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
        )
        output_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._pub = self.create_publisher(PoseArray, output_topic, output_qos)
        self._sub = self.create_subscription(
            Gate3BallState, input_topic, self._on_state, sensor_qos
        )
        self.get_logger().info(
            "{} -> {}; floor-to-table z offset={:.3f} m; active_only={}".format(
                input_topic, output_topic, self._table_z, self._active_only
            )
        )

    def _on_state(self, msg: Gate3BallState) -> None:
        state_key = (
            int(msg.shot_id), bool(msg.active),
            round(float(msg.position.x), 3),
            round(float(msg.position.y), 3),
            round(float(msg.position.z), 3),
        )
        if state_key != self._last_state_key:
            self._last_state_key = state_key
            self.get_logger().info(
                "state shot=%d active=%s p=(%.3f,%.3f,%.3f)"
                % (state_key[0], state_key[1], state_key[2], state_key[3], state_key[4])
            )
        if self._active_only and not bool(msg.active):
            return

        out = PoseArray()
        out.header = msg.header
        pose = Pose()
        pose.position.x = float(msg.position.x)
        pose.position.y = float(msg.position.y)
        pose.position.z = float(msg.position.z) - self._table_z
        pose.orientation.w = 1.0
        out.poses = [pose]
        self._pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Gate3StateToPoses()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
