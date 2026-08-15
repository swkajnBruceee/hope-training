#!/usr/bin/env python3
"""Convert the AimRT MuJoCo entity-ball pose into the HOPE planner stream.

AimRT publishes the physical ball in the simulator floor frame (floor z=0).
The HOPE planner models the table surface as z=0, so this bridge subtracts
the configured table height exactly once and republishes a one-slot PoseArray
on the planner's standard ``/poses`` topic.
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


class MujocoBallPoseBridge(Node):
    def __init__(self) -> None:
        super().__init__("mujoco_ball_pose_to_poses")
        self.declare_parameter("input_topic", "/sim/a3/ball_pose")
        self.declare_parameter("output_topic", "/poses")
        self.declare_parameter("table_surface_z", 0.76)

        qos_in = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
        )
        qos_out = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        self._table_z = float(self.get_parameter("table_surface_z").value)
        self._pub = self.create_publisher(PoseArray, output_topic, qos_out)
        self._sub = self.create_subscription(PoseStamped, input_topic, self._on_pose, qos_in)
        self.get_logger().info(
            f"{input_topic} -> {output_topic}; floor-to-table z offset={self._table_z:.3f} m"
        )

    def _on_pose(self, msg: PoseStamped) -> None:
        out = PoseArray()
        out.header = msg.header
        pose = Pose()
        pose.position.x = msg.pose.position.x
        pose.position.y = msg.pose.position.y
        pose.position.z = msg.pose.position.z - self._table_z
        pose.orientation = msg.pose.orientation
        out.poses = [pose]
        self._pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MujocoBallPoseBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
