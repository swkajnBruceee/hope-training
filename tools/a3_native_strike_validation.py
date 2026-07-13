#!/usr/bin/env python3
"""Validate A3 native standing + external arm/waist commands.

This script targets the official AimDK /motion/control interfaces, not the
local low-level /body_drive MuJoCo interface. Run it inside an AimDK/ROS2
environment where rclpy, sensor_msgs, ros2_plugin_proto, and aimdk are available.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


ARM_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

ARM_NOMINAL = [
    0.0,
    1.2,
    0.0,
    -0.5,
    1.5,
    0.0,
    0.0,
    0.0,
    -1.2,
    0.0,
    0.5,
    1.5,
    0.0,
    0.0,
]


@dataclass
class ProbeStats:
    arm_state_count: int = 0
    waist_state_count: int = 0
    imu_count: int = 0
    max_right_elbow_error: float = 0.0
    max_right_shoulder_roll_error: float = 0.0
    max_roll_rad: float = 0.0
    max_pitch_rad: float = 0.0
    first_arm_state_time: float | None = None
    last_arm_state_time: float | None = None
    notes: list[str] = field(default_factory=list)


def _check_dependencies() -> dict[str, str]:
    modules = [
        "rclpy",
        "sensor_msgs.msg",
        "ros2_plugin_proto.msg",
        "aimdk.protocol_pb2",
    ]
    status: dict[str, str] = {}
    for module in modules:
        try:
            __import__(module)
            status[module] = "OK"
        except Exception as exc:  # pragma: no cover - environment probe
            status[module] = f"MISSING: {type(exc).__name__}: {exc}"
    return status


def _create_header(seq: int, control_source: int = 1):
    from aimdk.protocol_pb2 import Header

    now = datetime.now(timezone.utc)
    ts = now.timestamp()
    header = Header()
    header.seq = seq
    header.timestamp.seconds = int(ts)
    header.timestamp.nanos = now.microsecond * 1000
    header.timestamp.ms_since_epoch = int(ts * 1000)
    header.control_source = control_source
    return header


def _quat_to_roll_pitch(x: float, y: float, z: float, w: float) -> tuple[float, float]:
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    return roll, pitch


def _post_action(host: str, port: int, command: str) -> Any:
    import requests

    now = datetime.utcnow()
    payload = {
        "header": {
            "timestamp": {
                "seconds": int(now.timestamp()),
                "nanos": now.microsecond * 1000,
                "ms_since_epoch": int(now.timestamp() * 1000),
            },
            "control_source": "ControlSource_SAFE",
            "uuid": "",
            "trace_id": "a3_native_strike_validation",
            "domin": "",
        },
        "command": command,
    }
    url = f"http://{host}:{port}/rpc/aimdk.protocol.MotionControlActionService/SetAction"
    response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=3.0)
    response.raise_for_status()
    return response.json()


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    from ros2_plugin_proto.msg import RosMsgWrapper
    from sensor_msgs.msg import Imu, JointState

    from aimdk.protocol_pb2 import MotionControlMoveWaistChannel

    qos = QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
    )

    class ProbeNode(Node):
        def __init__(self):
            super().__init__("a3_native_strike_validation")
            self.stats = ProbeStats()
            self.seq = 0
            self.start = time.monotonic()
            self.last_arm_cmd = list(ARM_NOMINAL)

            self.arm_pub = self.create_publisher(JointState, args.arm_command_topic, qos)
            self.waist_pub = self.create_publisher(RosMsgWrapper, args.waist_command_topic, qos)
            self.locomotion_pub = self.create_publisher(RosMsgWrapper, args.locomotion_topic, qos)

            self.create_subscription(JointState, args.arm_state_topic, self._on_arm_state, qos)
            if args.waist_state_topic:
                self.create_subscription(JointState, args.waist_state_topic, self._on_waist_state, qos)
            if args.imu_topic:
                self.create_subscription(Imu, args.imu_topic, self._on_imu, qos)

            self.timer = self.create_timer(1.0 / args.rate_hz, self._tick)

        def _on_arm_state(self, msg: JointState):
            now = time.monotonic()
            self.stats.arm_state_count += 1
            self.stats.first_arm_state_time = self.stats.first_arm_state_time or now
            self.stats.last_arm_state_time = now
            positions = dict(zip(msg.name, msg.position))
            for name, idx, attr in (
                ("right_elbow_joint", 10, "max_right_elbow_error"),
                ("right_shoulder_roll_joint", 8, "max_right_shoulder_roll_error"),
            ):
                if name in positions:
                    err = abs(float(positions[name]) - float(self.last_arm_cmd[idx]))
                    setattr(self.stats, attr, max(getattr(self.stats, attr), err))

        def _on_waist_state(self, _msg: JointState):
            self.stats.waist_state_count += 1

        def _on_imu(self, msg: Imu):
            self.stats.imu_count += 1
            q = msg.orientation
            roll, pitch = _quat_to_roll_pitch(q.x, q.y, q.z, q.w)
            self.stats.max_roll_rad = max(self.stats.max_roll_rad, abs(roll))
            self.stats.max_pitch_rad = max(self.stats.max_pitch_rad, abs(pitch))

        def _publish_locomotion_zero(self):
            payload = {
                "data": {
                    "mode": 0,
                    "forward_velocity": 0.0,
                    "lateral_velocity": 0.0,
                    "angular_velocity": 0.0,
                }
            }
            msg = RosMsgWrapper()
            msg.serialization_type = "json"
            msg.data = [bytes([x]) for x in json.dumps(payload).encode()]
            self.locomotion_pub.publish(msg)

        def _publish_arm(self, t: float):
            cmd = list(ARM_NOMINAL)
            if args.mode in ("arm", "combined"):
                phase = math.sin(2.0 * math.pi * args.frequency_hz * t)
                cmd[10] = ARM_NOMINAL[10] + args.arm_elbow_amp_rad * phase
                cmd[8] = ARM_NOMINAL[8] + args.arm_shoulder_roll_amp_rad * phase
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = list(ARM_NAMES)
            msg.position = cmd
            msg.velocity = [0.0] * len(cmd)
            msg.effort = [0.0] * len(cmd)
            self.last_arm_cmd = cmd
            self.arm_pub.publish(msg)

        def _publish_waist(self, t: float):
            if args.mode not in ("waist", "combined"):
                return
            phase = math.sin(2.0 * math.pi * args.frequency_hz * t)
            cmd = MotionControlMoveWaistChannel()
            cmd.header.CopyFrom(_create_header(self.seq))
            cmd.waist_pitch = args.waist_pitch_amp_rad * phase
            cmd.waist_roll = args.waist_roll_amp_rad * phase
            cmd.waist_yaw = args.waist_yaw_amp_rad * phase
            cmd.waist_height = args.waist_height_m
            raw = cmd.SerializeToString()
            msg = RosMsgWrapper()
            msg.serialization_type = "pb"
            msg.context = ["aimdk.protocol.MotionControlMoveWaistChannel"]
            msg.data = [bytes([x]) for x in raw]
            self.waist_pub.publish(msg)

        def _tick(self):
            t = time.monotonic() - self.start
            self.seq += 1
            self._publish_locomotion_zero()
            self._publish_arm(t)
            self._publish_waist(t)
            if t >= args.duration_s:
                self.timer.cancel()

    rclpy.init()
    node = ProbeNode()
    try:
        end = time.monotonic() + args.duration_s + 0.5
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        stats = node.stats
        node.destroy_node()
        rclpy.shutdown()

    if stats.arm_state_count == 0:
        stats.notes.append(f"no arm state received on {args.arm_state_topic}")
    if args.waist_state_topic and stats.waist_state_count == 0:
        stats.notes.append(f"no waist state received on {args.waist_state_topic}")
    if args.imu_topic and stats.imu_count == 0:
        stats.notes.append(f"no IMU received on {args.imu_topic}")

    return {
        "mode": args.mode,
        "duration_s": args.duration_s,
        "rate_hz": args.rate_hz,
        "arm_state_count": stats.arm_state_count,
        "waist_state_count": stats.waist_state_count,
        "imu_count": stats.imu_count,
        "max_right_elbow_error_rad": stats.max_right_elbow_error,
        "max_right_shoulder_roll_error_rad": stats.max_right_shoulder_roll_error,
        "max_roll_rad": stats.max_roll_rad,
        "max_pitch_rad": stats.max_pitch_rad,
        "notes": stats.notes,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["arm", "waist", "combined"], default="arm")
    p.add_argument("--duration-s", type=float, default=8.0)
    p.add_argument("--rate-hz", type=float, default=50.0)
    p.add_argument("--frequency-hz", type=float, default=0.25)
    p.add_argument("--arm-elbow-amp-rad", type=float, default=0.12)
    p.add_argument("--arm-shoulder-roll-amp-rad", type=float, default=0.08)
    p.add_argument("--waist-pitch-amp-rad", type=float, default=0.05)
    p.add_argument("--waist-roll-amp-rad", type=float, default=0.03)
    p.add_argument("--waist-yaw-amp-rad", type=float, default=0.05)
    p.add_argument("--waist-height-m", type=float, default=0.0)
    p.add_argument("--arm-command-topic", default="/motion/control/arm_joint_command")
    p.add_argument("--arm-state-topic", default="/motion/control/arm_joint_state")
    p.add_argument(
        "--waist-command-topic",
        default="/motion/control/move_waist/pb_3Aaimdk_2Eprotocol_2EMotionControlMoveWaistChannel",
    )
    p.add_argument("--waist-state-topic", default="")
    p.add_argument(
        "--locomotion-topic",
        default="/motion/control/locomotion_velocity/pb_3Aaimdk_2Eprotocol_2EMotionControlLocomotionVelocityChannel",
    )
    p.add_argument("--imu-topic", default="")
    p.add_argument("--robot-host", default="127.0.0.1")
    p.add_argument("--rpc-port", type=int, default=56322)
    p.add_argument("--set-action", default="", help="Optional action command string before publishing.")
    p.add_argument("--check-only", action="store_true", help="Only check Python dependencies.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    deps = _check_dependencies()
    if args.check_only:
        print(json.dumps({"dependencies": deps}, indent=2))
        return 0 if all(v == "OK" for v in deps.values()) else 2

    missing = {k: v for k, v in deps.items() if v != "OK"}
    if missing:
        print(json.dumps({"error": "missing dependencies", "dependencies": deps}, indent=2), file=sys.stderr)
        return 2

    if args.set_action:
        result = _post_action(args.robot_host, args.rpc_port, args.set_action)
        print(json.dumps({"set_action": args.set_action, "response": result}, indent=2, ensure_ascii=False))

    result = run_probe(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
