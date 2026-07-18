#!/usr/bin/env python3
"""Reset the local A3 MuJoCo model to its documented ``stand`` keyframe.

This is deliberately a simulator-only setup action.  It uses the simulator's
``/sim/a3/reset`` API, clears every velocity/controller value, and waits until
the published pelvis pose confirms the standing keyframe before returning.
The replay recorder performs the separate 3-second PD-STAND stability gate.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np


def quaternion_tilt_deg_xyzw(x: float, y: float, z: float, w: float) -> float:
    """Return the pelvis local-Z tilt from world vertical in degrees."""

    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        return math.inf
    x, y, z, w = quaternion / norm
    local_z_world_z = 1.0 - 2.0 * (x * x + y * y)
    return float(np.degrees(np.arccos(np.clip(local_z_world_z, -1.0, 1.0))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset-topic", default="/sim/a3/reset")
    parser.add_argument("--pelvis-topic", default="/sim/a3/pelvis_pose")
    parser.add_argument("--keyframe-id", type=int, default=0,
                        help="A3 pingpong model keyframe 0 is named stand.")
    parser.add_argument("--wait-s", type=float, default=5.0,
                        help="Maximum wait for reset subscriber and standing pose.")
    parser.add_argument("--publish-count", type=int, default=3)
    parser.add_argument("--publish-interval-s", type=float, default=0.05)
    parser.add_argument("--min-pelvis-height-m", type=float, default=0.75)
    parser.add_argument("--max-pelvis-tilt-deg", type=float, default=25.0)
    parser.add_argument("--ack-file", type=str,
                        help="Write an atomic passed=true JSON acknowledgement after upright-pose verification.")
    args = parser.parse_args()
    if (args.keyframe_id < 0 or args.wait_s <= 0.0 or args.publish_count < 1
            or args.publish_interval_s < 0.0 or args.min_pelvis_height_m <= 0.0
            or args.max_pelvis_tilt_deg < 0.0):
        parser.error("invalid reset or standing-pose argument")
    return args


def main() -> None:
    args = parse_args()
    try:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from mujoco_sim_msgs.msg import SimReset
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    except ImportError as exc:  # pragma: no cover - depends on local ROS overlay
        raise SystemExit(
            "ROS2 reset messages are unavailable. Source "
            "tools/setup_a3_mujoco_sim_env.sh before running this utility."
        ) from exc

    rclpy.init()
    node = rclpy.create_node("a3_mujoco_stand_keyframe_reset")
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
    latest: dict[str, float] = {}

    def on_pelvis(message: PoseStamped) -> None:
        pose = message.pose
        latest["height"] = float(pose.position.z)
        latest["tilt"] = quaternion_tilt_deg_xyzw(
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
        )

    publisher = node.create_publisher(SimReset, args.reset_topic, qos)
    node.create_subscription(PoseStamped, args.pelvis_topic, on_pelvis, qos)
    message = SimReset()
    message.mode = SimReset.MODE_KEYFRAME
    message.keyframe_id = args.keyframe_id
    message.set_base = False
    message.set_base_twist = False
    message.set_joints = False
    message.zero_all_velocities = True
    message.clear_ctrl = True
    deadline = time.monotonic() + args.wait_s
    try:
        while rclpy.ok() and publisher.get_subscription_count() < 1 and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        if publisher.get_subscription_count() < 1:
            raise RuntimeError(f"no reset subscriber on {args.reset_topic} within {args.wait_s:.1f}s")
        for _ in range(args.publish_count):
            publisher.publish(message)
            rclpy.spin_once(node, timeout_sec=args.publish_interval_s)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            if (latest.get("height", -math.inf) >= args.min_pelvis_height_m
                    and latest.get("tilt", math.inf) <= args.max_pelvis_tilt_deg):
                print(
                    "A3 stand keyframe applied: "
                    f"height={latest['height']:.3f} m, tilt={latest['tilt']:.2f} deg"
                )
                if args.ack_file:
                    from pathlib import Path

                    acknowledgement = Path(args.ack_file).expanduser().resolve()
                    acknowledgement.parent.mkdir(parents=True, exist_ok=True)
                    temporary = acknowledgement.with_suffix(acknowledgement.suffix + ".tmp")
                    temporary.write_text(
                        json.dumps({
                            "passed": True,
                            "min_pelvis_height_m": latest["height"],
                            "max_pelvis_tilt_deg": latest["tilt"],
                        }, separators=(",", ":")) + "\n",
                        encoding="utf-8",
                    )
                    temporary.replace(acknowledgement)
                return
        height = latest.get("height", math.nan)
        tilt = latest.get("tilt", math.nan)
        raise RuntimeError(
            "stand keyframe did not produce an upright pelvis: "
            f"height={height:.3f} m, tilt={tilt:.2f} deg"
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
