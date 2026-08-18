# Copyright (c) 2026 Intelligent Racing Inc. (dba Hitch Interactive)
# SPDX-License-Identifier: Apache-2.0
"""CLI entrypoint: run the reference runner against the MuJoCo sim.

    python -m a3_deploy_onnx_ref_pingpong --view --realtime

By default it loads the shipped runtime config, drives the in-process MuJoCo
bridge, and feeds an example (planner-less) command stream so the policy visibly
swings. Point ``--onnx`` at your exported ``policy.onnx`` (110-D hitter_pure).

Command feeds (mutually exclusive):
  * default      -- the built-in ExampleCommandFeed (planner-less demo);
  * ``--planner``-- subscribe the live planner's flat command stream
                    (std_msgs/Float64MultiArray on ``--command-topic``; needs only
                    a sourced ROS 2 env — no hope_msgs build; see
                    ros_command_source.py). This is the documented full
                    planner -> runner control path;
  * ``--idle``   -- no commands at all (robot just holds its stand).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .config import RuntimeConfig
from .initial_stance import get_initial_stance_offset
from .joint_order import JOINT_NAMES
from .racket_command import ExampleCommandFeed, QueueRacketCommandSource
from .ros_command_source import DEFAULT_COMMAND_TOPIC
from .runner import PingPongReferenceRunner

_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "hope_pingpong_runtime.yaml"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="a3_deploy_onnx_ref_pingpong", description=__doc__)
    p.add_argument("--config", type=Path, default=_DEFAULT_CONFIG,
                   help="runtime YAML (default: the shipped config)")
    p.add_argument("--backend", choices=["mujoco", "aimrt"], default="mujoco",
                   help="mujoco = in-process (default); aimrt = live-sim seam")
    p.add_argument("--onnx", type=Path, default=None, help="override the ONNX policy path")
    p.add_argument("--model-xml", type=Path, default=None,
                   help="override the a3_pingpong MJCF path (mujoco backend)")
    p.add_argument("--view", action="store_true", help="launch the MuJoCo passive viewer")
    p.add_argument("--realtime", action="store_true", help="pace the loop at wall-clock 50 Hz")
    p.add_argument("--video", type=Path, default=None,
                   help="record the MuJoCo replay to an MP4 at 50 fps")
    p.add_argument("--video-width", type=int, default=960)
    p.add_argument("--video-height", type=int, default=720)
    p.add_argument(
        "--initial-stance",
        choices=["standard", "right_front", "left_front", "width50_parallel"],
        default=None,
        help=(
            "initial MuJoCo pose; width50_parallel is the fixed 50 cm parallel "
            "Curriculum-FT target"
        ),
    )
    p.add_argument("--duration", type=float, default=None, help="run for N seconds")
    p.add_argument("--max-ticks", type=int, default=None, help="run for N control ticks")
    feed = p.add_mutually_exclusive_group()
    feed.add_argument("--planner", action="store_true",
                      help="consume the live planner's flat command stream "
                           "(std_msgs/Float64MultiArray; the full planner -> runner path; "
                           "needs only rclpy — no hope_msgs build)")
    feed.add_argument("--idle", action="store_true",
                      help="no command feed (robot just holds a stand)")
    p.add_argument("--command-topic", default=DEFAULT_COMMAND_TOPIC,
                   help=f"planner command topic for --planner (default: {DEFAULT_COMMAND_TOPIC})")
    p.add_argument("--warmup-ticks", type=int, default=None,
                   help="override HOPE PD_STAND warmup length (50 Hz ticks)")
    p.add_argument("--hold-recover-s", type=float, default=None,
                   help="policy recovery hold before official static stand")
    p.add_argument("--motion-blend-sec", type=float, default=None,
                   help="stand-to-policy q_des blend duration")
    p.add_argument("--auto-leg-hold", action="store_true",
                   help="native level-0 official leg+waist hold")
    p.add_argument("--no-official-stand", action="store_true",
                   help="disable official PD_STAND gains")
    p.add_argument("--leg-clamp-rad", type=float, default=None,
                   help="clamp released leg q_des around nominal pose")
    p.add_argument("--leg-smooth-alpha", type=float, default=None,
                   help="EMA alpha for released leg q_des")
    p.add_argument("--gain-scale", type=float, default=None)
    p.add_argument("--leg-gain-scale", type=float, default=None)
    p.add_argument("--ankle-gain-scale", type=float, default=None)
    p.add_argument(
        "--waist-pitch-offset-rad", type=float, default=0.0,
        help="extra waist pitch stance offset for stability comparison",
    )
    p.add_argument(
        "--hip-pitch-offset-rad", type=float, default=0.0,
        help=(
            "extra symmetric left/right hip pitch stance offset; in the A3 "
            "model negative values increase forward hip flexion"
        ),
    )
    p.add_argument(
        "--stability-csv", type=Path, default=None,
        help="write COM/support/tilt stability telemetry CSV",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = RuntimeConfig.load(args.config)
    if args.onnx is not None:
        cfg.onnx_path = args.onnx.resolve()
    if args.model_xml is not None:
        cfg.model_xml_path = args.model_xml.resolve()
    if args.warmup_ticks is not None:
        cfg.warmup_ticks = args.warmup_ticks
    if args.hold_recover_s is not None:
        cfg.hold_recover_s = args.hold_recover_s
    if args.motion_blend_sec is not None:
        cfg.motion_blend_ticks = int(round(args.motion_blend_sec / cfg.control_dt))
    if args.auto_leg_hold:
        cfg.auto_leg_hold = True
    if args.no_official_stand:
        cfg.official_stand = False
    if args.leg_clamp_rad is not None:
        cfg.leg_clamp_rad = args.leg_clamp_rad
    if args.leg_smooth_alpha is not None:
        cfg.leg_smooth_alpha = args.leg_smooth_alpha
    if args.gain_scale is not None:
        cfg.gain_scale = args.gain_scale
    if args.leg_gain_scale is not None:
        cfg.leg_gain_scale = args.leg_gain_scale
    if args.ankle_gain_scale is not None:
        cfg.ankle_gain_scale = args.ankle_gain_scale
    if args.initial_stance is not None:
        cfg.initial_stance = args.initial_stance

    # The MuJoCo bridge applies the selected stance to the reset qpos.  Keep
    # the same stance in subsequent q_des targets as well; otherwise a
    # width50_parallel reset is immediately pulled back toward the neutral
    # adapter posture during PD_STAND/policy execution.  Native deployment
    # applies this offset in its action adapter, so the reference runner must
    # do the same for bundle configs that carry the stance as a separate JSON.
    if cfg.initial_stance == "width50_parallel":
        stance = get_initial_stance_offset(cfg.initial_stance)
        if stance is not None:
            offsets, _ = stance
            if np.allclose(cfg.action_adapter.stance_offset, 0.0):
                cfg.action_adapter.stance_offset = np.asarray(
                    [offsets[name] for name in JOINT_NAMES], dtype=np.float64
                )
    if abs(float(args.waist_pitch_offset_rad)) > 0.0:
        cfg.action_adapter.stance_offset[2] += float(args.waist_pitch_offset_rad)
    if abs(float(args.hip_pitch_offset_rad)) > 0.0:
        for name in ("left_hip_pitch_joint", "right_hip_pitch_joint"):
            cfg.action_adapter.stance_offset[JOINT_NAMES.index(name)] += float(
                args.hip_pitch_offset_rad
            )

    if not Path(cfg.onnx_path).is_file():
        print(
            f"[ref] ONNX policy not found: {cfg.onnx_path}\n"
            f"      Export one with the training package "
            f"(policy.onnx) and pass --onnx, or set policy.onnx_path.",
            file=sys.stderr,
        )
        return 2

    # Sim bridge.
    if args.backend == "mujoco":
        from .sim_bridge import MujocoDirectBridge

        bridge = MujocoDirectBridge(
            cfg.model_xml_path, control_dt=cfg.control_dt, launch_viewer=args.view,
            video_path=args.video, video_width=args.video_width,
            video_height=args.video_height, initial_stance=cfg.initial_stance,
        )
    else:
        from .sim_bridge import AimrtSimBridge

        bridge = AimrtSimBridge()  # documented seam: raises with wiring guidance

    # Command source.
    if args.planner:
        from .ros_command_source import RosRacketCommandSource

        source = RosRacketCommandSource(topic=args.command_topic)
        print(f"[ref] consuming planner commands on {args.command_topic}", file=sys.stderr)
    elif args.idle:
        source = QueueRacketCommandSource()
    else:
        source = ExampleCommandFeed(dt=cfg.control_dt)

    max_ticks = args.max_ticks
    if args.duration is not None:
        max_ticks = int(round(args.duration / cfg.control_dt))

    runner = PingPongReferenceRunner(cfg, bridge, source)
    try:
        runner.run(
            max_ticks=max_ticks,
            realtime=args.realtime,
            stability_csv=args.stability_csv,
        )
    finally:
        close = getattr(source, "close", None)
        if close is not None:
            close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
