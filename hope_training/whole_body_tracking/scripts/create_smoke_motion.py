"""Create a tiny local A3 motion file for public smoke training.

The generated clip holds the prepared A3 articulation in its default standing
pose and writes the same `.npz` schema consumed by the tracking task. It is only
for pipeline verification; use retargeted ping-pong motions for real training.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import traceback

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate a local A3 stand-still smoke motion.")
parser.add_argument("--frames", type=int, default=120, help="Number of frames to record.")
parser.add_argument("--fps", type=int, default=50, help="Output motion FPS.")
parser.add_argument(
    "--output",
    type=Path,
    default=Path("sample_motions/agibot_a3_smoke_stand.npz"),
    help="Output .npz path.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from training.robots.agibot_a3 import AGIBOT_A3_CFG


@configclass
class SmokeMotionSceneCfg(InteractiveSceneCfg):
    """Minimal scene with a ground plane and one A3 articulation."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )
    robot: ArticulationCfg = AGIBOT_A3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _append_robot_state(log: dict[str, list], robot) -> None:
    log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
    log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
    log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
    log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
    log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
    log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())


def main() -> None:
    output = args_cli.output.expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output
    output.parent.mkdir(parents=True, exist_ok=True)

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / float(args_cli.fps)
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(SmokeMotionSceneCfg(num_envs=1, env_spacing=2.0))

    sim.reset()
    scene.reset()
    robot = scene["robot"]

    root_state = robot.data.default_root_state.clone()
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()

    log = {
        "fps": np.asarray([args_cli.fps], dtype=np.int64),
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }

    print(f"[create_smoke_motion] recording {args_cli.frames} frames at {args_cli.fps} fps", flush=True)
    with torch.inference_mode():
        for _ in range(int(args_cli.frames)):
            robot.write_root_state_to_sim(root_state)
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            sim.render()
            scene.update(sim.get_physics_dt())
            _append_robot_state(log, robot)

    for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        log[key] = np.stack(log[key], axis=0)

    np.savez(output, **log)
    print(f"[create_smoke_motion] wrote {output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    else:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
