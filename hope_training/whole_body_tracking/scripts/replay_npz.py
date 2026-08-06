"""Replay local or registry-hosted motion NPZ files in Isaac Lab."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument(
    "--motion_file",
    action="append",
    default=None,
    help="Local motion NPZ path. Repeat to replay multiple motions in sequence.",
)
parser.add_argument("--registry_name", type=str, default=None, help="Optional WandB registry motion name.")
parser.add_argument("--steps", type=int, default=300, help="Maximum replay steps before exit.")
parser.add_argument("--hold_steps", type=int, default=30, help="Frames to hold before and after each motion.")
parser.add_argument("--realtime", action="store_true", help="Throttle playback to the simulation timestep.")
parser.add_argument(
    "--view",
    choices=["default", "front"],
    default="default",
    help="Camera view; front looks along +X toward the A3 front side.",
)
parser.add_argument("--keep_open", action="store_true", help="Keep the Isaac window open after replay.")
parser.add_argument(
    "--robot",
    type=str,
    default="g1",
    choices=["g1", "agibot_a3"],
    help="Which robot model to replay the motion on.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if bool(args_cli.motion_file) == bool(args_cli.registry_name):
    parser.error("exactly one of --motion_file or --registry_name is required")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from training.robots.agibot_a3 import AGIBOT_A3_CFG
from training.robots.g1 import G1_CYLINDER_CFG
from training.tasks.tracking.mdp import MotionLoader

_ROBOT_CFG = {"g1": G1_CYLINDER_CFG, "agibot_a3": AGIBOT_A3_CFG}[args_cli.robot]


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Minimal local-only replay scene."""

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )

    robot: ArticulationCfg = _ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _resolve_motion_files() -> list[str]:
    if args_cli.motion_file is not None:
        return [str(Path(path).expanduser()) for path in args_cli.motion_file]

    registry_name = str(args_cli.registry_name)
    if ":" not in registry_name:
        registry_name += ":latest"
    import pathlib

    import wandb

    api = wandb.Api()
    artifact = api.artifact(registry_name)
    return [str(pathlib.Path(artifact.download()) / "motion.npz")]


def _set_camera(sim: SimulationContext, root_position: np.ndarray) -> None:
    if args_cli.view == "front":
        # A3 uses +X as the forward direction in the table-tennis scene. The
        # camera is placed in front of the robot and aligned with its lateral
        # centerline to compare paddle orientation without a side perspective.
        eye = root_position + np.array([2.8, 0.0, 0.45])
        target = root_position + np.array([0.0, 0.0, 0.12])
    else:
        eye = root_position + np.array([2.0, 2.0, 0.5])
        target = root_position
    sim.set_camera_view(eye, target)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()
    motion_files = _resolve_motion_files()
    max_steps = int(max(args_cli.steps, 1))
    for motion_file in motion_files:
        motion = MotionLoader(
            motion_file,
            torch.tensor([0], dtype=torch.long, device=sim.device),
            sim.device,
        )
        total_steps = min(max_steps, int(motion.time_step_total))
        print(f"[replay_npz] motion={motion_file} steps={total_steps}", flush=True)
        frame_indices = ([0] * max(0, int(args_cli.hold_steps))
                         + list(range(total_steps))
                         + [max(total_steps - 1, 0)] * max(0, int(args_cli.hold_steps)))
        for frame_idx in frame_indices:
            if not simulation_app.is_running():
                break
            time_step = torch.full(
                (scene.num_envs,), frame_idx, dtype=torch.long, device=sim.device
            )
            root_states = robot.data.default_root_state.clone()
            root_states[:, :3] = motion.body_pos_w[time_step][:, 0] + scene.env_origins
            root_states[:, 3:7] = motion.body_quat_w[time_step][:, 0]
            root_states[:, 7:10] = motion.body_lin_vel_w[time_step][:, 0]
            root_states[:, 10:] = motion.body_ang_vel_w[time_step][:, 0]

            robot.write_root_state_to_sim(root_states)
            robot.write_joint_state_to_sim(motion.joint_pos[time_step], motion.joint_vel[time_step])
            scene.write_data_to_sim()
            sim.render()
            scene.update(sim_dt)
            _set_camera(sim, root_states[0, :3].cpu().numpy())
            if args_cli.realtime:
                time.sleep(sim_dt)

        if not simulation_app.is_running():
            break

    print("[replay_npz] replay completed", flush=True)
    if args_cli.keep_open and not args_cli.headless:
        print("[replay_npz] keeping Isaac window open; close the window to exit", flush=True)
        while simulation_app.is_running():
            sim.render()
            time.sleep(0.05)


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.reset()
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
