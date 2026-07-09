"""Replay local or registry-hosted motion NPZ files in Isaac Lab."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument("--motion_file", type=str, default=None, help="Local motion NPZ path.")
parser.add_argument("--registry_name", type=str, default=None, help="Optional WandB registry motion name.")
parser.add_argument("--steps", type=int, default=300, help="Maximum replay steps before exit.")
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


def _resolve_motion_file() -> str:
    if args_cli.motion_file is not None:
        return str(Path(args_cli.motion_file).expanduser())

    registry_name = str(args_cli.registry_name)
    if ":" not in registry_name:
        registry_name += ":latest"
    import pathlib

    import wandb

    api = wandb.Api()
    artifact = api.artifact(registry_name)
    return str(pathlib.Path(artifact.download()) / "motion.npz")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()
    motion_file = _resolve_motion_file()
    motion = MotionLoader(
        motion_file,
        torch.tensor([0], dtype=torch.long, device=sim.device),
        sim.device,
    )
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)
    max_steps = int(max(args_cli.steps, 1))
    print(f"[replay_npz] motion={motion_file} steps={max_steps}", flush=True)

    for _ in range(max_steps):
        if not simulation_app.is_running():
            break
        time_steps += 1
        reset_ids = time_steps >= motion.time_step_total
        time_steps[reset_ids] = 0

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins
        root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
        root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(motion.joint_pos[time_steps], motion.joint_vel[time_steps])
        scene.write_data_to_sim()
        sim.render()
        scene.update(sim_dt)

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

    print("[replay_npz] replay completed", flush=True)


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
