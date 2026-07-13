"""Check whether a saved motion NPZ matches current A3 FK body order.

This is a no-physics diagnostic:

1. Load one motion frame.
2. Write root state and joint state directly into the current articulation.
3. Render/update once without stepping physics.
4. Compare robot.data.body_pos_w/body_quat_w with the NPZ body arrays.

If this fails badly, policy/reward evaluation is comparing the robot against a
reference that no longer matches the current asset/body order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="No-physics FK consistency check for motion NPZ files.")
parser.add_argument("--motion_file", type=str, default=None, help="Motion NPZ path.")
parser.add_argument("--manifest", type=str, default=None, help="Manifest path; uses motion_index entry.")
parser.add_argument("--motion_index", type=int, default=0, help="Manifest motion index.")
parser.add_argument("--frame", type=str, default="hit", help="'hit' or an integer frame index.")
parser.add_argument("--frame_z_offset", type=float, default=0.0, help="Z offset applied by manifest loader.")
parser.add_argument("--top_k", type=int, default=12, help="Print worst K bodies by pose error.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if bool(args_cli.motion_file) == bool(args_cli.manifest):
    parser.error("exactly one of --motion_file or --manifest is required")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_error_magnitude

from training.robots.agibot_a3 import AGIBOT_A3_CFG


@configclass
class FKCheckSceneCfg(InteractiveSceneCfg):
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )
    robot: ArticulationCfg = AGIBOT_A3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _resolve_motion_and_frame() -> tuple[Path, int, str]:
    if args_cli.motion_file:
        motion_file = Path(args_cli.motion_file).expanduser()
        frame = int(args_cli.frame) if str(args_cli.frame) != "hit" else 0
        return motion_file, frame, motion_file.stem

    manifest = Path(args_cli.manifest).expanduser()
    if not manifest.is_absolute():
        manifest = Path.cwd() / manifest
    with manifest.open("r", encoding="utf-8") as f:
        data = json.load(f)
    entry = data["motions"][int(args_cli.motion_index)]
    motion_file = Path(entry.get("library_motion_npz") or entry["motion_npz"]).expanduser()
    if str(args_cli.frame) == "hit":
        frame = int(entry.get("hit_event", {}).get("motion_hit_frame", 0))
    else:
        frame = int(args_cli.frame)
    return motion_file, frame, str(entry.get("episode_id", motion_file.stem))


def main() -> None:
    motion_file, frame, label = _resolve_motion_and_frame()
    motion = np.load(motion_file)
    joint_pos_np = np.asarray(motion["joint_pos"], dtype=np.float32)
    joint_vel_np = np.asarray(motion["joint_vel"], dtype=np.float32)
    body_pos_np = np.asarray(motion["body_pos_w"], dtype=np.float32)
    body_quat_np = np.asarray(motion["body_quat_w"], dtype=np.float32)
    frame = max(0, min(frame, joint_pos_np.shape[0] - 1))

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(FKCheckSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    scene.reset()
    robot: Articulation = scene["robot"]

    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] = torch.as_tensor(body_pos_np[frame, 0], device=sim.device).unsqueeze(0)
    root_state[:, 2] += float(args_cli.frame_z_offset)
    root_state[:, 3:7] = torch.as_tensor(body_quat_np[frame, 0], device=sim.device).unsqueeze(0)
    root_state[:, 7:] = 0.0
    joint_pos = torch.as_tensor(joint_pos_np[frame], dtype=torch.float32, device=sim.device).unsqueeze(0)
    joint_vel = torch.as_tensor(joint_vel_np[frame], dtype=torch.float32, device=sim.device).unsqueeze(0)

    robot.write_root_state_to_sim(root_state)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    scene.write_data_to_sim()
    sim.render()
    scene.update(sim.get_physics_dt())

    actual_pos = robot.data.body_pos_w[0].detach().cpu()
    actual_quat = robot.data.body_quat_w[0].detach().cpu()
    ref_pos = torch.as_tensor(body_pos_np[frame], dtype=torch.float32)
    ref_pos[:, 2] += float(args_cli.frame_z_offset)
    ref_quat = torch.as_tensor(body_quat_np[frame], dtype=torch.float32)

    pos_err = torch.linalg.norm(actual_pos - ref_pos, dim=-1)
    rot_err = torch.rad2deg(quat_error_magnitude(ref_quat, actual_quat))
    score = pos_err + 0.01 * rot_err
    order = torch.argsort(score, descending=True)

    print(f"[fk_check] label={label}")
    print(f"[fk_check] motion={motion_file}")
    print(f"[fk_check] frame={frame} frame_z_offset={float(args_cli.frame_z_offset):.4f}")
    print(f"[fk_check] num_bodies robot={len(robot.body_names)} npz={body_pos_np.shape[1]}")
    print(
        "[fk_check] max_pos_err_m={:.4f} mean_pos_err_m={:.4f} "
        "max_rot_err_deg={:.2f} mean_rot_err_deg={:.2f}".format(
            float(pos_err.max()), float(pos_err.mean()), float(rot_err.max()), float(rot_err.mean())
        )
    )
    print("rank,body_index,body_name,pos_err_m,rot_err_deg,actual_z,ref_z")
    for rank, idx_t in enumerate(order[: int(args_cli.top_k)], start=1):
        idx = int(idx_t)
        print(
            f"{rank},{idx},{robot.body_names[idx]},{float(pos_err[idx]):.5f},"
            f"{float(rot_err[idx]):.2f},{float(actual_pos[idx,2]):.4f},{float(ref_pos[idx,2]):.4f}"
        )

    simulation_app.close()


if __name__ == "__main__":
    main()
