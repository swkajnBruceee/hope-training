"""Check manifest reset FK consistency inside the actual task environment.

This uses the same Gym/IsaacLab task path as training/eval, pins one env to one
manifest motion, writes root + joint states, and compares current FK body poses
against the manifest NPZ before any policy action or physics rollout.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
del _REPO_ROOT

import hydra
from omegaconf import OmegaConf

from train import _apply_task_overrides


def _run(cfg, simulation_app):
    import pathlib

    import gymnasium as gym
    import torch
    from isaaclab.utils.math import quat_error_magnitude
    from isaaclab_tasks.utils import parse_env_cfg

    import training.tasks  # noqa: F401

    task_id = str(cfg.task.gym_task)
    env_cfg = parse_env_cfg(task_id, device=str(cfg.device), num_envs=1)
    _apply_task_overrides(env_cfg, cfg.task)
    env_cfg.sim.device = str(cfg.device)

    manifest_path = pathlib.Path(str(cfg.motion_manifest)).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = pathlib.Path.cwd() / manifest_path
    env_cfg.commands.motion.motion_manifest = str(manifest_path)
    env_cfg.commands.motion.motion_file = None
    env_cfg.commands.motion.manifest_subset_size = int(cfg.get("manifest_subset_size", 0) or 0) or None
    frame_z_offset = cfg.get("manifest_frame_z_offset", None)
    if frame_z_offset is None:
        frame_z_offset = cfg.task.get("manifest_frame_z_offset")
    if frame_z_offset is not None:
        env_cfg.commands.motion.manifest_frame_z_offset = float(frame_z_offset)

    env = gym.make(task_id, cfg=env_cfg, render_mode=None)
    device = env.unwrapped.device
    motion_cmd = env.unwrapped.command_manager.get_term("motion")
    robot = motion_cmd.robot

    motion_index = int(cfg.get("motion_index", 0))
    motion_index = max(0, min(motion_index, int(motion_cmd.motion.num_motions) - 1))
    if str(cfg.get("frame", "hit")) == "hit":
        frame = int(motion_cmd.motion.hit_frames[motion_index].detach().cpu())
    else:
        frame = int(cfg.get("frame"))
    frame = max(0, min(frame, int(motion_cmd.motion.motion_lengths[motion_index].detach().cpu()) - 1))

    env.reset()
    motion_cmd.motion_ids[0] = motion_index
    motion_cmd.time_steps[0] = frame

    root_pos = motion_cmd.motion._body_pos_w[motion_cmd.motion_ids[:1], motion_cmd.time_steps[:1], 0]
    root_pos = root_pos + env.unwrapped.scene.env_origins[:1]
    root_ori = motion_cmd.motion._body_quat_w[motion_cmd.motion_ids[:1], motion_cmd.time_steps[:1], 0]
    root_lin_vel = torch.zeros_like(motion_cmd.motion._body_lin_vel_w[motion_cmd.motion_ids[:1], motion_cmd.time_steps[:1], 0])
    root_ang_vel = torch.zeros_like(motion_cmd.motion._body_ang_vel_w[motion_cmd.motion_ids[:1], motion_cmd.time_steps[:1], 0])
    robot.write_joint_state_to_sim(motion_cmd.joint_pos[:1], torch.zeros_like(motion_cmd.joint_vel[:1]), env_ids=torch.tensor([0], device=device))
    robot.write_root_state_to_sim(torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1), env_ids=torch.tensor([0], device=device))
    env.unwrapped.scene.write_data_to_sim()
    env.unwrapped.sim.render()
    env.unwrapped.scene.update(env.unwrapped.physics_dt)

    actual_pos = robot.data.body_pos_w[0]
    actual_quat = robot.data.body_quat_w[0]
    ref_pos = motion_cmd.motion._body_pos_w[motion_index, frame] + env.unwrapped.scene.env_origins[:1]
    ref_quat = motion_cmd.motion._body_quat_w[motion_index, frame]

    pos_err = torch.linalg.norm(actual_pos - ref_pos, dim=-1)
    rot_err = torch.rad2deg(quat_error_magnitude(ref_quat, actual_quat))
    score = pos_err + 0.01 * rot_err
    order = torch.argsort(score, descending=True)
    top_k = int(cfg.get("top_k", 16))

    episode_id = motion_cmd.motion.episode_ids[motion_index]
    print(f"[reset_fk] manifest={manifest_path}", flush=True)
    print(f"[reset_fk] motion_index={motion_index} episode_id={episode_id} frame={frame}", flush=True)
    print(f"[reset_fk] robot_bodies={len(robot.body_names)} ref_bodies={ref_pos.shape[0]}", flush=True)
    print(
        "[reset_fk] max_pos_err_m={:.5f} mean_pos_err_m={:.5f} max_rot_err_deg={:.2f} mean_rot_err_deg={:.2f}".format(
            float(pos_err.max().detach().cpu()),
            float(pos_err.mean().detach().cpu()),
            float(rot_err.max().detach().cpu()),
            float(rot_err.mean().detach().cpu()),
        ),
        flush=True,
    )
    print("rank,body_index,body_name,pos_err_m,rot_err_deg,actual_z,ref_z", flush=True)
    for rank, idx_t in enumerate(order[:top_k], start=1):
        idx = int(idx_t.detach().cpu())
        print(
            f"{rank},{idx},{robot.body_names[idx]},{float(pos_err[idx].detach().cpu()):.5f},"
            f"{float(rot_err[idx].detach().cpu()):.2f},"
            f"{float(actual_pos[idx, 2].detach().cpu()):.4f},{float(ref_pos[idx, 2].detach().cpu()):.4f}",
            flush=True,
        )
    env.close()


@hydra.main(version_base=None, config_path="../cfg", config_name="play")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=bool(cfg.headless), device=str(cfg.device), enable_cameras=False)
    simulation_app = app_launcher.app
    try:
        _run(cfg, simulation_app)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
