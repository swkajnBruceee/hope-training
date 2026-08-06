#!/usr/bin/env python3
"""Verify that strike-ready reset perturbations reach physics and policy observations."""

from __future__ import annotations

import json
import pathlib
import sys

import hydra
from omegaconf import OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from train import _apply_task_overrides


def _policy_term_slice(env, name: str) -> slice:
    manager = env.observation_manager
    terms = manager.active_terms["policy"]
    dims = manager.group_obs_term_dim["policy"]
    index = terms.index(name)
    start = sum(int(dim[0]) for dim in dims[:index])
    return slice(start, start + int(dims[index][0]))


@hydra.main(version_base=None, config_path="../cfg", config_name="play")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, device=str(cfg.device))
    app = app_launcher.app
    try:
        import gymnasium as gym
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils import parse_env_cfg

        import training.tasks  # noqa: F401 -- gym registration

        task_id = str(cfg.task.gym_task)
        num_envs = int(cfg.num_envs)
        env_cfg = parse_env_cfg(task_id, device=str(cfg.device), num_envs=num_envs)
        _apply_task_overrides(env_cfg, cfg.task)
        env_cfg.sim.device = str(cfg.device)
        env_cfg.seed = int(cfg.get("seed", 20260722))
        manifest_value = cfg.task.get("motion_manifest", None)
        if manifest_value is not None:
            manifest_path = pathlib.Path(str(manifest_value)).expanduser()
            if not manifest_path.is_absolute():
                manifest_path = pathlib.Path.cwd() / manifest_path
            if not manifest_path.is_file():
                raise FileNotFoundError(manifest_path)
            env_cfg.commands.motion.motion_manifest = str(manifest_path)
            env_cfg.commands.motion.motion_file = None
            subset_size = cfg.task.get("manifest_subset_size", None)
            if subset_size is not None:
                env_cfg.commands.motion.manifest_subset_size = int(subset_size)
            frame_offset = cfg.task.get("manifest_frame_z_offset", None)
            if frame_offset is not None:
                env_cfg.commands.motion.manifest_frame_z_offset = float(frame_offset)

        # Match the proposed first robustness curriculum.  The audit must use
        # a nonzero range so it can prove the reset path, not merely inspect a
        # nominal ready reset.
        velocity_range = {
            "x": (float(cfg.get("vx_min", -0.08)), float(cfg.get("vx_max", 0.08))),
            "y": (float(cfg.get("vy_min", -0.08)), float(cfg.get("vy_max", 0.08))),
            "z": (0.0, 0.0),
            "roll": (float(cfg.get("roll_rate_min", -0.15)), float(cfg.get("roll_rate_max", 0.15))),
            "pitch": (float(cfg.get("pitch_rate_min", -0.15)), float(cfg.get("pitch_rate_max", 0.15))),
            "yaw": (0.0, 0.0),
        }
        env_cfg.commands.motion.velocity_range = velocity_range
        env_cfg.commands.motion.pose_range = {}
        env_cfg.commands.motion.reset_to_default_pose = True
        print("[reset-audit] creating environment", flush=True)
        env = gym.make(task_id, cfg=env_cfg)
        try:
            # Use the same wrapper initialization as PPO/evaluation.  A second
            # raw Gym reset after Isaac manager construction can block on this
            # task, whereas the wrapper owns exactly one reset transaction.
            print("[reset-audit] constructing RSL-RL wrapper", flush=True)
            vec_env = RslRlVecEnvWrapper(env)
            print("[reset-audit] reading initial observation", flush=True)
            obs = vec_env.get_observations()
            if isinstance(obs, tuple):
                obs = obs[0]
            unwrapped = env.unwrapped
            motion = unwrapped.command_manager.get_term("motion")
            robot = unwrapped.scene["robot"]
            policy_slice = _policy_term_slice(unwrapped, "base_lin_vel")
            observed_before = obs[:, policy_slice].clone()
            sampled_world = motion.last_reset_velocity_offset[:, :3].clone()
            hard_case = motion.last_reset_hard_case.clone()
            motion_ids = motion.motion_ids.clone()
            actual_world_before = robot.data.root_lin_vel_w.clone()
            actual_body_before = robot.data.root_lin_vel_b.clone()

            zero_action = torch.zeros(
                (num_envs, unwrapped.action_manager.total_action_dim), device=unwrapped.device
            )
            obs_after, _, _, _ = vec_env.step(zero_action)
            observed_after = obs_after[:, policy_slice].clone()
            actual_world_after = robot.data.root_lin_vel_w.clone()
            actual_body_after = robot.data.root_lin_vel_b.clone()

            write_error = actual_world_before - sampled_world
            observation_error = observed_before - actual_body_before
            report = {
                "task": task_id,
                "num_envs": num_envs,
                "velocity_range": velocity_range,
                "policy_base_lin_vel_terms": list(unwrapped.observation_manager.active_terms["policy"]),
                "policy_base_lin_vel_slice": [policy_slice.start, policy_slice.stop],
                "sampled_world_velocity_mean": sampled_world.mean(dim=0).cpu().tolist(),
                "sampled_world_velocity_abs_max": sampled_world.abs().amax(dim=0).cpu().tolist(),
                "hard_case_fraction": float(hard_case.float().mean().item()),
                "hard_case_motion_ids": motion_ids[hard_case].detach().cpu().tolist(),
                "write_error_abs_max": write_error.abs().amax(dim=0).cpu().tolist(),
                "write_error_rms": float(torch.sqrt(torch.mean(torch.square(write_error))).item()),
                "observation_error_abs_max": observation_error.abs().amax(dim=0).cpu().tolist(),
                "observation_error_rms": float(torch.sqrt(torch.mean(torch.square(observation_error))).item()),
                "first_step_body_velocity_abs_mean": actual_body_after.abs().mean(dim=0).cpu().tolist(),
                "first_step_body_velocity_abs_max": actual_body_after.abs().amax(dim=0).cpu().tolist(),
                "samples": [
                    {
                        "sampled_world_velocity": sampled_world[i].cpu().tolist(),
                        "hard_case": bool(hard_case[i].item()),
                        "motion_id": int(motion_ids[i].item()),
                        "actual_world_after_reset": actual_world_before[i].cpu().tolist(),
                        "actual_body_after_reset": actual_body_before[i].cpu().tolist(),
                        "policy_base_lin_vel_after_reset": observed_before[i].cpu().tolist(),
                        "actual_body_after_first_step": actual_body_after[i].cpu().tolist(),
                        "policy_base_lin_vel_after_first_step": observed_after[i].cpu().tolist(),
                    }
                    for i in range(num_envs)
                ],
            }
            output = pathlib.Path(str(cfg.get("output_json", "eval_outputs/strike_stabilizer_a/reset_perturbation_audit.json")))
            if not output.is_absolute():
                output = pathlib.Path.cwd() / output
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({
                "output": str(output),
                "write_error_rms": report["write_error_rms"],
                "observation_error_rms": report["observation_error_rms"],
                "sampled_world_velocity_abs_max": report["sampled_world_velocity_abs_max"],
            }), flush=True)
        finally:
            env.close()
    finally:
        app.close()


if __name__ == "__main__":
    main()
