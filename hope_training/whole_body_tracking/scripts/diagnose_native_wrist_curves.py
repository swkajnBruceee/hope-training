"""Dump reference/zero/policy joint curves for native-strike wrist diagnosis."""

from __future__ import annotations

import csv
import json
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
del _REPO_ROOT

import hydra
from omegaconf import OmegaConf

from train import _apply_task_overrides


def _obs_to_device(obs, device):
    if isinstance(obs, tuple):
        obs = obs[0]
    return obs.to(device)


def _sync_motion_state(env, motion_cmd, n, device):
    import torch

    ids = torch.arange(n, device=device, dtype=torch.long)
    motion_cmd.motion_ids[:n] = ids
    motion_cmd.time_steps[:n] = 0

    env_ids = torch.arange(n, device=device)
    root_pos = motion_cmd.motion._body_pos_w[motion_cmd.motion_ids[:n], motion_cmd.time_steps[:n], 0]
    root_pos = root_pos + env.scene.env_origins[:n]
    root_ori = motion_cmd.motion._body_quat_w[motion_cmd.motion_ids[:n], motion_cmd.time_steps[:n], 0]
    root_lin_vel = motion_cmd.motion._body_lin_vel_w[motion_cmd.motion_ids[:n], motion_cmd.time_steps[:n], 0]
    root_ang_vel = motion_cmd.motion._body_ang_vel_w[motion_cmd.motion_ids[:n], motion_cmd.time_steps[:n], 0]
    motion_cmd.robot.write_joint_state_to_sim(motion_cmd.joint_pos[:n], motion_cmd.joint_vel[:n], env_ids=env_ids)
    motion_cmd.robot.write_root_state_to_sim(
        torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1),
        env_ids=env_ids,
    )


def _run(cfg, simulation_app):
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg

    import training.tasks  # noqa: F401
    from training.utils.ppo_cfg import runner_kwargs

    task_id = str(cfg.task.gym_task)
    num_envs = int(cfg.num_envs) if cfg.num_envs is not None else 1
    if num_envs != 1:
        print(f"[WARN] forcing num_envs=1 for curve dump (got {num_envs})", flush=True)
        num_envs = 1

    env_cfg = parse_env_cfg(task_id, device=str(cfg.device), num_envs=num_envs)
    _apply_task_overrides(env_cfg, cfg.task)
    env_cfg.sim.device = str(cfg.device)
    env_cfg.seed = int(cfg.get("seed", 0) or 0)

    motion_manifest = cfg.motion_manifest if cfg.motion_manifest is not None else cfg.task.get("motion_manifest")
    if motion_manifest is None:
        raise ValueError("diagnose_native_wrist_curves.py requires motion_manifest=...")
    manifest_path = pathlib.Path(str(motion_manifest)).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = pathlib.Path.cwd() / manifest_path
    env_cfg.commands.motion.motion_manifest = str(manifest_path)
    env_cfg.commands.motion.motion_file = None
    subset_size = cfg.get("manifest_subset_size", None)
    if subset_size is not None:
        env_cfg.commands.motion.manifest_subset_size = int(subset_size) or None
    frame_z_offset = cfg.get("manifest_frame_z_offset", None)
    if frame_z_offset is None:
        frame_z_offset = cfg.task.get("manifest_frame_z_offset")
    if frame_z_offset is not None:
        env_cfg.commands.motion.manifest_frame_z_offset = float(frame_z_offset)

    agent_cfg = RslRlOnPolicyRunnerCfg(
        **runner_kwargs(OmegaConf.to_container(cfg.algo, resolve=True), str(cfg.task.experiment_name))
    )
    agent_cfg.device = str(cfg.device)

    checkpoint = cfg.get("checkpoint", None)
    if not checkpoint:
        raise ValueError("diagnose_native_wrist_curves.py requires checkpoint=...")
    checkpoint_path = pathlib.Path(str(checkpoint)).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = pathlib.Path.cwd() / checkpoint_path

    out_dir = pathlib.Path(str(cfg.get("out_dir", "eval_outputs/native_wrist_curves"))).expanduser()
    if not out_dir.is_absolute():
        out_dir = pathlib.Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(task_id, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)
    device = env.unwrapped.device
    print("[INFO] environment created", flush=True)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[INFO] loading checkpoint: {checkpoint_path}", flush=True)
    runner.load(str(checkpoint_path), load_optimizer=False)
    policy = runner.get_inference_policy(device=device)
    print("[INFO] policy loaded", flush=True)

    motion_cmd = env.unwrapped.command_manager.get_term("motion")
    robot = motion_cmd.robot
    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    native_joint_ids = getattr(action_term, "_joint_index_tensor")
    native_joint_names = [robot.data.joint_names[int(i)] for i in native_joint_ids.detach().cpu().tolist()]
    action_dim = int(env.unwrapped.action_manager.total_action_dim)

    focus = [
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    focus_local = [native_joint_names.index(name) for name in focus if name in native_joint_names]
    focus_names = [native_joint_names[i] for i in focus_local]

    max_steps = int(cfg.get("max_steps") or 80)
    rows: list[dict[str, float | int | str]] = []
    summary: dict[str, object] = {
        "manifest": str(manifest_path),
        "checkpoint": str(checkpoint_path),
        "max_steps": max_steps,
        "native_joint_names": native_joint_names,
        "focus_joint_names": focus_names,
        "rollouts": {},
    }

    def write_outputs():
        if not rows:
            return
        csv_path = out_dir / "joint_curves.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        json_path = out_dir / "summary.json"
        json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[INFO] wrote {csv_path}", flush=True)
        print(f"[INFO] wrote {json_path}", flush=True)

    def rollout(mode: str):
        print(f"[INFO] starting rollout: {mode}", flush=True)
        env.reset()
        _sync_motion_state(env.unwrapped, motion_cmd, 1, device)
        try:
            env.unwrapped.command_manager.get_term("racket_target")._resample_command(torch.arange(1, device=device))
        except Exception:
            pass
        obs = _obs_to_device(env.get_observations(), agent_cfg.device)
        episode_id = str(motion_cmd.motion.episode_ids[int(motion_cmd.motion_ids[0])])
        stroke = str(motion_cmd.motion.stroke_types[int(motion_cmd.motion_ids[0])])
        hit_step = int(round(float(env_cfg.commands.racket_target.strike_phase) * (int(motion_cmd.motion.time_step_total) - 1)))
        raw_abs_max = 0.0
        target_delta_abs_max = 0.0
        actual_delta_abs_max = 0.0

        for iter_idx in range(max_steps):
            if iter_idx % 5 == 0:
                print(f"[INFO] {mode} step {iter_idx}/{max_steps}", flush=True)
            if mode == "zero":
                actions = torch.zeros((1, action_dim), dtype=torch.float32, device=device)
            elif mode == "policy":
                with torch.inference_mode():
                    actions = policy(obs)
            else:
                raise ValueError(mode)
            with torch.inference_mode():
                obs, _, _, _ = env.step(actions.to(device))
                obs = _obs_to_device(obs, agent_cfg.device)

            step = int(motion_cmd.time_steps[0].item())
            ref = motion_cmd.joint_pos[0, native_joint_ids].detach()
            actual = robot.data.joint_pos[0, native_joint_ids].detach()
            target = action_term._processed_actions[0].detach()
            raw = action_term._raw_actions[0].detach()
            target_delta = target - ref
            actual_delta = actual - ref
            raw_abs_max = max(raw_abs_max, float(raw.abs().max().item()))
            target_delta_abs_max = max(target_delta_abs_max, float(target_delta.abs().max().item()))
            actual_delta_abs_max = max(actual_delta_abs_max, float(actual_delta.abs().max().item()))

            for local in focus_local:
                name = native_joint_names[local]
                rows.append(
                    {
                        "mode": mode,
                        "episode_id": episode_id,
                        "stroke_type": stroke,
                        "step": step,
                        "is_hit_step": int(step == hit_step),
                        "joint_name": name,
                        "q_ref_rad": float(ref[local].item()),
                        "q_actual_rad": float(actual[local].item()),
                        "q_target_rad": float(target[local].item()),
                        "raw_action": float(raw[local].item()),
                        "target_minus_ref_rad": float(target_delta[local].item()),
                        "actual_minus_ref_rad": float(actual_delta[local].item()),
                    }
                )

        summary["rollouts"][mode] = {
            "episode_id": episode_id,
            "stroke_type": stroke,
            "hit_step": hit_step,
            "raw_action_abs_max": raw_abs_max,
            "target_minus_ref_abs_max_rad": target_delta_abs_max,
            "actual_minus_ref_abs_max_rad": actual_delta_abs_max,
        }
        print(f"[INFO] finished rollout: {mode}", flush=True)
        write_outputs()

    rollout_mode = str(cfg.get("rollout_mode", "both"))
    if rollout_mode in ("zero", "both"):
        rollout("zero")
    if rollout_mode in ("policy", "both"):
        rollout("policy")
    write_outputs()
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
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
