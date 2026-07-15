"""Diagnose native-strike torso dynamics under physics tracking.

This script compares the manifest reference, the joint command target, and the
actual articulated robot state over the whole strike cycle. It is intended to
catch issues that kinematic NPZ replay hides, such as PD lag, torso wobble, and
loose waist/arm tracking.
"""

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
    motion_cmd.robot.write_joint_state_to_sim(
        motion_cmd.joint_pos[:n],
        motion_cmd.joint_vel[:n],
        env_ids=env_ids,
    )
    motion_cmd.robot.write_root_state_to_sim(
        torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1),
        env_ids=env_ids,
    )


def _angle_p95(values):
    import torch

    if values.numel() == 0:
        return float("nan")
    return float(torch.quantile(values.detach().flatten().float(), 0.95).cpu().item())


def _run(cfg, simulation_app):
    import gymnasium as gym
    import torch
    from isaaclab.utils.math import euler_xyz_from_quat, matrix_from_quat, quat_error_magnitude, wrap_to_pi
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg
    from rsl_rl.runners import OnPolicyRunner

    import training.tasks  # noqa: F401
    from training.utils.ppo_cfg import runner_kwargs

    task_id = str(cfg.task.gym_task)
    num_envs = int(cfg.num_envs) if cfg.num_envs is not None else 1
    env_cfg = parse_env_cfg(task_id, device=str(cfg.device), num_envs=num_envs)
    _apply_task_overrides(env_cfg, cfg.task)
    env_cfg.sim.device = str(cfg.device)
    env_cfg.seed = int(cfg.get("seed", 0) or 0)
    if bool(cfg.get("disable_pd_randomization", False)):
        env_cfg.events.randomize_pd_gains = None

    gain_overrides = {
        "waist_pitch_joint": (
            cfg.get("waist_pitch_stiffness", None),
            cfg.get("waist_pitch_damping", None),
        ),
        "waist_roll_joint": (
            cfg.get("waist_roll_stiffness", None),
            cfg.get("waist_roll_damping", None),
        ),
        "waist_yaw_joint": (
            cfg.get("waist_yaw_stiffness", None),
            cfg.get("waist_yaw_damping", None),
        ),
    }
    waist_actuator = env_cfg.scene.robot.actuators.get("waist")
    if waist_actuator is not None:
        for joint_name, (stiffness, damping) in gain_overrides.items():
            if stiffness is not None:
                if not isinstance(waist_actuator.stiffness, dict):
                    waist_actuator.stiffness = {joint_name: float(waist_actuator.stiffness)}
                waist_actuator.stiffness[joint_name] = float(stiffness)
                print(f"[INFO] override waist stiffness {joint_name}={float(stiffness)}", flush=True)
            if damping is not None:
                if not isinstance(waist_actuator.damping, dict):
                    waist_actuator.damping = {joint_name: float(waist_actuator.damping)}
                waist_actuator.damping[joint_name] = float(damping)
                print(f"[INFO] override waist damping {joint_name}={float(damping)}", flush=True)

    arm_stiffness_scale = cfg.get("arm_stiffness_scale", None)
    arm_damping_scale = cfg.get("arm_damping_scale", None)
    arm_actuator = env_cfg.scene.robot.actuators.get("arms")
    if arm_actuator is not None and (arm_stiffness_scale is not None or arm_damping_scale is not None):
        if arm_stiffness_scale is not None:
            scale = float(arm_stiffness_scale)
            if isinstance(arm_actuator.stiffness, dict):
                arm_actuator.stiffness = {k: float(v) * scale for k, v in arm_actuator.stiffness.items()}
            else:
                arm_actuator.stiffness = float(arm_actuator.stiffness) * scale
            print(f"[INFO] scale arm stiffness by {scale}", flush=True)
        if arm_damping_scale is not None:
            scale = float(arm_damping_scale)
            if isinstance(arm_actuator.damping, dict):
                arm_actuator.damping = {k: float(v) * scale for k, v in arm_actuator.damping.items()}
            else:
                arm_actuator.damping = float(arm_actuator.damping) * scale
            print(f"[INFO] scale arm damping by {scale}", flush=True)

    motion_manifest = cfg.motion_manifest if cfg.motion_manifest is not None else cfg.task.get("motion_manifest")
    if motion_manifest is None:
        raise ValueError("diagnose_native_torso_tracking.py requires motion_manifest=...")
    manifest_path = pathlib.Path(str(motion_manifest)).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = pathlib.Path.cwd() / manifest_path
    env_cfg.commands.motion.motion_manifest = str(manifest_path)
    env_cfg.commands.motion.motion_file = None
    subset_size = cfg.get("manifest_subset_size", None)
    if subset_size is not None:
        env_cfg.commands.motion.manifest_subset_size = int(subset_size) or None

    agent_cfg = RslRlOnPolicyRunnerCfg(
        **runner_kwargs(OmegaConf.to_container(cfg.algo, resolve=True), str(cfg.task.experiment_name))
    )
    agent_cfg.device = str(cfg.device)

    checkpoint = cfg.get("checkpoint", None)
    if not checkpoint:
        raise ValueError("diagnose_native_torso_tracking.py requires checkpoint=...")
    checkpoint_path = pathlib.Path(str(checkpoint)).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = pathlib.Path.cwd() / checkpoint_path

    out_dir = pathlib.Path(str(cfg.get("out_dir", "eval_outputs/native_torso_tracking"))).expanduser()
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
    body_name_to_id = {name: i for i, name in enumerate(robot.body_names)}
    torso_body_id = body_name_to_id["torso_Link"]
    native_joint_ids = getattr(action_term, "_joint_index_tensor")
    native_joint_names = [robot.data.joint_names[int(i)] for i in native_joint_ids.detach().cpu().tolist()]
    joint_local = {name: i for i, name in enumerate(native_joint_names)}
    focus_names = [
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
    focus_names = [name for name in focus_names if name in joint_local]
    action_dim = int(env.unwrapped.action_manager.total_action_dim)
    n_motions = int(motion_cmd.motion.num_motions)
    n = min(num_envs, n_motions)
    max_steps = int(cfg.get("max_steps") or int(motion_cmd.motion.motion_lengths[:n].max().item()))
    rollout_mode = str(cfg.get("rollout_mode", "both"))

    time_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "manifest": str(manifest_path),
        "checkpoint": str(checkpoint_path),
        "rollout_mode": rollout_mode,
        "max_steps": max_steps,
        "native_joint_names": native_joint_names,
        "focus_joint_names": focus_names,
        "rollouts": {},
    }

    def append_step(mode: str, step_index: int):
        motion_ids = motion_cmd.motion_ids[:n]
        time_steps = motion_cmd.time_steps[:n]
        body_quat = robot.data.body_quat_w[:n, torso_body_id]
        body_ang_vel = robot.data.body_ang_vel_w[:n, torso_body_id]
        if motion_cmd._use_motion_library:
            ref_quat = motion_cmd.motion._body_quat_w[motion_ids, time_steps, torso_body_id]
            ref_ang_vel = motion_cmd.motion._body_ang_vel_w[motion_ids, time_steps, torso_body_id]
        else:
            ref_quat = motion_cmd.motion._body_quat_w[time_steps, torso_body_id]
            ref_ang_vel = motion_cmd.motion._body_ang_vel_w[time_steps, torso_body_id]
        torso_roll, torso_pitch, torso_yaw = euler_xyz_from_quat(body_quat)
        ref_roll, ref_pitch, ref_yaw = euler_xyz_from_quat(ref_quat)
        torso_rot = matrix_from_quat(body_quat)
        ref_rot = matrix_from_quat(ref_quat)
        world_up = torch.zeros((n, 3), dtype=torch.float32, device=device)
        world_up[:, 2] = 1.0
        torso_tilt = torch.acos(torch.sum(torso_rot[:, :, 2] * world_up, dim=-1).clamp(-1.0, 1.0))
        ref_tilt = torch.acos(torch.sum(ref_rot[:, :, 2] * world_up, dim=-1).clamp(-1.0, 1.0))
        torso_ref_err = quat_error_magnitude(ref_quat, body_quat)
        ref = motion_cmd.joint_pos[:n, native_joint_ids]
        actual = robot.data.joint_pos[:n, native_joint_ids]
        target = action_term._processed_actions[:n]
        raw = action_term._raw_actions[:n]
        actual_minus_ref = actual - ref
        target_minus_ref = target - ref
        actual_minus_target = actual - target

        for env_i in range(n):
            row: dict[str, object] = {
                "mode": mode,
                "env_i": env_i,
                "episode_id": str(motion_cmd.motion.episode_ids[int(motion_ids[env_i])]),
                "stroke_type": str(motion_cmd.motion.stroke_types[int(motion_ids[env_i])]),
                "step_index": step_index,
                "motion_step": int(time_steps[env_i].item()),
                "torso_ref_err_deg": float(torch.rad2deg(torso_ref_err[env_i]).item()),
                "torso_roll_deg": float(torch.rad2deg(wrap_to_pi(torso_roll[env_i])).item()),
                "torso_pitch_deg": float(torch.rad2deg(wrap_to_pi(torso_pitch[env_i])).item()),
                "torso_yaw_deg": float(torch.rad2deg(wrap_to_pi(torso_yaw[env_i])).item()),
                "ref_torso_roll_deg": float(torch.rad2deg(wrap_to_pi(ref_roll[env_i])).item()),
                "ref_torso_pitch_deg": float(torch.rad2deg(wrap_to_pi(ref_pitch[env_i])).item()),
                "ref_torso_yaw_deg": float(torch.rad2deg(wrap_to_pi(ref_yaw[env_i])).item()),
                "torso_tilt_deg": float(torch.rad2deg(torso_tilt[env_i]).item()),
                "ref_torso_tilt_deg": float(torch.rad2deg(ref_tilt[env_i]).item()),
                "torso_tilt_delta_deg": float(torch.rad2deg(torch.abs(torso_tilt[env_i] - ref_tilt[env_i])).item()),
                "torso_ang_vel_norm": float(torch.linalg.norm(body_ang_vel[env_i]).item()),
                "ref_torso_ang_vel_norm": float(torch.linalg.norm(ref_ang_vel[env_i]).item()),
                "torso_ang_vel_err_norm": float(torch.linalg.norm(body_ang_vel[env_i] - ref_ang_vel[env_i]).item()),
                "joint_actual_minus_ref_max_abs": float(torch.abs(actual_minus_ref[env_i]).max().item()),
                "joint_actual_minus_target_max_abs": float(torch.abs(actual_minus_target[env_i]).max().item()),
                "target_minus_ref_max_abs": float(torch.abs(target_minus_ref[env_i]).max().item()),
                "raw_action_max_abs": float(torch.abs(raw[env_i]).max().item()),
            }
            for name in focus_names:
                i = joint_local[name]
                prefix = name.replace("_joint", "")
                row[f"{prefix}_ref"] = float(ref[env_i, i].item())
                row[f"{prefix}_actual"] = float(actual[env_i, i].item())
                row[f"{prefix}_actual_minus_ref"] = float(actual_minus_ref[env_i, i].item())
                row[f"{prefix}_actual_minus_target"] = float(actual_minus_target[env_i, i].item())
                row[f"{prefix}_target_minus_ref"] = float(target_minus_ref[env_i, i].item())
            time_rows.append(row)

    def write_outputs():
        csv_path = out_dir / "torso_tracking_timeseries.csv"
        if time_rows:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(time_rows[0].keys()))
                writer.writeheader()
                writer.writerows(time_rows)
            print(f"[INFO] wrote {csv_path}", flush=True)
        json_path = out_dir / "summary.json"
        json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[INFO] wrote {json_path}", flush=True)

    def rollout(mode: str):
        print(f"[INFO] starting rollout: {mode}", flush=True)
        env.reset()
        _sync_motion_state(env.unwrapped, motion_cmd, n, device)
        try:
            env.unwrapped.command_manager.get_term("racket_target")._resample_command(torch.arange(n, device=device))
        except Exception:
            pass
        obs = _obs_to_device(env.get_observations(), agent_cfg.device)
        start_idx = len(time_rows)
        for step_idx in range(max_steps):
            if step_idx % 20 == 0:
                print(f"[INFO] {mode} step {step_idx}/{max_steps}", flush=True)
            if mode == "zero":
                actions = torch.zeros((n, action_dim), dtype=torch.float32, device=device)
            elif mode == "policy":
                with torch.inference_mode():
                    actions = policy(obs)
            else:
                raise ValueError(mode)
            with torch.inference_mode():
                obs, _, _, _ = env.step(actions.to(device))
                obs = _obs_to_device(obs, agent_cfg.device)
            append_step(mode, step_idx)
        mode_rows = time_rows[start_idx:]
        summary["rollouts"][mode] = _summarize_rows(mode_rows, focus_names)
        print(f"[INFO] finished rollout: {mode}", flush=True)
        write_outputs()

    def _summarize_rows(rows: list[dict[str, object]], focus: list[str]) -> dict[str, object]:
        out: dict[str, object] = {"overall": {}, "motions": {}}
        fields = [
            "torso_ref_err_deg",
            "torso_pitch_deg",
            "torso_roll_deg",
            "torso_tilt_deg",
            "torso_tilt_delta_deg",
            "torso_ang_vel_norm",
            "torso_ang_vel_err_norm",
            "joint_actual_minus_ref_max_abs",
            "joint_actual_minus_target_max_abs",
            "target_minus_ref_max_abs",
        ]

        def stats(group: list[dict[str, object]]) -> dict[str, float]:
            tensors = {}
            for field in fields:
                vals = torch.tensor([float(r[field]) for r in group], dtype=torch.float32)
                tensors[field] = {
                    "mean": float(vals.mean().item()),
                    "p95": _angle_p95(vals),
                    "max_abs": float(vals.abs().max().item()),
                }
            for name in focus:
                prefix = name.replace("_joint", "")
                field = f"{prefix}_actual_minus_target"
                vals = torch.tensor([float(r[field]) for r in group], dtype=torch.float32)
                tensors[f"{prefix}_tracking_err_rad"] = {
                    "mean_abs": float(vals.abs().mean().item()),
                    "p95_abs": _angle_p95(vals.abs()),
                    "max_abs": float(vals.abs().max().item()),
                }
            return tensors

        out["overall"] = stats(rows)
        by_motion: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            by_motion.setdefault(str(row["episode_id"]), []).append(row)
        for episode_id, group in by_motion.items():
            out["motions"][episode_id] = {
                "stroke_type": str(group[0]["stroke_type"]),
                **stats(group),
            }
        return out

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
