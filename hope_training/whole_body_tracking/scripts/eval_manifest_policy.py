"""Finite-step per-motion evaluation for manifest-conditioned policies.

This is the policy counterpart of eval_manifest_zero_action.py. It pins env i
to motion i, loads a checkpoint, runs a short deterministic rollout, and prints
exact-hit racket errors per motion. Use it to diagnose K=2/K=4 failures before
spending more PPO time.
"""

import os
import sys
import json
import pathlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for _p in (_REPO_ROOT, os.path.normpath(os.path.join(_REPO_ROOT, "show"))):
    if _p not in sys.path:
        sys.path.insert(0, _p)
del _REPO_ROOT, _p

import hydra
from omegaconf import OmegaConf

from train import _apply_task_overrides


def _obs_to_device(obs, device):
    if isinstance(obs, tuple):
        obs = obs[0]
    return obs.to(device)


def _print_group_summary(rows):
    def summarize(name, group):
        if not group:
            return
        hit_comp = sum(1 for r in group if r[14]) / len(group)
        posture = sum(1 for r in group if r[15]) / len(group)
        whole = sum(1 for r in group if r[14] and r[15]) / len(group)
        pos_mean = sum(r[0] for r in group) / len(group)
        vel_mean = sum(r[1] for r in group) / len(group)
        normal_mean = sum(r[2] for r in group) / len(group)
        pelvis_margin_mean = sum(15.0 - r[9] for r in group) / len(group)
        torso_margin_mean = sum(20.0 - r[10] for r in group) / len(group)
        arm_margin_mean = sum(0.10 - r[12] for r in group) / len(group)
        worst = max(group, key=lambda r: (r[0], r[1], r[2]))
        print(
            f"[INFO] {name}: n={len(group)} hit_composite={hit_comp:.3f} posture={posture:.3f} whole_cycle={whole:.3f} "
            f"pos_mean={pos_mean:.4f} vel_mean={vel_mean:.4f} normal_mean={normal_mean:.2f} "
            f"pelvis_margin_mean={pelvis_margin_mean:.2f} torso_margin_mean={torso_margin_mean:.2f} "
            f"arm_margin_mean={arm_margin_mean:.4f} "
            f"worst={worst[17]} pos={worst[0]:.4f} vel={worst[1]:.4f} normal={worst[2]:.2f}",
            flush=True,
        )

    finite_rows = [r for r in rows if r[0] != float("inf")]
    summarize("overall", finite_rows)
    summarize("forehand", [r for r in finite_rows if r[16] == "forehand"])
    summarize("backhand", [r for r in finite_rows if r[16] == "backhand"])


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


def _apply_motion_reset_perturbation(env, motion_cmd, n, device, cfg, episode_ids):
    import torch
    from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, sample_uniform

    pose_range = getattr(motion_cmd.cfg, "pose_range", {}) or {}
    vel_range = getattr(motion_cmd.cfg, "velocity_range", {}) or {}
    joint_position_range = tuple(getattr(motion_cmd.cfg, "joint_position_range", (0.0, 0.0)) or (0.0, 0.0))
    bank_path = cfg.get("perturb_bank", None)
    write_bank_path = cfg.get("write_perturb_bank", None)
    if not pose_range and not vel_range and joint_position_range == (0.0, 0.0) and bank_path is None:
        return None

    env_ids = torch.arange(n, device=device)
    robot = motion_cmd.robot
    root_state = robot.data.root_state_w[env_ids].clone()
    joint_pos = robot.data.joint_pos[env_ids].clone()
    joint_vel = robot.data.joint_vel[env_ids].clone()
    if bank_path is not None:
        bank_file = pathlib.Path(str(bank_path)).expanduser()
        if not bank_file.is_absolute():
            bank_file = pathlib.Path.cwd() / bank_file
        bank_data = json.load(open(bank_file, "r", encoding="utf-8"))
        by_episode = {str(item["episode_id"]): item for item in bank_data.get("motions", [])}
        pose_samples = torch.tensor(
            [by_episode[e]["root_pose_delta"] for e in episode_ids], dtype=torch.float32, device=device
        )
        vel_samples = torch.tensor(
            [by_episode[e]["root_velocity_delta"] for e in episode_ids], dtype=torch.float32, device=device
        )
        joint_delta = torch.tensor(
            [by_episode[e]["joint_position_delta"] for e in episode_ids], dtype=torch.float32, device=device
        )
    else:
        pose_ranges = torch.tensor(
            [pose_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z", "roll", "pitch", "yaw")],
            device=device,
            dtype=torch.float32,
        )
        pose_samples = sample_uniform(pose_ranges[:, 0], pose_ranges[:, 1], (n, 6), device=device)
        vel_ranges = torch.tensor(
            [vel_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z", "roll", "pitch", "yaw")],
            device=device,
            dtype=torch.float32,
        )
        vel_samples = sample_uniform(vel_ranges[:, 0], vel_ranges[:, 1], (n, 6), device=device)
        if joint_position_range != (0.0, 0.0):
            joint_delta = sample_uniform(*joint_position_range, joint_pos.shape, device=device)
        else:
            joint_delta = torch.zeros_like(joint_pos)

    root_state[:, 0:3] += pose_samples[:, 0:3]
    ori_delta = quat_from_euler_xyz(pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5])
    root_state[:, 3:7] = quat_mul(ori_delta, root_state[:, 3:7])
    root_state[:, 7:10] += vel_samples[:, 0:3]
    root_state[:, 10:13] += vel_samples[:, 3:6]
    joint_pos += joint_delta
    soft_limits = robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])

    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    bank = {
        "motion_manifest": str(cfg.motion_manifest if cfg.motion_manifest is not None else cfg.task.get("motion_manifest")),
        "motions": [
            {
                "episode_id": episode_ids[i],
                "root_pose_delta": [float(x) for x in pose_samples[i].detach().cpu().tolist()],
                "root_velocity_delta": [float(x) for x in vel_samples[i].detach().cpu().tolist()],
                "joint_position_delta": [float(x) for x in joint_delta[i].detach().cpu().tolist()],
            }
            for i in range(n)
        ],
    }
    if write_bank_path is not None and bank_path is None:
        out = pathlib.Path(str(write_bank_path)).expanduser()
        if not out.is_absolute():
            out = pathlib.Path.cwd() / out
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(bank, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"[INFO] wrote perturbation bank: {out} ({n} motions)", flush=True)
    return bank


def _run(cfg, simulation_app):
    import pathlib

    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.utils.math import matrix_from_quat, quat_error_magnitude
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg

    import training.tasks  # noqa: F401
    from training.utils.ppo_cfg import runner_kwargs

    task_id = str(cfg.task.gym_task)
    num_envs = int(cfg.num_envs) if cfg.num_envs is not None else int(cfg.task.env.num_envs)

    env_cfg = parse_env_cfg(task_id, device=str(cfg.device), num_envs=num_envs)
    _apply_task_overrides(env_cfg, cfg.task)
    env_cfg.sim.device = str(cfg.device)
    env_cfg.seed = int(cfg.get("seed", 0) or 0)

    motion_manifest = cfg.motion_manifest if cfg.motion_manifest is not None else cfg.task.get("motion_manifest")
    if motion_manifest is None:
        raise ValueError("eval_manifest_policy.py requires motion_manifest=... or task.motion_manifest")
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
        raise ValueError("eval_manifest_policy.py requires checkpoint=...")
    resume_path = pathlib.Path(str(checkpoint)).expanduser()
    if not resume_path.is_absolute():
        resume_path = pathlib.Path.cwd() / resume_path
    print(f"[INFO] loading checkpoint: {resume_path}", flush=True)
    print(
        f"[INFO] using manifest: {manifest_path} "
        f"(subset_size={env_cfg.commands.motion.manifest_subset_size}, "
        f"frame_z_offset={env_cfg.commands.motion.manifest_frame_z_offset:.4f}m)",
        flush=True,
    )

    env = gym.make(task_id, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)
    device = env.unwrapped.device

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(str(resume_path), load_optimizer=False)
    policy = ppo_runner.get_inference_policy(device=device)

    motion_cmd = env.unwrapped.command_manager.get_term("motion")
    racket_cmd = env.unwrapped.command_manager.get_term("racket_target")
    robot = motion_cmd.robot
    body_name_to_id = {name: i for i, name in enumerate(robot.body_names)}
    pelvis_body_id = body_name_to_id.get("pelvis_link", 0)
    torso_body_id = body_name_to_id.get("torso_Link", pelvis_body_id)
    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    native_joint_ids = getattr(action_term, "_joint_index_tensor", None)
    action_scale = getattr(action_term, "_scale", None)
    if action_scale is not None:
        scale_abs_max = float(action_scale.abs().max().detach().cpu())
        scale_abs_mean = float(action_scale.abs().mean().detach().cpu())
        print(
            f"[INFO] action scale abs max/mean: {scale_abs_max:.6f}/{scale_abs_mean:.6f}",
            flush=True,
        )
    native_joint_names = []
    if native_joint_ids is not None:
        native_joint_names = [robot.data.joint_names[int(idx)] for idx in native_joint_ids.detach().cpu().tolist()]
    n_motions = int(motion_cmd.motion.num_motions)
    n = min(num_envs, n_motions)
    if n < n_motions:
        print(f"[WARN] num_envs={num_envs} < motions={n_motions}; evaluating first {n} motions only.", flush=True)

    env.reset()
    _sync_motion_state(env.unwrapped, motion_cmd, n, device)
    _apply_motion_reset_perturbation(
        env.unwrapped, motion_cmd, n, device, cfg, [str(x) for x in motion_cmd.motion.episode_ids[:n]]
    )
    try:
        racket_cmd._resample_command(torch.arange(n, device=device))
    except Exception:
        pass

    captured = {
        "pos_err": torch.full((n,), float("nan"), device=device),
        "vel_err": torch.full((n,), float("nan"), device=device),
        "normal_err": torch.full((n,), float("nan"), device=device),
        "pos_window": torch.full((n,), float("nan"), device=device),
        "action_abs": torch.full((n,), float("nan"), device=device),
        "pelvis_upright": torch.full((n,), float("nan"), device=device),
        "torso_upright": torch.full((n,), float("nan"), device=device),
        "ref_pelvis_upright": torch.full((n,), float("nan"), device=device),
        "ref_torso_upright": torch.full((n,), float("nan"), device=device),
        "pelvis_ref_err_deg": torch.full((n,), float("nan"), device=device),
        "torso_ref_err_deg": torch.full((n,), float("nan"), device=device),
        "joint_near_limit_frac": torch.full((n,), float("nan"), device=device),
        "arm_near_limit_frac": torch.full((n,), float("nan"), device=device),
        "joint_near_limit_mask": torch.zeros(
            (n, len(native_joint_ids) if native_joint_ids is not None else 0), dtype=torch.bool, device=device
        ),
        "captured": torch.zeros(n, dtype=torch.bool, device=device),
    }

    obs = _obs_to_device(env.get_observations(), agent_cfg.device)
    max_steps = int(cfg.get("max_steps") or 60)
    for _ in range(max_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions.to(device))
            obs = _obs_to_device(obs, agent_cfg.device)

            exact = torch.abs(racket_cmd.time_to_strike[:n]) <= (0.5 * env.unwrapped.step_dt + 1.0e-6)
            take = exact & (~captured["captured"])
            if bool(take.any()):
                normal_dot = torch.sum(
                    racket_cmd.racket_normal_w[:n][take] * racket_cmd.racket_target_normal_w[:n][take], dim=-1
                ).clamp(-1.0, 1.0)
                captured["pos_err"][take] = torch.linalg.norm(
                    racket_cmd.racket_pos_w[:n][take] - racket_cmd.racket_target_pos_w[:n][take], dim=-1
                )
                captured["vel_err"][take] = torch.linalg.norm(
                    racket_cmd.racket_lin_vel_w[:n][take] - racket_cmd.racket_target_vel_w[:n][take], dim=-1
                )
                captured["normal_err"][take] = torch.rad2deg(torch.acos(normal_dot))
                captured["pos_window"][take] = racket_cmd.metrics["racket_pos_error_at_strike"][:n][take]
                captured["action_abs"][take] = racket_cmd.metrics["action_abs_mean"][:n][take]
                body_quat = robot.data.body_quat_w[:n]
                motion_ids = motion_cmd.motion_ids[:n]
                time_steps = motion_cmd.time_steps[:n]
                if motion_cmd._use_motion_library:
                    ref_quat = motion_cmd.motion._body_quat_w[motion_ids, time_steps]
                else:
                    ref_quat = motion_cmd.motion._body_quat_w[time_steps]

                captured["pelvis_upright"][take] = matrix_from_quat(body_quat[take, pelvis_body_id])[:, 2, 2]
                captured["torso_upright"][take] = matrix_from_quat(body_quat[take, torso_body_id])[:, 2, 2]
                captured["ref_pelvis_upright"][take] = matrix_from_quat(ref_quat[take, pelvis_body_id])[:, 2, 2]
                captured["ref_torso_upright"][take] = matrix_from_quat(ref_quat[take, torso_body_id])[:, 2, 2]
                captured["pelvis_ref_err_deg"][take] = torch.rad2deg(
                    quat_error_magnitude(ref_quat[take, pelvis_body_id], body_quat[take, pelvis_body_id])
                )
                captured["torso_ref_err_deg"][take] = torch.rad2deg(
                    quat_error_magnitude(ref_quat[take, torso_body_id], body_quat[take, torso_body_id])
                )
                if native_joint_ids is not None:
                    joint_pos = robot.data.joint_pos[:n, native_joint_ids]
                    limits = robot.data.soft_joint_pos_limits[:n, native_joint_ids]
                    span = torch.clamp(limits[..., 1] - limits[..., 0], min=1.0e-6)
                    margin = torch.minimum(joint_pos - limits[..., 0], limits[..., 1] - joint_pos) / span
                    near_mask = margin[take] < 0.05
                    captured["joint_near_limit_frac"][take] = near_mask.float().mean(dim=-1)
                    non_waist_mask = torch.tensor(
                        [not name.startswith("waist_") for name in native_joint_names],
                        dtype=torch.bool,
                        device=device,
                    )
                    if bool(non_waist_mask.any()):
                        captured["arm_near_limit_frac"][take] = near_mask[:, non_waist_mask].float().mean(dim=-1)
                    captured["joint_near_limit_mask"][take] = near_mask
                captured["captured"][take] = True
        if not simulation_app.is_running():
            break

    names = motion_cmd.motion.episode_ids if hasattr(motion_cmd.motion, "episode_ids") else [str(i) for i in range(n)]
    strokes = motion_cmd.motion.stroke_types if hasattr(motion_cmd.motion, "stroke_types") else ["unknown"] * n

    pos_thresh = float(racket_cmd.cfg.strike_success_pos_thresh)
    vel_thresh = float(racket_cmd.cfg.strike_success_vel_thresh)
    normal_thresh = float(racket_cmd.cfg.strike_success_normal_thresh_deg)
    print("[INFO] policy manifest exact-hit evaluation:", flush=True)
    print(
        "rank,stroke,episode_id,pos_exact,vel_exact,normal_deg_exact,composite_pass,"
        "pos_window,action_abs_mean,pelvis_upright,torso_upright,ref_pelvis_upright,"
        "ref_torso_upright,pelvis_ref_err_deg,torso_ref_err_deg,joint_near_limit_frac,"
        "arm_near_limit_frac,joint_near_limit_names,posture_pass",
        flush=True,
    )

    rows = []
    for i in range(n):
        if not bool(captured["captured"][i]):
            rows.append(
                (
                    float("inf"),
                    float("inf"),
                    float("inf"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    "-",
                    False,
                    False,
                    strokes[i],
                    names[i],
                )
            )
            continue
        pos = float(captured["pos_err"][i].detach().cpu())
        vel = float(captured["vel_err"][i].detach().cpu())
        normal = float(captured["normal_err"][i].detach().cpu())
        pos_window = float(captured["pos_window"][i].detach().cpu())
        action_abs = float(captured["action_abs"][i].detach().cpu())
        pelvis_up = float(captured["pelvis_upright"][i].detach().cpu())
        torso_up = float(captured["torso_upright"][i].detach().cpu())
        ref_pelvis_up = float(captured["ref_pelvis_upright"][i].detach().cpu())
        ref_torso_up = float(captured["ref_torso_upright"][i].detach().cpu())
        pelvis_ref = float(captured["pelvis_ref_err_deg"][i].detach().cpu())
        torso_ref = float(captured["torso_ref_err_deg"][i].detach().cpu())
        joint_near_limit = float(captured["joint_near_limit_frac"][i].detach().cpu())
        arm_near_limit = float(captured["arm_near_limit_frac"][i].detach().cpu())
        near_names = "-"
        if native_joint_ids is not None:
            near_mask = captured["joint_near_limit_mask"][i].detach().cpu().tolist()
            near_names = "|".join(name for name, is_near in zip(native_joint_names, near_mask) if is_near) or "-"
        passed = pos <= pos_thresh and vel <= vel_thresh and normal <= normal_thresh
        posture_pass = (
            pelvis_ref <= 15.0
            and torso_ref <= 20.0
            and arm_near_limit <= 0.10
        )
        rows.append(
            (
                pos,
                vel,
                normal,
                pos_window,
                action_abs,
                pelvis_up,
                torso_up,
                ref_pelvis_up,
                ref_torso_up,
                pelvis_ref,
                torso_ref,
                joint_near_limit,
                arm_near_limit,
                near_names,
                passed,
                posture_pass,
                strokes[i],
                names[i],
            )
        )

    for rank, row in enumerate(sorted(rows, key=lambda r: (r[0], r[1], r[2])), start=1):
        (
            pos,
            vel,
            normal,
            pos_window,
            action_abs,
            pelvis_up,
            torso_up,
            ref_pelvis_up,
            ref_torso_up,
            pelvis_ref,
            torso_ref,
            joint_near_limit,
            arm_near_limit,
            near_names,
            passed,
            posture_pass,
            stroke,
            name,
        ) = row
        print(
            f"{rank},{stroke},{name},{pos:.4f},{vel:.4f},{normal:.2f},{int(passed)},"
            f"{pos_window:.4f},{action_abs:.4f},{pelvis_up:.4f},{torso_up:.4f},"
            f"{ref_pelvis_up:.4f},{ref_torso_up:.4f},{pelvis_ref:.2f},{torso_ref:.2f},"
            f"{joint_near_limit:.4f},{arm_near_limit:.4f},{near_names},{int(posture_pass)}",
            flush=True,
        )

    finite_rows = [r for r in rows if r[0] != float("inf")]
    if finite_rows:
        pass_rate = sum(1 for r in finite_rows if r[14]) / len(finite_rows)
        posture_rate = sum(1 for r in finite_rows if r[15]) / len(finite_rows)
        whole_rate = sum(1 for r in finite_rows if r[14] and r[15]) / len(finite_rows)
        print(f"[INFO] composite_pass_rate={pass_rate:.3f} ({len(finite_rows)} captured motions)", flush=True)
        print(f"[INFO] posture_pass_rate={posture_rate:.3f} ({len(finite_rows)} captured motions)", flush=True)
        print(f"[INFO] whole_cycle_pass_rate={whole_rate:.3f} ({len(finite_rows)} captured motions)", flush=True)
        _print_group_summary(rows)
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
    except Exception:
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
