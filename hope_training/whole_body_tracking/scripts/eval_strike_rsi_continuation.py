#!/usr/bin/env python3
"""Exact RSI direct-load smoke for native strike motions.

This is intentionally zero-residual and deterministic. It verifies that a
saved full state can be loaded at a motion phase and continue under the native
reference contract. It is not a training or deployment approval tool.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import hydra
from omegaconf import OmegaConf

from train import _apply_task_overrides


def _sync_motion_state(env, motion_cmd, n, device, start_steps):
    import torch

    motion_cmd.time_steps[:n] = torch.as_tensor(start_steps, dtype=torch.long, device=device)
    env_ids = torch.arange(n, device=device)
    root_pos = motion_cmd.motion._body_pos_w[motion_cmd.motion_ids[:n], motion_cmd.time_steps[:n], 0]
    root_pos = root_pos + env.scene.env_origins[:n]
    root_ori = motion_cmd.motion._body_quat_w[motion_cmd.motion_ids[:n], motion_cmd.time_steps[:n], 0]
    root_lin_vel = motion_cmd.motion._body_lin_vel_w[motion_cmd.motion_ids[:n], motion_cmd.time_steps[:n], 0]
    root_ang_vel = motion_cmd.motion._body_ang_vel_w[motion_cmd.motion_ids[:n], motion_cmd.time_steps[:n], 0]
    motion_cmd.robot.write_joint_state_to_sim(motion_cmd.joint_pos[:n], motion_cmd.joint_vel[:n], env_ids=env_ids)
    motion_cmd.robot.write_root_state_to_sim(
        torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1), env_ids=env_ids
    )

@hydra.main(version_base=None, config_path="../cfg", config_name="play")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    from isaaclab.app import AppLauncher

    # Hydra has already consumed the overrides; Isaac's launcher must receive
    # a clean argv just like the verified native evaluators.
    sys.argv = sys.argv[:1]
    app_launcher = AppLauncher(headless=bool(cfg.headless), device=str(cfg.device), enable_cameras=False)
    app = app_launcher.app
    try:
        print("[RSI] SimulationApp ready", flush=True)
        import gymnasium as gym
        import torch
        from isaaclab_tasks.utils import parse_env_cfg
        import training.tasks  # noqa: F401

        print("[RSI] task modules imported", flush=True)

        task_id = str(cfg.task.gym_task)
        sim_device = str(cfg.get("sim_device", None) or cfg.device)
        manifest = pathlib.Path(str(cfg.motion_manifest or cfg.task.motion_manifest)).expanduser()
        env_cfg = parse_env_cfg(task_id, device=sim_device, num_envs=1)
        _apply_task_overrides(env_cfg, cfg.task)
        env_cfg.sim.device = sim_device
        env_cfg.seed = int(cfg.get("seed", 0) or 0)
        env_cfg.commands.motion.motion_manifest = str(manifest)
        env_cfg.commands.motion.motion_file = None
        env_cfg.commands.motion.manifest_subset_size = None
        # Motion/contact debug markers are irrelevant for a headless numeric
        # audit.  Their stock assets live on NVIDIA Nucleus and can stall an
        # offline run for minutes while each remote frame marker times out.
        env_cfg.commands.motion.debug_vis = False
        if getattr(env_cfg.scene, "contact_forces", None) is not None:
            env_cfg.scene.contact_forces.debug_vis = False
        if getattr(env_cfg.scene, "terrain", None) is not None:
            env_cfg.scene.terrain.visual_material = None
        print(f"[RSI] creating task={task_id}", flush=True)
        env = gym.make(task_id, cfg=env_cfg, render_mode=None)
        print("[RSI] environment ready", flush=True)
        motion_cmd = env.unwrapped.command_manager.get_term("motion")
        racket_cmd = env.unwrapped.command_manager.get_term("racket_target")
        robot = motion_cmd.robot
        device = env.unwrapped.device
        bank = pathlib.Path(str(cfg.rsi_bank)).expanduser()
        cases = json.loads((bank / "continuation_cases.json").read_text(encoding="utf-8"))["cases"]
        if cfg.get("case_start", None) is not None:
            cases = cases[int(cfg.case_start):]
        if cfg.get("max_cases", None) is not None:
            cases = cases[: int(cfg.max_cases)]
        episode_ids = [str(x) for x in motion_cmd.motion.episode_ids]
        episode_to_id = {name: i for i, name in enumerate(episode_ids)}
        realized_bank = pathlib.Path(str(cfg.realized_bank)).expanduser() if cfg.get("realized_bank", None) else None
        realized_by_episode = {}
        realized_cache = {}
        if realized_bank is not None:
            realized_manifest = json.loads(
                (realized_bank / "realized_rsi_manifest.json").read_text(encoding="utf-8")
            )
            for entry in realized_manifest["entries"]:
                realized_by_episode[str(entry["episode_id"])] = realized_bank / str(entry["state_file"])
            print(
                f"[RSI] using realized rollout bank: {realized_bank} "
                f"({len(realized_by_episode)} motions)",
                flush=True,
            )
        rows = []
        capture_trace = bool(cfg.get("capture_trace", False))
        trace = []
        for case_index, case in enumerate(cases):
            episode_id = case["episode_id"]
            if episode_id not in episode_to_id:
                raise RuntimeError(f"case episode not loaded by manifest: {episode_id}")
            motion_id = episode_to_id[episode_id]
            frame = int(case["frame"])
            print(
                f"[RSI] case {case_index + 1}/{len(cases)} episode={episode_id} "
                f"phase={case['phase']} requested_frame={frame}",
                flush=True,
            )
            env.reset()
            motion_cmd.motion_ids[0] = motion_id
            loaded_step = frame
            loaded_joint_target = None
            loaded_root = None
            if realized_bank is None:
                _sync_motion_state(env.unwrapped, motion_cmd, 1, device, [frame])
            else:
                if episode_id not in realized_by_episode:
                    raise RuntimeError(f"realized bank has no episode: {episode_id}")
                state_path = realized_by_episode[episode_id]
                if episode_id not in realized_cache:
                    import numpy as np
                    with np.load(state_path, allow_pickle=False) as data:
                        realized_cache[episode_id] = {
                            key: np.asarray(data[key]) for key in data.files
                        }
                state = realized_cache[episode_id]
                steps = state["motion_step"].astype(int)
                state_idx = int(np.argmin(np.abs(steps - frame)))
                loaded_step = int(steps[state_idx])
                q = torch.as_tensor(state["joint_pos"][state_idx], dtype=robot.data.joint_pos.dtype, device=device).unsqueeze(0)
                dq = torch.as_tensor(state["joint_vel"][state_idx], dtype=robot.data.joint_vel.dtype, device=device).unsqueeze(0)
                root = torch.as_tensor(state["root_state_w"][state_idx], dtype=robot.data.root_state_w.dtype, device=device).unsqueeze(0)
                loaded_joint_target = torch.as_tensor(
                    state["joint_pos_target"][state_idx], dtype=robot.data.joint_pos_target.dtype, device=device
                ).unsqueeze(0)
                loaded_root = root.clone()
                motion_cmd.time_steps[0] = loaded_step
                robot.write_joint_state_to_sim(q, dq, env_ids=torch.tensor([0], device=device))
                robot.write_root_state_to_sim(root, env_ids=torch.tensor([0], device=device))
            try:
                racket_cmd._resample_command(torch.tensor([0], device=device))
                # _resample_command stamps the target and wrap detector but
                # intentionally does not recompute timing.  Exact RSI must
                # explicitly align the derived strike clock to motion.time_steps.
                racket_cmd._compute_strike_timing()
            except Exception:
                pass
            # Restore the residual-action/PD target contract at the same phase
            # before the first physics substep.  Without this priming, reset
            # leaves the actuator target at its default pose and creates an
            # artificial first-step impulse.
            zero = torch.zeros((1, env.unwrapped.action_manager.total_action_dim), device=device)
            env.unwrapped.action_manager.process_action(zero)
            env.unwrapped.action_manager.apply_action()
            if loaded_joint_target is not None:
                robot.set_joint_position_target(loaded_joint_target)
            env.unwrapped.scene.write_data_to_sim()
            start_q = robot.data.joint_pos[0].clone()
            if loaded_root is not None:
                start_root = loaded_root[0].clone()
            else:
                start_root = torch.cat(
                    [
                        motion_cmd.motion._body_pos_w[motion_id, loaded_step, 0] + env.unwrapped.scene.env_origins[0],
                        motion_cmd.motion._body_quat_w[motion_id, loaded_step, 0],
                        motion_cmd.motion._body_lin_vel_w[motion_id, loaded_step, 0],
                        motion_cmd.motion._body_ang_vel_w[motion_id, loaded_step, 0],
                    ]
                )
            initial_time_to_strike = float(racket_cmd.time_to_strike[0].detach().cpu())
            finite = True
            max_q_jump = 0.0
            first_joint_delta = float("nan")
            first_root_ang_vel_delta = float("nan")
            first_torque_abs_max = float("nan")
            hit = False
            hit_pos = float("nan")
            hit_vel = float("nan")
            hit_normal = float("nan")
            # Run only the remaining reference horizon.  No learned action is
            # used: the environment's native reference path is the baseline.
            motion_length = int(motion_cmd.motion.motion_lengths[motion_id].detach().cpu())
            rollout_count = max(1, motion_length - loaded_step)
            if cfg.get("max_rollout_steps", None) is not None:
                rollout_count = min(rollout_count, int(cfg.max_rollout_steps))
            for rollout_step in range(rollout_count):
                env.step(zero)
                q = robot.data.joint_pos[0]
                if rollout_step == 0:
                    first_joint_delta = float(torch.linalg.vector_norm(q - start_q).detach().cpu())
                    first_root_ang_vel_delta = float(
                        torch.linalg.vector_norm(robot.data.root_ang_vel_w[0] - start_root[10:13]).detach().cpu()
                    )
                    first_torque_abs_max = float(torch.abs(robot.data.applied_torque[0]).max().detach().cpu())
                if capture_trace:
                    trace.append({
                        "motion_step": motion_cmd.time_steps[0].detach().clone(),
                        "root_state_w": robot.data.root_state_w[0].detach().clone(),
                        "joint_pos": robot.data.joint_pos[0].detach().clone(),
                        "joint_vel": robot.data.joint_vel[0].detach().clone(),
                        "joint_pos_target": robot.data.joint_pos_target[0].detach().clone(),
                        "applied_torque": robot.data.applied_torque[0].detach().clone(),
                        "time_to_strike_s": racket_cmd.time_to_strike[0].detach().clone(),
                    })
                finite = finite and bool(torch.isfinite(q).all() and torch.isfinite(robot.data.root_state_w[0]).all())
                max_q_jump = max(max_q_jump, float(torch.linalg.vector_norm(q - start_q).detach().cpu()))
                exact = bool(torch.abs(racket_cmd.time_to_strike[0]) <= (0.5 * env.unwrapped.step_dt + 1.0e-6))
                if exact and not hit:
                    hit = True
                    hit_pos = float(torch.linalg.vector_norm(racket_cmd.racket_pos_w[0] - racket_cmd.racket_target_pos_w[0]).detach().cpu())
                    hit_vel = float(torch.linalg.vector_norm(racket_cmd.racket_lin_vel_w[0] - racket_cmd.racket_target_vel_w[0]).detach().cpu())
                    dot = torch.sum(racket_cmd.racket_normal_w[0] * racket_cmd.racket_target_normal_w[0]).clamp(-1.0, 1.0)
                    hit_normal = float(torch.rad2deg(torch.acos(dot)).detach().cpu())
            rows.append({"episode_id": episode_id, "phase": case["phase"], "frame": frame, "loaded_step": loaded_step, "realized_state": realized_bank is not None, "initial_time_to_strike_s": initial_time_to_strike, "finite": finite, "hit_seen": hit, "hit_pos_err_m": hit_pos, "hit_vel_err_mps": hit_vel, "hit_normal_err_deg": hit_normal, "first_joint_delta_norm": first_joint_delta, "first_root_ang_vel_delta_rad_s": first_root_ang_vel_delta, "first_torque_abs_max_nm": first_torque_abs_max, "start_state_root_norm": float(torch.linalg.vector_norm(start_root).detach().cpu()), "max_joint_delta_norm": max_q_jump})
            print(
                f"[RSI] case complete loaded_step={loaded_step} rollout_steps={rollout_count} "
                f"finite={finite} first_joint_delta={first_joint_delta:.6f} "
                f"first_torque_abs_max={first_torque_abs_max:.3f}",
                flush=True,
            )
        out = pathlib.Path(str(cfg.output_json)).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"stage": "strike_rsi_exact_direct_load_smoke_v1", "training_eligible": False, "cases": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if capture_trace and trace:
            import numpy as np
            trace_path = out.with_name(out.stem + "_trace.npz")
            np.savez_compressed(trace_path, **{name: np.stack([x[name].cpu().numpy() for x in trace]) for name in trace[0]})
            print(f"[RSI] wrote realized trace: {trace_path}", flush=True)
        print(json.dumps({"cases": len(rows), "finite_rate": sum(r["finite"] for r in rows) / len(rows), "hit_seen_rate": sum(r["hit_seen"] for r in rows) / len(rows), "output": str(out)}, ensure_ascii=False), flush=True)
        env.close()
    finally:
        app.close()


if __name__ == "__main__":
    main()
