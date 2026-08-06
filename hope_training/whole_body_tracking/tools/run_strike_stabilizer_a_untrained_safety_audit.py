#!/usr/bin/env python3
"""Paired bounded-random leg-residual safety smoke for Strike Stabilizer-A.

This deliberately does not assess learning.  It runs the real continuous swing
prefix, enables a bounded low-frequency random residual at each sampled phase,
and compares it to a paired zero-residual rollout.  The upper body and both
waist residual channels remain owned by the strike reference / structural mask.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import hydra
from omegaconf import OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from train import _apply_task_overrides


def _stats(values):
    import torch

    values = values.float()
    return {
        "min": float(values.min().item()),
        "mean": float(values.mean().item()),
        "median": float(values.median().item()),
        "p95": float(torch.quantile(values, 0.95).item()),
        "max": float(values.max().item()),
    }


def _policy_observation(vec_env):
    observation = vec_env.get_observations()
    return observation[0] if isinstance(observation, tuple) else observation


def _sync_motion_start(env, motion_cmd, motion_ids):
    """Start each paired rollout at the same known-safe frame-zero state."""

    import torch

    ids = torch.arange(len(motion_ids), device=env.device, dtype=torch.long)
    motion_cmd.motion_ids[:] = motion_ids
    motion_cmd.time_steps.zero_()
    root_pos, root_ori, root_lin_vel, root_ang_vel = motion_cmd._motion_root_state_w()
    root_state = torch.cat((root_pos, root_ori, root_lin_vel, root_ang_vel), dim=-1)
    motion_cmd.robot.write_joint_state_to_sim(
        motion_cmd.joint_pos, motion_cmd.joint_vel, env_ids=ids
    )
    motion_cmd.robot.write_root_state_to_sim(root_state, env_ids=ids)


@hydra.main(version_base=None, config_path="../cfg", config_name="play")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher

    video_value = cfg.get("video_path", None)
    video_path = pathlib.Path(str(video_value)) if video_value else None
    render_candidate_only = bool(cfg.get("render_candidate_only", False))
    app_launcher = AppLauncher(
        headless=bool(cfg.headless), device=str(cfg.device), enable_cameras=video_path is not None
    )
    simulation_app = app_launcher.app
    try:
        import gymnasium as gym
        import torch
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
        from isaaclab_tasks.utils import parse_env_cfg

        import training.tasks  # noqa: F401
        from training.utils.ppo_cfg import runner_kwargs

        pairs = int(cfg.get("pairs", 16))
        if pairs < 2:
            raise ValueError("pairs must be at least 2")
        action_bound = float(cfg.get("random_action_bound", 0.05))
        if not 0.0 < action_bound < 0.25:
            raise ValueError("random_action_bound must be inside (0, 0.25)")
        hold_steps = int(cfg.get("hold_steps", 5))
        if hold_steps < 1:
            raise ValueError("hold_steps must be positive")
        constant_leg_action = cfg.get("constant_leg_action", None)
        if constant_leg_action is not None:
            constant_leg_action = torch.as_tensor(constant_leg_action, dtype=torch.float32)
            if constant_leg_action.numel() != 12:
                raise ValueError("constant_leg_action must provide exactly 12 leg values")
        forced_handoff_step = cfg.get("handoff_step", None)
        if forced_handoff_step is not None and int(forced_handoff_step) < 0:
            raise ValueError("handoff_step must be non-negative")
        seed = int(cfg.get("seed", 20260722))
        case_offset = int(cfg.get("case_offset", 0))
        output_path = pathlib.Path(
            str(cfg.get("output_json", "eval_outputs/strike_stabilizer_a/untrained_leg_safety_smoke.json"))
        )
        task_id = str(cfg.task.gym_task)
        env_cfg = parse_env_cfg(task_id, device=str(cfg.device), num_envs=2 * pairs)
        _apply_task_overrides(env_cfg, cfg.task)
        env_cfg.sim.device = str(cfg.device)
        env_cfg.seed = seed
        # ``motion_manifest`` is a top-level task contract, not a generic
        # train.py override.  Mirror the capture/evaluation path explicitly so
        # this audit cannot silently fall back to a stale motion_file.
        manifest_path = pathlib.Path(
            str(cfg.motion_manifest or cfg.task.motion_manifest)
        ).expanduser()
        env_cfg.commands.motion.motion_manifest = str(manifest_path)
        env_cfg.commands.motion.motion_file = None
        env_cfg.commands.motion.manifest_subset_size = None
        frame_z_offset = cfg.get("manifest_frame_z_offset", None)
        if frame_z_offset is None:
            frame_z_offset = cfg.task.get("manifest_frame_z_offset", None)
        if frame_z_offset is not None:
            env_cfg.commands.motion.manifest_frame_z_offset = float(frame_z_offset)
        print(
            f"[strike-leg-safety] constructing task={task_id} envs={2 * pairs}",
            flush=True,
        )
        env = gym.make(task_id, cfg=env_cfg, render_mode="rgb_array" if video_path is not None else None)
        try:
            checkpoint = cfg.get("checkpoint", None)
            candidate_label = "bounded_random_legs"
            policy = None
            vec_env = None
            if checkpoint:
                checkpoint_path = pathlib.Path(str(checkpoint)).expanduser()
                if not checkpoint_path.is_absolute():
                    checkpoint_path = pathlib.Path.cwd() / checkpoint_path
                agent_cfg = RslRlOnPolicyRunnerCfg(
                    **runner_kwargs(OmegaConf.to_container(cfg.algo, resolve=True), str(cfg.task.experiment_name))
                )
                agent_cfg.device = str(cfg.device)
                vec_env = RslRlVecEnvWrapper(env)
                runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
                runner.load(str(checkpoint_path), load_optimizer=False)
                policy = runner.get_inference_policy(device=env.unwrapped.device)
                candidate_label = "learned_legs"
                print(f"[strike-leg-safety] deterministic checkpoint={checkpoint_path}", flush=True)
                if render_candidate_only:
                    print("[strike-leg-safety] video render is candidate-only; metrics are not a paired audit", flush=True)
            elif constant_leg_action is not None:
                candidate_label = "constant_legs"
            print("[strike-leg-safety] reset", flush=True)
            env.reset(seed=seed)
            unwrapped = env.unwrapped
            device = unwrapped.device
            robot = unwrapped.scene["robot"]
            motion = unwrapped.command_manager.get_term("motion")
            racket = unwrapped.command_manager.get_term("racket_target")
            action_term = unwrapped.action_manager.get_term("joint_pos")
            total = 2 * pairs
            case_ids = torch.arange(pairs, device=device, dtype=torch.long) + case_offset
            pair_motion_ids = case_ids % motion.motion.num_motions
            motion_ids = torch.cat((pair_motion_ids, pair_motion_ids))
            _sync_motion_start(unwrapped, motion, motion_ids)
            print("[strike-leg-safety] paired motion states synchronized", flush=True)
            all_ids = torch.arange(total, device=device)
            racket._resample_command(all_ids)
            racket._compute_strike_timing()
            policy_obs = _policy_observation(vec_env) if vec_env is not None else None

            candidate_handoffs = torch.tensor((0, 10, 18, 25, 30, 35, 50), device=device)
            pair_handoffs = candidate_handoffs[case_ids % len(candidate_handoffs)]
            if forced_handoff_step is not None:
                pair_handoffs.fill_(int(forced_handoff_step))
            unwrapped.strike_stabilizer_handoff_steps = torch.cat((pair_handoffs, pair_handoffs)).clone()
            swing_steps = int(motion.motion.motion_lengths.min().item()) - 1
            tail_steps = int(motion.cfg.hold_last_frame_steps)
            maximum_steps = swing_steps + tail_steps
            policy_steps = int(cfg.get("policy_steps", maximum_steps))
            policy_steps = min(policy_steps, maximum_steps)
            if policy_steps < 1:
                raise ValueError("policy_steps must be positive")

            zero_slice = slice(0, pairs)
            random_slice = slice(pairs, total)
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            held_random = torch.zeros((pairs, 12), device=device)
            if constant_leg_action is not None:
                constant_leg_action = constant_leg_action.to(device=device)
                print(
                    f"[strike-leg-safety] constant_leg_action={constant_leg_action.detach().cpu().tolist()}",
                    flush=True,
                )
            active = torch.ones(total, dtype=torch.bool, device=device)
            failure_count = torch.zeros(total, dtype=torch.long, device=device)
            finite = True
            max_torque = torch.zeros(total, device=device)
            min_joint_margin = torch.full((total,), float("inf"), device=device)
            contact_sum = torch.zeros(total, device=device)
            torso_omega_square_sum = torch.zeros(total, device=device)
            root_height_abs_sum = torch.zeros(total, device=device)
            raw_square_sum = torch.zeros(total, device=device)
            raw_action_sum_by_joint = torch.zeros((total, 14), device=device)
            raw_action_square_sum_by_joint = torch.zeros((total, 14), device=device)
            active_frame_count = torch.zeros(total, dtype=torch.long, device=device)
            effective_clip_count = torch.zeros(total, dtype=torch.long, device=device)
            sampled_exceed_count = torch.zeros(total, dtype=torch.long, device=device)
            active_action_values = torch.zeros(total, dtype=torch.long, device=device)
            waist_max = torch.zeros(total, device=device)
            torque_saturation_count = torch.zeros(total, dtype=torch.long, device=device)
            velocity_saturation_count = torch.zeros(total, dtype=torch.long, device=device)
            torque_saturation_run = torch.zeros(total, dtype=torch.long, device=device)
            velocity_saturation_run = torch.zeros(total, dtype=torch.long, device=device)
            max_torque_saturation_run = torch.zeros(total, dtype=torch.long, device=device)
            max_velocity_saturation_run = torch.zeros(total, dtype=torch.long, device=device)
            torque_saturation_count_by_joint = torch.zeros((total, 14), dtype=torch.long, device=device)
            velocity_saturation_count_by_joint = torch.zeros((total, 14), dtype=torch.long, device=device)
            torque_saturation_run_by_joint = torch.zeros((total, 14), dtype=torch.long, device=device)
            velocity_saturation_run_by_joint = torch.zeros((total, 14), dtype=torch.long, device=device)
            max_torque_saturation_run_by_joint = torch.zeros((total, 14), dtype=torch.long, device=device)
            max_velocity_saturation_run_by_joint = torch.zeros((total, 14), dtype=torch.long, device=device)
            tail_frame_count = torch.zeros(total, dtype=torch.long, device=device)
            tail_both_feet_count = torch.zeros(total, dtype=torch.long, device=device)
            tail_root_ang_vel_square_sum = torch.zeros(total, device=device)
            tail_root_lin_vel_square_sum = torch.zeros(total, device=device)
            tail_action_sum_by_joint = torch.zeros((total, 14), device=device)
            tail_action_square_sum_by_joint = torch.zeros((total, 14), device=device)
            tail_min_root_height = torch.full((total,), float("inf"), device=device)
            tail_max_base_torque = torch.zeros(total, device=device)
            video_frames = []
            termination_counts = {
                name: torch.zeros(total, dtype=torch.long, device=device)
                for name in unwrapped.termination_manager.active_terms
            }

            print(
                f"[strike-leg-safety] stepping swing={swing_steps} tail={tail_steps} total={policy_steps}",
                flush=True,
            )

            feet_sensor = unwrapped.scene.sensors["contact_forces"]
            feet_ids = feet_sensor.find_bodies(["left_ankle_roll_Link", "right_ankle_roll_Link"], preserve_order=True)[0]
            base_joint_ids = torch.as_tensor(action_term._base_joint_ids, device=device, dtype=torch.long)
            raw_clip = float(action_term.cfg.raw_clip)

            for step in range(policy_steps):
                if step % hold_steps == 0:
                    held_random.uniform_(-action_bound, action_bound, generator=generator)
                current_phase = motion.time_steps
                random_active = current_phase[random_slice] >= pair_handoffs
                action = torch.zeros((total, 14), device=device)
                if policy is None and constant_leg_action is None:
                    action[random_slice, :12] = held_random * random_active.unsqueeze(-1)
                elif constant_leg_action is not None:
                    action[random_slice, :12] = constant_leg_action * random_active.unsqueeze(-1)
                else:
                    with torch.inference_mode():
                        action[random_slice] = policy(policy_obs)[random_slice]
                        # The renderer follows env_0, which belongs to the zero branch in
                        # paired audits.  For a visual-only candidate replay, mirror the
                        # learned action into that branch.  Metrics from this mode are
                        # intentionally not used as a paired comparison.
                        if render_candidate_only:
                            action[zero_slice] = policy(policy_obs)[zero_slice]
                sampled_exceed_count[random_slice] += (torch.abs(action[random_slice, :12]) >= raw_clip).sum(dim=-1)
                active_action_values[random_slice] += random_active.to(torch.long) * 12

                try:
                    _obs, _reward, terminated, truncated, _extras = env.step(action)
                except BaseException as error:
                    print(
                        f"[strike-leg-safety] env.step failed at step={step + 1}: "
                        f"{type(error).__name__}: {error}",
                        flush=True,
                    )
                    raise
                if video_path is not None:
                    frame = env.render()
                    if frame is not None:
                        video_frames.append(frame)
                if vec_env is not None:
                    policy_obs = _policy_observation(vec_env)
                effective = action_term.raw_actions
                active_now = active.clone()
                torque = torch.abs(robot.data.applied_torque[:, base_joint_ids]).max(dim=-1).values
                max_torque = torch.where(active_now, torch.maximum(max_torque, torque), max_torque)
                torque_saturated = torch.abs(robot.data.applied_torque[:, base_joint_ids]) >= (
                    0.95 * robot.data.joint_effort_limits[:, base_joint_ids]
                )
                velocity_saturated = torch.abs(robot.data.joint_vel[:, base_joint_ids]) >= (
                    0.95 * robot.data.joint_vel_limits[:, base_joint_ids]
                )
                torque_saturation_count += torque_saturated.sum(dim=-1) * active_now
                velocity_saturation_count += velocity_saturated.sum(dim=-1) * active_now
                torque_saturation_count_by_joint += torque_saturated * active_now.unsqueeze(-1)
                velocity_saturation_count_by_joint += velocity_saturated * active_now.unsqueeze(-1)
                torque_saturation_run_by_joint = torch.where(
                    active_now.unsqueeze(-1) & torque_saturated,
                    torque_saturation_run_by_joint + 1,
                    torch.zeros_like(torque_saturation_run_by_joint),
                )
                velocity_saturation_run_by_joint = torch.where(
                    active_now.unsqueeze(-1) & velocity_saturated,
                    velocity_saturation_run_by_joint + 1,
                    torch.zeros_like(velocity_saturation_run_by_joint),
                )
                max_torque_saturation_run_by_joint = torch.maximum(
                    max_torque_saturation_run_by_joint, torque_saturation_run_by_joint
                )
                max_velocity_saturation_run_by_joint = torch.maximum(
                    max_velocity_saturation_run_by_joint, velocity_saturation_run_by_joint
                )
                torque_saturation_run = torch.where(
                    active_now & torque_saturated.any(dim=-1), torque_saturation_run + 1, torch.zeros_like(torque_saturation_run)
                )
                velocity_saturation_run = torch.where(
                    active_now & velocity_saturated.any(dim=-1), velocity_saturation_run + 1, torch.zeros_like(velocity_saturation_run)
                )
                max_torque_saturation_run = torch.maximum(max_torque_saturation_run, torque_saturation_run)
                max_velocity_saturation_run = torch.maximum(max_velocity_saturation_run, velocity_saturation_run)
                limits = robot.data.soft_joint_pos_limits[:, base_joint_ids]
                q = robot.data.joint_pos[:, base_joint_ids]
                margin = torch.minimum(q - limits[..., 0], limits[..., 1] - q).min(dim=-1).values
                min_joint_margin = torch.where(active_now, torch.minimum(min_joint_margin, margin), min_joint_margin)
                force = torch.linalg.vector_norm(feet_sensor.data.net_forces_w[:, feet_ids], dim=-1)
                contact_sum += (force > 10.0).float().mean(dim=-1) * active_now
                torso_id = robot.body_names.index("torso_Link")
                torso_omega_square_sum += torch.sum(torch.square(robot.data.body_ang_vel_w[:, torso_id]), dim=-1) * active_now
                root_height_abs_sum += torch.abs(robot.data.root_pos_w[:, 2]) * active_now
                raw_square_sum += torch.sum(torch.square(effective), dim=-1) * active_now
                raw_action_sum_by_joint += effective * active_now.unsqueeze(-1)
                raw_action_square_sum_by_joint += torch.square(effective) * active_now.unsqueeze(-1)
                active_frame_count += active_now.to(torch.long)
                effective_clip_count += ((torch.abs(effective) >= raw_clip - 1.0e-6).sum(dim=-1) * active_now)
                waist_max = torch.maximum(waist_max, torch.abs(effective[:, 12:]).max(dim=-1).values)
                tail_active = motion.time_steps >= (motion.motion.motion_lengths[motion.motion_ids] - 1)
                tail_active = tail_active & active_now
                tail_frame_count += tail_active.to(torch.long)
                tail_both_feet_count += ((force > 10.0).all(dim=-1) & tail_active).to(torch.long)
                tail_root_ang_vel_square_sum += torch.sum(torch.square(robot.data.root_ang_vel_b), dim=-1) * tail_active
                tail_root_lin_vel_square_sum += torch.sum(torch.square(robot.data.root_lin_vel_b), dim=-1) * tail_active
                tail_action_sum_by_joint += effective * tail_active.unsqueeze(-1)
                tail_action_square_sum_by_joint += torch.square(effective) * tail_active.unsqueeze(-1)
                tail_min_root_height = torch.where(
                    tail_active,
                    torch.minimum(tail_min_root_height, robot.data.root_pos_w[:, 2]),
                    tail_min_root_height,
                )
                tail_max_base_torque = torch.where(
                    tail_active, torch.maximum(tail_max_base_torque, torque), tail_max_base_torque
                )
                finite = finite and bool(
                    torch.isfinite(robot.data.root_state_w).all()
                    and torch.isfinite(robot.data.joint_pos).all()
                    and torch.isfinite(robot.data.joint_vel).all()
                    and torch.isfinite(effective).all()
                )
                done = (terminated | truncated) & active
                failure_count += (terminated & active).to(torch.long)
                for name, count in termination_counts.items():
                    count += (unwrapped.termination_manager.get_term(name) & done).to(torch.long)
                active[done] = False
                if step == 0 or (step + 1) % 20 == 0 or step + 1 == policy_steps:
                    print(
                        f"[strike-leg-safety] step={step + 1}/{policy_steps} active={int(active.sum())} finite={finite}",
                        flush=True,
                    )

            denom = float(policy_steps)
            action_joint_names = list(action_term.cfg.base_joint_names)

            def saturation_by_joint(count, max_run, mode_slice):
                return {
                    action_joint_names[index]: {
                        "fraction": float(count[mode_slice, index].sum().item() / (pairs * policy_steps)),
                        "max_consecutive_steps": int(max_run[mode_slice, index].max().item()),
                    }
                    for index in range(14)
                }

            def tail_metrics(mode_slice):
                count = tail_frame_count[mode_slice].clamp(min=1)
                return {
                    "observed_frames": _stats(tail_frame_count[mode_slice]),
                    "both_feet_contact_fraction": _stats(tail_both_feet_count[mode_slice].float() / count),
                    "root_angular_velocity_rms": _stats(
                        torch.sqrt(tail_root_ang_vel_square_sum[mode_slice] / count)
                    ),
                    "root_linear_velocity_rms": _stats(
                        torch.sqrt(tail_root_lin_vel_square_sum[mode_slice] / count)
                    ),
                    "minimum_root_height_m": _stats(tail_min_root_height[mode_slice]),
                    "max_base_torque_nm": _stats(tail_max_base_torque[mode_slice]),
                }

            def termination_metrics(mode_slice):
                return {name: int(count[mode_slice].sum().item()) for name, count in termination_counts.items()}

            def action_by_joint(mode_slice, *, tail_only: bool = False):
                if tail_only:
                    count = tail_frame_count[mode_slice].clamp(min=1).unsqueeze(-1)
                    action_sum = tail_action_sum_by_joint[mode_slice]
                    action_square_sum = tail_action_square_sum_by_joint[mode_slice]
                else:
                    count = active_frame_count[mode_slice].clamp(min=1).unsqueeze(-1)
                    action_sum = raw_action_sum_by_joint[mode_slice]
                    action_square_sum = raw_action_square_sum_by_joint[mode_slice]
                mean = action_sum / count
                rms = torch.sqrt(action_square_sum / count)
                return {
                    action_joint_names[index]: {
                        "mean": float(mean[:, index].mean().item()),
                        "rms": float(rms[:, index].mean().item()),
                    }
                    for index in range(14)
                }

            metrics = {
                "zero_residual": {
                    "survival_fraction": float((failure_count[zero_slice] == 0).float().mean().item()),
                    "max_base_torque_nm": _stats(max_torque[zero_slice]),
                    "min_soft_joint_margin_rad": _stats(min_joint_margin[zero_slice]),
                    "foot_contact_fraction": _stats(contact_sum[zero_slice] / denom),
                    "torso_angular_velocity_rms": _stats(torch.sqrt(torso_omega_square_sum[zero_slice] / denom)),
                    "torque_saturation_fraction": float(torque_saturation_count[zero_slice].sum().item() / (pairs * policy_steps * 14)),
                    "velocity_saturation_fraction": float(velocity_saturation_count[zero_slice].sum().item() / (pairs * policy_steps * 14)),
                    "max_consecutive_torque_saturation_steps": int(max_torque_saturation_run[zero_slice].max().item()),
                    "max_consecutive_velocity_saturation_steps": int(max_velocity_saturation_run[zero_slice].max().item()),
                    "termination_count_by_reason": termination_metrics(zero_slice),
                    "torque_saturation_by_joint": saturation_by_joint(
                        torque_saturation_count_by_joint, max_torque_saturation_run_by_joint, zero_slice
                    ),
                    "velocity_saturation_by_joint": saturation_by_joint(
                        velocity_saturation_count_by_joint, max_velocity_saturation_run_by_joint, zero_slice
                    ),
                    "post_swing_tail": tail_metrics(zero_slice),
                },
                candidate_label: {
                    "survival_fraction": float((failure_count[random_slice] == 0).float().mean().item()),
                    "max_base_torque_nm": _stats(max_torque[random_slice]),
                    "min_soft_joint_margin_rad": _stats(min_joint_margin[random_slice]),
                    "foot_contact_fraction": _stats(contact_sum[random_slice] / denom),
                    "torso_angular_velocity_rms": _stats(torch.sqrt(torso_omega_square_sum[random_slice] / denom)),
                    "raw_action_rms": float(torch.sqrt(raw_square_sum[random_slice].sum() / (pairs * policy_steps * 14)).item()),
                    "raw_action_by_joint": action_by_joint(random_slice),
                    "tail_action_by_joint": action_by_joint(random_slice, tail_only=True),
                    "sampled_raw_exceed_fraction": float(sampled_exceed_count[random_slice].sum().item() / max(int(active_action_values[random_slice].sum().item()), 1)),
                    "execution_clip_fraction": float(effective_clip_count[random_slice].sum().item() / (pairs * policy_steps * 14)),
                    "waist_action_max": float(waist_max[random_slice].max().item()),
                    "torque_saturation_fraction": float(torque_saturation_count[random_slice].sum().item() / (pairs * policy_steps * 14)),
                    "velocity_saturation_fraction": float(velocity_saturation_count[random_slice].sum().item() / (pairs * policy_steps * 14)),
                    "max_consecutive_torque_saturation_steps": int(max_torque_saturation_run[random_slice].max().item()),
                    "max_consecutive_velocity_saturation_steps": int(max_velocity_saturation_run[random_slice].max().item()),
                    "termination_count_by_reason": termination_metrics(random_slice),
                    "torque_saturation_by_joint": saturation_by_joint(
                        torque_saturation_count_by_joint, max_torque_saturation_run_by_joint, random_slice
                    ),
                    "velocity_saturation_by_joint": saturation_by_joint(
                        velocity_saturation_count_by_joint, max_velocity_saturation_run_by_joint, random_slice
                    ),
                    "post_swing_tail": tail_metrics(random_slice),
                },
            }
            runtime_safe = bool(
                finite
                and metrics["zero_residual"]["survival_fraction"] == 1.0
                and metrics[candidate_label]["survival_fraction"] == 1.0
                and metrics[candidate_label]["sampled_raw_exceed_fraction"] == 0.0
                and metrics[candidate_label]["execution_clip_fraction"] < 0.01
                and metrics[candidate_label]["waist_action_max"] <= 1.0e-7
                and metrics[candidate_label]["max_consecutive_torque_saturation_steps"]
                <= metrics["zero_residual"]["max_consecutive_torque_saturation_steps"] + 1
                and metrics[candidate_label]["max_consecutive_velocity_saturation_steps"]
                <= metrics["zero_residual"]["max_consecutive_velocity_saturation_steps"] + 1
                and metrics[candidate_label]["post_swing_tail"]["observed_frames"]["min"]
                >= max(tail_steps - 1, 1)
                and metrics[candidate_label]["post_swing_tail"]["both_feet_contact_fraction"]["min"] == 1.0
            )
            result = {
                "schema_version": 1,
                "audit_id": "strike_stabilizer_a_untrained_bounded_leg_safety_smoke_v1",
                "task": task_id,
                "simulation_only": True,
                "training_authorization": False,
                "pairs": pairs,
                "policy_steps": policy_steps,
                "swing_steps": swing_steps,
                "post_swing_tail_steps": tail_steps,
                "random_action_bound": action_bound,
                "constant_leg_action": (
                    constant_leg_action.detach().cpu().tolist() if constant_leg_action is not None else None
                ),
                "hold_steps": hold_steps,
                "seed": seed,
                "case_offset": case_offset,
                "candidate_mode": candidate_label,
                "motion_ids_per_pair": pair_motion_ids.detach().cpu().tolist(),
                "handoff_steps_per_pair": pair_handoffs.detach().cpu().tolist(),
                "metrics": metrics,
                "finite": finite,
                "runtime_safety_smoke_passed": runtime_safe,
                "notes": [
                    "The policy is never trained or loaded in this audit.",
                    "Random actions are applied only to the 12 leg channels after a continuous-prefix handoff.",
                    "Both waist residual channels remain structurally zero.",
                    "This smoke is not a PPO or checkpoint approval.",
                ],
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            if video_path is not None:
                import imageio.v2 as imageio

                video_path.parent.mkdir(parents=True, exist_ok=True)
                imageio.mimsave(video_path, video_frames, fps=50)
                print(f"[strike-leg-safety] wrote {len(video_frames)} frames -> {video_path}", flush=True)
            print(json.dumps({"passed": runtime_safe, "output": str(output_path)}, ensure_ascii=False))
            return result
        finally:
            env.close()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
