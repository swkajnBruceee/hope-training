#!/usr/bin/env python3
"""Deterministic full-cycle audit for the ready-pose Strike Stabilizer.

This intentionally does *not* restore a legacy floating-base motion state.
Every case starts from the current task's physical strike-ready reset pose,
passes through the configured prelude and swing, and retains the post-swing
tail. Each manifest motion is paired against the zero-residual baseline.
Optional reset-level perturbations are generated once and mirrored across each
pair, never injected during a swing.
"""

from __future__ import annotations

import json
import pathlib
import sys

import hydra
from omegaconf import OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from train import _apply_task_overrides


def _mean(value):
    return float(value.float().mean().item())


@hydra.main(version_base=None, config_path="../cfg", config_name="play")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher

    # Keep the launcher object alive for the entire evaluation.  Retaining
    # only ``.app`` lets Python collect AppLauncher and close Isaac before the
    # first env.step(), which looks like a clean but empty evaluation.
    app_launcher = AppLauncher(headless=bool(cfg.headless), device=str(cfg.device))
    app = app_launcher.app
    try:
        import gymnasium as gym
        import torch
        from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
        from isaaclab_tasks.utils import parse_env_cfg
        from rsl_rl.runners import OnPolicyRunner

        import training.tasks  # noqa: F401 -- gym registration
        from training.utils.ppo_cfg import runner_kwargs

        checkpoint = cfg.get("checkpoint", None)
        if checkpoint is None:
            raise ValueError("Pass checkpoint=<.../model_N.pt>")
        checkpoint_path = pathlib.Path(str(checkpoint)).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = pathlib.Path.cwd() / checkpoint_path
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

        task_id = str(cfg.task.gym_task)
        cases = int(cfg.get("cases", 8))
        if cases < 1:
            raise ValueError("cases must be positive")
        # Pin the task reset as well as Torch's action-free deterministic
        # inference.  Without this, identical 8-motion audits may differ in
        # reset-side event sampling and cannot serve as a checkpoint gate.
        seed = int(cfg.get("seed", 20260722))
        env_cfg = parse_env_cfg(task_id, device=str(cfg.device), num_envs=2 * cases)
        _apply_task_overrides(env_cfg, cfg.task)
        env_cfg.sim.device = str(cfg.device)
        env_cfg.seed = seed
        # Match scripts/train.py and scripts/play.py exactly: the task YAML
        # owns the local manifest, but parse_env_cfg does not inject it into
        # the command term by itself.
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
        perturbation_mode = str(cfg.get("perturbation", "none")).lower()
        fixed_perturbation = str(cfg.get("fixed_perturbation", "")).lower()
        record_traces = bool(cfg.get("record_traces", False))
        record_obs_zscores = bool(cfg.get("record_obs_zscores", False))
        fixed_modes = {
            "forward_velocity", "backward_velocity", "left_velocity", "right_velocity",
            "positive_pitch_rate", "negative_pitch_rate", "positive_roll_rate", "negative_roll_rate",
            "left_leg_pose", "right_leg_pose",
        }
        if perturbation_mode not in {"none", "small", "fixed"}:
            raise ValueError("perturbation must be 'none', 'small', or 'fixed'")
        if perturbation_mode == "fixed" and fixed_perturbation not in fixed_modes:
            raise ValueError(
                "fixed_perturbation must be one of " + ", ".join(sorted(fixed_modes))
            )
        env = gym.make(task_id, cfg=env_cfg)
        try:
            vec_env = RslRlVecEnvWrapper(env)
            agent_cfg = RslRlOnPolicyRunnerCfg(
                **runner_kwargs(OmegaConf.to_container(cfg.algo, resolve=True), str(cfg.task.experiment_name))
            )
            agent_cfg.device = str(cfg.device)
            runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(str(checkpoint_path), load_optimizer=False)
            policy = runner.get_inference_policy(device=env.unwrapped.device)

            # RslRlVecEnvWrapper constructs and resets the managed environment
            # before exposing observations.  Do not issue a second raw Gym
            # reset here: that reset path can block after the wrapper has
            # initialized Isaac's command managers.  The task reset state is
            # already the deterministic strike-ready pose; we only override
            # the future motion ids below.
            torch.manual_seed(seed)
            unwrapped = env.unwrapped
            device = unwrapped.device
            motion = unwrapped.command_manager.get_term("motion")
            racket = unwrapped.command_manager.get_term("racket_target")
            robot = unwrapped.scene["robot"]
            action_term = unwrapped.action_manager.get_term("joint_pos")

            motion_ids = torch.arange(cases, dtype=torch.long, device=device) % motion.motion.num_motions
            paired_ids = torch.cat((motion_ids, motion_ids))
            # Reset has already placed every environment at the configured
            # ready pose.  Only choose the future reference; never teleport.
            motion.motion_ids[:] = paired_ids
            motion.time_steps.zero_()
            motion.tail_steps.zero_()
            motion.prelude_elapsed_steps.zero_()
            unwrapped.strike_stabilizer_handoff_steps = torch.zeros(2 * cases, dtype=torch.long, device=device)
            env_ids = torch.arange(2 * cases, device=device)
            racket._resample_command(env_ids)
            racket._compute_strike_timing()

            # A seed by itself is not a meaningful robustness audit when the
            # task reset is deterministic. Build a small physical root-state
            # bank once, then mirror every case across zero/learned branches.
            perturbations = {
                "mode": perturbation_mode,
                "fixed_perturbation": fixed_perturbation if perturbation_mode == "fixed" else None,
                "root_roll_rad": [0.0] * cases,
                "root_pitch_rad": [0.0] * cases,
                "root_linear_velocity_xy_m_s": [[0.0, 0.0]] * cases,
                "root_angular_velocity_xy_rad_s": [[0.0, 0.0]] * cases,
                "leg_joint_position_offset_rad": [[0.0] * 12] * cases,
            }
            if perturbation_mode in {"small", "fixed"}:
                from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

                root_roll_pitch = torch.zeros((cases, 2), device=device)
                root_lin_xy = torch.zeros((cases, 2), device=device)
                root_ang_xy = torch.zeros((cases, 2), device=device)
                leg_joint_offset = torch.zeros((cases, 12), device=device)
                if perturbation_mode == "small":
                    generator = torch.Generator(device=device)
                    generator.manual_seed(seed)
                    root_roll_pitch.uniform_(-0.020, 0.020, generator=generator)
                    root_lin_xy.uniform_(-0.080, 0.080, generator=generator)
                    root_ang_xy.uniform_(-0.150, 0.150, generator=generator)
                else:
                    # Fixed probes isolate one recoverable reset error at a
                    # time.  Velocity/rate magnitudes match the small-bank
                    # envelope.  The leg probes are coherent 6-DOF encoder
                    # offsets, not a hand-authored recovery action.
                    if fixed_perturbation == "forward_velocity":
                        root_lin_xy[:, 0] = 0.080
                    elif fixed_perturbation == "backward_velocity":
                        root_lin_xy[:, 0] = -0.080
                    elif fixed_perturbation == "left_velocity":
                        root_lin_xy[:, 1] = 0.080
                    elif fixed_perturbation == "right_velocity":
                        root_lin_xy[:, 1] = -0.080
                    elif fixed_perturbation == "positive_pitch_rate":
                        root_ang_xy[:, 1] = 0.150
                    elif fixed_perturbation == "negative_pitch_rate":
                        root_ang_xy[:, 1] = -0.150
                    elif fixed_perturbation == "positive_roll_rate":
                        root_ang_xy[:, 0] = 0.150
                    elif fixed_perturbation == "negative_roll_rate":
                        root_ang_xy[:, 0] = -0.150
                    elif fixed_perturbation == "left_leg_pose":
                        leg_joint_offset[:, :6] = torch.tensor(
                            (-0.015, 0.010, 0.010, 0.030, -0.015, 0.010), device=device
                        )
                    elif fixed_perturbation == "right_leg_pose":
                        leg_joint_offset[:, 6:12] = torch.tensor(
                            (-0.015, -0.010, -0.010, 0.030, -0.015, -0.010), device=device
                        )
                paired_roll_pitch = torch.cat((root_roll_pitch, root_roll_pitch))
                paired_lin_xy = torch.cat((root_lin_xy, root_lin_xy))
                paired_ang_xy = torch.cat((root_ang_xy, root_ang_xy))
                paired_leg_joint_offset = torch.cat((leg_joint_offset, leg_joint_offset))
                root_state = robot.data.root_state_w.clone()
                orientation_delta = quat_from_euler_xyz(
                    paired_roll_pitch[:, 0], paired_roll_pitch[:, 1], torch.zeros(2 * cases, device=device)
                )
                root_state[:, 3:7] = quat_mul(orientation_delta, root_state[:, 3:7])
                root_state[:, 7:9] += paired_lin_xy
                root_state[:, 10:12] += paired_ang_xy
                robot.write_root_state_to_sim(root_state, env_ids=env_ids)
                if torch.any(paired_leg_joint_offset != 0.0):
                    joint_pos = robot.data.joint_pos.clone()
                    joint_vel = robot.data.joint_vel.clone()
                    joint_pos[:, action_term._base_joint_ids_tensor[:12]] += paired_leg_joint_offset
                    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
                perturbations = {
                    "mode": perturbation_mode,
                    "fixed_perturbation": fixed_perturbation if perturbation_mode == "fixed" else None,
                    "root_roll_rad": root_roll_pitch[:, 0].detach().cpu().tolist(),
                    "root_pitch_rad": root_roll_pitch[:, 1].detach().cpu().tolist(),
                    "root_linear_velocity_xy_m_s": root_lin_xy.detach().cpu().tolist(),
                    "root_angular_velocity_xy_rad_s": root_ang_xy.detach().cpu().tolist(),
                    "leg_joint_position_offset_rad": leg_joint_offset.detach().cpu().tolist(),
                }

            obs = vec_env.get_observations()
            if isinstance(obs, tuple):
                obs = obs[0]
            obs = obs.to(device)

            # The task may contain a final-pose hold, a return-to-ready phase,
            # and a final ready hold.  The environment's episode horizon is
            # the only authoritative full-cycle duration.
            policy_steps = int(unwrapped.max_episode_length) + 1
            print(
                f"[strike-checkpoint-eval] cases={cases} policy_steps={policy_steps} "
                f"motion_ids={motion_ids.detach().cpu().tolist()} perturbation={perturbation_mode}",
                flush=True,
            )
            active = torch.ones(2 * cases, dtype=torch.bool, device=device)
            completed = torch.zeros(2 * cases, dtype=torch.bool, device=device)
            failed = torch.zeros(2 * cases, dtype=torch.bool, device=device)
            min_height = torch.full((2 * cases,), float("inf"), device=device)
            tail_lin_sq = torch.zeros(2 * cases, device=device)
            tail_ang_sq = torch.zeros(2 * cases, device=device)
            tail_count = torch.zeros(2 * cases, dtype=torch.long, device=device)
            actor_leg_action_sq = torch.zeros(2 * cases, device=device)
            executed_leg_action_sq = torch.zeros(2 * cases, device=device)
            action_count = torch.zeros(2 * cases, dtype=torch.long, device=device)
            actor_leg_trust_exceed_count = torch.zeros(2 * cases, dtype=torch.long, device=device)
            actor_leg_envelope_exceed_count = torch.zeros(2 * cases, dtype=torch.long, device=device)
            executed_leg_saturation_count = torch.zeros(2 * cases, dtype=torch.long, device=device)
            phase_names = ("prelude", "swing", "final_hold", "return", "ready_hold")
            phase_count = torch.zeros(2 * cases, len(phase_names), dtype=torch.long, device=device)
            phase_actor_trust_exceed = torch.zeros(2 * cases, len(phase_names), 12, dtype=torch.long, device=device)
            phase_unbounded_trust_exceed = torch.zeros_like(phase_actor_trust_exceed)
            phase_executed_saturation = torch.zeros_like(phase_actor_trust_exceed)
            phase_actor_execution_gap_sq = torch.zeros(2 * cases, len(phase_names), device=device)
            phase_root_lin_sq = torch.zeros_like(phase_actor_execution_gap_sq)
            phase_root_ang_sq = torch.zeros_like(phase_actor_execution_gap_sq)
            # Optional read-only contract audit.  These statistics use the
            # checkpoint's frozen empirical normalizer, so they report how
            # far each rollout observation lies from the distribution seen
            # during training.  They never feed back into the policy or env.
            obs_dim = int(obs.shape[-1])
            phase_obs_abs_z_sum = torch.zeros(
                2 * cases, len(phase_names), obs_dim, device=device
            ) if record_obs_zscores else None
            phase_obs_abs_z_max = torch.zeros_like(phase_obs_abs_z_sum) if record_obs_zscores else None
            phase_obs_z_ge_3_count = torch.zeros_like(phase_obs_abs_z_sum, dtype=torch.long) if record_obs_zscores else None
            phase_obs_z_ge_5_count = torch.zeros_like(phase_obs_abs_z_sum, dtype=torch.long) if record_obs_zscores else None
            failure_step = torch.full((2 * cases,), -1, dtype=torch.long, device=device)
            failure_phase = torch.full((2 * cases,), -1, dtype=torch.long, device=device)
            learned_traces = [[] for _ in range(cases)] if record_traces else None
            motion_lengths = motion.motion.motion_lengths[motion.motion_ids]
            prelude_steps = int(motion.cfg.prelude_steps)
            hold_steps = int(motion.cfg.hold_last_frame_steps)
            return_steps = int(motion.cfg.return_to_default_steps)
            reasons = {
                name: torch.zeros(2 * cases, dtype=torch.long, device=device)
                for name in unwrapped.termination_manager.active_terms
            }
            policy_obs_terms = (
                ("base_lin_vel", 3),
                ("base_ang_vel", 3),
                ("joint_pos", 14),
                ("joint_vel", 14),
                ("previous_action", 14),
                ("projected_gravity", 3),
                ("racket_target_pos_b", 3),
                ("racket_target_vel_b", 3),
                ("racket_target_normal_b", 3),
                ("racket_pos_b", 3),
                ("racket_lin_vel_b", 3),
                ("racket_normal_b", 3),
                ("time_to_strike", 1),
                ("swing_type", 1),
                ("strike_joint_pos", 9),
                ("strike_joint_vel", 9),
                ("strike_reference_joint_pos", 9),
                ("strike_reference_joint_vel", 9),
                ("strike_reference_joint_vel_8", 9),
                ("strike_reference_joint_vel_16", 9),
                ("strike_phase", 1),
            )
            if record_obs_zscores and sum(width for _, width in policy_obs_terms) != obs_dim:
                raise RuntimeError(
                    f"Policy observation schema width {sum(width for _, width in policy_obs_terms)} "
                    f"does not match runtime obs width {obs_dim}"
                )

            for step in range(policy_steps):
                phase_index = torch.full((2 * cases,), 4, dtype=torch.long, device=device)
                phase_index[step < prelude_steps] = 0
                swing_end = prelude_steps + motion_lengths
                hold_end = swing_end + hold_steps
                return_end = hold_end + return_steps
                phase_index[(step >= prelude_steps) & (step < swing_end)] = 1
                phase_index[(step >= swing_end) & (step < hold_end)] = 2
                phase_index[(step >= hold_end) & (step < return_end)] = 3
                with torch.inference_mode():
                    if record_obs_zscores:
                        normalized_obs = runner.obs_normalizer(obs) if runner.empirical_normalization else obs
                        abs_z = torch.abs(normalized_obs)
                        for phase_id in range(len(phase_names)):
                            in_phase = active & (phase_index == phase_id)
                            mask = in_phase.unsqueeze(-1)
                            phase_obs_abs_z_sum[:, phase_id] += abs_z * mask
                            phase_obs_abs_z_max[:, phase_id] = torch.maximum(
                                phase_obs_abs_z_max[:, phase_id], abs_z * mask
                            )
                            phase_obs_z_ge_3_count[:, phase_id] += (
                                (abs_z >= 3.0).to(torch.long) * mask
                            )
                            phase_obs_z_ge_5_count[:, phase_id] += (
                                (abs_z >= 5.0).to(torch.long) * mask
                            )
                    learned_action = policy(obs)
                action = torch.zeros_like(learned_action)
                action[cases:] = learned_action[cases:]
                before_active = active.clone()
                tail = (motion.time_steps >= (motion.motion.motion_lengths[motion.motion_ids] - 1)) & before_active
                # ``action`` is the candidate action for this policy step.
                # action_term.raw_actions still describes the preceding step
                # until env.step() processes the new command.
                actor_leg_action_sq += torch.sum(torch.square(action[:, :12]), dim=-1) * before_active
                action_count += before_active.to(torch.long)
                actor_leg_trust_exceed_count += (
                    (torch.abs(action[:, :12]) >= 0.20).sum(dim=-1)
                    * before_active
                )
                actor_leg_envelope_exceed_count += (
                    (torch.abs(action[:, :12]) >= float(action_term.cfg.raw_clip) - 1.0e-6).sum(dim=-1)
                    * before_active
                )
                # ``env.step`` immediately resets terminated environments.
                # Keep the pre-step state for failure traces so the final
                # record describes the falling robot rather than its reset
                # ready pose.
                pre_root_height = robot.data.root_pos_w[:, 2].clone()
                pre_root_linear_velocity_b = robot.data.root_lin_vel_b.clone()
                pre_root_angular_velocity_b = robot.data.root_ang_vel_b.clone()
                for phase_id in range(len(phase_names)):
                    in_phase = before_active & (phase_index == phase_id)
                    phase_count[:, phase_id] += in_phase.to(torch.long)
                    phase_actor_trust_exceed[:, phase_id] += (
                        (torch.abs(action[:, :12]) >= 0.20).to(torch.long) * in_phase.unsqueeze(-1)
                    )

                # Run through the same RSL-RL wrapper used for training.  The
                # raw Gym environment returns a dict observation here, while
                # the policy needs the wrapper's tensor observation.
                next_obs, _, done_flags, _ = vec_env.step(action)
                executed_leg_action = action_term.raw_actions[:, :12]
                executed_leg_action_sq += torch.sum(torch.square(executed_leg_action), dim=-1) * before_active
                # A tanh-bound action never equals raw_clip exactly.  Count
                # the top 5% of the physical envelope as saturation so the
                # metric remains meaningful for both clip and smooth modes.
                executed_leg_saturation_count += (
                    (torch.abs(executed_leg_action) >= 0.95 * float(action_term.cfg.raw_clip)).sum(dim=-1)
                    * before_active
                )
                unbounded_leg_action = action_term.unbounded_actions[:, :12]
                for phase_id in range(len(phase_names)):
                    in_phase = before_active & (phase_index == phase_id)
                    phase_unbounded_trust_exceed[:, phase_id] += (
                        (torch.abs(unbounded_leg_action) >= 0.20).to(torch.long) * in_phase.unsqueeze(-1)
                    )
                    phase_executed_saturation[:, phase_id] += (
                        (torch.abs(executed_leg_action) >= 0.95 * float(action_term.cfg.raw_clip)).to(torch.long)
                        * in_phase.unsqueeze(-1)
                    )
                    phase_actor_execution_gap_sq[:, phase_id] += (
                        torch.mean(torch.square(action[:, :12] - executed_leg_action), dim=-1) * in_phase
                    )
                    phase_root_lin_sq[:, phase_id] += (
                        torch.sum(torch.square(robot.data.root_lin_vel_b), dim=-1) * in_phase
                    )
                    phase_root_ang_sq[:, phase_id] += (
                        torch.sum(torch.square(robot.data.root_ang_vel_b), dim=-1) * in_phase
                    )
                root_height = robot.data.root_pos_w[:, 2]
                min_height = torch.where(before_active, torch.minimum(min_height, root_height), min_height)
                tail_lin_sq += torch.sum(torch.square(robot.data.root_lin_vel_b), dim=-1) * tail
                tail_ang_sq += torch.sum(torch.square(robot.data.root_ang_vel_b), dim=-1) * tail
                tail_count += tail.to(torch.long)
                done = done_flags.to(torch.bool) & before_active
                time_out = unwrapped.termination_manager.get_term("time_out").to(torch.bool)
                completed |= time_out & done
                failed |= (~time_out) & done
                new_failure = (~time_out) & done & (failure_step < 0)
                failure_step[new_failure] = step + 1
                failure_phase[new_failure] = phase_index[new_failure]
                for name, values in reasons.items():
                    values += (unwrapped.termination_manager.get_term(name) & done).to(torch.long)
                if learned_traces is not None:
                    for local_id, env_id in enumerate(range(cases, 2 * cases)):
                        learned_traces[local_id].append(
                            {
                                "step": step + 1,
                                "phase": phase_names[int(phase_index[env_id].item())],
                                "active_before_step": bool(before_active[env_id].item()),
                                "done": bool(done[env_id].item()),
                                "time_out": bool(time_out[env_id].item()),
                                "root_height_m": float(pre_root_height[env_id].item()),
                                "root_linear_velocity_b_m_s": pre_root_linear_velocity_b[env_id].detach().cpu().tolist(),
                                "root_angular_velocity_b_rad_s": pre_root_angular_velocity_b[env_id].detach().cpu().tolist(),
                                "policy_leg_action": action[env_id, :12].detach().cpu().tolist(),
                                "unbounded_leg_action": unbounded_leg_action[env_id].detach().cpu().tolist(),
                                "executed_leg_action": executed_leg_action[env_id].detach().cpu().tolist(),
                            }
                        )
                active[done] = False
                obs = next_obs.to(device)
                if (step + 1) % 50 == 0 or step + 1 == policy_steps:
                    print(
                        f"[strike-checkpoint-eval] step={step + 1}/{policy_steps} "
                        f"active={int(active.sum())}",
                        flush=True,
                    )

            def pack(selection: slice):
                indices = torch.arange(2 * cases, device=device)[selection]
                rows = []
                for env_id in indices.tolist():
                    phase_metrics = {}
                    for phase_id, phase_name in enumerate(phase_names):
                        count = max(int(phase_count[env_id, phase_id].item()), 1)
                        phase_metrics[phase_name] = {
                            "steps": int(phase_count[env_id, phase_id].item()),
                            "actor_trust_exceed_by_joint": (
                                phase_actor_trust_exceed[env_id, phase_id].float() / count
                            ).detach().cpu().tolist(),
                            "unbounded_trust_exceed_by_joint": (
                                phase_unbounded_trust_exceed[env_id, phase_id].float() / count
                            ).detach().cpu().tolist(),
                            "executed_saturation_by_joint": (
                                phase_executed_saturation[env_id, phase_id].float() / count
                            ).detach().cpu().tolist(),
                            "actor_execution_gap_rms": float(
                                torch.sqrt(phase_actor_execution_gap_sq[env_id, phase_id] / count).item()
                            ),
                            "root_linear_velocity_rms": float(
                                torch.sqrt(phase_root_lin_sq[env_id, phase_id] / count).item()
                            ),
                            "root_angular_velocity_rms": float(
                                torch.sqrt(phase_root_ang_sq[env_id, phase_id] / count).item()
                            ),
                        }
                        if record_obs_zscores:
                            phase_metrics[phase_name]["observation_abs_zscore"] = {
                                "mean_abs_by_dim": (
                                    phase_obs_abs_z_sum[env_id, phase_id] / count
                                ).detach().cpu().tolist(),
                                "max_abs_by_dim": phase_obs_abs_z_max[env_id, phase_id].detach().cpu().tolist(),
                                "fraction_ge_3_by_dim": (
                                    phase_obs_z_ge_3_count[env_id, phase_id].float() / count
                                ).detach().cpu().tolist(),
                                "fraction_ge_5_by_dim": (
                                    phase_obs_z_ge_5_count[env_id, phase_id].float() / count
                                ).detach().cpu().tolist(),
                            }
                    rows.append(
                        {
                            "motion_id": int(motion_ids[env_id % cases].item()),
                            "completed_timeout": bool(completed[env_id].item()),
                            "failed": bool(failed[env_id].item()),
                            "failure_step": int(failure_step[env_id].item()),
                            "failure_phase": (
                                phase_names[int(failure_phase[env_id].item())]
                                if int(failure_phase[env_id].item()) >= 0 else None
                            ),
                            "minimum_root_height_m": float(min_height[env_id].item()),
                            "tail_root_linear_velocity_rms": float(
                                torch.sqrt(tail_lin_sq[env_id] / tail_count[env_id].clamp(min=1)).item()
                            ),
                            "tail_root_angular_velocity_rms": float(
                                torch.sqrt(tail_ang_sq[env_id] / tail_count[env_id].clamp(min=1)).item()
                            ),
                            "actor_leg_action_rms": float(
                                torch.sqrt(actor_leg_action_sq[env_id] / action_count[env_id].clamp(min=1)).item()
                            ),
                            "actor_leg_trust_exceed_fraction": float(
                                actor_leg_trust_exceed_count[env_id].item() / max(int(action_count[env_id].item()) * 12, 1)
                            ),
                            "actor_leg_envelope_exceed_fraction": float(
                                actor_leg_envelope_exceed_count[env_id].item() / max(int(action_count[env_id].item()) * 12, 1)
                            ),
                            "executed_leg_action_rms": float(
                                torch.sqrt(executed_leg_action_sq[env_id] / action_count[env_id].clamp(min=1)).item()
                            ),
                            "executed_leg_saturation_fraction": float(
                                executed_leg_saturation_count[env_id].item() / max(int(action_count[env_id].item()) * 12, 1)
                            ),
                            "termination_reasons": {name: int(values[env_id].item()) for name, values in reasons.items()},
                            "phase_metrics": phase_metrics,
                        }
                    )
                return rows

            zero_rows, learned_rows = pack(slice(0, cases)), pack(slice(cases, 2 * cases))
            def summary(rows):
                return {
                    "cases": len(rows),
                    "completed_timeout_fraction": sum(x["completed_timeout"] for x in rows) / len(rows),
                    "failure_fraction": sum(x["failed"] for x in rows) / len(rows),
                    "minimum_root_height_m_mean": sum(x["minimum_root_height_m"] for x in rows) / len(rows),
                    "tail_root_linear_velocity_rms_mean": sum(x["tail_root_linear_velocity_rms"] for x in rows) / len(rows),
                    "tail_root_angular_velocity_rms_mean": sum(x["tail_root_angular_velocity_rms"] for x in rows) / len(rows),
                    "actor_leg_action_rms_mean": sum(x["actor_leg_action_rms"] for x in rows) / len(rows),
                    "actor_leg_trust_exceed_fraction_mean": sum(x["actor_leg_trust_exceed_fraction"] for x in rows) / len(rows),
                    "actor_leg_envelope_exceed_fraction_mean": sum(x["actor_leg_envelope_exceed_fraction"] for x in rows) / len(rows),
                    "executed_leg_action_rms_mean": sum(x["executed_leg_action_rms"] for x in rows) / len(rows),
                    "executed_leg_saturation_fraction_mean": sum(x["executed_leg_saturation_fraction"] for x in rows) / len(rows),
                }

            result = {
                "checkpoint": str(checkpoint_path),
                "seed": seed,
                "policy_mode": "deterministic_actor_mean",
                **({"policy_observation_terms": policy_obs_terms} if record_obs_zscores else {}),
                "task_contract": {
                    "sequence": "ready_pose -> prelude -> full_swing -> final_pose_hold -> smooth_return_to_ready -> ready_hold",
                    "prelude_steps": int(motion.cfg.prelude_steps),
                    "final_pose_hold_steps": int(motion.cfg.hold_last_frame_steps),
                    "return_to_default_steps": int(motion.cfg.return_to_default_steps),
                    "episode_length_steps": int(unwrapped.max_episode_length),
                },
                "policy_steps": policy_steps,
                "paired_reset_perturbations": perturbations,
                "zero_residual": {"summary": summary(zero_rows), "cases": zero_rows},
                "learned_policy": {
                    "summary": summary(learned_rows),
                    "cases": learned_rows,
                    **({"traces": learned_traces} if learned_traces is not None else {}),
                },
            }
            output = pathlib.Path(str(cfg.get("output_json", "eval_outputs/strike_stabilizer_a/model_eval.json")))
            if not output.is_absolute():
                output = pathlib.Path.cwd() / output
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(json.dumps({"output": str(output), "learned": result["learned_policy"]["summary"]}, ensure_ascii=False))
        finally:
            env.close()
    except BaseException:
        # Hydra can otherwise suppress the useful traceback while the Kit app
        # is being torn down; retain it for finite evaluation diagnostics.
        import traceback

        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
