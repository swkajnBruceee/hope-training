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

from train import _apply_task_overrides, _as_bool


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
        robot_posture = sum(1 for r in group if r[26]) / len(group)
        wrist = sum(1 for r in group if r[33]) / len(group)
        whole = sum(1 for r in group if r[34]) / len(group)
        pos_mean = sum(r[0] for r in group) / len(group)
        vel_mean = sum(r[1] for r in group) / len(group)
        normal_mean = sum(r[2] for r in group) / len(group)
        pelvis_margin_mean = sum(15.0 - r[9] for r in group) / len(group)
        torso_margin_mean = sum(20.0 - r[10] for r in group) / len(group)
        arm_margin_mean = sum(0.10 - r[12] for r in group) / len(group)
        worst = max(group, key=lambda r: (r[0], r[1], r[2]))
        print(
            f"[INFO] {name}: n={len(group)} hit_composite={hit_comp:.3f} posture_ref={posture:.3f} "
            f"robot_posture={robot_posture:.3f} wrist_naturalness={wrist:.3f} whole_cycle={whole:.3f} "
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


def _robot_posture_tier(
    *,
    hit_pass: bool,
    robot_posture_pass: bool,
    wrist_naturalness_pass: bool,
    arm_near_limit: float,
    torso_tilt_delta: float,
) -> str:
    if hit_pass and robot_posture_pass and wrist_naturalness_pass:
        return "A_robot_usable_candidate"
    if hit_pass and robot_posture_pass and not wrist_naturalness_pass:
        return "B_wrist_retarget_required"
    if hit_pass and arm_near_limit <= 0.10 and torso_tilt_delta <= 20.0:
        return "B_robot_borderline"
    if hit_pass:
        return "C_requires_stance_or_retarget"
    return "D_task_fail"


def _angle_between_deg(a, b):
    import torch

    denom = torch.linalg.norm(a, dim=-1) * torch.linalg.norm(b, dim=-1)
    denom = torch.clamp(denom, min=1.0e-6)
    cos = torch.sum(a * b, dim=-1) / denom
    return torch.rad2deg(torch.acos(cos.clamp(-1.0, 1.0)))


def _reference_racket_pos_w(env, motion_cmd, racket_cmd, n, device):
    """Compute the racket position from the reference body pose at the current phase."""
    import torch
    from isaaclab.utils.math import quat_apply

    motion_ids = motion_cmd.motion_ids[:n]
    time_steps = motion_cmd.time_steps[:n]
    if motion_cmd._use_motion_library:
        body_pos = motion_cmd.motion._body_pos_w[motion_ids, time_steps]
        body_quat = motion_cmd.motion._body_quat_w[motion_ids, time_steps]
    else:
        body_pos = motion_cmd.motion._body_pos_w[time_steps]
        body_quat = motion_cmd.motion._body_quat_w[time_steps]
    body_pos = body_pos + env.scene.env_origins[:n].unsqueeze(1)

    if racket_cmd._racket_mode == "body":
        return body_pos[:, racket_cmd._racket_body_index]

    wrist_pos = body_pos[:, racket_cmd._wrist_body_index]
    wrist_quat = body_quat[:, racket_cmd._wrist_body_index]
    return wrist_pos + quat_apply(wrist_quat, racket_cmd._mount_offset[:n])


def _reference_racket_vel_w(env, motion_cmd, racket_cmd, n, device):
    """Compute reference TCP velocity from the same motion-library state."""
    import torch
    from isaaclab.utils.math import quat_apply

    motion_ids = motion_cmd.motion_ids[:n]
    time_steps = motion_cmd.time_steps[:n]
    if motion_cmd._use_motion_library:
        body_pos = motion_cmd.motion._body_pos_w[motion_ids, time_steps]
        body_quat = motion_cmd.motion._body_quat_w[motion_ids, time_steps]
        body_lin = motion_cmd.motion._body_lin_vel_w[motion_ids, time_steps]
        body_ang = motion_cmd.motion._body_ang_vel_w[motion_ids, time_steps]
    else:
        body_pos = motion_cmd.motion._body_pos_w[time_steps]
        body_quat = motion_cmd.motion._body_quat_w[time_steps]
        body_lin = motion_cmd.motion._body_lin_vel_w[time_steps]
        body_ang = motion_cmd.motion._body_ang_vel_w[time_steps]
    del body_pos
    if racket_cmd._racket_mode == "body":
        return body_lin[:, racket_cmd._racket_body_index]
    wrist_lin = body_lin[:, racket_cmd._wrist_body_index]
    wrist_ang = body_ang[:, racket_cmd._wrist_body_index]
    wrist_quat = body_quat[:, racket_cmd._wrist_body_index]
    offset_w = quat_apply(wrist_quat, racket_cmd._mount_offset[:n])
    return wrist_lin + torch.cross(wrist_ang, offset_w, dim=-1)


def _sync_motion_state(env, motion_cmd, n, device, start_steps=None):
    import torch

    ids = torch.arange(n, device=device, dtype=torch.long)
    motion_cmd.motion_ids[:n] = ids
    if start_steps is None:
        motion_cmd.time_steps[:n] = 0
    else:
        motion_cmd.time_steps[:n] = torch.as_tensor(start_steps, dtype=torch.long, device=device)

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

    from isaaclab.utils.math import euler_xyz_from_quat, matrix_from_quat, quat_error_magnitude, wrap_to_pi
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
    if (
        task_id == "HOPE-FloatingUnifiedUpperReferenceTracker-AgibotA3-v0"
        and frame_z_offset is not None
        and abs(float(frame_z_offset)) > 1.0e-8
    ):
        raise ValueError(
            "P5U unified upper tracker requires manifest_frame_z_offset=0.0: "
            "P5D scene-placed NPZ files already contain the world z anchor; "
            f"received {float(frame_z_offset):.6f} m (would double-apply the lift)."
        )
    if frame_z_offset is not None:
        env_cfg.commands.motion.manifest_frame_z_offset = float(frame_z_offset)
    if _as_bool(cfg.get("validate_stance_contract", False)):
        env_cfg.commands.motion.validate_stance_contract = True
        stance_mode = cfg.get("stance_contract_mode", None)
        if stance_mode is not None:
            env_cfg.commands.motion.stance_contract_mode = str(stance_mode)
        print(
            "[INFO] stance contract validation enabled for policy evaluation",
            flush=True,
        )

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
    right_elbow_body_id = body_name_to_id.get("right_elbow_Link", torso_body_id)
    right_wrist_body_id = body_name_to_id.get("right_wrist_yaw_Link", right_elbow_body_id)
    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    diagnostic = _as_bool(cfg.get("diagnostic", False))
    native_joint_ids = getattr(action_term, "_joint_index_tensor", None)
    # Some prior-guided P5D action terms intentionally do not expose their
    # internal upper-joint index tensor.  Diagnostics still must report real
    # soft-limit margins, so fall back to the full articulation joint set;
    # this does not alter the actor action contract.
    if diagnostic and native_joint_ids is None:
        native_joint_ids = torch.arange(robot.num_joints, device=device, dtype=torch.long)
    action_scale = getattr(action_term, "_scale", None)
    if action_scale is not None:
        scale_abs_max = float(action_scale.abs().max().detach().cpu())
        scale_abs_mean = float(action_scale.abs().mean().detach().cpu())
        print(
            f"[INFO] action scale abs max/mean: {scale_abs_max:.6f}/{scale_abs_mean:.6f}",
            flush=True,
        )
        if scale_abs_max <= 1.0e-9:
            print(
                "[WARN] policy evaluation has zero residual action scale; "
                "this is a zero-residual/reference replay, not the learned "
                "checkpoint behavior. Pass task.actions.native_residual_scale "
                "used during training for a policy evaluation.",
                flush=True,
            )
    native_joint_names = []
    if native_joint_ids is not None:
        native_joint_names = [robot.data.joint_names[int(idx)] for idx in native_joint_ids.detach().cpu().tolist()]
    native_joint_name_to_local = {name: i for i, name in enumerate(native_joint_names)}
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
        "torso_roll_abs_deg": torch.full((n,), float("nan"), device=device),
        "torso_pitch_abs_deg": torch.full((n,), float("nan"), device=device),
        "torso_yaw_deg": torch.full((n,), float("nan"), device=device),
        "torso_ref_yaw_delta_deg": torch.full((n,), float("nan"), device=device),
        "torso_tilt_abs_deg": torch.full((n,), float("nan"), device=device),
        "torso_ref_tilt_delta_deg": torch.full((n,), float("nan"), device=device),
        "min_joint_margin": torch.full((n,), float("nan"), device=device),
        "min_arm_margin": torch.full((n,), float("nan"), device=device),
        "joint_near_limit_frac": torch.full((n,), float("nan"), device=device),
        "arm_near_limit_frac": torch.full((n,), float("nan"), device=device),
        "joint_near_limit_mask": torch.zeros(
            (n, len(native_joint_ids) if native_joint_ids is not None else 0), dtype=torch.bool, device=device
        ),
        "right_wrist_roll_abs_deg": torch.full((n,), float("nan"), device=device),
        "right_wrist_pitch_abs_deg": torch.full((n,), float("nan"), device=device),
        "right_wrist_yaw_abs_deg": torch.full((n,), float("nan"), device=device),
        "right_wrist_bend_pitch_yaw_deg": torch.full((n,), float("nan"), device=device),
        "forearm_racket_angle_deg": torch.full((n,), float("nan"), device=device),
        "captured": torch.zeros(n, dtype=torch.bool, device=device),
    }
    if diagnostic:
        captured.update(
            {
                "target_pos": torch.full((n, 3), float("nan"), device=device),
                "reference_pos": torch.full((n, 3), float("nan"), device=device),
                "actual_pos": torch.full((n, 3), float("nan"), device=device),
                "target_vel": torch.full((n, 3), float("nan"), device=device),
                "reference_vel": torch.full((n, 3), float("nan"), device=device),
                "actual_vel": torch.full((n, 3), float("nan"), device=device),
                "reference_root_pos": torch.full((n, 3), float("nan"), device=device),
                "actual_root_pos": torch.full((n, 3), float("nan"), device=device),
                "root_translation_error": torch.full((n,), float("nan"), device=device),
                "target_reference_error": torch.full((n,), float("nan"), device=device),
                "reference_actual_error": torch.full((n,), float("nan"), device=device),
                "velocity_magnitude_error": torch.full((n,), float("nan"), device=device),
                "velocity_direction_error_deg": torch.full((n,), float("nan"), device=device),
                "best_pos_error": torch.full((n,), float("inf"), device=device),
                "best_pos_step": torch.full((n,), -1, dtype=torch.long, device=device),
                "best_pos_velocity_magnitude_error": torch.full((n,), float("nan"), device=device),
                "best_pos_velocity_direction_error_deg": torch.full((n,), float("nan"), device=device),
                "raw_action_max": torch.full((n,), float("nan"), device=device),
                "raw_action_mean": torch.full((n,), float("nan"), device=device),
                "residual_max": torch.full((n,), float("nan"), device=device),
                "residual_mean": torch.full((n,), float("nan"), device=device),
                "prior_contribution_max": torch.full((n,), float("nan"), device=device),
                "prior_contribution_mean": torch.full((n,), float("nan"), device=device),
                "tracker_residual_max": torch.full((n,), float("nan"), device=device),
                "tracker_residual_mean": torch.full((n,), float("nan"), device=device),
                "prior_contribution_vector": torch.full((n, 10), float("nan"), device=device),
                "tracker_residual_vector": torch.full((n, 10), float("nan"), device=device),
                "residual_clip_fraction": torch.full((n,), float("nan"), device=device),
                "safety_projection_max": torch.zeros((n,), device=device),
                "min_root_height": torch.full((n,), float("inf"), device=device),
                "min_root_upright": torch.full((n,), float("inf"), device=device),
                "physical_terminated": torch.zeros(n, dtype=torch.bool, device=device),
                "timeout_seen": torch.zeros(n, dtype=torch.bool, device=device),
                "terminated_step": torch.full((n,), -1, dtype=torch.long, device=device),
            }
        )

    obs = _obs_to_device(env.get_observations(), agent_cfg.device)
    max_steps = int(cfg.get("max_steps") or 60)
    trace_path = cfg.get("dump_action_trace", None)
    trace_steps = []
    trace_time_steps = []
    trace_motion_ids = []
    for step_idx in range(max_steps):
        with torch.inference_mode():
            actions = policy(obs)
            # Reference-only P5D audit: keep the frozen model_900/model_3396
            # execution prior inside the action term, while suppressing only
            # the newly learned public tracker residual.  This is evaluation
            # only and never changes training behavior.
            if _as_bool(cfg.get("zero_tracker_residual", False)):
                if str(cfg.task.gym_task) not in {
                    "HOPE-FloatingReferenceTracker-AgibotA3-v0",
                    "HOPE-FloatingPriorGuidedReferenceTracker-AgibotA3-v0",
                }:
                    raise ValueError("zero_tracker_residual is only valid for a P5D reference-tracker task")
                actions = torch.zeros_like(actions)
            if trace_path is not None:
                # process_actions runs inside env.step; retain the phase index
                # before advancing MotionCommand so the trace can be aligned
                # back to the source NPZ without exposing an ID to the actor.
                trace_time_steps.append(motion_cmd.time_steps[:n].detach().cpu().clone())
                trace_motion_ids.append(motion_cmd.motion_ids[:n].detach().cpu().clone())
            obs, _, terminated, truncated = env.step(actions.to(device))
            obs = _obs_to_device(obs, agent_cfg.device)
            if trace_path is not None and hasattr(action_term, "_upper_processed_actions"):
                trace_steps.append(
                    torch.cat(
                        (
                            action_term._upper_reference_actions[:n],
                            action_term._upper_primary_contribution[:n],
                            action_term._upper_coordinator_contribution[:n],
                            action_term._upper_processed_actions[:n],
                            action_term._upper_safety_override[:n],
                        ),
                        dim=-1,
                    ).detach().cpu().clone()
                )
            if diagnostic and hasattr(action_term, "_upper_safety_override"):
                captured["safety_projection_max"] = torch.maximum(
                    captured["safety_projection_max"],
                    action_term._upper_safety_override[:n].abs().max(dim=-1).values,
                )
            if diagnostic:
                # Keep recovery/termination evidence separate from the
                # hit-time task error.  A large reference->actual error is
                # expected for tracker training; a physical termination is
                # an execution-safety failure.
                done_tensor = torch.as_tensor(terminated, device=device, dtype=torch.bool)[:n]
                if torch.is_tensor(truncated):
                    timeout_tensor = torch.as_tensor(truncated, device=device, dtype=torch.bool)[:n]
                else:
                    timeout_tensor = torch.as_tensor(
                        truncated.get("time_outs", torch.zeros_like(done_tensor)),
                        device=device,
                        dtype=torch.bool,
                    )[:n]
                physical = done_tensor & (~timeout_tensor)
                captured["physical_terminated"] |= physical
                captured["timeout_seen"] |= timeout_tensor
                newly = physical & (captured["terminated_step"] < 0)
                captured["terminated_step"][newly] = step_idx + 1
                root_pos = robot.data.root_pos_w[:n]
                root_up = matrix_from_quat(robot.data.root_quat_w[:n])[:, 2, 2]
                captured["min_root_height"] = torch.minimum(
                    captured["min_root_height"], root_pos[:, 2]
                )
                captured["min_root_upright"] = torch.minimum(
                    captured["min_root_upright"], root_up
                )

                # Track the best actual TCP position over the whole replay,
                # not only at the marked hit frame.  This separates a phase
                # error from a pure geometric miss.
                current_pos_error = torch.linalg.norm(
                    racket_cmd.racket_pos_w[:n] - racket_cmd.racket_target_pos_w[:n], dim=-1
                )
                better = current_pos_error < captured["best_pos_error"]
                if bool(better.any()):
                    actual_vel_now = racket_cmd.racket_lin_vel_w[:n]
                    target_vel_now = racket_cmd.racket_target_vel_w[:n]
                    speed_actual = torch.linalg.norm(actual_vel_now, dim=-1)
                    speed_target = torch.linalg.norm(target_vel_now, dim=-1)
                    vel_dot = torch.sum(actual_vel_now * target_vel_now, dim=-1)
                    vel_denom = (speed_actual * speed_target).clamp_min(1.0e-6)
                    vel_dir = torch.rad2deg(torch.acos((vel_dot / vel_denom).clamp(-1.0, 1.0)))
                    captured["best_pos_error"][better] = current_pos_error[better]
                    captured["best_pos_step"][better] = step_idx + 1
                    captured["best_pos_velocity_magnitude_error"][better] = (
                        speed_actual - speed_target
                    ).abs()[better]
                    captured["best_pos_velocity_direction_error_deg"][better] = vel_dir[better]

            exact = torch.abs(racket_cmd.time_to_strike[:n]) <= (0.5 * env.unwrapped.step_dt + 1.0e-6)
            take = exact & (~captured["captured"])
            if bool(take.any()):
                if diagnostic:
                    target_pos = racket_cmd.racket_target_pos_w[:n]
                    actual_pos = racket_cmd.racket_pos_w[:n]
                    reference_pos = _reference_racket_pos_w(
                        env.unwrapped, motion_cmd, racket_cmd, n, device
                    )
                    raw_actions = action_term.raw_actions[:n]
                    processed_actions = action_term.processed_actions[:n]
                    if hasattr(action_term, "_reference_joint_pos_with_joint_lead"):
                        reference_joint_pos = action_term._reference_joint_pos_with_joint_lead(
                            motion_cmd, motion_cmd.time_steps
                        )
                        processed_for_audit = action_term.processed_actions
                    else:
                        # Prior-guided P5D stores the complete upper command
                        # and nominal upper reference in the shared action
                        # chain buffers.  Its public processed_actions tensor
                        # contains only the new residual, so using it here
                        # would hide the frozen model_900 contribution.
                        reference_joint_pos = action_term._upper_reference_actions
                        processed_for_audit = action_term._upper_processed_actions
                    residual = processed_for_audit - reference_joint_pos
                    prior_contribution = getattr(
                        action_term, "_upper_primary_contribution", torch.zeros_like(reference_joint_pos)
                    )
                    tracker_contribution = getattr(
                        action_term, "_upper_coordinator_contribution", torch.zeros_like(reference_joint_pos)
                    )
                    raw_clip = float(getattr(action_term.cfg, "raw_clip", 1.0))
                    captured["target_pos"][take] = target_pos[take]
                    captured["reference_pos"][take] = reference_pos[take]
                    captured["actual_pos"][take] = actual_pos[take]
                    motion_ids_now = motion_cmd.motion_ids[:n]
                    time_steps_now = motion_cmd.time_steps[:n]
                    if motion_cmd._use_motion_library:
                        reference_root_pos = motion_cmd.motion._body_pos_w[motion_ids_now, time_steps_now, 0]
                    else:
                        reference_root_pos = motion_cmd.motion._body_pos_w[time_steps_now, 0]
                    # ``env`` is the RSL-RL vector wrapper in this loop; the
                    # scene (and its per-environment origins) lives on the
                    # underlying IsaacLab environment.
                    reference_root_pos = reference_root_pos + env.unwrapped.scene.env_origins[:n]
                    actual_root_pos = robot.data.root_pos_w[:n]
                    captured["reference_root_pos"][take] = reference_root_pos[take]
                    captured["actual_root_pos"][take] = actual_root_pos[take]
                    captured["root_translation_error"][take] = torch.linalg.norm(
                        actual_root_pos[take] - reference_root_pos[take], dim=-1
                    )
                    target_vel = racket_cmd.racket_target_vel_w[:n]
                    actual_vel = racket_cmd.racket_lin_vel_w[:n]
                    reference_vel = _reference_racket_vel_w(
                        env.unwrapped, motion_cmd, racket_cmd, n, device
                    )
                    speed_actual = torch.linalg.norm(actual_vel, dim=-1)
                    speed_target = torch.linalg.norm(target_vel, dim=-1)
                    vel_dot = torch.sum(actual_vel * target_vel, dim=-1)
                    vel_denom = (speed_actual * speed_target).clamp_min(1.0e-6)
                    vel_dir = torch.rad2deg(torch.acos((vel_dot / vel_denom).clamp(-1.0, 1.0)))
                    captured["target_vel"][take] = target_vel[take]
                    captured["reference_vel"][take] = reference_vel[take]
                    captured["actual_vel"][take] = actual_vel[take]
                    captured["velocity_magnitude_error"][take] = (
                        speed_actual - speed_target
                    ).abs()[take]
                    captured["velocity_direction_error_deg"][take] = vel_dir[take]
                    captured["target_reference_error"][take] = torch.linalg.norm(
                        target_pos[take] - reference_pos[take], dim=-1
                    )
                    captured["reference_actual_error"][take] = torch.linalg.norm(
                        reference_pos[take] - actual_pos[take], dim=-1
                    )
                    captured["raw_action_max"][take] = raw_actions[take].abs().max(dim=-1).values
                    captured["raw_action_mean"][take] = raw_actions[take].abs().mean(dim=-1)
                    captured["residual_max"][take] = residual[take].abs().max(dim=-1).values
                    captured["residual_mean"][take] = residual[take].abs().mean(dim=-1)
                    captured["prior_contribution_max"][take] = prior_contribution[take].abs().max(dim=-1).values
                    captured["prior_contribution_mean"][take] = prior_contribution[take].abs().mean(dim=-1)
                    captured["tracker_residual_max"][take] = tracker_contribution[take].abs().max(dim=-1).values
                    captured["tracker_residual_mean"][take] = tracker_contribution[take].abs().mean(dim=-1)
                    captured["prior_contribution_vector"][take] = prior_contribution[take]
                    captured["tracker_residual_vector"][take] = tracker_contribution[take]
                    captured["residual_clip_fraction"][take] = (
                        raw_actions[take].abs() >= raw_clip - 1.0e-6
                    ).float().mean(dim=-1)
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
                body_pos = robot.data.body_pos_w[:n]
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
                torso_rot = matrix_from_quat(body_quat[take, torso_body_id])
                ref_torso_rot = matrix_from_quat(ref_quat[take, torso_body_id])
                torso_roll, torso_pitch, torso_yaw = euler_xyz_from_quat(body_quat[take, torso_body_id])
                _, _, ref_torso_yaw = euler_xyz_from_quat(ref_quat[take, torso_body_id])
                torso_up = torso_rot[:, :, 2]
                ref_torso_up = ref_torso_rot[:, :, 2]
                world_up = torch.zeros_like(torso_up)
                world_up[:, 2] = 1.0
                torso_tilt = torch.acos(torch.sum(torso_up * world_up, dim=-1).clamp(-1.0, 1.0))
                ref_torso_tilt = torch.acos(torch.sum(ref_torso_up * world_up, dim=-1).clamp(-1.0, 1.0))
                captured["torso_roll_abs_deg"][take] = torch.abs(torch.rad2deg(wrap_to_pi(torso_roll)))
                captured["torso_pitch_abs_deg"][take] = torch.abs(torch.rad2deg(wrap_to_pi(torso_pitch)))
                captured["torso_yaw_deg"][take] = torch.rad2deg(wrap_to_pi(torso_yaw))
                captured["torso_ref_yaw_delta_deg"][take] = torch.abs(torch.rad2deg(wrap_to_pi(torso_yaw - ref_torso_yaw)))
                captured["torso_tilt_abs_deg"][take] = torch.rad2deg(torso_tilt)
                captured["torso_ref_tilt_delta_deg"][take] = torch.abs(torch.rad2deg(torso_tilt - ref_torso_tilt))
                forearm_vec = body_pos[take, right_wrist_body_id] - body_pos[take, right_elbow_body_id]
                racket_vec = racket_cmd.racket_pos_w[:n][take] - body_pos[take, right_wrist_body_id]
                captured["forearm_racket_angle_deg"][take] = _angle_between_deg(forearm_vec, racket_vec)
                if native_joint_ids is not None:
                    joint_pos = robot.data.joint_pos[:n, native_joint_ids]
                    for joint_name, field in (
                        ("right_wrist_roll_joint", "right_wrist_roll_abs_deg"),
                        ("right_wrist_pitch_joint", "right_wrist_pitch_abs_deg"),
                        ("right_wrist_yaw_joint", "right_wrist_yaw_abs_deg"),
                    ):
                        local_idx = native_joint_name_to_local.get(joint_name)
                        if local_idx is not None:
                            captured[field][take] = torch.abs(torch.rad2deg(wrap_to_pi(joint_pos[take, local_idx])))
                    captured["right_wrist_bend_pitch_yaw_deg"][take] = torch.sqrt(
                        captured["right_wrist_pitch_abs_deg"][take] ** 2
                        + captured["right_wrist_yaw_abs_deg"][take] ** 2
                    )
                    limits = robot.data.soft_joint_pos_limits[:n, native_joint_ids]
                    span = torch.clamp(limits[..., 1] - limits[..., 0], min=1.0e-6)
                    margin = torch.minimum(joint_pos - limits[..., 0], limits[..., 1] - joint_pos) / span
                    near_mask = margin[take] < 0.05
                    captured["joint_near_limit_frac"][take] = near_mask.float().mean(dim=-1)
                    captured["min_joint_margin"][take] = margin[take].min(dim=-1).values
                    non_waist_mask = torch.tensor(
                        [not name.startswith("waist_") for name in native_joint_names],
                        dtype=torch.bool,
                        device=device,
                    )
                    if bool(non_waist_mask.any()):
                        captured["arm_near_limit_frac"][take] = near_mask[:, non_waist_mask].float().mean(dim=-1)
                        captured["min_arm_margin"][take] = margin[take][:, non_waist_mask].min(dim=-1).values
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
        "arm_near_limit_frac,joint_near_limit_names,posture_pass,"
        "torso_roll_abs_deg,torso_pitch_abs_deg,torso_yaw_deg,torso_ref_yaw_delta_deg,"
        "torso_tilt_abs_deg,torso_ref_tilt_delta_deg,min_joint_margin,min_arm_margin,"
        "robot_posture_pass,gate_tier,right_wrist_roll_abs_deg,right_wrist_pitch_abs_deg,"
        "right_wrist_yaw_abs_deg,right_wrist_bend_pitch_yaw_deg,forearm_racket_angle_deg,"
        "wrist_naturalness_pass,whole_cycle_pass",
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
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    False,
                    "D_not_captured",
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    False,
                    False,
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
        torso_roll = float(captured["torso_roll_abs_deg"][i].detach().cpu())
        torso_pitch = float(captured["torso_pitch_abs_deg"][i].detach().cpu())
        torso_yaw = float(captured["torso_yaw_deg"][i].detach().cpu())
        torso_yaw_delta = float(captured["torso_ref_yaw_delta_deg"][i].detach().cpu())
        torso_tilt = float(captured["torso_tilt_abs_deg"][i].detach().cpu())
        torso_tilt_delta = float(captured["torso_ref_tilt_delta_deg"][i].detach().cpu())
        min_joint_margin = float(captured["min_joint_margin"][i].detach().cpu())
        min_arm_margin = float(captured["min_arm_margin"][i].detach().cpu())
        wrist_roll = float(captured["right_wrist_roll_abs_deg"][i].detach().cpu())
        wrist_pitch = float(captured["right_wrist_pitch_abs_deg"][i].detach().cpu())
        wrist_yaw = float(captured["right_wrist_yaw_abs_deg"][i].detach().cpu())
        wrist_bend = float(captured["right_wrist_bend_pitch_yaw_deg"][i].detach().cpu())
        forearm_racket_angle = float(captured["forearm_racket_angle_deg"][i].detach().cpu())
        near_names = "-"
        if native_joint_ids is not None:
            near_mask = captured["joint_near_limit_mask"][i].detach().cpu().tolist()
            near_names = "|".join(name for name, is_near in zip(native_joint_names, near_mask) if is_near) or "-"
        passed = pos <= pos_thresh and vel <= vel_thresh and normal <= normal_thresh
        posture_pass = (
            pelvis_ref <= 15.0
            and torso_ref <= 25.0
            and arm_near_limit <= 0.10
            and min_arm_margin >= 0.05
        )
        robot_posture_pass = (
            pelvis_ref <= 15.0
            and torso_ref <= 25.0
            and torso_tilt_delta <= 20.0
            and torso_yaw_delta <= 15.0
            and torso_roll <= 25.0
            and torso_pitch <= 35.0
            and arm_near_limit <= 0.10
            and min_arm_margin >= 0.05
        )
        wrist_naturalness_pass = (
            wrist_roll <= 65.0
            and wrist_pitch <= 35.0
            and wrist_yaw <= 35.0
            and wrist_bend <= 45.0
            and forearm_racket_angle <= 75.0
        )
        whole_cycle_pass = bool(passed and robot_posture_pass and wrist_naturalness_pass)
        gate_tier = _robot_posture_tier(
            hit_pass=passed,
            robot_posture_pass=robot_posture_pass,
            wrist_naturalness_pass=wrist_naturalness_pass,
            arm_near_limit=arm_near_limit,
            torso_tilt_delta=torso_tilt_delta,
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
                torso_roll,
                torso_pitch,
                torso_yaw,
                torso_yaw_delta,
                torso_tilt,
                torso_tilt_delta,
                min_joint_margin,
                min_arm_margin,
                robot_posture_pass,
                gate_tier,
                wrist_roll,
                wrist_pitch,
                wrist_yaw,
                wrist_bend,
                forearm_racket_angle,
                wrist_naturalness_pass,
                whole_cycle_pass,
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
            torso_roll,
            torso_pitch,
            torso_yaw,
            torso_yaw_delta,
            torso_tilt,
            torso_tilt_delta,
            min_joint_margin,
            min_arm_margin,
            robot_posture_pass,
            gate_tier,
            wrist_roll,
            wrist_pitch,
            wrist_yaw,
            wrist_bend,
            forearm_racket_angle,
            wrist_naturalness_pass,
            whole_cycle_pass,
        ) = row
        print(
            f"{rank},{stroke},{name},{pos:.4f},{vel:.4f},{normal:.2f},{int(passed)},"
            f"{pos_window:.4f},{action_abs:.4f},{pelvis_up:.4f},{torso_up:.4f},"
            f"{ref_pelvis_up:.4f},{ref_torso_up:.4f},{pelvis_ref:.2f},{torso_ref:.2f},"
            f"{joint_near_limit:.4f},{arm_near_limit:.4f},{near_names},{int(posture_pass)},"
            f"{torso_roll:.2f},{torso_pitch:.2f},{torso_yaw:.2f},{torso_yaw_delta:.2f},"
            f"{torso_tilt:.2f},{torso_tilt_delta:.2f},{min_joint_margin:.4f},{min_arm_margin:.4f},"
            f"{int(robot_posture_pass)},{gate_tier},{wrist_roll:.2f},{wrist_pitch:.2f},"
            f"{wrist_yaw:.2f},{wrist_bend:.2f},{forearm_racket_angle:.2f},"
            f"{int(wrist_naturalness_pass)},{int(whole_cycle_pass)}",
            flush=True,
        )

    if diagnostic:
        print("[INFO] exact-hit alignment diagnostics:", flush=True)
        print(
            "rank,episode_id,target_xyz,reference_xyz,actual_xyz,target_minus_reference_m,"
            "reference_minus_actual_m,raw_action_max,raw_action_mean,residual_max_rad,"
            "residual_mean_rad,residual_clip_fraction,safety_projection_max_rad,"
            "target_vel_xyz,reference_vel_xyz,actual_vel_xyz,velocity_magnitude_error_mps,"
            "velocity_direction_error_deg,best_pos_error_m,best_pos_step,"
            "best_pos_velocity_magnitude_error_mps,best_pos_velocity_direction_error_deg,"
            "prior_contribution_max_rad,prior_contribution_mean_rad,"
            "tracker_residual_max_rad,tracker_residual_mean_rad,"
            "prior_contribution_vector_rad,tracker_residual_vector_rad,"
            "reference_root_xyz,actual_root_xyz,root_translation_error_m",
            flush=True,
        )
        for rank, i in enumerate(sorted(range(n), key=lambda j: float(captured["target_reference_error"][j])), start=1):
            target = captured["target_pos"][i].detach().cpu().tolist()
            reference = captured["reference_pos"][i].detach().cpu().tolist()
            actual = captured["actual_pos"][i].detach().cpu().tolist()
            print(
                f"{rank},{names[i]},"
                f"{target[0]:.4f}/{target[1]:.4f}/{target[2]:.4f},"
                f"{reference[0]:.4f}/{reference[1]:.4f}/{reference[2]:.4f},"
                f"{actual[0]:.4f}/{actual[1]:.4f}/{actual[2]:.4f},"
                f"{float(captured['target_reference_error'][i]):.4f},"
                f"{float(captured['reference_actual_error'][i]):.4f},"
                f"{float(captured['raw_action_max'][i]):.4f},"
                f"{float(captured['raw_action_mean'][i]):.4f},"
                f"{float(captured['residual_max'][i]):.6f},"
                f"{float(captured['residual_mean'][i]):.6f},"
                f"{float(captured['residual_clip_fraction'][i]):.4f},"
                f"{float(captured['safety_projection_max'][i]):.6f},"
                f"{captured['target_vel'][i, 0].item():.4f}/{captured['target_vel'][i, 1].item():.4f}/{captured['target_vel'][i, 2].item():.4f},"
                f"{captured['reference_vel'][i, 0].item():.4f}/{captured['reference_vel'][i, 1].item():.4f}/{captured['reference_vel'][i, 2].item():.4f},"
                f"{captured['actual_vel'][i, 0].item():.4f}/{captured['actual_vel'][i, 1].item():.4f}/{captured['actual_vel'][i, 2].item():.4f},"
                f"{float(captured['velocity_magnitude_error'][i]):.4f},"
                f"{float(captured['velocity_direction_error_deg'][i]):.2f},"
                f"{float(captured['best_pos_error'][i]):.4f},"
                f"{int(captured['best_pos_step'][i])},"
                f"{float(captured['best_pos_velocity_magnitude_error'][i]):.4f},"
                f"{float(captured['best_pos_velocity_direction_error_deg'][i]):.2f},"
                f"{float(captured['prior_contribution_max'][i]):.6f},"
                f"{float(captured['prior_contribution_mean'][i]):.6f},"
                f"{float(captured['tracker_residual_max'][i]):.6f},"
                f"{float(captured['tracker_residual_mean'][i]):.6f},"
                f"{'/'.join(f'{x:.6f}' for x in captured['prior_contribution_vector'][i].detach().cpu().tolist())},"
                f"{'/'.join(f'{x:.6f}' for x in captured['tracker_residual_vector'][i].detach().cpu().tolist())},"
                f"{captured['reference_root_pos'][i, 0].item():.4f}/{captured['reference_root_pos'][i, 1].item():.4f}/{captured['reference_root_pos'][i, 2].item():.4f},"
                f"{captured['actual_root_pos'][i, 0].item():.4f}/{captured['actual_root_pos'][i, 1].item():.4f}/{captured['actual_root_pos'][i, 2].item():.4f},"
                f"{float(captured['root_translation_error'][i]):.4f}",
                flush=True,
            )
        finite_diag = captured["captured"] & torch.isfinite(captured["target_reference_error"])
        if bool(finite_diag.any()):
            print(
                "[INFO] alignment means: "
                f"target-reference={float(captured['target_reference_error'][finite_diag].mean()):.4f}m "
                f"reference-actual={float(captured['reference_actual_error'][finite_diag].mean()):.4f}m "
                f"raw_max={float(captured['raw_action_max'][finite_diag].mean()):.4f} "
                f"residual_max={float(captured['residual_max'][finite_diag].mean()):.6f}rad "
                f"clip_fraction={float(captured['residual_clip_fraction'][finite_diag].mean()):.4f} "
                f"safety_projection_max={float(captured['safety_projection_max'][finite_diag].max()):.6f}rad",
                flush=True,
            )
            print(
                "[INFO] recovery audit: "
                f"physical_termination_count={int(captured['physical_terminated'][finite_diag].sum().item())}/"
                f"{int(finite_diag.sum().item())} "
                f"timeout_seen={int(captured['timeout_seen'][finite_diag].sum().item())}/"
                f"{int(finite_diag.sum().item())} "
                f"min_root_height={float(captured['min_root_height'][finite_diag].min().item()):.4f}m "
                f"min_root_upright={float(captured['min_root_upright'][finite_diag].min().item()):.4f}",
                flush=True,
            )
            print(
                "rank,episode_id,physical_terminated,terminated_step,timeout_seen,min_root_height_m,min_root_upright",
                flush=True,
            )
            for rank, i in enumerate(
                sorted(torch.where(finite_diag)[0].detach().cpu().tolist(), key=lambda j: names[j]), start=1
            ):
                print(
                    f"{rank},{names[i]},{int(captured['physical_terminated'][i].item())},"
                    f"{int(captured['terminated_step'][i].item())},{int(captured['timeout_seen'][i].item())},"
                    f"{float(captured['min_root_height'][i].item()):.4f},"
                    f"{float(captured['min_root_upright'][i].item()):.4f}",
                    flush=True,
                )

    finite_rows = [r for r in rows if r[0] != float("inf")]
    if finite_rows:
        pass_rate = sum(1 for r in finite_rows if r[14]) / len(finite_rows)
        posture_rate = sum(1 for r in finite_rows if r[15]) / len(finite_rows)
        robot_posture_rate = sum(1 for r in finite_rows if r[26]) / len(finite_rows)
        wrist_rate = sum(1 for r in finite_rows if r[33]) / len(finite_rows)
        whole_rate = sum(1 for r in finite_rows if r[34]) / len(finite_rows)
        print(f"[INFO] composite_pass_rate={pass_rate:.3f} ({len(finite_rows)} captured motions)", flush=True)
        print(f"[INFO] posture_pass_rate={posture_rate:.3f} ({len(finite_rows)} captured motions)", flush=True)
        print(f"[INFO] robot_posture_pass_rate={robot_posture_rate:.3f} ({len(finite_rows)} captured motions)", flush=True)
        print(f"[INFO] wrist_naturalness_pass_rate={wrist_rate:.3f} ({len(finite_rows)} captured motions)", flush=True)
        print(f"[INFO] whole_cycle_pass_rate={whole_rate:.3f} ({len(finite_rows)} captured motions)", flush=True)
        _print_group_summary(rows)
    if trace_path is not None and trace_steps:
        out = pathlib.Path(str(trace_path)).expanduser()
        if not out.is_absolute():
            out = pathlib.Path.cwd() / out
        out.parent.mkdir(parents=True, exist_ok=True)
        import numpy as np

        trace = torch.stack(trace_steps).numpy()
        time_steps = torch.stack(trace_time_steps).numpy()
        motion_ids = torch.stack(trace_motion_ids).numpy()
        np.savez_compressed(
            out,
            trace=trace,
            time_steps=time_steps,
            motion_ids=motion_ids,
            upper_joint_names=np.asarray(native_joint_names, dtype=object),
            fields=np.asarray(
                [
                    "reference",
                    "primary_contribution",
                    "tracker_contribution",
                    "processed_command",
                    "safety_projection",
                ],
                dtype=object,
            ),
        )
        print(f"[INFO] wrote action-chain trace: {out} shape={trace.shape}", flush=True)
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
