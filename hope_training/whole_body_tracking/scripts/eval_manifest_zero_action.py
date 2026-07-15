"""Evaluate manifest motions with zero residual action.

For HOPEA3NativeStrikeManifest, the action term is:

    joint_target = reference_joint_pos + residual * scale

So a zero action checks whether the current native-strike environment can execute
each manifest reference motion under the fixed-base waist/right-arm abstraction.
This is a cheap pre-filter before spending PPO time on a motion.
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


def _print_group_summary(rows):
    def summarize(name, group):
        if not group:
            return
        hit_comp = sum(1 for r in group if r[3]) / len(group)
        posture = sum(1 for r in group if r[15]) / len(group)
        robot_posture = sum(1 for r in group if r[26]) / len(group)
        wrist = sum(1 for r in group if r[33]) / len(group)
        whole = sum(1 for r in group if r[34]) / len(group)
        pos_mean = sum(r[0] for r in group) / len(group)
        vel_mean = sum(r[1] for r in group) / len(group)
        normal_mean = sum(r[2] for r in group) / len(group)
        pelvis_margin_mean = sum(15.0 - r[10] for r in group) / len(group)
        torso_margin_mean = sum(20.0 - r[11] for r in group) / len(group)
        arm_margin_mean = sum(0.10 - r[13] for r in group) / len(group)
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

    summarize("overall", rows)
    summarize("forehand", [r for r in rows if r[16] == "forehand"])
    summarize("backhand", [r for r in rows if r[16] == "backhand"])


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


def _robot_posture_tier(
    *,
    hit_pass: bool,
    robot_posture_pass: bool,
    wrist_naturalness_pass: bool,
    arm_near_limit: float,
    torso_tilt: float,
) -> str:
    if hit_pass and robot_posture_pass and wrist_naturalness_pass:
        return "A_robot_usable_candidate"
    if hit_pass and robot_posture_pass and not wrist_naturalness_pass:
        return "B_wrist_retarget_required"
    if hit_pass and arm_near_limit <= 0.10 and torso_tilt <= 35.0:
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


def _run(cfg, simulation_app):
    import pathlib

    import gymnasium as gym
    import torch

    from isaaclab.utils.math import euler_xyz_from_quat, matrix_from_quat, quat_error_magnitude, wrap_to_pi
    from isaaclab_tasks.utils import parse_env_cfg

    import training.tasks  # noqa: F401

    task_id = str(cfg.task.gym_task)
    num_envs = int(cfg.num_envs) if cfg.num_envs is not None else int(cfg.task.env.num_envs)

    env_cfg = parse_env_cfg(task_id, device=str(cfg.device), num_envs=num_envs)
    _apply_task_overrides(env_cfg, cfg.task)
    env_cfg.sim.device = str(cfg.device)
    env_cfg.seed = int(cfg.get("seed", 0) or 0)

    motion_manifest = cfg.motion_manifest if cfg.motion_manifest is not None else cfg.task.get("motion_manifest")
    if motion_manifest is None:
        raise ValueError("eval_manifest_zero_action.py requires motion_manifest=... or task.motion_manifest")
    manifest_path = pathlib.Path(str(motion_manifest)).expanduser()
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
    action_dim = env.unwrapped.action_manager.total_action_dim
    motion_cmd = env.unwrapped.command_manager.get_term("motion")
    racket_cmd = env.unwrapped.command_manager.get_term("racket_target")
    robot = motion_cmd.robot
    body_name_to_id = {name: i for i, name in enumerate(robot.body_names)}
    pelvis_body_id = body_name_to_id.get("pelvis_link", 0)
    torso_body_id = body_name_to_id.get("torso_Link", pelvis_body_id)
    right_elbow_body_id = body_name_to_id.get("right_elbow_Link", torso_body_id)
    right_wrist_body_id = body_name_to_id.get("right_wrist_yaw_Link", right_elbow_body_id)
    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    native_joint_ids = getattr(action_term, "_joint_index_tensor", None)
    native_joint_names = []
    if native_joint_ids is not None:
        native_joint_names = [robot.data.joint_names[int(idx)] for idx in native_joint_ids.detach().cpu().tolist()]
    native_joint_name_to_local = {name: i for i, name in enumerate(native_joint_names)}

    n_motions = int(motion_cmd.motion.num_motions)
    n = min(num_envs, n_motions)
    if n < n_motions:
        print(f"[WARN] num_envs={num_envs} < motions={n_motions}; evaluating first {n} motions only.", flush=True)

    env.reset()
    ids = torch.arange(n, device=device, dtype=torch.long)
    motion_cmd.motion_ids[:n] = ids
    motion_cmd.time_steps[:n] = 0
    # env.reset() samples motion ids and writes that sampled state to sim. For
    # this deterministic evaluator we overwrite the ids, so we must also sync
    # the robot state to the selected reference frame before stepping.
    env_ids = torch.arange(n, device=device)
    root_pos = motion_cmd.motion._body_pos_w[motion_cmd.motion_ids[:n], motion_cmd.time_steps[:n], 0]
    root_pos = root_pos + env.unwrapped.scene.env_origins[:n]
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
    _apply_motion_reset_perturbation(env.unwrapped, motion_cmd, n, device, cfg, [str(x) for x in motion_cmd.motion.episode_ids[:n]])
    try:
        racket_cmd._resample_command(env_ids)
    except Exception:
        pass

    if bool(cfg.get("print_reset_fk", False)):
        actual_pos = robot.data.body_pos_w[:n]
        actual_quat = robot.data.body_quat_w[:n]
        if motion_cmd._use_motion_library:
            ref_pos = motion_cmd.motion._body_pos_w[motion_cmd.motion_ids[:n], motion_cmd.time_steps[:n]]
            ref_pos = ref_pos + env.unwrapped.scene.env_origins[:n, None, :]
            ref_quat = motion_cmd.motion._body_quat_w[motion_cmd.motion_ids[:n], motion_cmd.time_steps[:n]]
        else:
            ref_pos = motion_cmd.motion._body_pos_w[motion_cmd.time_steps[:n]]
            ref_quat = motion_cmd.motion._body_quat_w[motion_cmd.time_steps[:n]]
        pos_err = torch.linalg.norm(actual_pos - ref_pos, dim=-1)
        rot_err = torch.rad2deg(quat_error_magnitude(ref_quat, actual_quat))
        print("[INFO] reset/no-step FK consistency:", flush=True)
        print(
            "max_pos_err_m={:.5f},mean_pos_err_m={:.5f},max_rot_err_deg={:.2f},mean_rot_err_deg={:.2f}".format(
                float(pos_err.max().detach().cpu()),
                float(pos_err.mean().detach().cpu()),
                float(rot_err.max().detach().cpu()),
                float(rot_err.mean().detach().cpu()),
            ),
            flush=True,
        )
        worst = torch.argsort(pos_err[0] + 0.01 * rot_err[0], descending=True)[:12]
        print("rank,body_index,body_name,pos_err_m,rot_err_deg,actual_z,ref_z", flush=True)
        for rank, idx_t in enumerate(worst, start=1):
            idx = int(idx_t.detach().cpu())
            print(
                f"{rank},{idx},{robot.body_names[idx]},{float(pos_err[0, idx].detach().cpu()):.5f},"
                f"{float(rot_err[0, idx].detach().cpu()):.2f},"
                f"{float(actual_pos[0, idx, 2].detach().cpu()):.4f},"
                f"{float(ref_pos[0, idx, 2].detach().cpu()):.4f}",
                flush=True,
            )

    zeros = torch.zeros((num_envs, action_dim), device=device)
    captured = {
        "pos": torch.full((n, 3), float("nan"), device=device),
        "vel": torch.full((n, 3), float("nan"), device=device),
        "normal": torch.full((n, 3), float("nan"), device=device),
        "target_pos": torch.full((n, 3), float("nan"), device=device),
        "target_vel": torch.full((n, 3), float("nan"), device=device),
        "target_normal": torch.full((n, 3), float("nan"), device=device),
        "pelvis_upright": torch.full((n,), float("nan"), device=device),
        "torso_upright": torch.full((n,), float("nan"), device=device),
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

    max_steps = int(cfg.get("max_steps") or 100)
    for _ in range(max_steps):
        env.step(zeros)
        exact = torch.abs(racket_cmd.time_to_strike[:n]) <= (0.5 * env.unwrapped.step_dt + 1.0e-6)
        take = exact & (~captured["captured"])
        if bool(take.any()):
            captured["pos"][take] = racket_cmd.racket_pos_w[:n][take]
            captured["vel"][take] = racket_cmd.racket_lin_vel_w[:n][take]
            captured["normal"][take] = racket_cmd.racket_normal_w[:n][take]
            captured["target_pos"][take] = racket_cmd.racket_target_pos_w[:n][take]
            captured["target_vel"][take] = racket_cmd.racket_target_vel_w[:n][take]
            captured["target_normal"][take] = racket_cmd.racket_target_normal_w[:n][take]
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

    print("[INFO] zero-action manifest executability:", flush=True)
    print(
        "rank,stroke,episode_id,pos_exact,vel_exact,normal_deg_exact,hit_composite_pass,pos_window,success10_window,"
        "action_abs_mean,joint_vel_abs_max,pelvis_upright,torso_upright,pelvis_ref_err_deg,"
        "torso_ref_err_deg,joint_near_limit_frac,arm_near_limit_frac,joint_near_limit_names,posture_pass,"
        "torso_roll_abs_deg,torso_pitch_abs_deg,torso_yaw_deg,torso_ref_yaw_delta_deg,"
        "torso_tilt_abs_deg,torso_ref_tilt_delta_deg,min_joint_margin,min_arm_margin,"
        "robot_posture_pass,gate_tier,right_wrist_roll_abs_deg,right_wrist_pitch_abs_deg,"
        "right_wrist_yaw_abs_deg,right_wrist_bend_pitch_yaw_deg,forearm_racket_angle_deg,"
        "wrist_naturalness_pass,whole_cycle_pass",
        flush=True,
    )
    rows = []
    for i in range(n):
        pos_t = captured["pos"][i]
        vel_t = captured["vel"][i]
        normal_t = captured["normal"][i]
        target_pos_t = captured["target_pos"][i]
        target_vel_t = captured["target_vel"][i]
        target_normal_t = captured["target_normal"][i]
        pos = float(torch.linalg.norm(pos_t - target_pos_t).detach().cpu())
        vel = float(torch.linalg.norm(vel_t - target_vel_t).detach().cpu())
        normal_dot = torch.sum(normal_t * target_normal_t).clamp(-1.0, 1.0)
        normal = float(torch.rad2deg(torch.acos(normal_dot)).detach().cpu())
        pos_window = pos
        success10 = float(pos <= 0.10)
        action_mean = float(racket_cmd.metrics["action_abs_mean"][i].detach().cpu())
        joint_vel = float(racket_cmd.metrics["joint_vel_abs_max"][i].detach().cpu())
        pelvis_up = float(captured["pelvis_upright"][i].detach().cpu())
        torso_up = float(captured["torso_upright"][i].detach().cpu())
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
        hit_pass = (
            pos <= float(racket_cmd.cfg.strike_success_pos_thresh)
            and vel <= float(racket_cmd.cfg.strike_success_vel_thresh)
            and normal <= float(racket_cmd.cfg.strike_success_normal_thresh_deg)
        )
        posture_pass = (
            pelvis_ref <= 15.0
            and torso_ref <= 20.0
            and arm_near_limit <= 0.10
        )
        robot_posture_pass = (
            pelvis_ref <= 15.0
            and torso_tilt <= 32.0
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
        whole_cycle_pass = bool(hit_pass and robot_posture_pass and wrist_naturalness_pass)
        gate_tier = _robot_posture_tier(
            hit_pass=hit_pass,
            robot_posture_pass=robot_posture_pass,
            wrist_naturalness_pass=wrist_naturalness_pass,
            arm_near_limit=arm_near_limit,
            torso_tilt=torso_tilt,
        )
        rows.append(
            (
                pos,
                vel,
                normal,
                hit_pass,
                pos_window,
                success10,
                action_mean,
                joint_vel,
                pelvis_up,
                torso_up,
                pelvis_ref,
                torso_ref,
                joint_near_limit,
                arm_near_limit,
                near_names,
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
    for rank, row in enumerate(sorted(rows, key=lambda r: (r[0], r[1])), start=1):
        (
            pos,
            vel,
            normal,
            hit_pass,
            pos_window,
            success10,
            action_mean,
            joint_vel,
            pelvis_up,
            torso_up,
            pelvis_ref,
            torso_ref,
            joint_near_limit,
            arm_near_limit,
            near_names,
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
            f"{rank},{stroke},{name},{pos:.4f},{vel:.4f},{normal:.2f},{int(hit_pass)},{pos_window:.4f},"
            f"{success10:.3f},{action_mean:.4f},{joint_vel:.4f},{pelvis_up:.4f},"
            f"{torso_up:.4f},{pelvis_ref:.2f},{torso_ref:.2f},{joint_near_limit:.4f},"
            f"{arm_near_limit:.4f},{near_names},{int(posture_pass)},"
            f"{torso_roll:.2f},{torso_pitch:.2f},{torso_yaw:.2f},{torso_yaw_delta:.2f},"
            f"{torso_tilt:.2f},{torso_tilt_delta:.2f},{min_joint_margin:.4f},{min_arm_margin:.4f},"
            f"{int(robot_posture_pass)},{gate_tier},{wrist_roll:.2f},{wrist_pitch:.2f},"
            f"{wrist_yaw:.2f},{wrist_bend:.2f},{forearm_racket_angle:.2f},"
            f"{int(wrist_naturalness_pass)},{int(whole_cycle_pass)}",
            flush=True,
        )
    if rows:
        hit_rate = sum(1 for r in rows if r[3]) / len(rows)
        posture_rate = sum(1 for r in rows if r[15]) / len(rows)
        robot_posture_rate = sum(1 for r in rows if r[26]) / len(rows)
        wrist_rate = sum(1 for r in rows if r[33]) / len(rows)
        whole_rate = sum(1 for r in rows if r[34]) / len(rows)
        print(f"[INFO] hit_composite_pass_rate={hit_rate:.3f} ({len(rows)} captured motions)", flush=True)
        print(f"[INFO] posture_pass_rate={posture_rate:.3f} ({len(rows)} captured motions)", flush=True)
        print(f"[INFO] robot_posture_pass_rate={robot_posture_rate:.3f} ({len(rows)} captured motions)", flush=True)
        print(f"[INFO] wrist_naturalness_pass_rate={wrist_rate:.3f} ({len(rows)} captured motions)", flush=True)
        print(f"[INFO] whole_cycle_pass_rate={whole_rate:.3f} ({len(rows)} captured motions)", flush=True)
        _print_group_summary(rows)

    write_path = cfg.get("write_native_manifest", None)
    if write_path:
        write_path = pathlib.Path(str(write_path)).expanduser()
        if not write_path.is_absolute():
            write_path = pathlib.Path.cwd() / write_path
        with open(manifest_path, "r", encoding="utf-8") as f:
            source_manifest = json.load(f)
        source_entries = list(source_manifest.get("motions", []))
        source_by_episode = {str(e.get("episode_id", i)): e for i, e in enumerate(source_entries)}

        env_origins = env.unwrapped.scene.env_origins[:n]
        out_entries = []
        pos_local = captured["pos"] - env_origins
        # Manifest positions are stored before MotionLibraryLoader applies
        # manifest_frame_z_offset. Captured FK is post-offset simulation state,
        # so remove the configured offset before writing the manifest.
        pos_local[:, 2] -= float(env_cfg.commands.motion.manifest_frame_z_offset)
        pos_cpu = pos_local.detach().cpu()
        vel_cpu = captured["vel"].detach().cpu()
        normal_cpu = captured["normal"].detach().cpu()
        captured_cpu = captured["captured"].detach().cpu()
        for i in range(n):
            name = names[i]
            if not bool(captured_cpu[i]):
                print(f"[WARN] did not capture exact strike for {name}; skipping native manifest entry.", flush=True)
                continue
            src = dict(source_by_episode.get(str(name), motion_cmd.motion.entries[i]))
            hit_event = dict(src.get("hit_event", {}))
            hit_event["motion_hit_frame"] = int(motion_cmd.motion.hit_frame[i].detach().cpu())
            src["hit_event"] = hit_event
            original_target = dict(src.get("strike_target", {}))
            normal = normal_cpu[i]
            normal = normal / max(float(torch.linalg.norm(normal)), 1.0e-6)
            src["strike_target"] = {
                **original_target,
                "racket_position_m": [float(x) for x in pos_cpu[i].tolist()],
                "racket_velocity_mps": [float(x) for x in vel_cpu[i].tolist()],
                "racket_normal_w": [float(x) for x in normal.tolist()],
            }
            src["native_calibration"] = {
                "target_source": "zero_residual_native_fk",
                "source_manifest": str(manifest_path),
                "original_strike_target": original_target,
                "notes": (
                    "Strike target was replaced by the racket state reached by the current "
                    "HOPEA3NativeStrikeManifest fixed-base waist+right-arm zero-residual executor. "
                    "Use for native-RL executability calibration, not as a substitute for ball-planner targets."
                ),
            }
            out_entries.append(src)

        out_manifest = {
            **source_manifest,
            "motions": out_entries,
            "native_calibration": {
                "target_source": "zero_residual_native_fk",
                "source_manifest": str(manifest_path),
                "num_motions": len(out_entries),
                "frame_z_offset": float(env_cfg.commands.motion.manifest_frame_z_offset),
                "task": task_id,
            },
        }
        write_path.parent.mkdir(parents=True, exist_ok=True)
        with open(write_path, "w", encoding="utf-8") as f:
            json.dump(out_manifest, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"[INFO] wrote native-calibrated manifest: {write_path} ({len(out_entries)} motions)", flush=True)

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
