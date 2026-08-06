"""Decompose native-strike position error into geometry, command, and dynamics.

The diagnostic evaluates selected manifest motions in a fixed-base Isaac scene.
At the exact command hit frame it records:

    T: manifest target racket position
    R: offline reference-motion FK racket position
    C: FK of the processed joint command with the reference root
    J: FK of the actual joint state with the reference root
    A: simulated actual racket position

It also performs a static FK injection check for the reference pose and emits a
hit-window CSV for phase/lag analysis.  This is diagnostic-only and never edits
the motion manifest or training configuration.
"""

from __future__ import annotations

import csv
import json
import os
import pathlib
import sys
from typing import Any

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


def _sync_motion_state(env, motion_cmd, motion_ids, device):
    import torch

    n = len(motion_ids)
    ids = torch.arange(n, device=device, dtype=torch.long)
    selected = torch.as_tensor(motion_ids, dtype=torch.long, device=device)
    motion_cmd.motion_ids[:n] = selected
    motion_cmd.time_steps[:n] = 0
    motion_cmd.tail_steps[:n] = 0
    motion_cmd.prelude_elapsed_steps[:n] = 0
    motion_cmd._prev_motion_steps = motion_cmd.time_steps.clone()
    env_ids = torch.arange(n, device=device, dtype=torch.long)
    root_pos = motion_cmd.motion._body_pos_w[selected, 0, 0] + env.scene.env_origins[:n]
    root_ori = motion_cmd.motion._body_quat_w[selected, 0, 0]
    root_lin = motion_cmd.motion._body_lin_vel_w[selected, 0, 0]
    root_ang = motion_cmd.motion._body_ang_vel_w[selected, 0, 0]
    motion_cmd.robot.write_joint_state_to_sim(motion_cmd.joint_pos[:n], motion_cmd.joint_vel[:n], env_ids=env_ids)
    motion_cmd.robot.write_root_state_to_sim(
        torch.cat([root_pos, root_ori, root_lin, root_ang], dim=-1), env_ids=env_ids
    )
    env.scene.write_data_to_sim()
    env.sim.forward()
    motion_cmd._update_command()


def _reference_racket_pos(env, motion_cmd, racket_cmd, motion_ids, steps):
    import torch
    from isaaclab.utils.math import quat_apply

    mids = torch.as_tensor(motion_ids, dtype=torch.long, device=env.device)
    steps = torch.as_tensor(steps, dtype=torch.long, device=env.device)
    body_pos = motion_cmd.motion._body_pos_w[mids, steps]
    body_quat = motion_cmd.motion._body_quat_w[mids, steps]
    body_pos = body_pos + env.scene.env_origins[: len(motion_ids)].unsqueeze(1)
    if racket_cmd._racket_mode == "body":
        return body_pos[:, racket_cmd._racket_body_index]
    wrist_pos = body_pos[:, racket_cmd._wrist_body_index]
    wrist_quat = body_quat[:, racket_cmd._wrist_body_index]
    return wrist_pos + quat_apply(wrist_quat, racket_cmd._mount_offset[: len(motion_ids)])


def _reference_root(env, motion_cmd, motion_ids, steps, env_ids=None):
    import torch

    mids = torch.as_tensor(motion_ids, dtype=torch.long, device=env.device)
    steps = torch.as_tensor(steps, dtype=torch.long, device=env.device)
    if env_ids is None:
        env_ids = torch.arange(len(motion_ids), device=env.device, dtype=torch.long)
    else:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    pos = motion_cmd.motion._body_pos_w[mids, steps, 0] + env.scene.env_origins[env_ids]
    quat = motion_cmd.motion._body_quat_w[mids, steps, 0]
    return pos, quat


def _probe_fk(env, motion_cmd, racket_cmd, q_full, root_pos, root_quat):
    """Evaluate racket FK for supplied pose without advancing physics."""
    import torch

    n = q_full.shape[0]
    ids = torch.arange(n, device=env.device, dtype=torch.long)
    zeros_j = torch.zeros_like(q_full)
    zeros_root = torch.zeros((n, 6), dtype=torch.float32, device=env.device)
    if q_full.is_cuda:
        torch.cuda.synchronize(q_full.device)
    print(
        f"[decompose] probe q finite={bool(torch.isfinite(q_full).all())} "
        f"range=({float(q_full.min()):.3f},{float(q_full.max()):.3f}) "
        f"root finite={bool(torch.isfinite(root_pos).all() and torch.isfinite(root_quat).all())}",
        flush=True,
    )
    print("[decompose] probe write joint", flush=True)
    motion_cmd.robot.write_joint_state_to_sim(q_full, zeros_j, env_ids=ids)
    print("[decompose] probe write root", flush=True)
    motion_cmd.robot.write_root_state_to_sim(
        torch.cat([root_pos, root_quat, zeros_root], dim=-1), env_ids=ids
    )
    # The articulation write methods update PhysX immediately and invalidate the
    # kinematics buffers.  Avoid a full simulator forward here: this is a
    # static kinematic probe, not a dynamics step, and forward can be very slow
    # or unstable for poses copied from a motion clip.
    print("[decompose] probe compute racket", flush=True)
    racket_cmd._compute_racket_state()
    print("[decompose] probe read racket", flush=True)
    return racket_cmd.racket_pos_w[:n].detach().clone()


def _vec(value):
    return [float(x) for x in value.detach().cpu().tolist()]


def _norm(value):
    import torch

    return float(torch.linalg.norm(value).item())


def _error(a, b):
    value = a - b
    return {"xyz_m": _vec(value), "norm_m": _norm(value)}


def _best_lag(rows: list[dict[str, Any]], max_lag: int = 10) -> dict[str, Any]:
    by_step = {int(row["command_motion_step"]): row for row in rows}
    candidates = []
    for lag in range(-max_lag, max_lag + 1):
        errors = []
        for step, row in by_step.items():
            ref = by_step.get(step + lag)
            if ref is None:
                continue
            a = row["A_xyz_m"]
            r = ref["R_xyz_m"]
            errors.append(sum((float(a[i]) - float(r[i])) ** 2 for i in range(3)) ** 0.5)
        if errors:
            candidates.append((sum(errors) / len(errors), lag, len(errors)))
    if not candidates:
        return {"lag_steps": None, "mean_error_m": None, "samples": 0}
    mean_error, lag, samples = min(candidates)
    return {"lag_steps": int(lag), "mean_error_m": float(mean_error), "samples": int(samples)}


def _best_joint_lag(rows: list[dict[str, Any]], joint_names: list[str], max_lag: int = 10) -> dict[str, Any]:
    """Find the reference-frame lead/lag that best explains each joint's tracking error."""
    by_step = {int(row["command_motion_step"]): row for row in rows}
    result = {}
    for joint_idx, joint_name in enumerate(joint_names):
        candidates = []
        for lag in range(-max_lag, max_lag + 1):
            errors = []
            for step, row in by_step.items():
                ref = by_step.get(step + lag)
                if ref is None:
                    continue
                actual = float(row["native_q_actual_rad"][joint_idx])
                reference = float(ref["native_q_ref_rad"][joint_idx])
                errors.append(abs(actual - reference))
            if errors:
                candidates.append((sum(errors) / len(errors), lag, len(errors)))
        if candidates:
            error, lag, samples = min(candidates)
            result[joint_name] = {
                "lag_steps": int(lag),
                "mean_abs_error_rad": float(error),
                "samples": int(samples),
            }
    return result


def _run(cfg, simulation_app):
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from training.utils.ppo_cfg import runner_kwargs
    import training.tasks  # noqa: F401

    task_id = str(cfg.task.gym_task)
    motion_ids = [int(x) for x in cfg.get("motion_ids", [2, 4])]
    num_envs = len(motion_ids)
    env_cfg = parse_env_cfg(task_id, device=str(cfg.device), num_envs=num_envs)
    _apply_task_overrides(env_cfg, cfg.task)
    env_cfg.sim.device = str(cfg.device)
    manifest = cfg.motion_manifest if cfg.motion_manifest is not None else cfg.task.get("motion_manifest")
    if manifest is None:
        raise ValueError("motion_manifest is required")
    manifest_path = pathlib.Path(str(manifest)).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = pathlib.Path.cwd() / manifest_path
    env_cfg.commands.motion.motion_manifest = str(manifest_path)
    env_cfg.commands.motion.motion_file = None
    env_cfg.commands.motion.manifest_subset_size = int(cfg.get("manifest_subset_size", 6))
    frame_z_offset = cfg.get("manifest_frame_z_offset", None)
    if frame_z_offset is not None:
        env_cfg.commands.motion.manifest_frame_z_offset = float(frame_z_offset)

    checkpoint = pathlib.Path(str(cfg.checkpoint)).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = pathlib.Path.cwd() / checkpoint
    agent_cfg = RslRlOnPolicyRunnerCfg(
        **runner_kwargs(OmegaConf.to_container(cfg.algo, resolve=True), str(cfg.task.experiment_name))
    )
    agent_cfg.device = str(cfg.device)
    env = gym.make(task_id, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)
    device = env.unwrapped.device
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=device)
    motion_cmd = env.unwrapped.command_manager.get_term("motion")
    racket_cmd = env.unwrapped.command_manager.get_term("racket_target")
    robot = motion_cmd.robot
    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    native_ids = getattr(action_term, "_joint_index_tensor")
    native_names = [motion_cmd.robot.joint_names[int(j)] for j in native_ids.detach().cpu().tolist()]
    lead_scan_joint = cfg.get("lead_scan_joint", None)
    lead_scan_values = [float(x) for x in cfg.get("lead_scan_values", [])]
    if lead_scan_values:
        if lead_scan_joint not in native_names:
            raise ValueError(f"lead_scan_joint must be one of {native_names}, got {lead_scan_joint}")
        if len(lead_scan_values) != num_envs:
            raise ValueError("lead_scan_values length must equal the number of motion_ids/envs")
        base_lead = action_term._joint_reference_lookahead_steps
        if base_lead.ndim == 1:
            base_lead = base_lead.unsqueeze(0)
        lead_matrix = base_lead.repeat(num_envs, 1)
        lead_matrix[:, native_names.index(lead_scan_joint)] = torch.as_tensor(
            lead_scan_values, dtype=torch.float32, device=device
        )
        action_term._joint_reference_lookahead_steps = lead_matrix
    action_dim = int(env.unwrapped.action_manager.total_action_dim)
    hit_frames = [int(motion_cmd.motion.hit_frame[mid].item()) for mid in motion_ids]
    episode_ids = [str(motion_cmd.motion.episode_ids[mid]) for mid in motion_ids]
    out_dir = pathlib.Path(str(cfg.get("out_dir", "eval_outputs/strike_error_decomposition"))).expanduser()
    if not out_dir.is_absolute():
        out_dir = pathlib.Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    max_steps = int(cfg.get("max_steps", 100))
    rollout_mode = str(cfg.get("rollout_mode", "both"))
    static_fk_enabled = _as_bool(cfg.get("static_fk", False))
    single_strike = _as_bool(cfg.get("single_strike", True))
    if single_strike:
        # The strike-only candidate intentionally freezes after the hit. Do not
        # let the motion library wrap and compare a new target to the old hit.
        max_steps = min(max_steps, max(hit_frames) + 9)
    summary: dict[str, Any] = {
        "manifest": str(manifest_path),
        "checkpoint": str(checkpoint),
        "motion_ids": motion_ids,
        "episode_ids": episode_ids,
        "hit_frames": hit_frames,
        "max_steps": max_steps,
        "rollout_mode": rollout_mode,
        "static_fk_enabled": static_fk_enabled,
        "single_strike": single_strike,
        "lead_scan_joint": lead_scan_joint,
        "lead_scan_values": lead_scan_values,
        "static_fk": {},
        "rollouts": {},
    }
    all_rows: list[dict[str, Any]] = []

    def rollout(mode: str):
        print(f"[decompose] rollout start mode={mode} motions={episode_ids}", flush=True)
        env.reset()
        _sync_motion_state(env.unwrapped, motion_cmd, motion_ids, device)
        racket_cmd._resample_command(torch.arange(num_envs, device=device))
        obs = _obs_to_device(env.get_observations(), agent_cfg.device)
        rows: list[dict[str, Any]] = []
        exact_states: dict[int, dict[str, torch.Tensor]] = {}
        for step_idx in range(max_steps):
            command_steps = motion_cmd.time_steps[:num_envs].clone()
            command_q_full = motion_cmd.motion.joint_pos[motion_cmd.motion_ids[:num_envs], command_steps].clone()
            command_qvel_full = motion_cmd.motion.joint_vel[motion_cmd.motion_ids[:num_envs], command_steps].clone()
            with torch.inference_mode():
                if mode == "zero":
                    actions = torch.zeros((num_envs, action_dim), dtype=torch.float32, device=device)
                else:
                    actions = policy(obs)
                action_term.process_actions(actions)
                q_cmd_native = action_term.processed_actions[:num_envs].clone()
                raw_actions = action_term.raw_actions[:num_envs].clone()
                obs, _, _, _ = env.step(actions.to(device))
                obs = _obs_to_device(obs, agent_cfg.device)

            steps_list = [int(x) for x in command_steps.detach().cpu().tolist()]
            ref_pos = _reference_racket_pos(env.unwrapped, motion_cmd, racket_cmd, motion_ids, steps_list)
            target_pos = racket_cmd.racket_target_pos_w[:num_envs].detach().clone()
            actual_pos = racket_cmd.racket_pos_w[:num_envs].detach().clone()
            for i, motion_step in enumerate(steps_list):
                if abs(motion_step - hit_frames[i]) <= 10:
                    actual_native_q = robot.data.joint_pos[i, native_ids].detach().clone()
                    row = {
                        "mode": mode,
                        "episode_id": episode_ids[i],
                        "env_i": i,
                        "lead_scan_value": lead_scan_values[i] if lead_scan_values else None,
                        "control_step": step_idx,
                        "command_motion_step": motion_step,
                        "T_xyz_m": _vec(target_pos[i]),
                        "R_xyz_m": _vec(ref_pos[i]),
                        "A_xyz_m": _vec(actual_pos[i]),
                        "T_minus_R": _error(target_pos[i], ref_pos[i]),
                        "T_minus_A": _error(target_pos[i], actual_pos[i]),
                        "raw_action_abs_max": float(raw_actions[i].abs().max().item()),
                        "residual_abs_max_rad": float((q_cmd_native[i] - command_q_full[i, native_ids]).abs().max().item()),
                        "native_q_ref_rad": _vec(command_q_full[i, native_ids]),
                        "native_q_cmd_rad": _vec(q_cmd_native[i]),
                        "native_q_actual_rad": _vec(actual_native_q),
                    }
                    rows.append(row)
                if motion_step == hit_frames[i] and i not in exact_states:
                    root_pos, root_quat = _reference_root(
                        env.unwrapped, motion_cmd, motion_ids[i:i + 1], [motion_step], env_ids=[i]
                    )
                    actual_root_pos = robot.data.root_pos_w[i:i + 1].detach().clone()
                    actual_root_quat = robot.data.root_quat_w[i:i + 1].detach().clone()
                    actual_q_full = robot.data.joint_pos[i:i + 1].detach().clone()
                    # PhysX exposes the spatial Jacobian at each non-root body.
                    # Convert the wrist Jacobian to the racket TCP Jacobian so
                    # the command-tracking error can be ranked by joint.
                    jacobians = robot.root_physx_view.get_jacobians()
                    wrist_jac = jacobians[i, racket_cmd._wrist_body_index - 1, :, :]
                    wrist_quat = robot.data.body_quat_w[i, racket_cmd._wrist_body_index]
                    from isaaclab.utils.math import quat_apply

                    offset_w = quat_apply(wrist_quat, racket_cmd._mount_offset[i])
                    skew = torch.zeros((3, 3), dtype=torch.float32, device=device)
                    skew[0, 1], skew[0, 2] = -offset_w[2], offset_w[1]
                    skew[1, 0], skew[1, 2] = offset_w[2], -offset_w[0]
                    skew[2, 0], skew[2, 1] = -offset_w[1], offset_w[0]
                    tcp_jac = wrist_jac[:3, :] - skew @ wrist_jac[3:, :]
                    q_cmd_full = command_q_full[i:i + 1].clone()
                    q_cmd_full[:, native_ids] = q_cmd_native[i:i + 1]
                    exact_states[i] = {
                        "target": target_pos[i:i + 1].clone(),
                        "reference_offline": ref_pos[i:i + 1].clone(),
                        "actual": actual_pos[i:i + 1].clone(),
                        "root_pos": root_pos.clone(),
                        "root_quat": root_quat.clone(),
                        "actual_root_pos": actual_root_pos,
                        "actual_root_quat": actual_root_quat,
                        "q_ref": command_q_full[i:i + 1].clone(),
                        "q_cmd": q_cmd_full,
                        "q_actual": actual_q_full,
                        "tcp_jac_native": tcp_jac[:, native_ids].detach().clone(),
                    }
            if (step_idx + 1) % 10 == 0 or step_idx + 1 == max_steps:
                print(
                    f"[decompose] rollout mode={mode} step={step_idx + 1}/{max_steps} "
                    f"motion_steps={steps_list}",
                    flush=True,
                )
        print(f"[decompose] rollout complete mode={mode} exact_states={list(exact_states)}", flush=True)
        static_rows = []
        for i, state in exact_states.items():
            if not static_fk_enabled:
                summary["static_fk"][f"{mode}:{episode_ids[i]}"] = {
                    "mode": mode,
                    "episode_id": episode_ids[i],
                    "status": "disabled",
                }
                continue
            # Probe one motion at a time to avoid changing the other selected motion's state.
            print(f"[decompose] static probes mode={mode} episode={episode_ids[i]}", flush=True)
            r_static = _probe_fk(
                env.unwrapped, motion_cmd, racket_cmd,
                state["q_ref"], state["root_pos"], state["root_quat"],
            )
            c_static = _probe_fk(
                env.unwrapped, motion_cmd, racket_cmd,
                state["q_cmd"], state["root_pos"], state["root_quat"],
            )
            j_static = _probe_fk(
                env.unwrapped, motion_cmd, racket_cmd,
                state["q_actual"], state["root_pos"], state["root_quat"],
            )
            static = {
                "mode": mode,
                "episode_id": episode_ids[i],
                "motion_id": motion_ids[i],
                "hit_frame": hit_frames[i],
                "T": _vec(state["target"][0]),
                "R_offline": _vec(state["reference_offline"][0]),
                "R_static": _vec(r_static[0]),
                "C": _vec(c_static[0]),
                "J": _vec(j_static[0]),
                "A": _vec(state["actual"][0]),
                "T_minus_R": _error(state["target"][0], state["reference_offline"][0]),
                "R_static_minus_R_offline": _error(r_static[0], state["reference_offline"][0]),
                "C_minus_R": _error(c_static[0], r_static[0]),
                "J_minus_C": _error(j_static[0], c_static[0]),
                "A_minus_J": _error(state["actual"][0], j_static[0]),
                "T_minus_A": _error(state["target"][0], state["actual"][0]),
            }
            static_rows.append(static)
            summary["static_fk"][f"{mode}:{episode_ids[i]}"] = static
            print(f"[decompose] static probes complete mode={mode} episode={episode_ids[i]}", flush=True)
        all_rows.extend(rows)
        exact_report = []
        for i, state in exact_states.items():
            q_ref = state["q_ref"]
            q_cmd = state["q_cmd"]
            q_actual = state["q_actual"]
            root_pos_error = state["actual_root_pos"] - state["root_pos"]
            q_delta_native = q_cmd[0, native_ids] - q_actual[0, native_ids]
            tcp_jac_native = state["tcp_jac_native"]
            predicted_native_delta = tcp_jac_native * q_delta_native.unsqueeze(0)
            target_error = state["target"][0] - state["actual"][0]
            target_error_unit = target_error / torch.clamp(torch.linalg.norm(target_error), min=1.0e-6)
            exact_report.append(
                {
                    "episode_id": episode_ids[i],
                    "motion_id": motion_ids[i],
                    "lead_scan_value": lead_scan_values[i] if lead_scan_values else None,
                    "hit_frame": hit_frames[i],
                    "target_xyz_m": _vec(state["target"][0]),
                    "reference_xyz_m": _vec(state["reference_offline"][0]),
                    "actual_xyz_m": _vec(state["actual"][0]),
                    "reference_minus_actual_xyz_m": _vec(state["reference_offline"][0] - state["actual"][0]),
                    "reference_root_xyz_m": _vec(state["root_pos"][0]),
                    "actual_root_xyz_m": _vec(state["actual_root_pos"][0]),
                    "root_error_xyz_m": _vec(root_pos_error[0]),
                    "root_error_norm_m": _norm(root_pos_error[0]),
                    "q_ref_rad": _vec(q_ref[0]),
                    "q_cmd_rad": _vec(q_cmd[0]),
                    "q_actual_rad": _vec(q_actual[0]),
                    "q_actual_minus_cmd_abs_max_rad": float((q_actual - q_cmd).abs().max().item()),
                    "q_actual_minus_ref_abs_max_rad": float((q_actual - q_ref).abs().max().item()),
                    "native_joint_names": native_names,
                    "native_q_ref_rad": _vec(q_ref[0, native_ids]),
                    "native_q_cmd_rad": _vec(q_cmd[0, native_ids]),
                    "native_q_actual_rad": _vec(q_actual[0, native_ids]),
                    "native_q_actual_minus_cmd_abs_max_rad": float(
                        (q_actual[:, native_ids] - q_cmd[:, native_ids]).abs().max().item()
                    ),
                    "native_q_actual_minus_ref_abs_max_rad": float(
                        (q_actual[:, native_ids] - q_ref[:, native_ids]).abs().max().item()
                    ),
                    "native_tcp_jacobian_xyz_m_per_rad": [
                        _vec(tcp_jac_native[:, j]) for j in range(tcp_jac_native.shape[1])
                    ],
                    "native_predicted_delta_xyz_m": [
                        _vec(predicted_native_delta[:, j]) for j in range(predicted_native_delta.shape[1])
                    ],
                    "native_predicted_delta_norm_m": [
                        _norm(predicted_native_delta[:, j]) for j in range(predicted_native_delta.shape[1])
                    ],
                    "native_predicted_delta_along_target_error_m": [
                        float(torch.dot(predicted_native_delta[:, j], target_error_unit).item())
                        for j in range(predicted_native_delta.shape[1])
                    ],
                }
            )
        summary.setdefault("exact_states", {})[mode] = exact_report
        summary["rollouts"][mode] = {
            "rows": len(rows),
            "static": static_rows,
            "lag_by_motion": {
                episode_id: _best_lag([r for r in rows if r["episode_id"] == episode_id])
                for episode_id in episode_ids
            },
            "joint_lag_by_motion": {
                episode_id: _best_joint_lag(
                    [r for r in rows if r["episode_id"] == episode_id], native_names
                )
                for episode_id in episode_ids
            },
        }

    if rollout_mode in ("zero", "both"):
        rollout("zero")
    if rollout_mode in ("policy", "both"):
        rollout("policy")

    csv_path = out_dir / "strike_error_window.csv"
    if all_rows:
        keys = sorted({key for row in all_rows for key in row.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            for row in all_rows:
                writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, list) or isinstance(value, dict) else value for key, value in row.items()})
    summary_path = out_dir / "strike_error_decomposition.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[decompose] wrote {csv_path}", flush=True)
    print(f"[decompose] wrote {summary_path}", flush=True)
    for key, value in summary["static_fk"].items():
        if value.get("status") != "ok":
            print(f"[decompose] {key}: static FK {value.get('status')}", flush=True)
            continue
        print(
            f"[decompose] {key}: T-R={value['T_minus_R']['norm_m']:.4f}m "
            f"Rstatic-R={value['R_static_minus_R_offline']['norm_m']:.4f}m "
            f"C-R={value['C_minus_R']['norm_m']:.4f}m "
            f"J-C={value['J_minus_C']['norm_m']:.4f}m "
            f"A-J={value['A_minus_J']['norm_m']:.4f}m "
            f"T-A={value['T_minus_A']['norm_m']:.4f}m",
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
