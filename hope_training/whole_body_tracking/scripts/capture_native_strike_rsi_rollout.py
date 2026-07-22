#!/usr/bin/env python3
"""Capture realized zero-residual strike rollouts for RSI diagnostics.

Unlike ``build_strike_rsi_bank.py`` (reference-data inventory), this tool saves
the state actually realized by the simulator together with controller targets
and phase state.  Generated files remain diagnostic-only for direct loading.
They may also qualify continuous prefix handoff, where the simulator is rolled
from frame zero to a sampled phase without resetting physics or contacts.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import hydra
from omegaconf import OmegaConf

from train import _apply_task_overrides


def _sync_start(env, motion_cmd, count, device):
    import torch

    ids = torch.arange(count, device=device, dtype=torch.long)
    motion_cmd.motion_ids[:count] = ids
    motion_cmd.time_steps[:count] = 0
    root_pos = motion_cmd.motion._body_pos_w[ids, 0, 0] + env.scene.env_origins[:count]
    root_state = torch.cat(
        [
            root_pos,
            motion_cmd.motion._body_quat_w[ids, 0, 0],
            motion_cmd.motion._body_lin_vel_w[ids, 0, 0],
            motion_cmd.motion._body_ang_vel_w[ids, 0, 0],
        ],
        dim=-1,
    )
    motion_cmd.robot.write_joint_state_to_sim(motion_cmd.joint_pos[:count], motion_cmd.joint_vel[:count], env_ids=ids)
    motion_cmd.robot.write_root_state_to_sim(root_state, env_ids=ids)


def _action_joint_names(action_term, robot) -> list[str]:
    """Resolve action ownership for both native and Base14 composers."""

    if hasattr(action_term, "_joint_index_tensor"):
        joint_ids = action_term._joint_index_tensor.detach().cpu().tolist()
    elif hasattr(action_term, "_base_joint_ids"):
        joint_ids = list(action_term._base_joint_ids)
    else:
        raise RuntimeError(
            f"Cannot resolve controlled joints for action term {type(action_term).__name__}"
        )
    return [robot.data.joint_names[int(index)] for index in joint_ids]


@hydra.main(version_base=None, config_path="../cfg", config_name="play")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=bool(cfg.headless), device=str(cfg.device), enable_cameras=False)
    simulation_app = app_launcher.app
    try:
        import gymnasium as gym
        import numpy as np
        import torch
        from isaaclab_tasks.utils import parse_env_cfg
        import training.tasks  # noqa: F401

        task_id = str(cfg.task.gym_task)
        manifest_path = pathlib.Path(str(cfg.motion_manifest or cfg.task.motion_manifest)).expanduser()
        env_cfg = parse_env_cfg(task_id, device=str(cfg.device), num_envs=int(cfg.num_envs))
        _apply_task_overrides(env_cfg, cfg.task)
        env_cfg.sim.device = str(cfg.device)
        env_cfg.seed = int(cfg.get("seed", 0) or 0)
        env_cfg.commands.motion.motion_manifest = str(manifest_path)
        env_cfg.commands.motion.motion_file = None
        env_cfg.commands.motion.manifest_subset_size = None
        frame_z_offset = cfg.get("manifest_frame_z_offset", None)
        if frame_z_offset is None:
            frame_z_offset = cfg.task.get("manifest_frame_z_offset", None)
        if frame_z_offset is not None:
            env_cfg.commands.motion.manifest_frame_z_offset = float(frame_z_offset)
        ground_z_offset = cfg.get("manifest_ground_z_offset", None)
        if ground_z_offset is None:
            ground_z_offset = cfg.task.get("manifest_ground_z_offset", None)
        if ground_z_offset is not None:
            env_cfg.commands.motion.manifest_ground_z_offset = float(ground_z_offset)
        env = gym.make(task_id, cfg=env_cfg, render_mode=None)
        motion_cmd = env.unwrapped.command_manager.get_term("motion")
        racket_cmd = env.unwrapped.command_manager.get_term("racket_target")
        action_term = env.unwrapped.action_manager.get_term("joint_pos")
        robot = motion_cmd.robot
        device = env.unwrapped.device
        count = min(int(cfg.num_envs), int(motion_cmd.motion.num_motions))
        env.reset()
        _sync_start(env.unwrapped, motion_cmd, count, device)
        env_ids = torch.arange(count, device=device)
        racket_cmd._resample_command(env_ids)
        racket_cmd._compute_strike_timing()
        constant_action = float(cfg.get("constant_action", 0.0) or 0.0)
        zero = torch.full(
            (int(cfg.num_envs), env.unwrapped.action_manager.total_action_dim),
            constant_action,
            device=device,
        )

        core_fields = [
            "motion_step", "root_state_w", "joint_pos", "joint_vel", "joint_pos_target",
            "applied_torque", "raw_action", "processed_action", "time_to_strike_s",
            "racket_pos_w", "racket_lin_vel_w", "racket_normal_w",
            "racket_target_pos_w", "racket_target_vel_w", "racket_target_normal_w",
        ]
        has_handoff = hasattr(env.unwrapped, "strike_stabilizer_handoff_steps")
        if has_handoff:
            core_fields.append("policy_handoff_step")
        full_context = bool(cfg.get("capture_full_context", False))
        if full_context:
            core_fields.extend(["body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "policy_observation"])
        fields = {name: [] for name in core_fields}
        max_length = int(motion_cmd.motion.motion_lengths[:count].max().detach().cpu())
        capture_steps = max_length - 1
        if cfg.get("max_capture_steps", None) is not None:
            capture_steps = min(capture_steps, int(cfg.max_capture_steps))
        print(f"[RSI capture] steps={capture_steps} envs={count}", flush=True)
        for capture_index in range(capture_steps):
            step_result = env.step(zero)
            obs = step_result[0]
            if capture_steps <= 5:
                print(f"[RSI capture] simulated step {capture_index + 1}", flush=True)
            values = {
                "motion_step": motion_cmd.time_steps[:count],
                "root_state_w": robot.data.root_state_w[:count],
                "joint_pos": robot.data.joint_pos[:count],
                "joint_vel": robot.data.joint_vel[:count],
                "joint_pos_target": robot.data.joint_pos_target[:count],
                "applied_torque": robot.data.applied_torque[:count],
                "raw_action": action_term.raw_actions[:count],
                "processed_action": action_term.processed_actions[:count],
                "time_to_strike_s": racket_cmd.time_to_strike[:count],
                "racket_pos_w": racket_cmd.racket_pos_w[:count],
                "racket_lin_vel_w": racket_cmd.racket_lin_vel_w[:count],
                "racket_normal_w": racket_cmd.racket_normal_w[:count],
                "racket_target_pos_w": racket_cmd.racket_target_pos_w[:count],
                "racket_target_vel_w": racket_cmd.racket_target_vel_w[:count],
                "racket_target_normal_w": racket_cmd.racket_target_normal_w[:count],
            }
            if has_handoff:
                values["policy_handoff_step"] = env.unwrapped.strike_stabilizer_handoff_steps[:count]
            if full_context:
                policy_obs = obs["policy"] if isinstance(obs, dict) else obs
                values.update({
                    "body_pos_w": robot.data.body_pos_w[:count],
                    "body_quat_w": robot.data.body_quat_w[:count],
                    "body_lin_vel_w": robot.data.body_lin_vel_w[:count],
                    "body_ang_vel_w": robot.data.body_ang_vel_w[:count],
                    "policy_observation": policy_obs[:count],
                })
            for name, value in values.items():
                # Keep the short rollout on-device and synchronize once at the
                # end.  Per-field CPU copies made every control step dominated
                # runtime and made a 79-frame capture take minutes.
                fields[name].append(value.detach().clone())
            if capture_steps <= 5:
                print(f"[RSI capture] buffered step {capture_index + 1}", flush=True)
            if (capture_index + 1) % 10 == 0 or capture_index + 1 == capture_steps:
                print(f"[RSI capture] completed {capture_index + 1}/{capture_steps}", flush=True)

        stacked = {name: torch.stack(values, dim=0).cpu().numpy() for name, values in fields.items()}
        out_dir = pathlib.Path(str(cfg.output_dir)).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        entries = []
        episode_ids = [str(x) for x in motion_cmd.motion.episode_ids[:count]]
        for env_index, episode_id in enumerate(episode_ids):
            length = min(int(motion_cmd.motion.motion_lengths[env_index].detach().cpu()) - 1, capture_steps)
            payload = {name: value[:length, env_index] for name, value in stacked.items()}
            file_name = f"{episode_id}.npz"
            np.savez_compressed(out_dir / file_name, **payload)
            entries.append({"episode_id": episode_id, "state_file": file_name, "captured_frames": length})
        finite = all(np.isfinite(value).all() for value in stacked.values() if value.dtype.kind in "fc")
        manifest = {
            "schema_version": 1,
            "stage": "strike_realized_prefix_capture_v2",
            "training_eligible": False,
            "direct_load_eligible": False,
            "continuous_prefix_handoff_candidate": bool(finite),
            "task_id": task_id,
            "source_manifest": str(manifest_path),
            "manifest_frame_z_offset_m": float(env_cfg.commands.motion.manifest_frame_z_offset),
            "manifest_ground_z_offset_m": float(env_cfg.commands.motion.manifest_ground_z_offset),
            "joint_names": list(robot.data.joint_names),
            "action_joint_names": _action_joint_names(action_term, robot),
            "action_dim": int(action_term.action_dim),
            "all_values_finite": bool(finite),
            "captured_context": list(fields),
            "missing_context_for_direct_load": [
                "physx_solver_warm_start_state",
                "contact_manifold_history",
                "actuator_hidden_state_not_exposed",
                "ball_state_not_present_in_strike_task",
                "filter_state_not_present",
            ],
            "entries": entries,
        }
        (out_dir / "rsi_capture_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"captured": len(entries), "frames_per_motion": [e["captured_frames"] for e in entries], "output_dir": str(out_dir)}, ensure_ascii=False), flush=True)
        env.close()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
