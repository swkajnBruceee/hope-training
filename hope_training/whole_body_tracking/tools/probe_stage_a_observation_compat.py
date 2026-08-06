#!/usr/bin/env python3
"""Static same-state probe for the legacy Stage-A yaw-frame contract."""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

import hydra
import torch
from omegaconf import OmegaConf

from evaluate_f0_migration import (
    CheckpointPolicy,
    _make_env,
    _prepare_episode,
    _path,
    _group_obs,
    _vec,
)
from stage_a_compat import adapt_stage_a_observation_legacy_yaw_pi, validate_stage_a_legacy_layout


def _set_state(raw, robot, root_state, joint_pos, root_quat, env_ids):
    state = root_state.clone()
    state[:, 3:7] = torch.as_tensor(root_quat, device=raw.device, dtype=state.dtype)
    state[:, 7:] = 0.0
    joint_vel = torch.zeros_like(robot.data.joint_vel)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.write_root_state_to_sim(state, env_ids=env_ids)
    raw.scene.write_data_to_sim()
    # Refresh cached body transforms without advancing the physical state.
    raw.sim.forward()
    raw.scene.update(0.0)


@hydra.main(version_base=None, config_path="../cfg", config_name="play")
def main(cfg: Any):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True, device=str(cfg.device))
    try:
        base = pathlib.Path.cwd()
        cases = int(cfg.get("cases", 6))
        seed = int(cfg.get("seed", 20260725))
        stage_path = _path(
            str(cfg.get("stage_a_checkpoint")), base
        )
        stage_a = CheckpointPolicy(str(stage_path), torch.device(str(cfg.device)))
        if stage_a.obs_dim != 126 or stage_a.action_dim != 14:
            raise RuntimeError("Stage-A checkpoint must be 126-D / 14-D")

        env = _make_env(cfg, fixed=False, cases=cases, seed=seed)
        try:
            raw = env.unwrapped
            (
                motion,
                racket,
                _action_term,
                _root0,
                _foot0,
                _foot_ids,
                _foot_names,
                target_pos,
                target_vel,
                target_normal,
                _stance_names,
                _stance_pos,
            ) = _prepare_episode(env, cases, seed)
            if cases > motion.motion.num_motions:
                raise ValueError("cases exceeds manifest motion count")

            ids = torch.arange(cases, device=raw.device)
            hit_frames = motion.motion.hit_frame[motion.motion_ids]
            motion.time_steps[:] = hit_frames
            motion.prelude_elapsed_steps[:] = int(getattr(motion, "prelude_steps", 0))
            racket.racket_target_pos_w[:] = target_pos
            racket.racket_target_vel_w[:] = target_vel
            racket.racket_target_normal_w[:] = target_normal
            racket._compute_strike_timing()

            robot = raw.scene["robot"]
            root_state = robot.data.root_state_w.clone()
            # Use the exact same joint state and world position in both probes.
            # Only the root yaw frame changes.
            joint_pos = motion.motion.joint_pos[motion.motion_ids, hit_frames].clone()
            term_names = list(raw.observation_manager._group_obs_term_names["stage_a"])
            term_dims = [
                int(shape[0])
                for shape in raw.observation_manager.group_obs_term_dim["stage_a"]
            ]
            validate_stage_a_legacy_layout(term_names, term_dims)

            # Disable observation noise for a mathematical frame comparison.
            term_cfgs = raw.observation_manager._group_obs_term_cfgs["stage_a"]
            saved_noise = [term.noise for term in term_cfgs]
            for term in term_cfgs:
                term.noise = None
            try:
                _set_state(raw, robot, root_state, joint_pos, (1.0, 0.0, 0.0, 0.0), ids)
                racket._compute_racket_state()
                old_obs = _group_obs(raw, "stage_a").clone()

                _set_state(raw, robot, root_state, joint_pos, (0.0, 0.0, 0.0, 1.0), ids)
                racket._compute_racket_state()
                new_obs = _group_obs(raw, "stage_a").clone()
            finally:
                for term, noise in zip(term_cfgs, saved_noise):
                    term.noise = noise

            adapted_obs = adapt_stage_a_observation_legacy_yaw_pi(new_obs)
            old_action = stage_a(old_obs)
            adapted_action = stage_a(adapted_obs)
            diff = torch.abs(old_obs - adapted_obs)
            action_diff = torch.abs(old_action - adapted_action)

            results = []
            cursor = 0
            for name, width in zip(term_names, term_dims):
                span = slice(cursor, cursor + width)
                term_diff = diff[:, span]
                results.append(
                    {
                        "motion_id": int(motion.motion_ids[len(results)].item()),
                        "hit_frame": int(hit_frames[len(results)].item()),
                        "term": name,
                        "width": width,
                        "max_abs_old_minus_adapted_new": float(term_diff.max().item()),
                        "mean_abs_old_minus_adapted_new": float(term_diff.mean().item()),
                    }
                )
                cursor += width

            per_motion = []
            for env_id in range(cases):
                per_motion.append(
                    {
                        "motion_id": int(motion.motion_ids[env_id].item()),
                        "hit_frame": int(hit_frames[env_id].item()),
                        "obs_max_abs_diff": float(diff[env_id].max().item()),
                        "obs_mean_abs_diff": float(diff[env_id].mean().item()),
                        "action_max_abs_diff": float(action_diff[env_id].max().item()),
                        "action_mean_abs_diff": float(action_diff[env_id].mean().item()),
                        "old_action": _vec(old_action[env_id]),
                        "adapted_action": _vec(adapted_action[env_id]),
                    }
                )
            report = {
                "stage_a_checkpoint": str(stage_path),
                "cases": cases,
                "root_position_same": True,
                "old_root_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                "new_root_quat_wxyz": [0.0, 0.0, 0.0, 1.0],
                "motion_time_is_hit_frame": True,
                "observation_noise_disabled": True,
                "term_slices": results,
                "per_motion": per_motion,
            }
            output = _path(
                str(cfg.get("output", "eval_outputs/stagea_180_observation_probe.json")),
                base,
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
            print(json.dumps({"output": str(output), "per_motion": per_motion}, indent=2), flush=True)
        finally:
            env.close()
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
