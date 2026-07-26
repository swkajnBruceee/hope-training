#!/usr/bin/env python3
"""Deterministic zero-correction audit for frozen model_3396 + model_900."""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

import hydra
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from train import _apply_task_overrides  # noqa: E402


@hydra.main(version_base=None, config_path="../cfg", config_name="play")
def main(cfg: Any):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True, device=str(cfg.device))
    try:
        import gymnasium as gym
        from isaaclab_tasks.utils import parse_env_cfg
        import training.tasks  # noqa: F401

        cases = int(cfg.get("cases", 6))
        print("[upper-audit] parsing env config", flush=True)
        env_cfg = parse_env_cfg(str(cfg.task.gym_task), device=str(cfg.device), num_envs=cases)
        _apply_task_overrides(env_cfg, cfg.task)
        env_cfg.seed = int(cfg.get("seed", 0))
        print("[upper-audit] creating environment", flush=True)
        env = gym.make(str(cfg.task.gym_task), cfg=env_cfg)
        try:
            print("[upper-audit] resetting environment", flush=True)
            env.reset(seed=env_cfg.seed)
            print("[upper-audit] reset complete", flush=True)
            raw = env.unwrapped
            motion = raw.command_manager.get_term("motion")
            racket = raw.command_manager.get_term("racket_target")
            if motion.motion.num_motions < cases:
                raise RuntimeError("The manifest has fewer motions than requested cases")
            ids = torch.arange(cases, device=raw.device)
            motion.motion_ids[:] = ids
            motion.time_steps.zero_()
            motion.tail_steps.zero_()
            motion.prelude_elapsed_steps.zero_()
            racket._resample_command(ids)
            if not getattr(racket.cfg, "manifest_base_aligned", False):
                racket.racket_target_pos_w[:] = raw.scene.env_origins + motion.motion.strike_pos_w[ids]
            racket.racket_target_vel_w[:] = motion.motion.strike_vel_w[ids]
            racket.racket_target_normal_w[:] = motion.motion.strike_normal_w[ids]
            target_pos = racket.racket_target_pos_w.clone()
            target_vel = racket.racket_target_vel_w.clone()
            target_normal = racket.racket_target_normal_w.clone()
            hit = motion.motion.hit_frame[motion.motion_ids]
            robot = raw.scene["robot"]
            root0 = robot.data.root_pos_w.clone()
            root_max = torch.zeros(cases, device=raw.device)
            exact: list[dict[str, float] | None] = [None] * cases
            zero = torch.zeros((cases, raw.action_manager.total_action_dim), device=raw.device)
            for step in range(int(raw.max_episode_length)):
                racket.racket_target_pos_w[:] = target_pos
                racket.racket_target_vel_w[:] = target_vel
                racket.racket_target_normal_w[:] = target_normal
                env.step(zero)
                if step == 0:
                    print("[upper-audit] first control step complete", flush=True)
                root_max = torch.maximum(root_max, torch.linalg.vector_norm(robot.data.root_pos_w - root0, dim=-1))
                active = motion.prelude_elapsed_steps >= int(motion.prelude_steps)
                at_hit = active & (motion.time_steps == hit)
                for env_id in torch.nonzero(at_hit, as_tuple=False).flatten().tolist():
                    if exact[env_id] is None:
                        racket._compute_racket_state()
                        error = racket.racket_target_pos_w[env_id] - racket.racket_pos_w[env_id]
                        exact[env_id] = {
                            "motion_id": int(motion.motion_ids[env_id].item()),
                            "position_error_m": float(torch.linalg.vector_norm(error).item()),
                            "position_error_x_m": float(error[0].item()),
                            "position_error_y_m": float(error[1].item()),
                            "position_error_z_m": float(error[2].item()),
                            "root_displacement_m": float(root_max[env_id].item()),
                            "hit_control_step": step + 1,
                        }
                if all(row is not None for row in exact):
                    break
            if any(row is None for row in exact):
                raise RuntimeError(f"Exact hit was not reached for {[i for i, row in enumerate(exact) if row is None]}")
            rows = [row for row in exact if row is not None]
            report = {
                "task": str(cfg.task.gym_task),
                "correction_action": "all_zero",
                "upper_prelude_release_steps": int(raw.action_manager.get_term("joint_pos").cfg.upper_prelude_release_steps),
                "results": rows,
                "mean_position_error_m": sum(row["position_error_m"] for row in rows) / len(rows),
                "mean_root_displacement_m": sum(row["root_displacement_m"] for row in rows) / len(rows),
            }
            output = pathlib.Path(str(cfg.get("output", "eval_outputs/upper_contract/upper_correction_zero.json")))
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2), flush=True)
        finally:
            env.close()
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
