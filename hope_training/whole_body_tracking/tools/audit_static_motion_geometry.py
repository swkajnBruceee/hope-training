#!/usr/bin/env python3
"""Static hit-frame geometry audit for the A3 motion/target contract.

This audit writes one stored motion pose into the current Isaac articulation and
does not call ``env.step`` or advance physics.  It compares:

* NPZ body poses and wrist-mounted racket FK;
* current simulator body poses and the same wrist-mounted racket FK;
* manifest target position;
* NPZ joint values against the simulator articulation order.

The output is diagnostic-only.  It never edits motion files or manifests.
"""

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


def _path(value: str, base: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _vec(value: torch.Tensor) -> list[float]:
    return [float(x) for x in value.detach().cpu().reshape(-1).tolist()]


def _norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value).detach().cpu())


def _error(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    delta = actual - reference
    return {"xyz_m": _vec(delta), "norm_m": _norm(delta)}


def _quat_error_deg(actual: torch.Tensor, reference: torch.Tensor) -> float:
    from isaaclab.utils.math import quat_error_magnitude

    return float(torch.rad2deg(quat_error_magnitude(reference, actual)).detach().cpu())


def _run(cfg: Any) -> dict[str, Any]:
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg

    import training.tasks  # noqa: F401

    task_id = str(cfg.task.gym_task)
    manifest_value = cfg.get("motion_manifest") or cfg.task.get("motion_manifest")
    if manifest_value is None:
        raise ValueError("motion_manifest is required")
    manifest_path = _path(str(manifest_value), pathlib.Path.cwd())
    num_envs = int(cfg.get("num_envs", 6) or 6)
    env_cfg = parse_env_cfg(task_id, device=str(cfg.device), num_envs=num_envs)
    _apply_task_overrides(env_cfg, cfg.task)
    env_cfg.sim.device = str(cfg.device)
    env_cfg.seed = int(cfg.get("seed", 0) or 0)
    env_cfg.scene.robot.spawn.fix_base = True
    env_cfg.commands.motion.motion_manifest = str(manifest_path)
    env_cfg.commands.motion.motion_file = None
    env_cfg.commands.motion.manifest_subset_size = None
    frame_offset = cfg.get("manifest_frame_z_offset", None)
    if frame_offset is None:
        frame_offset = cfg.task.get("manifest_frame_z_offset", 0.0)
    env_cfg.commands.motion.manifest_frame_z_offset = float(frame_offset or 0.0)

    env = gym.make(task_id, cfg=env_cfg, render_mode=None)
    try:
        raw = env.unwrapped
        robot = raw.scene["robot"]
        motion = raw.command_manager.get_term("motion")
        racket = raw.command_manager.get_term("racket_target")
        env.reset()

        count = min(num_envs, int(motion.motion.num_motions))
        if count <= 0:
            raise RuntimeError("motion manifest contains no motions")
        env_ids = torch.arange(count, device=raw.device, dtype=torch.long)
        motion_ids = torch.arange(count, device=raw.device, dtype=torch.long)
        motion.motion_ids[:count] = motion_ids
        hit_frames = motion.motion.hit_frame[motion_ids]
        motion.time_steps[:count] = hit_frames
        motion.tail_steps[:count] = 0
        motion.prelude_elapsed_steps[:count] = 0

        # The motion library has already applied the configured z offset to both
        # body_pos_w and strike_pos_w.  Use those exact tensors for the paired audit.
        q_npz = motion.motion.joint_pos[motion_ids, hit_frames].clone()
        body_npz = motion.motion._body_pos_w[motion_ids, hit_frames].clone()
        quat_npz = motion.motion._body_quat_w[motion_ids, hit_frames].clone()
        root_pos = body_npz[:, 0] + raw.scene.env_origins[:count]
        root_quat = quat_npz[:, 0]
        zeros_j = torch.zeros_like(q_npz)
        zeros_root = torch.zeros((count, 6), dtype=torch.float32, device=raw.device)
        robot.write_joint_state_to_sim(q_npz, zeros_j, env_ids=env_ids)
        robot.write_root_state_to_sim(
            torch.cat([root_pos, root_quat, zeros_root], dim=-1), env_ids=env_ids
        )
        raw.scene.write_data_to_sim()

        # No env.step(), sim.step(), render(), or scene.update(): this is a pure
        # kinematic read immediately after the state injection.
        racket._compute_racket_state()
        sim_body = robot.data.body_pos_w[:count].detach().clone()
        sim_quat = robot.data.body_quat_w[:count].detach().clone()
        sim_q = robot.data.joint_pos[:count].detach().clone()
        sim_racket = racket.racket_pos_w[:count].detach().clone()
        sim_racket_quat = racket.racket_quat_w[:count].detach().clone()

        action_term = raw.action_manager.get_term("joint_pos")
        upper_ids = action_term._upper_joint_ids_tensor
        q_default_upper = robot.data.default_joint_pos[:count].clone()
        q_default_upper[:, upper_ids] = q_npz[:, upper_ids]
        robot.write_joint_state_to_sim(q_default_upper, zeros_j, env_ids=env_ids)
        raw.scene.write_data_to_sim()
        racket._compute_racket_state()
        default_upper_racket = racket.racket_pos_w[:count].detach().clone()

        # Restore the original all-NPZ pose before leaving the probe.  The
        # report below contains both variants, so a leg/head/left-arm mismatch
        # cannot be mistaken for an upper-body geometry mismatch.
        robot.write_joint_state_to_sim(q_npz, zeros_j, env_ids=env_ids)
        raw.scene.write_data_to_sim()
        racket._compute_racket_state()

        racket_target = motion.motion.strike_pos_w[motion_ids].clone()
        if getattr(racket.cfg, "manifest_base_aligned", False):
            raise RuntimeError("audit expects manifest_base_aligned=false for absolute target comparison")
        racket_target = racket_target + raw.scene.env_origins[:count]

        mount = racket._mount_offset[:count]
        from isaaclab.utils.math import quat_apply, quat_mul

        wrist_index = int(racket._wrist_body_index)
        npz_wrist = body_npz[:, wrist_index]
        npz_wrist_quat = quat_npz[:, wrist_index]
        npz_racket = npz_wrist + quat_apply(npz_wrist_quat, mount)
        npz_racket_quat = quat_mul(npz_wrist_quat, racket._mount_quat[:count])

        body_names = list(robot.body_names)
        joint_names = list(robot.joint_names)
        body_rows = []
        for i in range(count):
            per_body = []
            for body_index, name in enumerate(body_names):
                per_body.append(
                    {
                        "index": body_index,
                        "name": name,
                        "sim_pos_m": _vec(sim_body[i, body_index]),
                        "npz_pos_m": _vec(body_npz[i, body_index] + raw.scene.env_origins[i]),
                        "sim_minus_npz": _error(
                            sim_body[i, body_index], body_npz[i, body_index] + raw.scene.env_origins[i]
                        ),
                        "orientation_error_deg": _quat_error_deg(
                            sim_quat[i, body_index], quat_npz[i, body_index]
                        ),
                    }
                )
            body_rows.append(per_body)

        results = []
        for i in range(count):
            joint_rows = []
            for joint_index, name in enumerate(joint_names):
                delta = sim_q[i, joint_index] - q_npz[i, joint_index]
                joint_rows.append(
                    {
                        "npz_index": joint_index,
                        "sim_index": joint_index,
                        "name": name,
                        "npz_rad": float(q_npz[i, joint_index].detach().cpu()),
                        "sim_rad": float(sim_q[i, joint_index].detach().cpu()),
                        "sim_minus_npz_rad": float(delta.detach().cpu()),
                    }
                )
            body_errors = torch.linalg.vector_norm(
                sim_body[i] - (body_npz[i] + raw.scene.env_origins[i]), dim=-1
            )
            results.append(
                {
                    "motion_id": int(motion_ids[i].cpu()),
                    "episode_id": str(motion.motion.episode_ids[int(motion_ids[i].cpu())]),
                    "hit_frame": int(hit_frames[i].cpu()),
                    "root_sim_m": _vec(robot.data.root_pos_w[i]),
                    "root_npz_m": _vec(root_pos[i]),
                    "root_error": _error(robot.data.root_pos_w[i], root_pos[i]),
                    "joint_max_abs_error_rad": float(torch.max(torch.abs(sim_q[i] - q_npz[i])).cpu()),
                    "joint_rows": joint_rows,
                    "worst_body_error_m": float(body_errors.max().cpu()),
                    "mean_body_error_m": float(body_errors.mean().cpu()),
                    "racket_target_m": _vec(racket_target[i]),
                    "npz_body_fk_racket_m": _vec(npz_racket[i] + raw.scene.env_origins[i]),
                    "sim_joint_fk_racket_m": _vec(sim_racket[i]),
                    "npz_body_fk_minus_target": _error(
                        npz_racket[i] + raw.scene.env_origins[i], racket_target[i]
                    ),
                    "sim_joint_fk_minus_npz_body_fk": _error(
                        sim_racket[i], npz_racket[i] + raw.scene.env_origins[i]
                    ),
                    "sim_joint_fk_minus_target": _error(sim_racket[i], racket_target[i]),
                    "sim_default_body_with_npz_upper_racket_m": _vec(default_upper_racket[i]),
                    "sim_default_body_with_npz_upper_minus_target": _error(
                        default_upper_racket[i], racket_target[i]
                    ),
                    "racket_orientation_error_vs_npz_deg": _quat_error_deg(
                        sim_racket_quat[i], npz_racket_quat[i]
                    ),
                    "wrist_body_index": wrist_index,
                    "wrist_body_name": body_names[wrist_index],
                    "mount_offset_m": _vec(mount[i]),
                    "body_rows": body_rows[i],
                }
            )

        return {
            "task": task_id,
            "manifest": str(manifest_path),
            "frame_z_offset_m": float(env_cfg.commands.motion.manifest_frame_z_offset),
            "physics_advanced": False,
            "body_names": body_names,
            "joint_names": joint_names,
            "racket_fk_contract": {
                "mode": str(racket._racket_mode),
                "wrist_body_index": wrist_index,
                "wrist_body_name": body_names[wrist_index],
                "mount_offset_m": _vec(racket._mount_offset[0]),
                "mount_quat_wxyz": _vec(racket._mount_quat[0]),
            },
            "results": results,
        }
    finally:
        env.close()


@hydra.main(version_base=None, config_path="../cfg", config_name="play")
def main(cfg: Any) -> None:
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, device=str(cfg.device), enable_cameras=False)
    simulation_app = app_launcher.app
    try:
        report = _run(cfg)
        output = _path(str(cfg.get("output", "eval_outputs/static_motion_geometry_audit.json")), pathlib.Path.cwd())
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        for row in report["results"]:
            print(
                f"[static] {row['episode_id']}: "
                f"bodyFK-target={row['npz_body_fk_minus_target']['norm_m']:.4f}m "
                f"simFK-bodyFK={row['sim_joint_fk_minus_npz_body_fk']['norm_m']:.4f}m "
                f"simFK-target={row['sim_joint_fk_minus_target']['norm_m']:.4f}m "
                f"body_max={row['worst_body_error_m']:.4f}m "
                f"qmax={row['joint_max_abs_error_rad']:.6f}rad",
                flush=True,
            )
        print(f"[static] wrote {output}", flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
