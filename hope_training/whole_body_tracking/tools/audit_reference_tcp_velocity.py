#!/usr/bin/env python3
"""Audit the reference TCP velocity using the active simulator FK contract.

This is deliberately a read-only, pre-hit kinematic audit.  It writes stored
motion poses into the current articulation without advancing physics, then
derives the racket TCP velocity from the same URDF, TCP mount and control
period used at runtime.  The derivative only uses frames ending at hit_frame;
the strike-only tail is never sampled because it is intentionally stationary.

NPZ body velocity fields are reported for provenance but are *not* treated as
the reference truth: retargeting/resampling can leave them inconsistent with
the stored pose trajectory.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from train import _apply_task_overrides  # noqa: E402


def _path(value: str, base: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _vec(value: torch.Tensor | np.ndarray) -> list[float]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    else:
        value = np.asarray(value).reshape(-1).tolist()
    return [float(x) for x in value]


def _delta(lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, Any]:
    value = lhs - rhs
    return {
        "xyz_mps": _vec(value),
        "norm_mps": float(torch.linalg.vector_norm(value).detach().cpu()),
    }


def _reference_tcp_position(raw: Any, robot: Any, racket: Any, motion: Any, motion_id: int, frame: int) -> torch.Tensor:
    """Inject a single stored pose and read the current simulator TCP FK."""
    device = raw.device
    env_id = torch.zeros(1, dtype=torch.long, device=device)
    motion_tensor_id = torch.tensor([motion_id], dtype=torch.long, device=device)
    frame_tensor = torch.tensor([frame], dtype=torch.long, device=device)
    q = motion.motion.joint_pos[motion_tensor_id, frame_tensor]
    body_pos = motion.motion._body_pos_w[motion_tensor_id, frame_tensor]
    body_quat = motion.motion._body_quat_w[motion_tensor_id, frame_tensor]
    root_pos = body_pos[:, 0] + raw.scene.env_origins[:1]
    root_quat = body_quat[:, 0]
    zero_joint_vel = torch.zeros_like(q)
    zero_root_vel = torch.zeros((1, 6), dtype=torch.float32, device=device)
    robot.write_joint_state_to_sim(q, zero_joint_vel, env_ids=env_id)
    robot.write_root_state_to_sim(
        torch.cat([root_pos, root_quat, zero_root_vel], dim=-1), env_ids=env_id
    )
    raw.scene.write_data_to_sim()
    racket._compute_racket_state()
    return racket.racket_pos_w[0].detach().clone()


def _load_actual_by_motion(path: pathlib.Path | None) -> dict[int, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    rows: dict[int, dict[str, Any]] = {}
    for row in report.get("results", []):
        # Existing deterministic reports store components as flat scalar fields.
        # Normalize them here rather than requiring a new rollout just to read
        # the same exact-hit sample.
        if "actual_velocity_mps" not in row and all(
            key in row
            for key in (
                "racket_velocity_x_mps",
                "racket_velocity_y_mps",
                "racket_velocity_z_mps",
            )
        ):
            row = dict(row)
            row["actual_velocity_mps"] = [
                row["racket_velocity_x_mps"],
                row["racket_velocity_y_mps"],
                row["racket_velocity_z_mps"],
            ]
        rows[int(row["motion_id"])] = row
    return rows


def _run(cfg: Any) -> dict[str, Any]:
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab.utils.math import quat_apply

    import training.tasks  # noqa: F401

    task_id = str(cfg.task.gym_task)
    manifest_value = cfg.get("motion_manifest") or cfg.task.get("motion_manifest")
    if manifest_value is None:
        raise ValueError("motion_manifest is required")
    manifest_path = _path(str(manifest_value), pathlib.Path.cwd())
    actual_report_value = cfg.get("actual_report", None)
    actual_report_path = (
        _path(str(actual_report_value), pathlib.Path.cwd()) if actual_report_value else None
    )
    actual_by_motion = _load_actual_by_motion(actual_report_path)
    fit_frames = int(cfg.get("fit_frames", 5) or 5)
    if fit_frames < 2:
        raise ValueError("fit_frames must be at least 2")

    env_cfg = parse_env_cfg(task_id, device=str(cfg.device), num_envs=1)
    _apply_task_overrides(env_cfg, cfg.task)
    env_cfg.sim.device = str(cfg.device)
    env_cfg.seed = int(cfg.get("seed", 0) or 0)
    # This is a FK probe, not a floating/fixed dynamics comparison.  Root pose
    # is explicitly injected below, so fixing the base prevents physics state
    # from entering this purely geometric contract audit.
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

        control_dt = float(env_cfg.sim.dt * env_cfg.decimation)
        wrist_index = int(racket._wrist_body_index)
        results: list[dict[str, Any]] = []
        for motion_id in range(int(motion.motion.num_motions)):
            hit = int(motion.motion.hit_frame[motion_id].detach().cpu())
            length = int(motion.motion.motion_lengths[motion_id].detach().cpu())
            first = hit - fit_frames + 1
            if first < 0:
                raise RuntimeError(
                    f"motion {motion_id} hit_frame={hit} has fewer than {fit_frames} pre-hit frames"
                )
            if hit >= length:
                raise RuntimeError(f"motion {motion_id} hit_frame={hit} outside length={length}")

            frames = list(range(first, hit + 1))
            positions = torch.stack(
                [_reference_tcp_position(raw, robot, racket, motion, motion_id, frame) for frame in frames]
            )
            # Fit p(t) = a*t^2 + b*t + c at t=0 (the hit frame) using only
            # pre-hit samples.  b is a low-noise, causal derivative estimate.
            times = np.asarray([(frame - hit) * control_dt for frame in frames], dtype=np.float64)
            coeff = np.polyfit(times, positions.detach().cpu().numpy(), deg=min(2, len(frames) - 1))
            canonical_fit = torch.as_tensor(coeff[-2], dtype=torch.float32, device=raw.device)
            canonical_backward = (positions[-1] - positions[-2]) / control_dt

            tensor_id = torch.tensor(motion_id, dtype=torch.long, device=raw.device)
            target = motion.motion.strike_vel_w[tensor_id].detach().clone()
            wrist_pos = motion.motion._body_pos_w[tensor_id, hit, wrist_index]
            wrist_quat = motion.motion._body_quat_w[tensor_id, hit, wrist_index]
            offset = racket._mount_offset[0]
            offset_world = quat_apply(wrist_quat.unsqueeze(0), offset.unsqueeze(0))[0]
            declared = (
                motion.motion._body_lin_vel_w[tensor_id, hit, wrist_index]
                + torch.linalg.cross(
                    motion.motion._body_ang_vel_w[tensor_id, hit, wrist_index], offset_world, dim=-1
                )
            )
            actual = actual_by_motion.get(motion_id)
            actual_velocity = actual.get("actual_velocity_mps") if actual else None
            actual_tensor = (
                torch.tensor(actual_velocity, dtype=torch.float32, device=raw.device)
                if actual_velocity is not None
                else None
            )
            row: dict[str, Any] = {
                "motion_id": motion_id,
                "episode_id": str(motion.motion.episode_ids[motion_id]),
                "motion_length_frames": length,
                "hit_frame": hit,
                "control_dt_s": control_dt,
                "fit_frames": frames,
                "tail_sampled": False,
                "manifest_task_velocity_mps": _vec(target),
                "canonical_reference_tcp_velocity_pre_hit_fit_mps": _vec(canonical_fit),
                "canonical_reference_tcp_velocity_pre_hit_backward_mps": _vec(canonical_backward),
                "npz_declared_tcp_velocity_mps": _vec(declared),
                "npz_declared_minus_canonical_mps": _delta(declared, canonical_fit),
                "task_minus_reference_mps": _delta(target, canonical_fit),
                "reference_tcp_positions_m": [_vec(p) for p in positions],
            }
            if actual_tensor is not None:
                row["actual_velocity_mps"] = _vec(actual_tensor)
                row["reference_minus_actual_mps"] = _delta(canonical_fit, actual_tensor)
                row["task_minus_actual_mps"] = _delta(target, actual_tensor)
                row["actual_source"] = str(actual_report_path)
            else:
                row["actual_velocity_mps"] = None
                row["actual_source"] = None
            results.append(row)

        return {
            "purpose": "Read-only canonical reference TCP velocity audit",
            "task": task_id,
            "manifest": str(manifest_path),
            "actual_report": str(actual_report_path) if actual_report_path else None,
            "reference_contract": {
                "source": "current_simulator_fk_from_stored_joint_and_root_pose",
                "control_dt_s": control_dt,
                "derivative": "causal quadratic fit over pre-hit frames; backward difference retained as cross-check",
                "does_not_sample_strike_only_tail": True,
                "wrist_body_index": wrist_index,
                "wrist_body_name": str(robot.body_names[wrist_index]),
                "tcp_mount_offset_m": _vec(racket._mount_offset[0]),
            },
            "npz_velocity_interpretation": "provenance_only_not_reference_truth",
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
        output = _path(
            str(cfg.get("output", "eval_outputs/reference_velocity_contract/reference_tcp_velocity.json")),
            pathlib.Path.cwd(),
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        for row in report["results"]:
            task = np.asarray(row["manifest_task_velocity_mps"])
            reference = np.asarray(row["canonical_reference_tcp_velocity_pre_hit_fit_mps"])
            print(
                f"[velocity] {row['episode_id']}: target={task.round(3).tolist()} "
                f"reference={reference.round(3).tolist()} "
                f"|target-reference|={row['task_minus_reference_mps']['norm_mps']:.3f}m/s",
                flush=True,
            )
        print(f"[velocity] wrote {output}", flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
