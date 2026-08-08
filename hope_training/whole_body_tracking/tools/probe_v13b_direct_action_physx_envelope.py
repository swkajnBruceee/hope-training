#!/usr/bin/env python3
"""Parallel PhysX envelope probe for the V1.3B 22-joint direct action."""
from __future__ import annotations

import json
import math
import os
import pathlib
import sys
from typing import Any

import torch
from isaaclab.app import AppLauncher

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TASK_ID = os.environ.get(
    "V13B_PROBE_TASK",
    "HOPE-FloatingTargetConditionedReferenceFreeV13B-AgibotA3-v0",
)
DEVICE = os.environ.get("V13B_PROBE_DEVICE", "cuda:0")
MOTION_MANIFEST = os.environ.get("V13B_PROBE_MOTION_MANIFEST", "")
# Fine levels locate a safe per-joint scale when the nominal READY pose is
# close to an asymmetric URDF limit; the requested +/-0.25..1.00 points are
# still retained verbatim.  The 0.01--0.04 points are important for joints
# whose READY pose is close to a one-sided limit (notably shoulder roll).
LEVELS = (0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.15, 0.20, 0.25, 0.50, 0.75, 1.00)
_LEVEL_OVERRIDE = os.environ.get("V13B_PROBE_LEVELS", "").strip()
if _LEVEL_OVERRIDE:
    LEVELS = tuple(float(x) for x in _LEVEL_OVERRIDE.split(",") if x.strip())
SETTLE_STEPS = 24
CONTROL_STEPS = 24
SOFT_LIMIT_MARGIN_RAD = 0.02
MAX_TILT_RAD = math.radians(45.0)
MIN_ROOT_HEIGHT_M = 0.82
MAX_TORQUE_FRAC = 0.95
MAX_VELOCITY_FRAC = 0.95


def _tilt(robot) -> torch.Tensor:
    g = robot.data.projected_gravity_b
    return torch.arccos(torch.clamp(-g[:, 2], -1.0, 1.0))


def _finite(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().detach().cpu())


def main():
    app = AppLauncher(headless=True, device=DEVICE, enable_cameras=False).app
    try:
        import gymnasium as gym
        import training.tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg

        case_specs = [
            {"channel": ch, "sign": sign, "normalized_level": level}
            for ch in range(22)
            for sign in (-1.0, 1.0)
            for level in LEVELS
        ]
        num_envs = len(case_specs)
        cfg = parse_env_cfg(TASK_ID, device=DEVICE, num_envs=num_envs)
        cfg.scene.num_envs = num_envs
        cfg.episode_length_s = max(float(getattr(cfg, "episode_length_s", 10.0)), 10.0)
        if MOTION_MANIFEST:
            if getattr(cfg.commands, "motion", None) is None:
                raise RuntimeError("V13B_PROBE_MOTION_MANIFEST was supplied but the selected task has no motion command")
            cfg.commands.motion.motion_manifest = str(pathlib.Path(MOTION_MANIFEST).expanduser().resolve())
            cfg.commands.motion.motion_file = None
        env = gym.make(TASK_ID, cfg=cfg, render_mode=None)
        try:
            env.reset()
            raw = env.unwrapped
            robot = raw.scene["robot"]
            action_term = raw.action_manager.get_term("joint_pos")
            backend_names = list(robot.joint_names)
            controlled_names = list(action_term.cfg.base_joint_names)[:12] + list(action_term.cfg.upper_joint_names)
            controlled_ids = [backend_names.index(name) for name in controlled_names]
            direct_scales = torch.cat([action_term._lower_scale_direct[0], action_term._upper_scale_direct[0]])
            device = raw.device

            # Restore the same audited READY state in every vectorized case.
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] = torch.tensor((-0.5, -0.7625, 1.04), device=device)
            root_state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device)
            root_state[:, 7:13] = 0.0
            robot.write_root_state_to_sim(root_state)
            robot.write_joint_state_to_sim(action_term._ready_full.clone(), torch.zeros_like(action_term._ready_full))
            raw.action_manager.reset(list(range(num_envs)))
            raw.scene.write_data_to_sim()
            raw.sim.forward()

            zero = torch.zeros((num_envs, 26), device=device)
            settle_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
            for _ in range(SETTLE_STEPS):
                out = env.step(zero)
                settle_done |= (out[2] | out[3])

            action = torch.zeros((num_envs, 26), device=device)
            for i, spec in enumerate(case_specs):
                action[i, spec["channel"]] = spec["sign"] * spec["normalized_level"]

            max_tilt = torch.zeros(num_envs, device=device)
            min_height = torch.full((num_envs,), float("inf"), device=device)
            min_margin = torch.full((num_envs,), float("inf"), device=device)
            max_torque_frac = torch.zeros(num_envs, device=device)
            max_velocity_frac = torch.zeros(num_envs, device=device)
            max_action_sat = torch.zeros(num_envs, device=device)
            finite = torch.ones(num_envs, dtype=torch.bool, device=device)
            done = settle_done.clone()
            done_step = torch.zeros(num_envs, dtype=torch.long, device=device)

            for step in range(CONTROL_STEPS):
                step_action = torch.where(done.unsqueeze(-1), zero, action)
                out = env.step(step_action)
                done_now = out[2] | out[3]
                newly_done = done_now & ~done
                done_step[newly_done] = step + 1
                done |= done_now
                data = robot.data
                finite &= torch.isfinite(data.joint_pos).all(dim=-1)
                min_height = torch.minimum(min_height, data.root_pos_w[:, 2])
                max_tilt = torch.maximum(max_tilt, _tilt(robot))
                limits = data.soft_joint_pos_limits[:, controlled_ids]
                q = data.joint_pos[:, controlled_ids]
                min_margin = torch.minimum(min_margin, torch.minimum(q - limits[..., 0], limits[..., 1] - q).min(dim=-1).values)
                effort = data.joint_effort_limits[:, controlled_ids].clamp_min(1.0e-6)
                vel_lim = data.joint_vel_limits[:, controlled_ids].clamp_min(1.0e-6)
                max_torque_frac = torch.maximum(max_torque_frac, (data.applied_torque[:, controlled_ids].abs() / effort).max(dim=-1).values)
                max_velocity_frac = torch.maximum(max_velocity_frac, (data.joint_vel[:, controlled_ids].abs() / vel_lim).max(dim=-1).values)
                processed = getattr(action_term, "_raw_actions", step_action)
                max_action_sat = torch.maximum(max_action_sat, (processed[:, :22] - step_action[:, :22]).abs().max(dim=-1).values)

            cases = []
            for i, spec in enumerate(case_specs):
                scale = float(direct_scales[spec["channel"]].detach().cpu())
                row = {
                    **spec,
                    "joint_name": controlled_names[spec["channel"]],
                    "joint_index": controlled_ids[spec["channel"]],
                    "scale_rad": scale,
                    "commanded_peak_rad": abs(float(spec["normalized_level"])) * scale,
                    "settle_done": bool(settle_done[i].detach().cpu()),
                    "done": bool(done[i].detach().cpu()),
                    "done_step": int(done_step[i].detach().cpu()),
                    "finite": bool(finite[i].detach().cpu()),
                    "min_root_height_m": float(min_height[i].detach().cpu()),
                    "max_tilt_rad": float(max_tilt[i].detach().cpu()),
                    "min_soft_limit_margin_rad": float(min_margin[i].detach().cpu()),
                    "max_torque_fraction": float(max_torque_frac[i].detach().cpu()),
                    "max_velocity_fraction": float(max_velocity_frac[i].detach().cpu()),
                    "max_action_saturation_abs": float(max_action_sat[i].detach().cpu()),
                }
                row["safe"] = bool(
                    not row["done"]
                    and row["finite"]
                    and row["min_root_height_m"] >= MIN_ROOT_HEIGHT_M
                    and row["max_tilt_rad"] <= MAX_TILT_RAD
                    and row["min_soft_limit_margin_rad"] >= SOFT_LIMIT_MARGIN_RAD
                    and row["max_torque_fraction"] <= MAX_TORQUE_FRAC
                    and row["max_velocity_fraction"] <= MAX_VELOCITY_FRAC
                )
                cases.append(row)
            per_joint = []
            for channel, name in enumerate(controlled_names):
                jc = [c for c in cases if c["channel"] == channel]
                # A PPO action scale must be safe for both action signs.  Do
                # not take the maximum over either sign: that would certify a
                # scale which is only safe in one direction for asymmetric
                # URDF limits.
                safe_by_sign = {}
                max_safe_by_sign = {}
                for sign in (-1.0, 1.0):
                    signed = [c for c in jc if c["sign"] == sign]
                    safe_levels = [c["normalized_level"] for c in signed if c["safe"]]
                    safe_by_sign[str(int(sign))] = safe_levels
                    max_safe_by_sign[str(int(sign))] = max(safe_levels, default=0.0)
                max_safe = min(max_safe_by_sign.values(), default=0.0)
                scale = float(direct_scales[channel].detach().cpu())
                per_joint.append({
                    "channel": channel,
                    "joint_name": name,
                    "joint_index": controlled_ids[channel],
                    "current_scale_rad": scale,
                    "safe_levels_by_sign": safe_by_sign,
                    "max_safe_normalized_level_by_sign": max_safe_by_sign,
                    "max_safe_normalized_level": max_safe,
                    "max_safe_commanded_rad": max_safe * scale,
                    "all_four_levels_safe": max_safe >= 1.0,
                    "symmetric_safe": max_safe > 0.0,
                })
            result = {
                "status": "physx_dynamic_envelope_probe_complete",
                "task": TASK_ID,
                "num_envs": num_envs,
                "settle_steps_per_case": SETTLE_STEPS,
                "control_steps_per_case": CONTROL_STEPS,
                "levels": list(LEVELS),
                "controlled_joint_names": controlled_names,
                "current_scale_status": getattr(raw, "v13b_direct_scale_status", None),
                "self_collision_enabled": False,
                "self_collision_check": "disabled_by_articulation_contract",
                "criteria": {
                    "min_root_height_m": MIN_ROOT_HEIGHT_M,
                    "max_tilt_deg": math.degrees(MAX_TILT_RAD),
                    "min_soft_limit_margin_rad": SOFT_LIMIT_MARGIN_RAD,
                    "max_torque_fraction": MAX_TORQUE_FRAC,
                    "max_velocity_fraction": MAX_VELOCITY_FRAC,
                },
                "cases": cases,
                "per_joint": per_joint,
                "all_cases_safe": all(c["safe"] for c in cases),
                "all_channels_have_safe_level": all(p["max_safe_normalized_level"] > 0.0 for p in per_joint),
                "all_channels_have_symmetric_safe_level": all(p["symmetric_safe"] for p in per_joint),
            }
            out_dir_name = os.environ.get(
                "V13B_PROBE_OUTPUT_DIR",
                "target_conditioned_v13b_direct_action_probe",
            )
            out_dir = ROOT / "eval_outputs" / out_dir_name
            out_dir.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(result, indent=2)
            (out_dir / "physx_dynamic_envelope_raw.json").write_text(payload, encoding="utf-8")
            (out_dir / "physx_dynamic_envelope_summary.json").write_text(
                json.dumps({k: result[k] for k in ("status", "current_scale_status", "criteria", "per_joint", "all_cases_safe", "all_channels_have_safe_level", "all_channels_have_symmetric_safe_level")}, indent=2),
                encoding="utf-8",
            )
            print(json.dumps({"status": result["status"], "all_cases_safe": result["all_cases_safe"], "all_channels_have_safe_level": result["all_channels_have_safe_level"], "all_channels_have_symmetric_safe_level": result["all_channels_have_symmetric_safe_level"]}, indent=2), flush=True)
        finally:
            env.close()
    finally:
        app.close()


if __name__ == "__main__":
    main()
