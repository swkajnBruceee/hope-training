"""Run the bounded D0--D1/D3--D6 fall/recovery admission smoke audit.

This is an evaluation-only harness.  It never loads PPO and never writes a
checkpoint.  D0/D1/D3/D4/D5/D6 use the real IsaacLab/PhysX A3 plant with a
zero action and explicitly injected physical states.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch


TASK = "HOPE-FloatingF0-AgibotA3-v0"
# D4 defaults to the lightweight F0 plant so the audit never silently loads a
# checkpoint.  A caller may explicitly select a prior-backed task for a
# separate diagnostic, but that result must still be recorded with its task
# identity and cannot silently replace the default evidence.
D4_TASK = TASK
MANIFEST = "sample_motions/p2_data260708_backhand_strike_only_v1/manifest.json"


def _snapshot(state) -> dict:
    return {
        "risk_score": float(state.risk_score[0].item()),
        "risk_level": int(state.risk_level[0].item()),
        "fall_reason": int(state.fall_reason[0].item()),
        "predicted_unrecoverable": bool(state.predicted_unrecoverable[0].item()),
        "confirmed_fall": bool(state.confirmed_fall[0].item()),
        "recovery_ready": bool(state.recovery_ready[0].item()),
        "recovery_progress": float(state.recovery_progress[0].item()),
        "forward_tilt_rad": float(state.forward_tilt_rad[0].item()),
        "torso_forward_tilt_rad": float(state.torso_forward_tilt_rad[0].item()),
        "relative_root_height_m": float(state.relative_root_height_m[0].item()),
        "support_min_margin_m": float(state.support_margins[0].min().item()),
        "foot_slip_max_mps": float(state.foot_slip_mps[0].max().item()),
    }


def _tilt_quat_y(angle_rad: float, device: torch.device) -> torch.Tensor:
    # Isaac/IsaacLab quaternions use wxyz.
    return torch.tensor(
        [math.cos(angle_rad / 2.0), 0.0, math.sin(angle_rad / 2.0), 0.0],
        dtype=torch.float32,
        device=device,
    )


def _inject_root_state(env, *, tilt_rad: float | None = None, velocity_x: float = 0.0) -> None:
    robot = env.unwrapped.scene["robot"]
    pos = robot.data.root_pos_w[:1].detach().clone()
    quat = robot.data.root_quat_w[:1].detach().clone()
    if tilt_rad is not None:
        quat[:] = _tilt_quat_y(float(tilt_rad), env.unwrapped.device)
    velocity = robot.data.root_lin_vel_w[:1].detach().clone()
    velocity[:, 0] = float(velocity_x)
    robot.write_root_pose_to_sim(torch.cat((pos, quat), dim=-1))
    robot.write_root_velocity_to_sim(torch.cat((velocity, robot.data.root_ang_vel_w[:1].detach().clone()), dim=-1))
    env.unwrapped.scene.write_data_to_sim()
    env.unwrapped.sim.forward()
    env.unwrapped.scene.update(dt=env.unwrapped.step_dt)


def _make_env(gym, parse_env_cfg, scenario: str, device: str, d4_task: str | None = None):
    task = (d4_task or D4_TASK) if scenario == "D4" else TASK
    print(f"[fall-audit] creating {task} for {scenario}", flush=True)
    cfg = parse_env_cfg(task, device=device, num_envs=1)
    cfg.commands.motion.motion_manifest = os.path.abspath(MANIFEST)
    cfg.commands.racket_target.target_mode = "manifest"
    if scenario == "D4":
        # The post-hit delay case needs the nominal zero-action reference to
        # survive through the hit marker before the injected fall.  Use the
        # documented high-gain Isaac comparison profile only for this audit;
        # it is not a training or teacher-qualification claim.  Crucially,
        # Keep the strict termination manager; this default probe intentionally
        # does not load any checkpoint.
        cfg.apply_native_actuator_profile("official_pd_stand_approx")
    # Keep the robot in the ready/prelude region for isolated recovery cases.
    if scenario in {"D0", "D1", "D3", "D5", "D6"}:
        cfg.commands.motion.prelude_steps = 1000
    return gym.make(task, cfg=cfg)


def _run_one(gym, parse_env_cfg, unified_fall_state, scenario: str, device: str, d4_task: str | None = None) -> dict:
    env = _make_env(gym, parse_env_cfg, scenario, device, d4_task)
    print(f"[fall-audit] env created for {scenario}", flush=True)
    observations, _ = env.reset()
    print(f"[fall-audit] reset complete for {scenario}", flush=True)
    records = []
    terminal_snapshots = []
    current_step = {"value": 0}
    original_reset_idx = env.unwrapped._reset_idx

    def _audit_reset_idx(env_ids):
        # ManagerBasedRLEnv resets inside step() immediately after computing
        # termination.  Capture the unified state before that reset erases the
        # physical evidence, including active termination terms where exposed.
        state_before_reset = unified_fall_state(env.unwrapped)
        snapshot = _snapshot(state_before_reset)
        snapshot.update({
            "step": int(current_step["value"]),
            "pre_reset": True,
            "reset_env_ids": torch.as_tensor(env_ids).detach().cpu().tolist(),
        })
        try:
            manager = env.unwrapped.termination_manager
            names = list(getattr(manager, "active_terms", ()))
            snapshot["active_termination_terms"] = [
                name for name in names
                if bool(torch.as_tensor(manager.get_term(name))[0].item())
            ]
        except Exception:
            snapshot["active_termination_terms"] = []
        terminal_snapshots.append(snapshot)
        return original_reset_idx(env_ids)

    env.unwrapped._reset_idx = _audit_reset_idx
    terminated_step = None
    injected = False
    restored = False
    max_steps = {"D0": 35, "D1": 55, "D3": 30, "D4": 110, "D5": 70, "D6": 35}[scenario]
    try:
        for step in range(1, max_steps + 1):
            current_step["value"] = step
            if scenario in {"D1", "D3", "D5", "D6"} and step == 10:
                if scenario == "D6":
                    _inject_root_state(env, velocity_x=0.30)
                else:
                    # 0.18--0.20 rad is deliberately inside the recoverable
                    # envelope; 0.95 rad is the clearly unrecoverable case.
                    _inject_root_state(env, tilt_rad=0.20 if scenario in {"D1", "D5"} else 0.95)
                injected = True
            if scenario in {"D1", "D5"} and step == 14:
                _inject_root_state(env, tilt_rad=0.0, velocity_x=0.0)
                restored = True
            # D4 is deliberately post-hit: 50-step prelude + ~30-frame clip.
            if scenario == "D4" and step == 86:
                _inject_root_state(env, tilt_rad=0.95)
                injected = True
            action = torch.zeros((1, env.unwrapped.action_manager.total_action_dim), device=env.unwrapped.device)
            if step == 1:
                print(f"[fall-audit] first action step for {scenario}", flush=True)
            observations, reward, terminated, truncated, info = env.step(action)
            state = unified_fall_state(env.unwrapped)
            row = _snapshot(state)
            if bool(torch.as_tensor(terminated)[0].item()) and terminal_snapshots:
                # Replace the post-reset state with the captured physical
                # state; retain the returned reset flags below.
                row = dict(terminal_snapshots[-1])
            row.update(
                {
                    "step": step,
                    "terminated": bool(torch.as_tensor(terminated)[0].item()),
                    "truncated": bool(torch.as_tensor(truncated)[0].item()),
                    "injected": injected,
                    "restored": restored,
                }
            )
            records.append(row)
            if row["terminated"] and terminated_step is None:
                terminated_step = step
                break
        if scenario == "D0":
            # D0 is a nominal READY hold, not a full swing success claim.
            status = "PASS" if terminated_step is None and not any(r["confirmed_fall"] for r in records) else "FAIL"
        elif scenario == "D1":
            status = "PASS" if terminated_step is None and any(r["recovery_ready"] for r in records) else "FAIL"
        elif scenario == "D3":
            status = "PASS" if terminated_step is not None else "FAIL"
        elif scenario == "D4":
            status = "PASS" if terminated_step is not None and any(r["injected"] for r in records) else "FAIL"
        elif scenario == "D5":
            status = "PASS" if any(r["recovery_ready"] for r in records) else "FAIL"
        else:  # D6
            status = "PASS" if any(r["foot_slip_max_mps"] > 0.08 or r["predicted_unrecoverable"] for r in records) else "FAIL"
        return {
            "scenario": scenario,
            "task": (d4_task or D4_TASK) if scenario == "D4" else TASK,
            "status": status,
            "terminated_step": terminated_step,
            "records": records,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="eval_outputs/fall_recovery_physx_audit_v1.json")
    parser.add_argument("--scenario", choices=("D0", "D1", "D3", "D4", "D5", "D6"), default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--d4-task", default=None, help="explicit prior-backed D4 task for a separate diagnostic")
    args = parser.parse_args()

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, device=args.device).app
    try:
        import gymnasium as gym
        import training.tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg
        from training.tasks.tracking.mdp.fall_state import unified_fall_state

        scenarios = (args.scenario,) if args.scenario is not None else ("D0", "D1", "D3", "D4", "D5", "D6")
        report = {
            "schema_version": "fall_recovery_physx_audit/v1",
            "task": TASK,
            "scenario_tasks": {"D4": args.d4_task or D4_TASK, "D0/D1/D3/D5/D6": TASK},
            "manifest": os.path.abspath(MANIFEST),
            "training_started": False,
            "scenarios": [_run_one(gym, parse_env_cfg, unified_fall_state, s, args.device, args.d4_task) for s in scenarios],
        }
        path = Path(args.output).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"output": str(path), "statuses": {x["scenario"]: x["status"] for x in report["scenarios"]}}, ensure_ascii=False), flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
