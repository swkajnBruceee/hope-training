#!/usr/bin/env python3
"""One-env reset/zero-action diagnostic for V1.3B.

This is read-only: it creates one environment, records the state immediately
after reset, applies one zero 26-D action, and records the state/termination
signals after that control step.  It is intended to distinguish an invalid
initial pose from a bad first action or fall gate.
"""

from __future__ import annotations

import json
import pathlib
import sys

import hydra
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from train import _apply_task_overrides  # noqa: E402


def _vec(x: torch.Tensor) -> list[float]:
    return [float(v) for v in x.detach().cpu().reshape(-1).tolist()]


def _snapshot(env, label: str) -> dict:
    raw = env.unwrapped
    robot = raw.scene["robot"]
    data = robot.data
    body_names = list(robot.body_names)
    wanted = ["torso_Link", "left_ankle_roll_Link", "right_ankle_roll_Link"]
    rows = {}
    for name in wanted:
        ids, resolved = robot.find_bodies([name], preserve_order=True)
        if resolved == [name]:
            idx = int(ids[0])
            rows[name] = {"index": idx, "pos_w": _vec(data.body_pos_w[0, idx]), "quat_wxyz": _vec(data.body_quat_w[0, idx])}
    term = {}
    for name in ("base_height", "strict_fall", "recovery_tilt", "non_foot_ground_contact"):
        try:
            value = raw.termination_manager.get_term(name)
            term[name] = bool(value[0].detach().cpu()) if torch.is_tensor(value) else str(value)
        except Exception as exc:
            term[name] = f"unavailable:{type(exc).__name__}"
    action_term = raw.action_manager.get_term("joint_pos")
    return {
        "label": label,
        "root_pos_w": _vec(data.root_pos_w[0]),
        "root_quat_wxyz": _vec(data.root_quat_w[0]),
        "root_lin_vel_w": _vec(data.root_lin_vel_w[0]),
        "root_ang_vel_w": _vec(data.root_ang_vel_w[0]),
        "joint_pos_first_22": _vec(data.joint_pos[0, :22]),
        "body": rows,
        "termination": term,
        "env_done": bool(getattr(raw, "reset_buf", torch.tensor([False]))[0].detach().cpu()) if hasattr(raw, "reset_buf") else None,
        "direct_action_scale_status": getattr(raw, "v13b_direct_scale_status", None),
        "ready_target_first_22": _vec(action_term._ready_full[0, :22]),
        "raw_action_first_26": _vec(getattr(action_term, "_raw_actions", torch.zeros((1, 26), device=raw.device))[0]),
        "body_names_count": len(body_names),
        "body_names": body_names,
        "racket_body_index": body_names.index("pingpang_red_Link") if "pingpang_red_Link" in body_names else None,
    }


def _run() -> dict:
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg
    import training.tasks  # noqa: F401

    task_id = "HOPE-FloatingTargetConditionedReferenceFreeV13B-AgibotA3-v0"
    env_cfg = parse_env_cfg(task_id, device="cuda:0", num_envs=128)
    env = gym.make(task_id, cfg=env_cfg, render_mode=None)
    try:
        obs, info = env.reset()
        before = _snapshot(env, "after_reset_before_step")
        zero = torch.zeros((128, 26), device=env.unwrapped.device)
        step_out = env.step(zero)
        after = _snapshot(env, "after_one_zero_action_step")
        env.reset()
        torch.manual_seed(7)
        random_action = torch.randn((128, 26), device=env.unwrapped.device) * 0.25
        random_out = env.step(random_action)
        random_after = _snapshot(env, "after_one_gaussian025_action_step")
        done = random_out[2] | random_out[3]
        return {
            "task": task_id,
            "obs_policy_shape": list(obs["policy"].shape) if isinstance(obs, dict) and "policy" in obs else None,
            "step_return_len": len(step_out),
            "before": before,
            "after": after,
            "random_action": _vec(random_action[0]),
            "zero_done_count": int((step_out[2] | step_out[3]).sum().detach().cpu()),
            "random_done_count": int(done.sum().detach().cpu()),
            "random_step_return_len": len(random_out),
            "random_after": random_after,
        }
    finally:
        env.close()


@hydra.main(version_base=None, config_path="../cfg", config_name="play")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, device="cuda:0", enable_cameras=False).app
    try:
        report = _run()
        output = pathlib.Path("/tmp/v13b_reset_state.json")
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
