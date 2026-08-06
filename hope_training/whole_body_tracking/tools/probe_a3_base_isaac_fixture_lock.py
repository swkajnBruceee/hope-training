#!/usr/bin/env python3
"""Probe whether PhysX joint limits can implement ``single_joint_fixture_v1``.

This is deliberately a short, fixture-only diagnostic.  It does not train a
policy and it does not approve free-base Stand or hardware deployment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--joint", default="left_hip_roll_joint")
parser.add_argument("--delta-rad", type=float, default=0.08)
parser.add_argument("--steps", type=int, default=160)
parser.add_argument("--pre-steps", type=int, default=0)
parser.add_argument(
    "--lock-mode",
    choices=("equal_limit", "state_restore"),
    default="state_restore",
)
parser.add_argument(
    "--root-mode", choices=("fixed", "state_restore"), default="fixed"
)
parser.add_argument(
    "--ground-contact", action=argparse.BooleanOptionalAction, default=True
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from training.robots.agibot_a3 import AGIBOT_A3_CFG


def main() -> None:
    try:
        sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
        )
        if args_cli.ground_contact:
            ground_cfg = sim_utils.GroundPlaneCfg()
            ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

        robot_cfg = AGIBOT_A3_CFG.replace(prim_path="/World/Robot")
        # Request a real world-to-root fixed joint in the converted articulation.
        robot_cfg.spawn.fix_base = args_cli.root_mode == "fixed"
        robot = Articulation(robot_cfg)
        sim.reset()

        joint_names = list(robot.data.joint_names)
        if args_cli.joint not in joint_names:
            raise ValueError(f"unknown joint {args_cli.joint!r}")
        selected = joint_names.index(args_cli.joint)
        baseline = robot.data.default_joint_pos.clone()
        fixture_root_state = robot.data.default_root_state.clone()
        zero_velocity = torch.zeros_like(robot.data.default_joint_vel)
        robot.write_joint_state_to_sim(baseline, zero_velocity)
        robot.reset()

        original_limits = robot.data.joint_pos_limits.clone()
        locked_indices = [index for index in range(robot.num_joints) if index != selected]
        locked_ids = torch.tensor(locked_indices, dtype=torch.long, device=sim.device)
        locked_values = baseline[:, locked_ids]
        if args_cli.lock_mode == "equal_limit":
            equal_limits = torch.stack((locked_values, locked_values), dim=-1)
            robot.write_joint_position_limit_to_sim(
                equal_limits, joint_ids=locked_ids, warn_limit_violation=False
            )

        root_start = robot.data.root_pos_w.clone()
        joint_start = robot.data.joint_pos.clone()
        target = baseline.clone()
        excited_target = baseline.clone()
        excited_target[:, selected] += float(args_cli.delta_rad)
        low = float(original_limits[0, selected, 0])
        high = float(original_limits[0, selected, 1])
        excited_target[:, selected].clamp_(min=low, max=high)

        max_locked_error = 0.0
        max_root_drift = 0.0
        max_selected_delta = 0.0
        pre_end_q = float(joint_start[0, selected])
        active_q: list[float] = []
        incoming_force_shape: list[int] | None = None
        incoming_force_finite = False
        total_steps = max(1, int(args_cli.pre_steps) + int(args_cli.steps))
        for step in range(total_steps):
            current_target = target if step < int(args_cli.pre_steps) else excited_target
            robot.set_joint_position_target(current_target)
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(sim.get_physics_dt())
            if args_cli.root_mode == "state_restore":
                robot.write_root_state_to_sim(fixture_root_state)
            if args_cli.lock_mode == "state_restore":
                robot.write_joint_state_to_sim(
                    locked_values,
                    torch.zeros_like(locked_values),
                    joint_ids=locked_ids,
                )
            max_locked_error = max(
                max_locked_error,
                float(torch.max(torch.abs(robot.data.joint_pos[:, locked_ids] - locked_values))),
            )
            max_root_drift = max(
                max_root_drift,
                float(torch.linalg.vector_norm(robot.data.root_pos_w - root_start, dim=-1).max()),
            )
            max_selected_delta = max(
                max_selected_delta,
                float(torch.abs(robot.data.joint_pos[:, selected] - joint_start[:, selected]).max()),
            )
            if step == int(args_cli.pre_steps) - 1:
                pre_end_q = float(robot.data.joint_pos[0, selected])
            if step >= int(args_cli.pre_steps):
                active_q.append(float(robot.data.joint_pos[0, selected]))
            incoming = robot.root_physx_view.get_link_incoming_joint_force()
            incoming_force_shape = list(incoming.shape)
            incoming_force_finite = bool(torch.all(torch.isfinite(incoming)))

        reported_limits = robot.data.joint_pos_limits[0, locked_ids]
        limit_width = reported_limits[:, 1] - reported_limits[:, 0]
        payload = {
            "probe": "a3_isaac_joint_fixture_lock_probe_v1",
            "lock_mode": args_cli.lock_mode,
            "root_mode": args_cli.root_mode,
            "ground_contact_enabled": bool(args_cli.ground_contact),
            "device": str(sim.device),
            "physics_dt_s": float(sim.get_physics_dt()),
            "is_fixed_base": bool(robot.is_fixed_base),
            "selected_joint": args_cli.joint,
            "requested_delta_rad": float(args_cli.delta_rad),
            "pre_steps": int(args_cli.pre_steps),
            "pre_end_joint_q_rad": pre_end_q,
            "active_steady_joint_q_rad": float(sum(active_q[-20:]) / len(active_q[-20:])),
            "active_steady_delta_from_pre_rad": float(
                sum(active_q[-20:]) / len(active_q[-20:]) - pre_end_q
            ),
            "max_selected_delta_rad": max_selected_delta,
            "locked_joint_count": len(locked_indices),
            "max_reported_locked_limit_width_rad": float(torch.max(torch.abs(limit_width))),
            "max_locked_joint_error_rad": max_locked_error,
            "max_root_position_drift_m": max_root_drift,
            "incoming_joint_force_shape": incoming_force_shape,
            "incoming_joint_force_all_finite": incoming_force_finite,
            "constraint_reaction_semantics_verified": False,
            "training_or_deployment_approved": False,
        }
        print(json.dumps(payload, indent=2), flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
