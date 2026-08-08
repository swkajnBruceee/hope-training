#!/usr/bin/env python3
"""PhysX contact calibration probe for the A3 ping-pong racket.

The robot is fixed in the table-tennis scene.  The ball is placed on each
candidate racket-frame axis and launched a short distance toward the racket.
The probe records the actual ``pingpang_red_Link`` body index, the signed
ball-centre displacement and the ball contact-sensor force.  It does not edit
training configuration or calibration YAML.
"""

from __future__ import annotations

import json
import pathlib
import sys

import torch
from isaaclab.app import AppLauncher

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _vec(x: torch.Tensor) -> list[float]:
    return [float(v) for v in x.detach().cpu().reshape(-1).tolist()]


def main() -> None:
    launcher = AppLauncher(headless=True, device="cuda:0", enable_cameras=False)
    app = launcher.app
    try:
        from isaaclab.scene import InteractiveScene
        from isaaclab.sim import SimulationContext
        from isaaclab.utils.math import matrix_from_quat
        from training.tasks.table_tennis.config.agibot_a3.table_tennis_env_cfg import AgibotA3TableTennisEnvCfg

        cfg = AgibotA3TableTennisEnvCfg()
        cfg.sim.device = "cuda:0"
        cfg.scene.num_envs = 1
        cfg.scene.robot.spawn.fix_base = True
        sim = SimulationContext(cfg.sim)
        scene = InteractiveScene(cfg.scene)
        sim.reset()
        scene.reset()
        robot = scene["robot"]
        ball = scene["ball"]
        sensor = scene.sensors["racket_ball_contact"]
        body_name = "pingpang_red_Link"
        ids, names = robot.find_bodies([body_name], preserve_order=True)
        if names != [body_name]:
            raise RuntimeError(f"racket body resolution mismatch: {names}")
        body_index = int(ids[0])
        body_pos = robot.data.body_pos_w[0, body_index].detach().clone()
        body_quat = robot.data.body_quat_w[0, body_index].detach().clone()
        basis = matrix_from_quat(body_quat.unsqueeze(0))[0]
        # The URDF/MuJoCo mesh audit predicts local +Y, but test all axes and signs.
        candidates = []
        for axis in range(3):
            for sign in (-1.0, 1.0):
                normal = basis[:, axis] * sign
                normal = normal / torch.linalg.vector_norm(normal).clamp_min(1.0e-8)
                candidates.append((axis, sign, normal))

        radius = 0.020
        distances = (0.010, 0.015, 0.020, 0.025, 0.030)
        rows = []
        state = ball.data.default_root_state.clone()
        for axis, sign, normal in candidates:
            for distance in distances:
                # Reset ball in the current environment-local world frame.  A
                # small inward velocity makes contact observable without relying
                # on gravity or on a long trajectory.
                center = body_pos + normal * distance
                state[0, :3] = center
                state[0, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=state.device)
                state[0, 7:10] = -normal * 1.0
                state[0, 10:13] = 0.0
                ball.write_root_state_to_sim(state)
                scene.write_data_to_sim()
                sim.forward()
                max_force = 0.0
                contact_steps = 0
                # 8 physics steps at 360 Hz; record the first contact impulse.
                for _ in range(8):
                    sim.step()
                    scene.update(cfg.sim.dt)
                    force = sensor.data.net_forces_w[0, sensor.body_ids]
                    force_norm = float(torch.linalg.vector_norm(force, dim=-1).amax().detach().cpu())
                    max_force = max(max_force, force_norm)
                    if force_norm > 0.05:
                        contact_steps += 1
                actual_center = ball.data.root_pos_w[0].detach().clone()
                d = actual_center - body_pos
                signed = float(torch.dot(d, normal).detach().cpu())
                lateral = d - signed * normal
                rows.append(
                    {
                        "local_axis": axis,
                        "sign": sign,
                        "initial_distance_m": distance,
                        "initial_center_w_m": _vec(center),
                        "final_center_w_m": _vec(actual_center),
                        "measured_d_m": _vec(d),
                        "signed_d_along_normal_m": signed,
                        "lateral_residual_m": float(torch.linalg.vector_norm(lateral).detach().cpu()),
                        "max_ball_contact_force_n": max_force,
                        "contact_steps": contact_steps,
                    }
                )
        result = {
            "status": "physx_contact_probe_complete",
            "robot_body_name": body_name,
            "robot_body_index": body_index,
            "robot_body_names": list(robot.body_names),
            "body_pos_w_m": _vec(body_pos),
            "body_quat_wxyz": _vec(body_quat),
            "ball_radius_m": radius,
            "contact_sensor": "racket_ball_contact",
            "normal_candidates": [
                {"local_axis": axis, "sign": sign, "normal_w": _vec(normal)}
                for axis, sign, normal in candidates
            ],
            "rows": rows,
            "qualification_note": "A candidate is admissible only when PhysX contact force is observed with small lateral residual; no YAML was promoted by this script.",
        }
        output = pathlib.Path("/tmp/v13b_racket_contact_physx_probe.json")
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        sim.clear_instance()
    finally:
        app.close()


if __name__ == "__main__":
    main()
