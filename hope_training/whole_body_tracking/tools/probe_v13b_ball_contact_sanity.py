#!/usr/bin/env python3
"""Sanity check that the table-tennis ball contact sensor reports a known table contact."""
from __future__ import annotations

import json
import pathlib
import sys
import torch
from isaaclab.app import AppLauncher

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main():
    app = AppLauncher(headless=True, device="cuda:0", enable_cameras=False).app
    try:
        import gymnasium as gym
        import training.tasks  # noqa: F401
        from training.tasks.table_tennis.config.agibot_a3.table_tennis_env_cfg import AgibotA3HitFixedBaseTouchEnvCfg

        cfg = AgibotA3HitFixedBaseTouchEnvCfg()
        cfg.scene.num_envs = 1
        cfg.scene.robot.init_state.pos = (-0.5, -0.7625, 1.04)
        cfg.sim.device = "cuda:0"
        env = gym.make("HOPE-TableTennis-AgibotA3-HitFixedBaseTouch-v0", cfg=cfg, render_mode=None)
        try:
            env.reset()
            raw = env.unwrapped
            ball = raw.scene["ball"]
            sensor = raw.scene.sensors["racket_ball_contact"]
            state = ball.data.default_root_state.clone()
            # Table surface is z=0 in the HOPE frame; put the 40 mm ball on it.
            state[0, :3] = torch.tensor((1.0, -0.7625, 0.20), device=state.device)
            state[0, 7:10] = torch.tensor((0.0, 0.0, -5.0), device=state.device)
            state[0, 10:13] = 0.0
            ball.write_root_state_to_sim(state)
            raw.scene.write_data_to_sim()
            raw.sim.forward()
            rows = []
            for _ in range(100):
                raw.sim.step()
                raw.scene.update(raw.sim.get_rendering_dt())
                f = float(torch.linalg.vector_norm(sensor.data.net_forces_w[0], dim=-1).amax().detach().cpu())
                fm = getattr(sensor.data, "force_matrix_w", None)
                fm_max = None if fm is None else float(torch.linalg.vector_norm(fm[0], dim=-1).amax().detach().cpu())
                rows.append({"ball_pos": ball.data.root_pos_w[0].detach().cpu().tolist(), "force_n": f, "force_matrix_max_n": fm_max})
            result = {"status": "ball_contact_sanity_complete", "rows": rows}
            pathlib.Path("/tmp/v13b_ball_contact_sanity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps(result, indent=2), flush=True)
        finally:
            env.close()
    finally:
        app.close()

if __name__ == "__main__":
    main()
