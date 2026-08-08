#!/usr/bin/env python3
"""Check how long the V1.3B fixed READY target survives with zero action."""
from __future__ import annotations
import pathlib, sys
import hydra
import torch
from omegaconf import OmegaConf

@hydra.main(version_base=None, config_path="../cfg", config_name="play")
def main(cfg):
    OmegaConf.resolve(cfg); OmegaConf.set_struct(cfg, False); sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device="cuda:0", enable_cameras=False).app
    try:
        import gymnasium as gym
        from isaaclab_tasks.utils import parse_env_cfg
        import training.tasks  # noqa: F401
        task = "HOPE-FloatingTargetConditionedReferenceFreeV13B-AgibotA3-v0"
        ecfg = parse_env_cfg(task, device="cuda:0", num_envs=1)
        env = gym.make(task, cfg=ecfg, render_mode=None)
        try:
            env.reset()
            action = torch.zeros((1, 26), device=env.unwrapped.device)
            first = None
            for i in range(240):
                out = env.step(action)
                done = bool((out[2] | out[3])[0].detach().cpu())
                if done:
                    first = {"step": i + 1, "terminated": bool(out[2][0].detach().cpu()), "truncated": bool(out[3][0].detach().cpu())}
                    break
            root = env.unwrapped.scene["robot"].data.root_pos_w[0]
            print({"first_done": first, "survived_control_steps": 240 if first is None else first["step"] - 1, "root_pos_after": [float(x) for x in root.detach().cpu()]}, flush=True)
        finally:
            env.close()
    finally:
        app.close()

if __name__ == "__main__":
    main()
