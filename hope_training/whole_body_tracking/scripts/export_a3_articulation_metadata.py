#!/usr/bin/env python3
"""Export A3 articulation joint/body metadata from the current Isaac asset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Export A3 articulation joint/body order.")
parser.add_argument("--out", type=Path, default=Path("docs/a3_articulation_metadata.json"))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from training.robots.agibot_a3 import AGIBOT_A3_CFG


@configclass
class MetadataSceneCfg(InteractiveSceneCfg):
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )
    robot: ArticulationCfg = AGIBOT_A3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main() -> None:
    try:
        sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
        sim = SimulationContext(sim_cfg)
        scene = InteractiveScene(MetadataSceneCfg(num_envs=1, env_spacing=2.0))
        sim.reset()
        scene.reset()
        robot: Articulation = scene["robot"]

        joint_names = list(robot.data.joint_names)
        body_names = list(robot.body_names)
        payload = {
            "asset": "agibot_a3",
            "source": "AGIBOT_A3_CFG current Isaac articulation",
            "joint_names": joint_names,
            "body_names": body_names,
            "num_joints": len(joint_names),
            "num_bodies": len(body_names),
        }
        args_cli.out.parent.mkdir(parents=True, exist_ok=True)
        args_cli.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[a3_metadata] wrote {args_cli.out}", flush=True)
        print("JOINT_NAMES=" + json.dumps(joint_names), flush=True)
        print("BODY_NAMES=" + json.dumps(body_names), flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
