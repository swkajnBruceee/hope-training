#!/usr/bin/env python3
"""Inspect the generated Isaac USD racket collision prims without stepping physics."""
from __future__ import annotations

import json
import pathlib
import sys

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
        import omni.usd
        from pxr import UsdGeom

        cfg = AgibotA3HitFixedBaseTouchEnvCfg()
        cfg.scene.num_envs = 1
        cfg.scene.robot.init_state.pos = (-0.5, -0.7625, 1.04)
        cfg.sim.device = "cuda:0"
        env = gym.make("HOPE-TableTennis-AgibotA3-HitFixedBaseTouch-v0", cfg=cfg, render_mode=None)
        try:
            env.reset()
            stage = omni.usd.get_context().get_stage()
            rows = []
            bbox_cache = UsdGeom.BBoxCache(UsdGeom.Tokens.default_, [UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide])
            for prim in stage.Traverse():
                path = str(prim.GetPath())
                name = prim.GetName().lower()
                if "/robot/" not in path.lower() and not path.lower().startswith("/colliders/"):
                    continue
                if not any(k in name or k in path.lower() for k in ("racket", "pingpang", "wrist_yaw", "collision")):
                    continue
                row = {
                    "path": path,
                    "name": prim.GetName(),
                    "type": prim.GetTypeName(),
                    "applied_schemas": list(prim.GetAppliedSchemas()),
                    "properties": [str(p.GetName()) for p in prim.GetProperties() if any(k in str(p.GetName()).lower() for k in ("collision", "physics", "mesh", "size", "extent", "purpose"))],
                }
                for attr_name in ("physics:collisionEnabled", "physics:rigidBodyEnabled", "physxContactReport:threshold"):
                    attr = prim.GetAttribute(attr_name)
                    if attr and attr.HasValue():
                        row[attr_name] = str(attr.Get())
                if any(k in path.lower() for k in ("right_wrist_yaw_link", "pingpang_red_link", "collisions")):
                    try:
                        box = bbox_cache.ComputeWorldBound(prim)
                        row["world_bound_min"] = [float(v) for v in box.ComputeAlignedRange().GetMin()]
                        row["world_bound_max"] = [float(v) for v in box.ComputeAlignedRange().GetMax()]
                    except Exception as exc:
                        row["world_bound_error"] = repr(exc)
                    row["children"] = [str(c.GetPath()) for c in prim.GetChildren()]
                rows.append(row)
            result = {"status": "usd_collision_inspection_complete", "rows": rows}
            pathlib.Path("/tmp/v13b_usd_racket_collision.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps(result, indent=2), flush=True)
        finally:
            env.close()
    finally:
        app.close()


if __name__ == "__main__":
    main()
