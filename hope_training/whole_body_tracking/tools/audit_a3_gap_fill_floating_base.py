#!/usr/bin/env python3
"""Lightweight floating-base PhysX replay for an already FK-expanded A3 bank."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--post-steps", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, AssetBaseCfg, ArticulationCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from training.robots.agibot_a3 import AGIBOT_A3_CFG  # noqa: E402


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(20.0, 20.0)),
    )
    robot: ArticulationCfg = AGIBOT_A3_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=AGIBOT_A3_CFG.spawn.replace(fix_base=False),
    )


def tilt_deg(quat: torch.Tensor) -> float:
    x, y = quat[1], quat[2]
    z_axis_world = 1.0 - 2.0 * (x * x + y * y)
    return float(torch.rad2deg(torch.acos(torch.clamp(z_axis_world, -1.0, 1.0))).item())


def main() -> int:
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sim = SimulationContext(sim_utils.SimulationCfg(device=args.device))
    sim.cfg.dt = 0.01
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset(); scene.reset()
    robot = scene["robot"]
    names = list(robot.joint_names)
    device = sim.device
    rows = []
    for entry in manifest["motions"]:
        item = {"episode_id": entry["episode_id"], "source_goal_id": entry["source_goal_id"], "status": "FLOATING_BASE_REPLAY_FAILED"}
        try:
            path = manifest_path.parent / entry["motion_npz"]
            with np.load(path, allow_pickle=False) as data:
                q = np.asarray(data["joint_pos"], dtype=np.float32)
                qd = np.asarray(data["joint_vel"], dtype=np.float32)
                src_names = [str(x) for x in np.asarray(data["joint_names"]).reshape(-1)]
                hit = int(np.asarray(data["hit_frame"]).reshape(-1)[0])
            ids = torch.as_tensor([names.index(x) for x in src_names], dtype=torch.long, device=device)
            root = robot.data.default_root_state.clone()
            q0 = robot.data.default_joint_pos.clone(); qd0 = robot.data.default_joint_vel.clone()
            root[:, 7:] = 0.0
            robot.write_root_state_to_sim(root); robot.write_joint_state_to_sim(q0, qd0)
            robot.set_joint_position_target(q0); robot.set_joint_velocity_target(qd0)
            scene.write_data_to_sim(); sim.forward(); scene.update(0.0)
            min_hard = math.inf; min_soft = math.inf; min_height = math.inf; max_tilt = 0.0; max_qerr = 0.0; finite = True
            frames = list(range(q.shape[0])) + [q.shape[0] - 1] * max(0, args.post_steps)
            for frame in frames:
                target = robot.data.default_joint_pos.clone()
                target[:, ids] = torch.as_tensor(q[frame], device=device).unsqueeze(0)
                robot.set_joint_position_target(target); robot.set_joint_velocity_target(torch.zeros_like(target))
                scene.write_data_to_sim(); sim.step(render=False); scene.update(sim.get_physics_dt())
                actual = robot.data.joint_pos[0]; hard = robot.data.joint_pos_limits[0]; soft = robot.data.soft_joint_pos_limits[0]
                hm = torch.minimum(actual-hard[:,0], hard[:,1]-actual); sm = torch.minimum(actual-soft[:,0], soft[:,1]-actual)
                finite_now = torch.isfinite(actual).all() and torch.isfinite(robot.data.root_pos_w[0]).all()
                finite = finite and bool(finite_now)
                min_hard = min(min_hard, float(hm.min().item())); min_soft = min(min_soft, float(sm.min().item()))
                min_height = min(min_height, float(robot.data.root_pos_w[0,2].item())); max_tilt = max(max_tilt, tilt_deg(robot.data.root_quat_w[0]))
                max_qerr = max(max_qerr, float(torch.abs(actual-target[0]).max().item()))
            if not finite or min_hard < -1e-4: status = "FLOATING_BASE_REPLAY_FAIL"
            elif min_soft < -1e-4 or min_height < 0.45 or max_tilt > 20.0: status = "FLOATING_BASE_REPLAY_WARNING"
            else: status = "FLOATING_BASE_REPLAY_PASS"
            item.update({"status": status, "frames_replayed": len(frames), "hit_frame": hit, "finite_state": finite, "minimum_actual_hard_joint_margin_rad": min_hard, "minimum_actual_soft_joint_margin_rad": min_soft, "minimum_root_height_w_m": min_height, "max_root_tilt_deg": max_tilt, "max_joint_tracking_error_rad": max_qerr, "floating_base": True, "physics_qualified": False})
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(item)
    report = {"schema_version": "a3_gap_fill_floating_base_replay/v1", "status": "completed", "training_started": False, "physics_qualified": False, "teacher_approved": False, "training_admission": False, "rows": rows, "summary": {"total": len(rows), "pass": sum(x["status"] == "FLOATING_BASE_REPLAY_PASS" for x in rows), "warning": sum(x["status"] == "FLOATING_BASE_REPLAY_WARNING" for x in rows), "fail": sum(x["status"] == "FLOATING_BASE_REPLAY_FAIL" for x in rows)}}
    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)
    app.close(wait_for_replicator=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
