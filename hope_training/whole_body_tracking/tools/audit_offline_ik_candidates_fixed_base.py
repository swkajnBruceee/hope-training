#!/usr/bin/env python3
"""Replay raw offline IK upper-reference candidates through fixed-base PhysX.

The offline IK wrapper intentionally emits a 10-DOF upper reference (waist plus
the right arm), while Isaac Lab's A3 articulation has 31 actuated joints.  This
tool keeps that distinction explicit: it fills the remaining joints with the
local A3 reset pose, injects each raw reference frame into a *fixed-base*
articulation, advances PhysX, and writes diagnostics.  Passing this audit is
only evidence of a fixed-base replay; it is not a floating-base qualification,
teacher-data approval, or a training admission decision.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import threading
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--motion_file", action="append", default=None, help="Raw trajectory_100hz.npz; repeatable.")
parser.add_argument("--candidate_index", type=Path, default=None, help="Final candidate index CSV; used if motion_file is omitted.")
parser.add_argument("--max_candidates", type=int, default=1, help="Maximum index rows to replay (default: 1).")
parser.add_argument("--output", type=Path, required=True, help="JSON report path.")
parser.add_argument("--post_steps", type=int, default=5, help="Extra fixed-base steps at the last frame.")
parser.add_argument("--settle_steps", type=int, default=2, help="Reset/settle steps before each candidate.")
parser.add_argument("--progress_jsonl", type=Path, default=None, help="Optional append-only per-candidate progress log.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.motion_file and args_cli.candidate_index is None:
    parser.error("provide --motion_file or --candidate_index")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from training.robots.agibot_a3 import AGIBOT_A3_CFG  # noqa: E402


@configclass
class FixedBaseSceneCfg(InteractiveSceneCfg):
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )
    robot: ArticulationCfg = AGIBOT_A3_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=AGIBOT_A3_CFG.spawn.replace(fix_base=True),
    )


def _resolve_motion_files() -> list[Path]:
    if args_cli.motion_file:
        return [Path(item).expanduser().resolve() for item in args_cli.motion_file]
    rows: list[dict[str, str]] = []
    with args_cli.candidate_index.expanduser().open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args_cli.max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    return [Path(row["trajectory_npz"]).expanduser().resolve() for row in rows[: args_cli.max_candidates]]


def _scalar(data: np.lib.npyio.NpzFile, name: str, default: float | None = None) -> float:
    if name not in data:
        if default is None:
            raise KeyError(name)
        return float(default)
    return float(np.asarray(data[name]).reshape(-1)[0])


def _finite_tensor(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())


def _reset_robot(sim: SimulationContext, scene: InteractiveScene, robot: Articulation) -> None:
    root = robot.data.default_root_state.clone()
    q = robot.data.default_joint_pos.clone()
    qd = robot.data.default_joint_vel.clone()
    robot.write_root_state_to_sim(root)
    robot.write_joint_state_to_sim(q, qd)
    robot.set_joint_position_target(q)
    robot.set_joint_velocity_target(qd)
    scene.write_data_to_sim()
    sim.forward()
    scene.update(0.0)
    for _ in range(max(0, int(args_cli.settle_steps))):
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(sim.get_physics_dt())


def _run() -> dict:
    motion_files = _resolve_motion_files()
    if not motion_files:
        raise ValueError("no motion files resolved")

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.01
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(FixedBaseSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    scene.reset()
    robot = scene["robot"]
    device = sim.device

    joint_names = list(robot.joint_names)
    raw_joint_ids_cache: dict[tuple[str, ...], torch.Tensor] = {}
    rows: list[dict] = []
    progress_handle = None
    if args_cli.progress_jsonl is not None:
        args_cli.progress_jsonl.expanduser().parent.mkdir(parents=True, exist_ok=True)
        progress_handle = args_cli.progress_jsonl.expanduser().open("w", encoding="utf-8", buffering=1)
    for motion_file in motion_files:
        item: dict = {
            "motion_file": str(motion_file),
            "status": "FIXED_BASE_PHYSX_REPLAY_FAILED",
            "qualification": "PENDING_SEPARATE_FLOATING_BASE_AND_TASK_GATES",
            "self_collision_observable": False,
        }
        try:
            with np.load(motion_file, allow_pickle=False) as data:
                raw_names = tuple(str(x) for x in np.asarray(data["joint_names"]).reshape(-1))
                q_raw = np.asarray(data["joint_pos"], dtype=np.float32)
                qd_raw = np.asarray(data["joint_vel"], dtype=np.float32)
                hit_frame = int(_scalar(data, "hit_frame", q_raw.shape[0] - 1))
                requested_time = _scalar(data, "requested_strike_time_s", hit_frame * 0.01)
                item["source_goal_id"] = str(np.asarray(data["source_goal_id"]).reshape(-1)[0]) if "source_goal_id" in data else motion_file.stem
                item["selected_swing_type"] = str(np.asarray(data["selected_swing_type"]).reshape(-1)[0]) if "selected_swing_type" in data else "unknown"
            if q_raw.ndim != 2 or qd_raw.shape != q_raw.shape:
                raise ValueError(f"raw q/qd shape mismatch: {q_raw.shape} vs {qd_raw.shape}")
            if len(raw_names) != q_raw.shape[1] or not np.isfinite(q_raw).all() or not np.isfinite(qd_raw).all():
                raise ValueError("raw joint names/trajectory are invalid or non-finite")
            if not (0 <= hit_frame < q_raw.shape[0]):
                raise ValueError(f"invalid hit_frame={hit_frame} for {q_raw.shape[0]} frames")
            missing = [name for name in raw_names if name not in joint_names]
            if missing:
                raise ValueError(f"raw joints missing from A3 articulation: {missing}")
            key = raw_names
            if key not in raw_joint_ids_cache:
                raw_joint_ids_cache[key] = torch.as_tensor(
                    [joint_names.index(name) for name in raw_names], dtype=torch.long, device=device
                )
            raw_ids = raw_joint_ids_cache[key]

            _reset_robot(sim, scene, robot)
            q_default = robot.data.default_joint_pos.clone()
            qd_default = robot.data.default_joint_vel.clone()
            q_ref = q_default.repeat(q_raw.shape[0], 1)
            qd_ref = qd_default.repeat(q_raw.shape[0], 1)
            q_ref[:, raw_ids.cpu().numpy()] = torch.as_tensor(q_raw, device=device)
            qd_ref[:, raw_ids.cpu().numpy()] = torch.as_tensor(qd_raw, device=device)
            if not _finite_tensor(q_ref) or not _finite_tensor(qd_ref):
                raise ValueError("expanded 31-DOF trajectory is non-finite")

            min_hard_margin = math.inf
            min_soft_margin = math.inf
            max_abs_q = 0.0
            max_abs_qd = 0.0
            max_body_abs = 0.0
            finite_state = True
            frames = list(range(q_raw.shape[0])) + [q_raw.shape[0] - 1] * max(0, int(args_cli.post_steps))
            for frame in frames:
                q_now = q_ref[frame : frame + 1]
                qd_now = qd_ref[frame : frame + 1]
                robot.write_joint_state_to_sim(q_now, qd_now)
                robot.set_joint_position_target(q_now)
                robot.set_joint_velocity_target(qd_now)
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(sim.get_physics_dt())

                actual_q = robot.data.joint_pos[0]
                actual_qd = robot.data.joint_vel[0]
                hard = robot.data.joint_pos_limits[0]
                soft = robot.data.soft_joint_pos_limits[0]
                hard_margin = torch.minimum(actual_q - hard[:, 0], hard[:, 1] - actual_q)
                soft_margin = torch.minimum(actual_q - soft[:, 0], soft[:, 1] - actual_q)
                body_pos = robot.data.body_pos_w[0]
                finite_now = _finite_tensor(actual_q) and _finite_tensor(actual_qd) and _finite_tensor(body_pos)
                finite_state = finite_state and finite_now
                min_hard_margin = min(min_hard_margin, float(hard_margin.min().item()))
                min_soft_margin = min(min_soft_margin, float(soft_margin.min().item()))
                max_abs_q = max(max_abs_q, float(actual_q.abs().max().item()))
                max_abs_qd = max(max_abs_qd, float(actual_qd.abs().max().item()))
                max_body_abs = max(max_body_abs, float(body_pos.abs().max().item()))
            if not finite_state or min_hard_margin < -1e-4:
                replay_status = "FIXED_BASE_PHYSX_REPLAY_FAILED"
            elif min_soft_margin < -1e-4:
                replay_status = "FIXED_BASE_PHYSX_SOFT_LIMIT_WARNING"
            else:
                replay_status = "FIXED_BASE_PHYSX_REPLAY_PASS"
            item.update(
                {
                    "status": replay_status,
                    "frames_replayed": len(frames),
                    "hit_frame": hit_frame,
                    "requested_strike_time_s": requested_time,
                    "finite_state": finite_state,
                    "minimum_actual_hard_joint_margin_rad": min_hard_margin,
                    "minimum_actual_soft_joint_margin_rad": min_soft_margin,
                    "max_abs_actual_joint_position_rad": max_abs_q,
                    "max_abs_actual_joint_velocity_radps": max_abs_qd,
                    "max_abs_body_position_w_m": max_body_abs,
                    "fixed_base": True,
                    "physics_qualified": False,
                }
            )
        except Exception as exc:  # keep batch reports useful and fail closed
            item["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(item)
        if progress_handle is not None:
            progress_handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    if progress_handle is not None:
        progress_handle.close()

    args_cli.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "a3_offline_ik_fixed_base_physx_replay/v1",
        "status": "completed",
        "fixed_base_physx_evidence_only": True,
        "physics_qualified": False,
        "teacher_approved": False,
        "training_admission": False,
        "notice_contract": "THIRD_PARTY_NOTICES.md",
        "rows": rows,
    }
    args_cli.output.expanduser().write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args_cli.output.expanduser()), "count": len(rows), "statuses": {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})}}, ensure_ascii=False), flush=True)
    return report


try:
    _run()
finally:
    # Isaac Sim 4.5 can block in its normal close path when another Kit process
    # owns the shared cache.  The report/progress files are already flushed;
    # use a short watchdog so this audit cannot linger and consume resources.
    watchdog = threading.Timer(8.0, os._exit, args=(0,))
    watchdog.daemon = True
    watchdog.start()
    simulation_app.close()
