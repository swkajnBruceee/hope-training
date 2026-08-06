#!/usr/bin/env python3
"""Audit canonical motion hit poses in the formal HOPE P1 match scene.

The probe is deliberately non-training and non-rollout.  It injects each
canonical hit-frame root/joint state into the formal table-tennis scene and
reads Isaac FK without applying a policy action.  Both the task-default racket
body and the validated wrist-offset point are reported so a collapsed/fixed
URDF body cannot silently change the TCP contract.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--canonical-manifest", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--task", default="HOPE-PingPong-AgibotA3-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import matrix_from_quat  # noqa: E402

from training.tasks.table_tennis import geometry  # noqa: E402
from training.tasks.table_tennis.config.agibot_a3.table_tennis_env_cfg import (  # noqa: E402
    AgibotA3TableTennisEnvCfg,
)
from training.tasks.table_tennis.mdp.racket import racket_spatial_state_w  # noqa: E402


def _vector(value: torch.Tensor | np.ndarray) -> list[float]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return [float(item) for item in np.asarray(value).reshape(-1)]


def _angle_deg(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    dot = torch.sum(lhs * rhs).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.acos(dot)).detach().cpu())


def _norm(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(lhs - rhs).detach().cpu())


def _load_contracts(path: Path) -> tuple[dict, list[str]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    source_path = Path(manifest["source_manifest"])
    source = json.loads(source_path.read_text(encoding="utf-8"))
    joint_names = source["momentum_preview_contract"]["joint_names"]
    if manifest.get("contract_version") != "motion_prior_base_heading_frame0/v1":
        raise ValueError("unsupported canonical motion contract")
    if len(manifest.get("motions", [])) == 0:
        raise ValueError("canonical manifest contains no motions")
    return manifest, list(joint_names)


def _run() -> dict:
    manifest_path = args_cli.canonical_manifest.expanduser().resolve()
    manifest, source_joint_names = _load_contracts(manifest_path)

    env_cfg = AgibotA3TableTennisEnvCfg()
    env_cfg.sim.device = str(args_cli.device)
    # FK is invariant to fixing the base.  Pinning it makes the probe incapable
    # of turning into an accidental dynamics rollout.
    env_cfg.scene.robot.spawn.fix_base = True
    env_cfg.scene.num_envs = 1
    # Keep collision geometry identical but strip decoration/material assets.
    # On an offline workstation those visual-only resources can remain in an
    # asynchronous USD loading state and delay SimulationApp shutdown forever.
    for asset_name in ("floor", "table", "net", "ball"):
        asset_cfg = getattr(env_cfg.scene, asset_name)
        if hasattr(asset_cfg.spawn, "visual_material"):
            asset_cfg.spawn.visual_material = None
    for decoration_name in (
        "net_post_left",
        "net_post_right",
        "center_line",
        "light",
        "sky_light",
    ):
        setattr(env_cfg.scene, decoration_name, None)
    sim = SimulationContext(env_cfg.sim)
    scene = InteractiveScene(env_cfg.scene)
    sim.reset()
    scene.reset()
    raw = SimpleNamespace(
        scene=scene,
        device=sim.device,
        num_envs=1,
    )
    robot = scene["robot"]
    env_id = torch.zeros(1, dtype=torch.long, device=raw.device)

    missing = [name for name in source_joint_names if name not in robot.joint_names]
    if missing:
        raise RuntimeError(f"formal P1 robot is missing source joints: {missing}")
    destination_ids = torch.as_tensor(
        [robot.joint_names.index(name) for name in source_joint_names],
        dtype=torch.long,
        device=raw.device,
    )
    p1_anchor = robot.data.default_root_state[0, :3].detach().clone()
    normal_axis = 1
    rows = []
    for entry in manifest["motions"]:
        npz_path = manifest_path.parent / entry["canonical_motion_npz"]
        with np.load(npz_path, allow_pickle=False) as data:
            hit = int(entry["hit_frame"])
            source_q = torch.as_tensor(data["joint_pos"][hit], device=raw.device)
            source_qd = torch.as_tensor(data["joint_vel"][hit], device=raw.device)
            root_position = p1_anchor + torch.as_tensor(
                data["body_pos_b0"][hit, 0], device=raw.device
            )
            root_quaternion = torch.as_tensor(
                data["body_quat_b0_wxyz"][hit, 0], device=raw.device
            )
            root_linear_velocity = torch.as_tensor(
                data["body_lin_vel_b0"][hit, 0], device=raw.device
            )
            root_angular_velocity = torch.as_tensor(
                data["body_ang_vel_b0"][hit, 0], device=raw.device
            )

        q = robot.data.default_joint_pos[0].detach().clone()
        qd = torch.zeros_like(q)
        q[destination_ids] = source_q
        qd[destination_ids] = source_qd
        robot.write_joint_state_to_sim(q.unsqueeze(0), qd.unsqueeze(0), env_ids=env_id)
        robot.write_root_state_to_sim(
            torch.cat(
                (
                    root_position,
                    root_quaternion,
                    root_linear_velocity,
                    root_angular_velocity,
                )
            ).unsqueeze(0),
            env_ids=env_id,
        )
        scene.write_data_to_sim()
        sim.forward()
        scene.update(0.0)

        default_pos, default_vel, default_quat, _ = racket_spatial_state_w(
            raw, SceneEntityCfg("robot")
        )
        wrist_pos, wrist_vel, wrist_quat, _ = racket_spatial_state_w(
            raw,
            SceneEntityCfg("robot"),
            racket_body_name="__force_wrist_offset_racket_fk__",
        )
        default_normal = matrix_from_quat(default_quat)[:, :, normal_axis][0]
        wrist_normal = matrix_from_quat(wrist_quat)[:, :, normal_axis][0]

        target = entry["strike_target_b0"]
        target_pos = p1_anchor + torch.tensor(
            target["racket_position_b0_m"], device=raw.device
        )
        target_vel = torch.tensor(target["racket_velocity_b0_mps"], device=raw.device)
        target_normal = torch.tensor(target["racket_normal_b0"], device=raw.device)
        ball_pos = p1_anchor + torch.tensor(target["ball_position_b0_m"], device=raw.device)

        soft_limits = robot.data.soft_joint_pos_limits[0]
        hard_limits = robot.data.joint_pos_limits[0]
        soft_margin = torch.minimum(q - soft_limits[:, 0], soft_limits[:, 1] - q)
        hard_margin = torch.minimum(q - hard_limits[:, 0], hard_limits[:, 1] - q)
        minimum_soft_index = int(torch.argmin(soft_margin).detach().cpu())
        minimum_hard_index = int(torch.argmin(hard_margin).detach().cpu())
        table_xy = (
            0.0 <= float(target_pos[0]) <= geometry.TABLE_LENGTH
            and -geometry.TABLE_WIDTH <= float(target_pos[1]) <= 0.0
        )
        rows.append(
            {
                "motion_id": int(entry["motion_id"]),
                "episode_id": str(entry["episode_id"]),
                "hit_frame": hit,
                "formal_p1_root_position_w_m": _vector(root_position),
                "task_target_position_w_m": _vector(target_pos),
                "task_target_velocity_w_mps": _vector(target_vel),
                "task_target_normal_w": _vector(target_normal),
                "ball_center_target_w_m": _vector(ball_pos),
                "default_task_fk": {
                    "position_w_m": _vector(default_pos[0]),
                    "velocity_w_mps": _vector(default_vel[0]),
                    "normal_w": _vector(default_normal),
                    "position_error_to_task_target_m": _norm(default_pos[0], target_pos),
                    "velocity_error_to_task_target_mps": _norm(default_vel[0], target_vel),
                    "normal_error_to_task_target_deg": _angle_deg(default_normal, target_normal),
                },
                "validated_wrist_offset_fk": {
                    "position_w_m": _vector(wrist_pos[0]),
                    "velocity_w_mps": _vector(wrist_vel[0]),
                    "normal_w": _vector(wrist_normal),
                    "position_error_to_task_target_m": _norm(wrist_pos[0], target_pos),
                    "velocity_error_to_task_target_mps": _norm(wrist_vel[0], target_vel),
                    "normal_error_to_task_target_deg": _angle_deg(wrist_normal, target_normal),
                },
                "default_vs_wrist_offset_position_m": _norm(default_pos[0], wrist_pos[0]),
                "minimum_soft_joint_margin_rad": float(soft_margin.min().detach().cpu()),
                "minimum_soft_joint_margin_joint": robot.joint_names[minimum_soft_index],
                "minimum_hard_joint_margin_rad": float(hard_margin.min().detach().cpu()),
                "minimum_hard_joint_margin_joint": robot.joint_names[minimum_hard_index],
                "geometry": {
                    "racket_xy_over_table": table_xy,
                    "racket_height_above_table_m": float(target_pos[2]),
                    "ball_height_above_table_m": float(ball_pos[2]),
                    "racket_before_near_edge_m": max(-float(target_pos[0]), 0.0),
                },
            }
        )

    return {
        "purpose": "read-only canonical motion FK audit in formal HOPE P1 match scene",
        "task": args_cli.task,
        "canonical_manifest": str(manifest_path),
        "canonical_contract_version": manifest["contract_version"],
        "scene_frame": "HOPE world: table surface origin, P1 faces +X",
        "formal_p1_anchor_w_m": _vector(p1_anchor),
        "simulation_steps_advanced": 0,
        "policy_actions_applied": 0,
        "training_started": False,
        "qualification_scope": [
            "joint-name mapping",
            "formal-scene placement",
            "hit-frame FK position/orientation/declared velocity",
            "soft and hard joint-limit margins",
            "coarse table geometry",
        ],
        "not_qualified": [
            "dynamic tracking",
            "self collision",
            "table collision along the full trajectory",
            "balance",
            "physical ball contact TCP",
            "Planner/control clock mapping",
        ],
        "motions": rows,
    }


def main() -> None:
    try:
        report = _run()
        output = args_cli.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(output)
        print(output, flush=True)
    finally:
        # An offline Isaac installation can leave USD resource operations in a
        # permanent loading state during close(), even though the atomic report
        # is already complete.  Prefer normal cleanup but never leave this
        # read-only audit occupying the GPU indefinitely.
        if os.name == "posix":
            signal.signal(signal.SIGALRM, lambda *_: os._exit(0))
            signal.alarm(10)
        simulation_app.close(wait_for_replicator=False)
        if os.name == "posix":
            signal.alarm(0)


if __name__ == "__main__":
    main()
