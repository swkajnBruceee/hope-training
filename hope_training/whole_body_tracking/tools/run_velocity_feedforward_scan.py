#!/usr/bin/env python3
"""Safety-gated A/B joint-velocity feedforward scan on frozen V2.

The scan never trains PPO, changes the motion data, changes the position-lead
contract, or changes either frozen prior.  It only varies the velocity phase
contract and scalar beta for the three audited right-arm joints.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
)
JOINT_SETS = {
    "all": JOINTS,
    "shoulders": JOINTS[:2],
    "pitch": (JOINTS[0],),
    "yaw": (JOINTS[1],),
    "elbow": (JOINTS[2],),
}


def _csv(value: str, cast: Any) -> tuple[Any, ...]:
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def _summarize(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = report["results"]
    by_motion = {int(row["motion_id"]): row for row in rows}
    return {
        "report": str(report_path),
        "safety_pass_all": all(bool(row["safety_pass"]) for row in rows),
        "stability_pass_all": all(bool(row["stability_pass"]) for row in rows),
        "mean_position_error_m": sum(float(row["position_error_m"]) for row in rows) / len(rows),
        "mean_velocity_error_mps": sum(float(row["velocity_error_mps"]) for row in rows) / len(rows),
        "mean_normal_error_deg": sum(float(row["normal_error_deg"]) for row in rows) / len(rows),
        "mean_root_displacement_m": sum(float(row["max_root_displacement_m"]) for row in rows) / len(rows),
        "hard_motions": {
            str(motion_id): {
                "position_error_m": float(by_motion[motion_id]["position_error_m"]),
                "velocity_error_mps": float(by_motion[motion_id]["velocity_error_mps"]),
                "velocity_error_xyz_mps": [
                    float(by_motion[motion_id][f"velocity_error_{axis}_mps"])
                    for axis in ("x", "y", "z")
                ],
                "actual_velocity_xyz_mps": [
                    float(by_motion[motion_id][f"racket_velocity_{axis}_mps"])
                    for axis in ("x", "y", "z")
                ],
            }
            for motion_id in (4, 5)
        },
    }


def _command(
    args: argparse.Namespace, mode: str, beta: float, joint_names: tuple[str, ...], report_path: Path
) -> list[str]:
    python = os.environ.get("HOPE_ISAAC_PYTHON")
    if not python:
        raise RuntimeError("source setup_train_env.sh before running this tool")
    return [
        python,
        "scripts/train.py",
        "task=HOPEA3JointCoordinatorV2",
        "algo=ppo_joint_coordinator",
        "headless=true",
        "logger=tensorboard",
        f"device={args.device}",
        f"seed={args.seed}",
        "num_envs=6",
        "resume=true",
        f"checkpoint={args.checkpoint}",
        f"++task.actions.joint_velocity_feedforward_mode={mode}",
        f"++task.actions.joint_velocity_feedforward_beta={beta}",
        "++task.actions.joint_velocity_feedforward_joint_names=["
        + ",".join(joint_names)
        + "]",
        f"++task.actions.joint_velocity_feedforward_post_hit_decay_steps={args.post_hit_decay_steps}",
        "+audit_policy_action=true",
        f"+audit_post_hit_steps={args.post_hit_steps}",
        f"+audit_output={json.dumps(str(report_path), ensure_ascii=False)}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=(
            "logs/rsl_rl/agibot_a3_joint_coordinator_v2_20260726/"
            "2026-07-26_18-10-06_joint_coordinator_v2_256x1000/model_900.pt"
        ),
    )
    parser.add_argument("--output-dir", default="eval_outputs/velocity_feedforward/v2_ab_scan_20260726")
    parser.add_argument("--modes", default="position_lead,task_phase")
    parser.add_argument("--betas", default="0.25,0.50,0.75,1.00")
    parser.add_argument(
        "--joint-sets",
        default="all",
        help="comma-separated sets from: " + ", ".join(JOINT_SETS),
    )
    parser.add_argument("--post-hit-steps", type=int, default=20)
    parser.add_argument("--post-hit-decay-steps", type=int, default=6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    workspace = Path.cwd()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = (workspace / checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    args.checkpoint = str(checkpoint)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (workspace / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = _csv(args.modes, str)
    if not set(modes) <= {"position_lead", "task_phase"}:
        raise ValueError(f"unsupported modes: {modes}")
    betas = _csv(args.betas, float)
    if not betas or any(not 0.0 < beta <= 1.0 for beta in betas):
        raise ValueError("betas must be in (0, 1]")
    joint_set_names = _csv(args.joint_sets, str)
    unknown_joint_sets = set(joint_set_names) - set(JOINT_SETS)
    if unknown_joint_sets:
        raise ValueError(f"unsupported joint sets: {sorted(unknown_joint_sets)}")

    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.environ.get("HOPE_WBT_PYTHONPATH", environment.get("PYTHONPATH", ""))
    candidates: list[dict[str, Any]] = []
    for joint_set_name in joint_set_names:
        joint_names = JOINT_SETS[joint_set_name]
        for mode in modes:
            for beta in betas:
                suffix = "" if joint_set_name == "all" else f"_{joint_set_name}"
                report_path = output_dir / f"{mode}_beta_{beta:.2f}{suffix}.json"
                if args.force or not report_path.is_file():
                    print(
                        f"[velocity-ff] running joints={joint_set_name} mode={mode} beta={beta:.2f}",
                        flush=True,
                    )
                    result = subprocess.run(
                        _command(args, mode, beta, joint_names, report_path),
                        cwd=workspace,
                        env=environment,
                        text=True,
                        stdout=sys.stdout,
                        stderr=sys.stderr,
                    )
                    if result.returncode != 0:
                        raise RuntimeError(
                            f"velocity feedforward scan failed: {joint_set_name=} {mode=} {beta=}"
                        )
                summary = _summarize(report_path)
                summary.update({"joint_set": joint_set_name, "joint_names": joint_names, "mode": mode, "beta": beta})
                candidates.append(summary)
                print(
                    "[velocity-ff] complete "
                    f"joints={joint_set_name} {mode} beta={beta:.2f}: "
                    f"pos={summary['mean_position_error_m'] * 100:.2f}cm "
                    f"vel={summary['mean_velocity_error_mps']:.3f}m/s "
                    f"safe={summary['safety_pass_all']} stable={summary['stability_pass_all']}",
                    flush=True,
                )

    safe = [item for item in candidates if item["safety_pass_all"] and item["stability_pass_all"]]
    # Rank only admissible candidates.  The report deliberately preserves the
    # complete Pareto data so deployment selection cannot hide a position or
    # stability regression behind a lower mean speed error.
    ranking = sorted(
        safe,
        key=lambda item: (
            item["mean_velocity_error_mps"],
            item["mean_position_error_m"],
            item["mean_root_displacement_m"],
        ),
    )
    payload = {
        "purpose": "frozen V2 A/B joint velocity feedforward scan",
        "contract": {
            "task": "HOPEA3JointCoordinatorV2",
            "checkpoint": str(checkpoint),
            "position_lead": {
                "right_shoulder_pitch_joint": 12,
                "right_shoulder_yaw_joint": 12,
                "right_elbow_joint": 0,
            },
            "available_velocity_joint_sets": JOINT_SETS,
            "post_hit_decay_steps": args.post_hit_decay_steps,
            "post_hit_safety_window_steps": args.post_hit_steps,
            "no_training": True,
        },
        "candidates": candidates,
        "safe_ranking": ranking,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[velocity-ff] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
