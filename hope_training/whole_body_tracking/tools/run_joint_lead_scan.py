#!/usr/bin/env python3
"""Run a deterministic, safety-gated absolute joint-lead scan for V2.

Each candidate is an isolated process so the Isaac simulation and policy state
cannot leak between leads.  The frozen coordinator checkpoint, frozen upper
prior, legacy Stage-A prior, manifest, target, seed, ready prelude and all
rewards are inherited from HOPEA3JointCoordinatorV2.  Only the named joint's
absolute reference lead changes.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
from typing import Any


DEFAULT_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
)
DEFAULT_LEADS = (0, 4, 8, 12, 16, 20)


def _parse_csv(value: str, cast: Any) -> tuple[Any, ...]:
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def _lead_mapping(scanned_joint: str, lead: int) -> str:
    # Full replacement is intentional: numbers are absolute.  It prevents the
    # historical 12-step shoulder lead from being added twice during a scan.
    values = {
        "right_shoulder_pitch_joint": 12,
        "right_shoulder_yaw_joint": 12,
        "right_elbow_joint": 0,
    }
    values[scanned_joint] = lead
    return "{" + ",".join(f"{name}: {value}.0" for name, value in values.items()) + "}"


def _metrics(report_path: pathlib.Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = report["results"]
    safe = all(bool(row.get("safety_pass")) for row in rows)
    stable = all(bool(row.get("stability_pass")) for row in rows)
    return {
        "report": str(report_path),
        "safety_pass_all": safe,
        "stability_pass_all": stable,
        "mean_position_error_m": sum(float(row["position_error_m"]) for row in rows) / len(rows),
        "mean_velocity_error_mps": sum(float(row["velocity_error_mps"]) for row in rows) / len(rows),
        "mean_normal_error_deg": sum(float(row["normal_error_deg"]) for row in rows) / len(rows),
        "mean_root_displacement_m": sum(float(row["max_root_displacement_m"]) for row in rows) / len(rows),
        "hard_motion_velocity_error_mps": {
            str(row["motion_id"]): float(row["velocity_error_mps"])
            for row in rows
            if int(row["motion_id"]) in (4, 5)
        },
        "motions": [
            {
                "motion_id": int(row["motion_id"]),
                "position_error_m": float(row["position_error_m"]),
                "velocity_error_mps": float(row["velocity_error_mps"]),
                "velocity_error_xyz_mps": [
                    float(row["velocity_error_x_mps"]),
                    float(row["velocity_error_y_mps"]),
                    float(row["velocity_error_z_mps"]),
                ],
                "actual_velocity_xyz_mps": [
                    float(row["racket_velocity_x_mps"]),
                    float(row["racket_velocity_y_mps"]),
                    float(row["racket_velocity_z_mps"]),
                ],
                "normal_error_deg": float(row["normal_error_deg"]),
                "root_displacement_m": float(row["max_root_displacement_m"]),
                "foot_slip_mps": float(row["max_loaded_foot_tangential_speed_mps"]),
                "torque_saturation_fraction": float(row["torque_saturation_fraction"]),
                "velocity_saturation_fraction": float(row["velocity_saturation_fraction"]),
                "safety_pass": bool(row["safety_pass"]),
                "stability_pass": bool(row["stability_pass"]),
                "first_failure_step": row.get("first_failure_step"),
                "termination_reasons": row.get("termination_reasons", []),
            }
            for row in rows
        ],
    }


def _command(args: argparse.Namespace, joint: str, lead: int, report: pathlib.Path) -> list[str]:
    python = os.environ.get("HOPE_ISAAC_PYTHON")
    if not python:
        raise RuntimeError("source setup_train_env.sh before running this tool (HOPE_ISAAC_PYTHON missing)")
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
        f"++task.actions.joint_reference_lookahead_steps={_lead_mapping(joint, lead)}",
        "+audit_policy_action=true",
        f"+audit_post_hit_steps={args.post_hit_steps}",
        # Hydra parses the override itself, not a shell.  JSON quoting keeps an
        # absolute workspace path with non-ASCII directory names a scalar.
        f"+audit_output={json.dumps(str(report), ensure_ascii=False)}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="logs/rsl_rl/agibot_a3_joint_coordinator_v2_20260726/2026-07-26_18-10-06_joint_coordinator_v2_256x1000/model_900.pt",
    )
    parser.add_argument("--output-dir", default="eval_outputs/joint_lead_scan/v2_absolute_20260726")
    parser.add_argument("--joints", default=",".join(DEFAULT_JOINTS))
    parser.add_argument("--leads", default=",".join(map(str, DEFAULT_LEADS)))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--post-hit-steps", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    workspace = pathlib.Path.cwd()
    checkpoint = pathlib.Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = (workspace / checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    args.checkpoint = str(checkpoint)
    output_dir = pathlib.Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (workspace / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    joints = _parse_csv(args.joints, str)
    leads = _parse_csv(args.leads, int)
    unknown = set(joints) - set(DEFAULT_JOINTS)
    if unknown:
        raise ValueError(f"unsupported scan joints: {sorted(unknown)}")

    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.environ.get("HOPE_WBT_PYTHONPATH", environment.get("PYTHONPATH", ""))
    candidates: list[dict[str, Any]] = []
    for joint in joints:
        for lead in leads:
            report = output_dir / f"{joint}_absolute_lead_{lead:02d}.json"
            if not report.is_file() or args.force:
                print(f"[lead-scan] running joint={joint} absolute_lead={lead}", flush=True)
                result = subprocess.run(
                    _command(args, joint, lead, report),
                    cwd=workspace,
                    env=environment,
                    text=True,
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                )
                if result.returncode != 0:
                    raise RuntimeError(f"lead scan failed: joint={joint}, lead={lead}")
            metric = _metrics(report)
            metric.update({"joint": joint, "absolute_lead_steps": lead})
            candidates.append(metric)
            print(
                f"[lead-scan] complete joint={joint} lead={lead}: "
                f"pos={metric['mean_position_error_m'] * 100:.2f}cm "
                f"vel={metric['mean_velocity_error_mps']:.3f}m/s "
                f"safe={metric['safety_pass_all']} stable={metric['stability_pass_all']}",
                flush=True,
            )

    safe = [item for item in candidates if item["safety_pass_all"] and item["stability_pass_all"]]
    # The summary deliberately does not choose a deployment lead: optimization
    # needs the per-motion velocity directions and the later small combination
    # scan.  This ranking only identifies safe, promising coarse candidates.
    ranking = sorted(
        safe,
        key=lambda item: (
            (item["hard_motion_velocity_error_mps"].get("4", 99.0) + item["hard_motion_velocity_error_mps"].get("5", 99.0)) / 2.0,
            item["mean_velocity_error_mps"],
            item["mean_position_error_m"],
            item["mean_root_displacement_m"],
        ),
    )
    summary = {
        "purpose": "deterministic safety-gated absolute lead scan",
        "contract": {
            "task": "HOPEA3JointCoordinatorV2",
            "checkpoint": str(checkpoint),
            "seed": args.seed,
            "num_envs": 6,
            "post_hit_steps": args.post_hit_steps,
            "absolute_lead_semantics": True,
            "fixed_other_leads": {
                "right_shoulder_pitch_joint": 12,
                "right_shoulder_yaw_joint": 12,
                "right_elbow_joint": 0,
            },
            "safety_gate": "every motion must finish hit plus post-hit window with no termination, finite state, root height >= 0.65m",
        },
        "candidates": candidates,
        "safe_ranking_for_followup_combination_scan": ranking,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[lead-scan] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
