#!/usr/bin/env python3
"""Paired deterministic repeatability audit for the selected velocity contract."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any


CHECKPOINT = (
    "logs/rsl_rl/agibot_a3_joint_coordinator_v2_20260726/"
    "2026-07-26_18-10-06_joint_coordinator_v2_256x1000/model_900.pt"
)
SHOULDERS = ("right_shoulder_pitch_joint", "right_shoulder_yaw_joint")


def _run(
    workspace: Path,
    checkpoint: Path,
    output: Path,
    seed: int,
    candidate: bool,
    device: str,
) -> dict[str, Any]:
    python = os.environ.get("HOPE_ISAAC_PYTHON")
    if not python:
        raise RuntimeError("source setup_train_env.sh before running this tool")
    command = [
        python,
        "scripts/train.py",
        "task=HOPEA3JointCoordinatorV2",
        "algo=ppo_joint_coordinator",
        "headless=true",
        "logger=tensorboard",
        f"device={device}",
        f"seed={seed}",
        "num_envs=6",
        "resume=true",
        f"checkpoint={checkpoint}",
        "+audit_policy_action=true",
        "+audit_post_hit_steps=20",
        f"+audit_output={json.dumps(str(output), ensure_ascii=False)}",
    ]
    if candidate:
        command.extend(
            [
                "++task.actions.joint_velocity_feedforward_mode=task_phase",
                "++task.actions.joint_velocity_feedforward_beta=0.75",
                "++task.actions.joint_velocity_feedforward_joint_names=["
                + ",".join(SHOULDERS)
                + "]",
                "++task.actions.joint_velocity_feedforward_post_hit_decay_steps=6",
            ]
        )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.environ.get("HOPE_WBT_PYTHONPATH", environment.get("PYTHONPATH", ""))
    result = subprocess.run(command, cwd=workspace, env=environment, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{'candidate' if candidate else 'baseline'} seed={seed} failed:\n"
            + "\n".join((result.stdout + "\n" + result.stderr).splitlines()[-80:])
        )
    return json.loads(output.read_text(encoding="utf-8"))


def _metrics(report: dict[str, Any]) -> dict[str, Any]:
    rows = report["results"]
    return {
        "mean_position_error_m": sum(float(row["position_error_m"]) for row in rows) / len(rows),
        "mean_velocity_error_mps": sum(float(row["velocity_error_mps"]) for row in rows) / len(rows),
        "mean_normal_error_deg": sum(float(row["normal_error_deg"]) for row in rows) / len(rows),
        "mean_root_displacement_m": sum(float(row["max_root_displacement_m"]) for row in rows) / len(rows),
        "safety_pass_all": all(bool(row["safety_pass"]) for row in rows),
        "stability_pass_all": all(bool(row["stability_pass"]) for row in rows),
        "position_pass_count_10cm": sum(float(row["position_error_m"]) < 0.10 for row in rows),
        "normal_pass_count_20deg": sum(float(row["normal_error_deg"]) < 20.0 for row in rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--output-dir", default="eval_outputs/velocity_feedforward/v2_b_shoulders_beta075_repeatability_20260726")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    workspace = Path.cwd()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = (workspace / checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (workspace / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    if not seeds:
        raise ValueError("at least one seed is required")

    pairs: list[dict[str, Any]] = []
    for seed in seeds:
        base_path = output_dir / f"seed_{seed:02d}_v2_baseline.json"
        candidate_path = output_dir / f"seed_{seed:02d}_ff_b_shoulders_beta075.json"
        if args.force or not base_path.is_file():
            print(f"[repeatability] baseline seed={seed}", flush=True)
            base = _run(workspace, checkpoint, base_path, seed, False, args.device)
        else:
            base = json.loads(base_path.read_text(encoding="utf-8"))
        if args.force or not candidate_path.is_file():
            print(f"[repeatability] candidate seed={seed}", flush=True)
            candidate = _run(workspace, checkpoint, candidate_path, seed, True, args.device)
        else:
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        baseline = _metrics(base)
        selected = _metrics(candidate)
        pair = {
            "seed": seed,
            "baseline_report": str(base_path),
            "candidate_report": str(candidate_path),
            "baseline": baseline,
            "candidate": selected,
            "delta_candidate_minus_baseline": {
                key: selected[key] - baseline[key]
                for key in (
                    "mean_position_error_m",
                    "mean_velocity_error_mps",
                    "mean_normal_error_deg",
                    "mean_root_displacement_m",
                )
            },
        }
        delta = pair["delta_candidate_minus_baseline"]
        pair["pass"] = bool(
            baseline["safety_pass_all"]
            and selected["safety_pass_all"]
            and selected["stability_pass_all"]
            and selected["position_pass_count_10cm"] == 6
            and selected["normal_pass_count_20deg"] == 6
            and delta["mean_position_error_m"] < 0.0
            and delta["mean_velocity_error_mps"] < 0.0
            and delta["mean_root_displacement_m"] <= 0.001
        )
        pairs.append(pair)
        print(
            f"[repeatability] seed={seed} pos {baseline['mean_position_error_m'] * 100:.2f} -> "
            f"{selected['mean_position_error_m'] * 100:.2f}cm, vel "
            f"{baseline['mean_velocity_error_mps']:.3f} -> {selected['mean_velocity_error_mps']:.3f}, "
            f"pass={pair['pass']}",
            flush=True,
        )

    first = pairs[0]
    repeatability = {}
    for contract in ("baseline", "candidate"):
        for metric in (
            "mean_position_error_m",
            "mean_velocity_error_mps",
            "mean_normal_error_deg",
            "mean_root_displacement_m",
        ):
            values = [float(pair[contract][metric]) for pair in pairs]
            repeatability[f"{contract}_{metric}_range"] = max(values) - min(values)
    payload = {
        "purpose": "paired deterministic repeatability, not perturbation robustness",
        "candidate_contract": {
            "position_lead_steps": {"right_shoulder_pitch_joint": 12, "right_shoulder_yaw_joint": 12},
            "velocity_phase": "task_phase",
            "velocity_beta": 0.75,
            "velocity_joints": SHOULDERS,
            "post_hit_decay_steps": 6,
        },
        "checkpoint": str(checkpoint),
        "seeds": seeds,
        "pairs": pairs,
        "repeatability_ranges": repeatability,
        "passed": all(pair["pass"] for pair in pairs),
        "deterministic_baseline_reference": first["baseline"],
        "deterministic_candidate_reference": first["candidate"],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[repeatability] wrote {summary_path}; passed={payload['passed']}", flush=True)


if __name__ == "__main__":
    main()
