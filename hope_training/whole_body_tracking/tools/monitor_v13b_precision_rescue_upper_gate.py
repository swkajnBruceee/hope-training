#!/usr/bin/env python3
"""External, read-only teacher-off gate worker for PrecisionRescue.

Run this in a separate Isaac process (normally GPU1) while PPO runs on GPU0.
It evaluates saved policy checkpoints with upper prior forced to zero and
writes an atomic, monotone approval token consumed by the training runner.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "tools" / "evaluate_v13b_precision_rescue_candidate.py"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--gate-file", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--source-progress", required=True, type=float)
    parser.add_argument("--source-lower-alpha", required=True, type=float)
    parser.add_argument("--source-upper-alpha", required=True, type=float)
    parser.add_argument("--schedule-total-updates", required=True, type=int)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--interval-updates", type=int, default=200)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--min-survival", type=float, default=0.95)
    parser.add_argument("--min-hit-rate", type=float, default=0.95)
    parser.add_argument("--max-position-error-m", type=float, default=0.03)
    parser.add_argument("--max-normal-error-deg", type=float, default=35.0)
    parser.add_argument("--max-velocity-error-mps", type=float, default=1.2)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def _smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def _lower_alpha(progress: float) -> float:
    knots = ((0.00, 1.00), (0.10, 1.00), (0.25, 0.85), (0.45, 0.55), (0.60, 0.30), (0.70, 0.00), (1.00, 0.00))
    progress = max(0.0, min(1.0, progress))
    for (left_p, left_v), (right_p, right_v) in zip(knots[:-1], knots[1:]):
        if progress <= right_p:
            return left_v + _smoothstep((progress - left_p) / max(right_p - left_p, 1.0e-8)) * (right_v - left_v)
    return 0.0


def _checkpoint_iteration(path: Path) -> int | None:
    if path.stem.startswith("model_"):
        try:
            return int(path.stem.removeprefix("model_"))
        except ValueError:
            return None
    return None


def _latest_eligible_checkpoint(args: argparse.Namespace, minimum_iteration: int) -> tuple[int, Path] | None:
    rows = []
    for path in args.run_dir.glob("model_*.pt"):
        iteration = _checkpoint_iteration(path)
        if iteration is not None and iteration >= minimum_iteration:
            rows.append((iteration, path))
    if not rows:
        return None
    iteration, path = max(rows)
    # Never try to load a checkpoint while the trainer may still be writing it.
    if time.time() - path.stat().st_mtime < 15.0:
        return None
    return iteration, path


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_gate(path: Path) -> dict:
    if not path.is_file():
        return {"approved_withdrawal_index": 0, "consecutive_passes": 0, "last_evaluated_iteration": -1}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {"approved_withdrawal_index": 0, "consecutive_passes": 0, "last_evaluated_iteration": -1}


def _is_pass(report: dict, args: argparse.Namespace) -> bool:
    metrics = report.get("metrics", {})
    values = (
        metrics.get("survival_10s"), metrics.get("exact_hit_rate"),
        metrics.get("position_error_m"), metrics.get("normal_error_deg"), metrics.get("velocity_error_mps"),
    )
    if report.get("status") != "pass" or any(value is None or not math.isfinite(float(value)) for value in values):
        return False
    survival, hits, position, normal, velocity = (float(value) for value in values)
    return (
        survival >= args.min_survival and hits >= args.min_hit_rate and position <= args.max_position_error_m
        and normal <= args.max_normal_error_deg and velocity <= args.max_velocity_error_mps
    )


def main() -> None:
    args = _args()
    if args.schedule_total_updates < 2 or args.interval_updates <= 0:
        raise SystemExit("schedule-total-updates and interval-updates must be positive")
    source = args.source_checkpoint.resolve()
    reports_dir = args.gate_file.parent / "upper_off_reports"
    last_attempted = -1
    while True:
        minimum = max(300, last_attempted + args.interval_updates)
        candidate = _latest_eligible_checkpoint(args, minimum)
        if candidate is not None:
            iteration, checkpoint = candidate
            progress = min(1.0, args.source_progress + (1.0 - args.source_progress) * iteration / (args.schedule_total_updates - 1))
            lower = min(args.source_lower_alpha, _lower_alpha(progress))
            report_path = reports_dir / f"upper_off_model_{iteration}.json"
            command = [
                sys.executable, str(EVALUATOR), "--task-mode", "precision_rescue",
                "--rescue-schedule-total-updates", str(args.schedule_total_updates),
                "--checkpoint", str(checkpoint), "--iteration", str(iteration), "--set", "native",
                "--condition", "upper_off", "--progress", str(progress),
                "--source-lower-alpha", str(lower), "--source-upper-alpha", str(args.source_upper_alpha),
                "--episodes", str(args.episodes), "--max-steps", str(args.max_steps), "--device", args.device,
                "--output", str(report_path),
            ]
            result = subprocess.run(command, cwd=ROOT, check=False)
            report = json.loads(report_path.read_text(encoding="utf-8")) if result.returncode == 0 and report_path.is_file() else {}
            gate = _read_gate(args.gate_file)
            passed = _is_pass(report, args)
            streak = int(gate.get("consecutive_passes", 0)) + 1 if passed else 0
            approved = int(gate.get("approved_withdrawal_index", 0))
            if streak >= 2:
                approved += 1
                streak = 0
            gate = {
                "contract": "v13b_precision_rescue_upper_off_gate_v1",
                "run_id": args.run_id,
                "source_checkpoint": str(source),
                "approved_withdrawal_index": approved,
                "consecutive_passes": streak,
                "last_evaluated_iteration": iteration,
                "last_probe_pass": passed,
                "last_report": str(report_path),
            }
            _write_atomic(args.gate_file, gate)
            print(json.dumps(gate, indent=2), flush=True)
            last_attempted = iteration
            if args.once:
                return
        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
