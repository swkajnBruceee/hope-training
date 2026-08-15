"""Evaluate a checkpoint sweep on fixed Isaac validation seeds.

The sweep intentionally uses raw training-side virtual-ball counts rather than the
command term's decayed EMA metrics.  Its primary selection quantity is
``legal_landings / strike_attempts``.  Each checkpoint is evaluated independently
on the same seed set and budget.

Example:
    python scripts/sweep_checkpoints.py \
        --run-dir logs/rsl_rl/agibot_a3_hitter_pingpong/<run> \
        --checkpoints 0,100,500,1000,1500,2000,2500,2999 \
        --seeds 100,101,102,103,104 \
        --num-envs 32 --num-steps 750 --device cuda:1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import statistics
import subprocess
import sys
from typing import Any


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("list must not be empty")
    return values


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def run_one(args: argparse.Namespace, checkpoint: pathlib.Path, seed: int, output: pathlib.Path) -> dict[str, Any]:
    command = [
        sys.executable,
        str(pathlib.Path(__file__).with_name("evaluate.py")),
        "--checkpoint",
        str(checkpoint),
        "--num-envs",
        str(args.num_envs),
        "--num-steps",
        str(args.num_steps),
        "--device",
        args.device,
        "--seed",
        str(seed),
        "--diagnostics",
        "--algo-config",
        args.algo_config,
        "--json-out",
        str(output),
    ]
    print("[sweep]", " ".join(command), flush=True)
    if args.dry_run:
        return {"checkpoint": str(checkpoint), "seed": seed, "dry_run": True}
    # evaluate.py must be launched with the same source overlays as the normal
    # ``hope_isaac_py`` entry point.  In particular, IsaacLab is a source checkout
    # here and the local whole_body_tracking package must win over any installed copy.
    isaaclab_root = (args.repo_root / "../../../external_repos/IsaacLab").resolve()
    source_paths = [
        args.repo_root / "source" / "whole_body_tracking",
        isaaclab_root / "source" / "isaaclab",
        isaaclab_root / "source" / "isaaclab_tasks",
        isaaclab_root / "source" / "isaaclab_assets",
        isaaclab_root / "source" / "isaaclab_rl",
    ]
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = os.pathsep.join(str(path) for path in source_paths) + (
        os.pathsep + child_env["PYTHONPATH"] if child_env.get("PYTHONPATH") else ""
    )
    completed = subprocess.run(command, cwd=str(args.repo_root), env=child_env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"evaluation failed for {checkpoint.name}, seed={seed}")
    with output.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    result["checkpoint"] = str(checkpoint)
    return result


def summarize(checkpoint: pathlib.Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    count_rows = [row["internal_virtual_counts"] for row in rows]
    rate_rows = [row["internal_virtual_rates"] for row in rows]
    def vals(name: str) -> list[float]:
        return [float(row[name]) for row in rate_rows]

    count_names = ("attempts", "hits", "net_clears", "valid_landings", "legal_landings")
    totals = {name: int(sum(int(row[name]) for row in count_rows)) for name in count_names}
    seed_stats = {}
    for name in ("hit_per_attempt", "net_per_attempt", "legal_per_attempt", "legal_per_hit", "valid_land_per_attempt"):
        m, s = mean_std(vals(name))
        seed_stats[name] = {"mean": m, "std": s, "values": vals(name)}

    reset_rates = [float(row.get("reset_rate_per_1k_steps", 0.0)) for row in rows]
    land_errors = [
        float(row.get("isaac_internal_metrics_mean", {}).get("virtual_land_err_m", float("nan")))
        for row in rows
    ]
    reset_mean, reset_std = mean_std(reset_rates)
    finite_land_errors = [value for value in land_errors if math.isfinite(value)]
    land_mean, land_std = mean_std(finite_land_errors)
    aggregate = {
        "hit_per_attempt": ratio(totals["hits"], totals["attempts"]),
        "net_per_attempt": ratio(totals["net_clears"], totals["attempts"]),
        "legal_per_attempt": ratio(totals["legal_landings"], totals["attempts"]),
        "legal_per_hit": ratio(totals["legal_landings"], totals["hits"]),
        "valid_land_per_attempt": ratio(totals["valid_landings"], totals["attempts"]),
    }
    return {
        "checkpoint": checkpoint.name,
        "checkpoint_path": str(checkpoint),
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "seeds": args.seeds,
        "counts_aggregate": totals,
        "rates_aggregate": aggregate,
        "rates_seed_mean_std": seed_stats,
        "reset_per_1k_steps": {"mean": reset_mean, "std": reset_std, "values": reset_rates},
        "land_err_m": {"mean": land_mean, "std": land_std, "values": finite_land_errors},
        "raw_results": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=pathlib.Path)
    parser.add_argument("--checkpoints", type=parse_int_list, default=list(range(0, 3000, 100)) + [2999])
    parser.add_argument("--seeds", type=parse_int_list, default=[100, 101, 102, 103, 104])
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=750)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--algo-config", default="cfg/algo/ppo_residual.yaml")
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.repo_root = pathlib.Path(__file__).resolve().parents[1]
    args.run_dir = args.run_dir if args.run_dir.is_absolute() else args.repo_root / args.run_dir
    args.output = args.output or (args.run_dir / "checkpoint_sweep_seed100-104.json")
    args.output = args.output if args.output.is_absolute() else args.repo_root / args.output
    args.output.parent.mkdir(parents=True, exist_ok=True)

    summaries = []
    for iteration in args.checkpoints:
        checkpoint = args.run_dir / f"model_{iteration}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        rows = []
        for seed in args.seeds:
            output = args.run_dir / "sweep" / f"model_{iteration}_seed{seed}.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            rows.append(run_one(args, checkpoint, seed, output))
        if not args.dry_run:
            summaries.append(summarize(checkpoint, rows, args))

    if args.dry_run:
        return 0
    payload = {
        "protocol": {
            "primary_metric": "legal_landings / strike_attempts",
            "secondary_order": ["net_clears / strike_attempts", "hits / strike_attempts", "reset_per_1k_steps", "virtual_land_err_m"],
            "seeds": args.seeds,
            "num_envs": args.num_envs,
            "num_steps": args.num_steps,
        },
        "summaries": summaries,
    }
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    csv_path = args.output.with_suffix(".csv")
    fields = ["checkpoint", "attempts", "hit_per_attempt", "net_per_attempt", "legal_per_attempt", "legal_per_hit", "reset_per_1k_mean", "reset_per_1k_std", "land_err_mean", "land_err_std"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({
                "checkpoint": summary["checkpoint"],
                "attempts": summary["counts_aggregate"]["attempts"],
                "hit_per_attempt": summary["rates_aggregate"]["hit_per_attempt"],
                "net_per_attempt": summary["rates_aggregate"]["net_per_attempt"],
                "legal_per_attempt": summary["rates_aggregate"]["legal_per_attempt"],
                "legal_per_hit": summary["rates_aggregate"]["legal_per_hit"],
                "reset_per_1k_mean": summary["reset_per_1k_steps"]["mean"],
                "reset_per_1k_std": summary["reset_per_1k_steps"]["std"],
                "land_err_mean": summary["land_err_m"]["mean"],
                "land_err_std": summary["land_err_m"]["std"],
            })
    print(f"[sweep] wrote {args.output}")
    print(f"[sweep] wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
