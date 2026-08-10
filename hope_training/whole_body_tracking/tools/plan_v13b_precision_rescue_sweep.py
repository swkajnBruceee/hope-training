#!/usr/bin/env python3
"""Create the two-stage, Common-set-first Rescue checkpoint sweep plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "eval_outputs/v13b_complete_priors_precision_rescue/checkpoint_selection/checkpoint_metrics.json"
OUT = ROOT / "eval_outputs/v13b_complete_priors_precision_rescue/checkpoint_selection/sweep_plan.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=1000)
    parser.add_argument("--coarse-episodes", type=int, default=128)
    parser.add_argument("--finalists", type=int, default=8)
    parser.add_argument("--fine-episodes", type=int, default=512)
    args = parser.parse_args()
    payload = json.loads(INVENTORY.read_text(encoding="utf-8"))
    pool = payload["candidate_pool"]
    latest = max(pool, key=lambda row: row["iteration"])
    coarse = [row for row in pool if row["iteration"] % args.stride == 0]
    forced = {7000, latest["iteration"]}
    for row in pool:
        if row["iteration"] in forced and row not in coarse:
            coarse.append(row)
    coarse.sort(key=lambda row: row["iteration"])
    plan = {
        "status": "pending_execution",
        "sets": {
            "native": "fixed targets sampled from each checkpoint's own historical curriculum snapshot",
            "common": "one fixed seed/goal list at the final current-local distribution; this is decisive for prior ablation and source selection",
        },
        "stage_1": {
            "episodes_per_condition": args.coarse_episodes,
            "conditions": ["historical", "upper_off", "all_off"],
            "candidates": coarse,
            "hard_filter": {"common_survival_min": .95, "common_position_error_max_m": .03},
            "ranking": ["common_upper_off_normal", "common_upper_off_velocity", "common_all_off", "prior_reliance_gap"],
        },
        "stage_2": {
            "select_top_pareto": args.finalists,
            "episodes_per_condition": args.fine_episodes,
            "conditions": ["historical", "half_upper", "upper_off", "all_off"],
            "sets": ["native", "common"],
            "model7000_is_candidate_only": True,
        },
        "selection_rule": "latest transfer-ready actor before autonomous upper normal/velocity collapse; never select by historical reward alone",
        "no_learning_replay_after_stage_2": ["wide_normal_episode_sum", "wide_velocity_episode_sum", "temporal_effective_frames", "velocity_position_gate", "normal_error", "velocity_error"],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
