#!/usr/bin/env python3
"""Create the log-only 10-checkpoint shortlist for PrecisionRescue."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "eval_outputs/v13b_complete_priors_precision_rescue/checkpoint_selection"
HISTORY = BASE / "history_scalars.json"
INVENTORY = BASE / "checkpoint_metrics.json"
OUT = BASE / "log_shortlist.json"
SELECTED = {
    3000: "early stable baseline",
    5000: "early high-combined plateau",
    7000: "requested early-transfer candidate; never privileged",
    8000: "historical combined-success peak neighbourhood",
    10000: "post-peak, still low normal/velocity error",
    12000: "mid-transfer checkpoint",
    13000: "last low-normal pre-acceleration checkpoint",
    14000: "normal-error acceleration onset",
    15000: "sustained-degradation boundary",
    16000: "early-collapse boundary / negative control",
}
WINDOW = 200


def main() -> None:
    history = json.loads(HISTORY.read_text(encoding="utf-8"))["tags"]
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    by_iteration = {int(row["iteration"]): row for row in inventory["candidate_pool"]}
    rows = []
    for iteration, reason in SELECTED.items():
        if iteration not in by_iteration:
            raise RuntimeError(f"checkpoint missing: model_{iteration}.pt")
        row = dict(by_iteration[iteration])
        row["log_selection_reason"] = reason
        for name in ("position_error_m", "velocity_error_mps", "normal_error_deg", "combined_success"):
            points = [item["value"] for item in history[name] if iteration - WINDOW <= item["step"] <= iteration]
            row[f"trailing_{WINDOW}_{name}"] = sum(points) / len(points) if points else None
        rows.append(row)
    report = {
        "status": "pass",
        "contract": "v13b_precision_rescue_log_coarse_selection_v1",
        "window_updates": WINDOW,
        "shortlist": rows,
        "stage_1": {
            "sets": {"common": ["historical", "upper_off", "all_off"], "native": ["historical"]},
            "episodes": 128,
            "reason": "Common-set prior ablations decide transfer readiness; Native Historical only preserves each checkpoint's own-difficulty context without duplicating all ablations.",
        },
        "stage_2": {"sets": ["native", "common"], "conditions": ["historical", "half_upper", "upper_off", "all_off"], "episodes": 256, "finalists": "3-5"},
        "selection_prohibited": True,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    # Two balanced shards let the two 4090s evaluate independent checkpoint
    # groups while each shard still reuses one Isaac scene and one runner.
    for index, shard in enumerate((rows[::2], rows[1::2])):
        (BASE / f"log_shortlist_gpu{index}.json").write_text(
            json.dumps({**report, "shortlist": shard, "shard": index}, indent=2) + "\n",
            encoding="utf-8",
        )
    print(OUT)


if __name__ == "__main__":
    main()
