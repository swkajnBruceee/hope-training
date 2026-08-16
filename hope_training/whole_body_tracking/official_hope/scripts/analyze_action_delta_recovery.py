"""Analyze natural gate-minus-baseline policy-action deltas on the paired prefix."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


OFFSETS = (0.05, 0.10, 0.20, 0.30, 0.50, 0.80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--gate-result", required=True)
    parser.add_argument("--json-out", required=True)
    return parser.parse_args()


def vec_norm(values) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values))


def stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p90": ordered[max(0, math.ceil(0.90 * len(values)) - 1)],
    }


def main() -> int:
    args = parse_args()
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    gate = json.loads(Path(args.gate).read_text(encoding="utf-8"))
    result = json.loads(Path(args.gate_result).read_text(encoding="utf-8"))

    first_mismatch = {}
    for mismatch in result.get("paired_recipe_mismatches", []):
        if "env_id" in mismatch and "event_index" in mismatch:
            first_mismatch.setdefault(int(mismatch["env_id"]), int(mismatch["event_index"]))

    def outcome_map(payload):
        return {
            (int(row["env_id"]), int(row["paired_recipe_index"])): row
            for row in payload.get("rows", [])
            if row.get("paired_recipe_index") is not None
        }

    def post_map(payload):
        return {
            (
                int(row["env_id"]),
                int(row["source_paired_recipe_index"]),
                round(float(row["offset_s"]), 6),
            ): row
            for row in payload.get("post_strike_state_rows", [])
            if row.get("source_paired_recipe_index") is not None
            and row.get("offset_s") is not None
        }

    base_outcomes = outcome_map(baseline)
    gate_outcomes = outcome_map(gate)
    base_post = post_map(baseline)
    gate_post = post_map(gate)
    reset_by_key = defaultdict(list)
    for event in gate.get("reset_events", []):
        reset_by_key[(int(event["env_id"]), int(event.get("paired_recipe_index", -1)))].append(event)

    candidates = []
    for key, gate_row in gate_outcomes.items():
        env_id, source_index = key
        if not gate_row.get("fh_correction_applied", False):
            continue
        next_key = (env_id, source_index + 1)
        if next_key not in gate_outcomes or next_key not in base_outcomes:
            continue
        if env_id in first_mismatch and source_index + 1 >= first_mismatch[env_id]:
            continue
        reset = reset_by_key.get(next_key, [])
        next_row = gate_outcomes[next_key]
        reset_before = [
            event for event in reset
            if int(event.get("global_step", 10**18)) <= int(next_row.get("global_step", 10**18))
        ]
        if reset_before:
            outcome = "RESET_BEFORE_NEXT_SHOT"
        elif next_row.get("failure_code") == "LEGAL":
            outcome = "LEGAL"
        else:
            outcome = "FAILURE"
        candidates.append({"env_id": env_id, "source_index": source_index, "outcome": outcome})

    by_offset = {}
    for offset in OFFSETS:
        groups = defaultdict(lambda: defaultdict(list))
        n = 0
        for sample in candidates:
            key = (sample["env_id"], sample["source_index"], round(float(offset), 6))
            base_row = base_post.get(key)
            gate_row = gate_post.get(key)
            if not base_row or not gate_row:
                continue
            base_action = base_row.get("policy_action")
            gate_action = gate_row.get("policy_action")
            if base_action is None or gate_action is None or len(base_action) != len(gate_action):
                continue
            delta = [float(g) - float(b) for b, g in zip(base_action, gate_action)]
            group = groups[sample["outcome"]]
            group["action_delta_norm"].append(vec_norm(delta))
            group["action_delta_mean_abs"].append(statistics.mean(abs(value) for value in delta))
            group["baseline_action_norm"].append(vec_norm(base_action))
            group["gate_action_norm"].append(vec_norm(gate_action))
            n += 1
        by_offset[str(offset)] = {
            "n": n,
            "outcome_counts": {group: len(values["action_delta_norm"]) for group, values in groups.items()},
            "features": {
                group: {name: stats(values) for name, values in values.items()}
                for group, values in groups.items()
            },
        }

    report = {
        "schema_version": 1,
        "baseline": str(Path(args.baseline)),
        "gate": str(Path(args.gate)),
        "gate_result": str(Path(args.gate_result)),
        "sample_selection": {
            "candidates": len(candidates),
            "strict_prefix_only": True,
            "natural_action_association_not_causal": True,
        },
        "outcome_counts": dict(Counter(sample["outcome"] for sample in candidates)),
        "by_offset": by_offset,
    }
    output = Path(args.json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
