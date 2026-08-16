"""Analyze a baseline/gate paired-recipe audit without mixing post-divergence samples."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--gate-result", required=True)
    parser.add_argument("--json-out", required=True)
    return parser.parse_args()


def legal(row: dict) -> bool:
    return row.get("failure_code") == "LEGAL"


def paired_stats(rows: list[tuple[dict, dict]]) -> dict:
    n = len(rows)
    base_legal = sum(legal(base) for base, _ in rows)
    gate_legal = sum(legal(gate) for _, gate in rows)
    return {
        "n": n,
        "baseline_legal_rate": base_legal / n if n else 0.0,
        "gate_legal_rate": gate_legal / n if n else 0.0,
        "delta_legal_pp": 100.0 * (gate_legal - base_legal) / n if n else 0.0,
        "baseline_failure_code": dict(Counter(base["failure_code"] for base, _ in rows)),
        "gate_failure_code": dict(Counter(gate["failure_code"] for _, gate in rows)),
    }


def state_delta(rows: list[tuple[dict, dict]], field: str) -> dict:
    values = []
    for base, gate in rows:
        a = base.get(field)
        b = gate.get(field)
        if a is None or b is None:
            continue
        values.append(max(abs(x - y) for x, y in zip(a, b)))
    return {
        "n": len(values),
        "mean_max_abs_delta": statistics.mean(values) if values else 0.0,
        "max_abs_delta": max(values) if values else 0.0,
    }


def main() -> int:
    args = parse_args()
    baseline_rows = json.loads(Path(args.baseline).read_text(encoding="utf-8"))["rows"]
    gate_rows = json.loads(Path(args.gate).read_text(encoding="utf-8"))["rows"]
    gate_result = json.loads(Path(args.gate_result).read_text(encoding="utf-8"))
    baseline = {(row["env_id"], row["paired_recipe_index"]): row for row in baseline_rows}
    gate = {(row["env_id"], row["paired_recipe_index"]): row for row in gate_rows}

    common = []
    current = []
    for key in sorted(set(baseline) & set(gate)):
        pair = (baseline[key], gate[key])
        common.append(pair)
        base, gated = pair
        if base["clip_id"] == 0 and not base["venue_tuple_selected"] and gated.get(
            "fh_correction_applied", False
        ):
            current.append(pair)

    # The evaluator deliberately stores only the first 100 mismatch records in the result JSON.
    # This prefix is therefore a conservative, explicitly labeled subset of strict paired data.
    first_mismatch: dict[int, int] = {}
    for mismatch in gate_result.get("paired_recipe_mismatches", []):
        if "event_index" in mismatch:
            first_mismatch.setdefault(int(mismatch["env_id"]), int(mismatch["event_index"]))
    strict_current = [
        pair
        for pair in current
        if pair[0]["env_id"] in first_mismatch
        and pair[0]["paired_recipe_index"] < first_mismatch[pair[0]["env_id"]]
    ]
    strict_next = []
    for base, gated in strict_current:
        key = (base["env_id"], base["paired_recipe_index"] + 1)
        if key in baseline and key in gate and key[1] < first_mismatch[key[0]]:
            strict_next.append((baseline[key], gate[key]))

    report = {
        "schema_version": 1,
        "baseline_rows": len(baseline_rows),
        "gate_rows": len(gate_rows),
        "paired_recipe_mismatch_count": gate_result.get("paired_recipe_mismatch_count", 0),
        "known_first_mismatch_envs": len(first_mismatch),
        "current_fh": {
            "common_event_index_exploratory": paired_stats(current),
            "strict_prefix": paired_stats(strict_current),
            "state_delta_strict_prefix": {
                field: state_delta(strict_current, field)
                for field in ("robot_root_lin_vel_w", "robot_root_ang_vel_w")
            },
        },
        "next_shot_after_current_fh": {
            "strict_prefix": paired_stats(strict_next),
            "state_delta_strict_prefix": {
                field: state_delta(strict_next, field)
                for field in ("robot_root_lin_vel_w", "robot_root_ang_vel_w")
            },
        },
        "interpretation": {
            "strict_prefix_is_causal_safe": True,
            "common_event_index_is_causal_safe": False,
            "post_divergence_gate_rows_must_not_be_used_as_paired_evidence": True,
        },
    }
    output = Path(args.json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
