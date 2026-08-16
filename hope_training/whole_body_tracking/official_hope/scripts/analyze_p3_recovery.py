"""Analyze post-strike recovery state deltas on the causal paired prefix.

This report deliberately separates three things:
* the current corrected FH shot;
* the immediately following shot;
* reset/topology events, which are not automatically attributable to FH.

Rows after the first recipe mismatch for an environment are excluded from all
paired state/outcome conclusions.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


OFFSETS = (0.05, 0.10, 0.20, 0.30, 0.50, 0.80)
VECTOR_FIELDS = (
    "robot_root_pos_env",
    "robot_root_lin_vel_w",
    "robot_root_ang_vel_w",
    "racket_pos_env",
    "racket_velocity",
)
ARRAY_FIELDS = ("robot_joint_pos", "robot_joint_vel")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--gate-result", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--transplant", default=None)
    parser.add_argument("--transplant-result", default=None)
    return parser.parse_args()


def is_legal(row: dict) -> bool:
    return row.get("failure_code") == "LEGAL"


def paired_outcome(rows: list[tuple[dict, dict]]) -> dict:
    n = len(rows)
    base_legal = sum(is_legal(base) for base, _ in rows)
    gate_legal = sum(is_legal(gate) for _, gate in rows)
    return {
        "n": n,
        "baseline_legal_rate": base_legal / n if n else 0.0,
        "gate_legal_rate": gate_legal / n if n else 0.0,
        "delta_legal_pp": 100.0 * (gate_legal - base_legal) / n if n else 0.0,
        "baseline_failure_code": dict(Counter(base["failure_code"] for base, _ in rows)),
        "gate_failure_code": dict(Counter(gate["failure_code"] for _, gate in rows)),
    }


def component_delta(rows: list[tuple[dict, dict]], field: str) -> dict:
    deltas = []
    for base, gate in rows:
        a = base.get(field)
        b = gate.get(field)
        if a is None or b is None or len(a) != len(b):
            continue
        deltas.append([float(y) - float(x) for x, y in zip(a, b)])
    if not deltas:
        return {"n": 0}
    width = len(deltas[0])
    signed = [statistics.mean(row[i] for row in deltas) for i in range(width)]
    mean_abs = [statistics.mean(abs(row[i]) for row in deltas) for i in range(width)]
    max_abs = [max(abs(row[i]) for row in deltas) for i in range(width)]
    max_norm = max(math.sqrt(sum(value * value for value in row)) for row in deltas)
    return {
        "n": len(deltas),
        "mean_signed_delta_gate_minus_baseline": signed,
        "mean_abs_delta": mean_abs,
        "max_abs_delta": max_abs,
        "max_l2_delta": max_norm,
    }


def first_mismatch_map(gate_result: dict) -> dict[int, int]:
    first = {}
    for mismatch in gate_result.get("paired_recipe_mismatches", []):
        if "env_id" in mismatch and "event_index" in mismatch:
            first.setdefault(int(mismatch["env_id"]), int(mismatch["event_index"]))
    return first


def strict_pair(
    baseline_rows: list[dict], gate_rows: list[dict], first_mismatch: dict[int, int]
) -> tuple[dict[tuple[int, int], tuple[dict, dict]], list[tuple[dict, dict]]]:
    base = {(row["env_id"], row["paired_recipe_index"]): row for row in baseline_rows}
    gate = {(row["env_id"], row["paired_recipe_index"]): row for row in gate_rows}
    pairs = {}
    for key in sorted(set(base) & set(gate)):
        env, index = key
        if env in first_mismatch and index < first_mismatch[env]:
            pairs[key] = (base[key], gate[key])
    current = [
        pair
        for pair in pairs.values()
        if pair[1].get("fh_correction_applied", False)
        and not pair[0].get("venue_tuple_selected", False)
        and pair[0].get("clip_id") == 0
    ]
    return pairs, current


def strict_post_pairs(
    baseline: dict, gate: dict, first_mismatch: dict[int, int]
) -> dict[float, list[tuple[dict, dict]]]:
    base_rows = baseline.get("post_strike_state_rows", [])
    gate_rows = gate.get("post_strike_state_rows", [])
    base = {
        (row["env_id"], row["source_paired_recipe_index"], round(row["offset_s"], 6)): row
        for row in base_rows
        if "offset_s" in row and "source_paired_recipe_index" in row
    }
    gate_map = {
        (row["env_id"], row["source_paired_recipe_index"], round(row["offset_s"], 6)): row
        for row in gate_rows
        if "offset_s" in row and "source_paired_recipe_index" in row
    }
    result = defaultdict(list)
    for key in sorted(set(base) & set(gate_map)):
        env, source_index, offset = key
        if env not in first_mismatch or source_index >= first_mismatch[env]:
            continue
        base_row, gate_row = base[key], gate_map[key]
        if not gate_row.get("source_fh_correction_applied", False):
            continue
        result[offset].append((base_row, gate_row))
    return dict(result)


def reset_summary(baseline: dict, gate: dict, first_mismatch: dict[int, int]) -> dict:
    def by_env(data: dict) -> dict[int, list[dict]]:
        result = defaultdict(list)
        for event in data.get("reset_events", []):
            result[int(event["env_id"])].append(event)
        return result

    base = by_env(baseline)
    gated = by_env(gate)
    first_difference = []
    reason_pairs = Counter()
    for env, limit in sorted(first_mismatch.items()):
        b = [event for event in base[env] if event["paired_recipe_index"] <= limit]
        g = [event for event in gated[env] if event["paired_recipe_index"] <= limit]
        b_sig = [(event["paired_recipe_index"], tuple(event["termination_reasons"])) for event in b]
        g_sig = [(event["paired_recipe_index"], tuple(event["termination_reasons"])) for event in g]
        if b_sig != g_sig:
            first_difference.append({"env_id": env, "first_mismatch_event": limit, "baseline": b_sig, "gate": g_sig})
            reason_pairs[(tuple(b_sig), tuple(g_sig))] += 1
    return {
        "known_first_mismatch_envs": len(first_mismatch),
        "envs_with_reset_sequence_difference_before_or_at_first_mismatch": len(first_difference),
        "examples": first_difference[:20],
        "reset_sequence_difference_example_count": sum(reason_pairs.values()),
    }


def transplant_outcomes(
    baseline_rows: list[dict], transplant_rows: list[dict], first_mismatch: dict[int, int]
) -> dict:
    base = {(row["env_id"], row["paired_recipe_index"]): row for row in baseline_rows}
    tr = {(row["env_id"], row["paired_recipe_index"]): row for row in transplant_rows}
    pairs = []
    for key in sorted(set(base) & set(tr)):
        env, index = key
        if env in first_mismatch and index < first_mismatch[env]:
            pairs.append((base[key], tr[key]))
    current = [pair for pair in pairs if pair[1].get("fh_correction_applied", False)]
    next_rows = []
    for base_row, _ in current:
        key = (base_row["env_id"], base_row["paired_recipe_index"] + 1)
        if key in base and key in tr and key[1] < first_mismatch.get(key[0], -1):
            next_rows.append((base[key], tr[key]))
    return {"current_fh": paired_outcome(current), "next_shot": paired_outcome(next_rows)}


def main() -> int:
    args = parse_args()
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    gate = json.loads(Path(args.gate).read_text(encoding="utf-8"))
    gate_result = json.loads(Path(args.gate_result).read_text(encoding="utf-8"))
    first_mismatch = first_mismatch_map(gate_result)
    _, current = strict_pair(baseline["rows"], gate["rows"], first_mismatch)
    post_pairs = strict_post_pairs(baseline, gate, first_mismatch)

    post_report = {}
    for offset in sorted(post_pairs):
        rows = post_pairs[offset]
        post_report[str(offset)] = {
            "n": len(rows),
            "root_pos": component_delta(rows, "robot_root_pos_env"),
            "root_lin_vel": component_delta(rows, "robot_root_lin_vel_w"),
            "root_ang_vel": component_delta(rows, "robot_root_ang_vel_w"),
            "joint_pos": component_delta(rows, "robot_joint_pos"),
            "joint_vel": component_delta(rows, "robot_joint_vel"),
            "racket_pos": component_delta(rows, "racket_pos_env"),
            "racket_velocity": component_delta(rows, "racket_velocity"),
        }

    report = {
        "schema_version": 1,
        "baseline_rows": len(baseline["rows"]),
        "gate_rows": len(gate["rows"]),
        "strict_current_fh": {
            "outcome": paired_outcome(current),
            "n": len(current),
        },
        "strict_post_strike_state_delta": post_report,
        "reset_attribution": reset_summary(baseline, gate, first_mismatch),
        "causal_scope": {
            "strict_prefix_only": True,
            "first_mismatch_envs_from_gate_result": len(first_mismatch),
            "post_divergence_rows_excluded": True,
            "reset_sequence_difference_is_attribution_evidence_not_proof": True,
        },
    }
    if args.transplant:
        transplant = json.loads(Path(args.transplant).read_text(encoding="utf-8"))
        transplant_first_mismatch = first_mismatch
        if args.transplant_result:
            transplant_result = json.loads(Path(args.transplant_result).read_text(encoding="utf-8"))
            transplant_first_mismatch = first_mismatch_map(transplant_result)
        report["transplant"] = transplant_outcomes(
            baseline["rows"], transplant["rows"], transplant_first_mismatch
        )
        report["transplant_metadata"] = transplant.get("state_transplant", {})
        report["transplant_causal_scope"] = {
            "first_mismatch_envs": len(transplant_first_mismatch),
            "strict_prefix_only": True,
            "post_transplant_recipe_divergence_excluded": True,
        }

    output = Path(args.json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
