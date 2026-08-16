"""Aggregate true in-process snapshot branches at the fixed next-shot event."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


BRANCHES = {0: "G0", 1: "G-ang", 2: "G-lin", 3: "G-upper", 4: "G-allvel"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True, help="SOURCE_ID=TELEMETRY_PATH")
    parser.add_argument("--json-out", required=True)
    return parser.parse_args()


def legal(row: dict) -> bool:
    return row.get("failure_code") == "LEGAL"


def same_task(rows: dict[int, dict]) -> bool:
    if len(rows) != len(BRANCHES):
        return False
    reference = rows[0]
    fields = ("clip_id", "incoming_velocity", "incoming_spin", "planner_racket_velocity", "planner_racket_normal")
    return all(
        all(rows[env_id].get(field) == reference.get(field) for field in fields)
        for env_id in BRANCHES
    )


TASK_FIELDS = (
    "clip_id",
    "incoming_velocity",
    "incoming_spin",
    "planner_racket_velocity",
    "planner_racket_normal",
)


def normalize_reset_before_next_shot(
    rows: dict[int, dict], reset_events: list[dict]
) -> tuple[dict[int, dict], dict[int, list[dict]]]:
    """Turn a pre-event1 termination into the outcome of the fixed next-shot attempt.

    After a reset, paired replay may emit event 1 again with a new live topology.  That
    later row is not the causal next-shot outcome.  The termination itself is the failure,
    while the task fields must remain those of the canonical G0 event.
    """
    canonical = rows[0]
    reset_by_env: dict[int, list[dict]] = {}
    for event in reset_events:
        if int(event.get("paired_recipe_index", -1)) != 1:
            continue
        env_id = int(event.get("env_id", -1))
        reset_by_env.setdefault(env_id, []).append(event)
    normalized = {}
    for env_id, row in rows.items():
        candidates = reset_by_env.get(env_id, [])
        row_step = int(row.get("global_step", 10**18))
        before_row = [event for event in candidates if int(event.get("global_step", 10**18)) <= row_step]
        if not before_row:
            normalized[env_id] = row
            continue
        updated = dict(row)
        for field in TASK_FIELDS:
            updated[field] = canonical.get(field)
        updated["failure_code"] = "RESET_BEFORE_NEXT_SHOT"
        updated["capture_gate"] = False
        updated["net_crossed"] = False
        updated["net_clear"] = False
        updated["landing_valid"] = False
        updated["on_opponent"] = False
        updated["reset_before_next_shot"] = True
        updated["reset_reasons"] = sorted(
            {
                reason
                for event in before_row
                for reason in event.get("termination_reasons", [])
            }
        )
        normalized[env_id] = updated
    return normalized, reset_by_env


def main() -> int:
    args = parse_args()
    usable = []
    excluded = []
    warnings = []
    for item in args.source:
        source_id_text, path_text = item.split("=", 1)
        source_id = int(source_id_text)
        path = Path(path_text)
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = {
            int(row["env_id"]): row
            for row in payload.get("rows", [])
            if int(row.get("paired_recipe_index", -1)) == 1
        }
        if len(rows) != len(BRANCHES):
            excluded.append({"source_id": source_id, "reason": "missing_event1_branch", "env_ids": sorted(rows)})
            continue
        rows, reset_by_env = normalize_reset_before_next_shot(
            rows, payload.get("reset_events", [])
        )
        if not same_task(rows):
            excluded.append({"source_id": source_id, "reason": "event1_task_mismatch"})
            continue
        result_path = path.with_suffix("").with_suffix(".json")
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
        mismatches = result.get("paired_recipe_mismatches", [])
        reset_env_ids = set(reset_by_env)
        event1_mismatches = [
            m
            for m in mismatches
            if int(m.get("event_index", -1)) == 1
            and int(m.get("env_id", -1)) not in reset_env_ids
        ]
        if event1_mismatches:
            excluded.append(
                {
                    "source_id": source_id,
                    "reason": "event1_topology_mismatch",
                    "mismatches": event1_mismatches,
                }
            )
            continue
        pre_snapshot_mismatches = [m for m in mismatches if int(m.get("event_index", -1)) == 0]
        if pre_snapshot_mismatches:
            warnings.append(
                {
                    "source_id": source_id,
                    "reason": "pre_snapshot_topology_warning",
                    "mismatches": pre_snapshot_mismatches,
                }
            )
        usable.append({"source_id": source_id, "rows": rows, "mismatches": mismatches})

    report = {
        "schema_version": 1,
            "causal_scope": {
            "snapshot": "same Isaac process, env0 gate state cloned to G-ang/G-lin/G-upper/G-allvel",
            "event": 1,
            "required_fields_equal": ["clip_id", "incoming_velocity", "incoming_spin", "planner_racket_velocity", "planner_racket_normal"],
            "pre_event1_reset_policy": "RESET_BEFORE_NEXT_SHOT uses canonical G0 task fields; post-reset rows are not used",
            "later_events_excluded": True,
        },
        "usable_source_ids": [item["source_id"] for item in usable],
        "excluded": excluded,
        "warnings": warnings,
        "n_sources": len(usable),
        "branches": {},
    }
    for env_id, name in BRANCHES.items():
        g0_rows = [item["rows"][0] for item in usable]
        branch_rows = [item["rows"][env_id] for item in usable]
        transitions = Counter((a.get("failure_code"), b.get("failure_code")) for a, b in zip(g0_rows, branch_rows))
        g0_legal = sum(legal(row) for row in g0_rows)
        branch_legal = sum(legal(row) for row in branch_rows)
        report["branches"][name] = {
            "n": len(branch_rows),
            "g0_legal": g0_legal,
            "branch_legal": branch_legal,
            "g0_legal_rate": g0_legal / len(g0_rows) if g0_rows else 0.0,
            "branch_legal_rate": branch_legal / len(branch_rows) if branch_rows else 0.0,
            "delta_legal_pp": 100.0 * (branch_legal - g0_legal) / len(g0_rows) if g0_rows else 0.0,
            "fail_to_legal": sum(not legal(a) and legal(b) for a, b in zip(g0_rows, branch_rows)),
            "legal_to_fail": sum(legal(a) and not legal(b) for a, b in zip(g0_rows, branch_rows)),
            "transition_matrix": {f"{a}->{b}": count for (a, b), count in sorted(transitions.items())},
            "g0_failure_codes": dict(Counter(row.get("failure_code") for row in g0_rows)),
            "branch_failure_codes": dict(Counter(row.get("failure_code") for row in branch_rows)),
        }

    output = Path(args.json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
