"""Aggregate single-env fixed-recipe transplant branches across source shots."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True, help="SOURCE_ID=PREFIX")
    parser.add_argument("--json-out", required=True)
    return parser.parse_args()


def load(prefix: str, branch: str) -> dict[int, dict]:
    payload = json.loads(Path(f"{prefix}{branch}.telemetry.json").read_text(encoding="utf-8"))
    return {int(row["paired_recipe_index"]): row for row in payload["rows"]}


def legal(row: dict) -> bool:
    return row.get("failure_code") == "LEGAL"


def main() -> int:
    args = parse_args()
    sources = []
    for item in args.source:
        source_id, prefix = item.split("=", 1)
        source_id = int(source_id)
        rows = {branch: load(prefix, branch) for branch in ("g0", "gang", "glin", "gupper", "gallvel")}
        if not all(0 in branch_rows and 1 in branch_rows for branch_rows in rows.values()):
            continue
        if any(rows[branch][0]["failure_code"] != rows["g0"][0]["failure_code"] for branch in rows):
            continue
        if any(rows[branch][1].get("clip_id") != rows["g0"][1].get("clip_id") for branch in rows):
            continue
        sources.append((source_id, rows))

    report = {
        "schema_version": 1,
        "usable_sources": [source_id for source_id, _ in sources],
        "n_sources": len(sources),
        "branches": {},
        "causal_scope": {
            "current_event": 0,
            "next_event": 1,
            "current_failure_code_equal_to_g0_required": True,
            "next_clip_equal_to_g0_required": True,
            "later_events_excluded": True,
        },
    }
    for branch in ("gang", "glin", "gupper", "gallvel"):
        current_g0 = [rows["g0"][0] for _, rows in sources]
        current_branch = [rows[branch][0] for _, rows in sources]
        next_g0 = [rows["g0"][1] for _, rows in sources]
        next_branch = [rows[branch][1] for _, rows in sources]
        transitions = Counter((a["failure_code"], b["failure_code"]) for a, b in zip(next_g0, next_branch))
        g0_legal = sum(legal(row) for row in next_g0)
        branch_legal = sum(legal(row) for row in next_branch)
        report["branches"][branch] = {
            "current_fh_failure_codes": dict(Counter(row["failure_code"] for row in current_branch)),
            "next_g0_legal": g0_legal,
            "next_branch_legal": branch_legal,
            "next_delta_legal_pp": 100.0 * (branch_legal - g0_legal) / len(next_g0) if next_g0 else 0.0,
            "fail_to_legal": sum(not legal(a) and legal(b) for a, b in zip(next_g0, next_branch)),
            "legal_to_fail": sum(legal(a) and not legal(b) for a, b in zip(next_g0, next_branch)),
            "transition_matrix": {f"{a}->{b}": count for (a, b), count in sorted(transitions.items())},
        }
    output = Path(args.json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
