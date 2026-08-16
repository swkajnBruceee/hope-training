"""Compare fixed single-shot recovery branches against an untouched gate branch."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g0", required=True, help="Unmodified gate branch telemetry JSON.")
    parser.add_argument(
        "--branch",
        action="append",
        default=[],
        help="Named branch in NAME=PATH form; may be repeated.",
    )
    parser.add_argument("--json-out", required=True)
    return parser.parse_args()


def load_rows(path: str) -> tuple[dict, dict[tuple[int, int], dict]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = {(int(row["env_id"]), int(row["paired_recipe_index"])): row for row in payload["rows"]}
    return payload, rows


def legal(row: dict) -> bool:
    return row.get("failure_code") == "LEGAL"


def outcome(rows: list[dict]) -> dict:
    counts = Counter(row.get("failure_code") for row in rows)
    return {
        "n": len(rows),
        "legal": sum(legal(row) for row in rows),
        "legal_rate": sum(legal(row) for row in rows) / len(rows) if rows else 0.0,
        "failure_code": dict(counts),
    }


def branch_report(g0: dict[tuple[int, int], dict], branch: dict[tuple[int, int], dict]) -> dict:
    current_keys = sorted(
        key for key, row in g0.items() if key[1] == 0 and row.get("fh_correction_applied", False) and key in branch
    )
    next_keys = [
        (env_id, 1)
        for env_id, _ in current_keys
        if (env_id, 1) in g0 and (env_id, 1) in branch
    ]
    current_g0 = [g0[key] for key in current_keys]
    current_branch = [branch[key] for key in current_keys]
    next_g0 = [g0[key] for key in next_keys]
    next_branch = [branch[key] for key in next_keys]
    transitions = Counter((base["failure_code"], alt["failure_code"]) for base, alt in zip(next_g0, next_branch))
    same_next_recipe = sum(
        base.get("clip_id") == alt.get("clip_id")
        and base.get("paired_recipe_index") == alt.get("paired_recipe_index")
        for base, alt in zip(next_g0, next_branch)
    )
    return {
        "current_fh": {
            "g0": outcome(current_g0),
            "branch": outcome(current_branch),
        },
        "next_shot": {
            "g0": outcome(next_g0),
            "branch": outcome(next_branch),
            "delta_legal_pp": 100.0 * (sum(legal(alt) for alt in next_branch) - sum(legal(base) for base in next_g0)) / len(next_g0) if next_g0 else 0.0,
            "fail_to_legal": sum(not legal(base) and legal(alt) for base, alt in zip(next_g0, next_branch)),
            "legal_to_fail": sum(legal(base) and not legal(alt) for base, alt in zip(next_g0, next_branch)),
            "transition_matrix": {f"{a}->{b}": count for (a, b), count in sorted(transitions.items())},
            "same_next_recipe_count": same_next_recipe,
            "same_next_recipe_rate": same_next_recipe / len(next_g0) if next_g0 else 0.0,
        },
        "paired_env_ids": [env_id for env_id, _ in next_keys],
    }


def main() -> int:
    args = parse_args()
    g0_payload, g0 = load_rows(args.g0)
    branches = {}
    for item in args.branch:
        if "=" not in item:
            raise ValueError(f"branch must be NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        payload, rows = load_rows(path)
        branches[name] = {
            "report": branch_report(g0, rows),
            "telemetry": {
                "state_transplant_applied_events": len(payload.get("state_transplant", {}).get("applied_events", [])),
                "post_strike_rows": len(payload.get("post_strike_state_rows", [])),
            },
        }
    report = {
        "schema_version": 1,
        "g0_rows": len(g0),
        "g0_transplant_applied_events": len(g0_payload.get("state_transplant", {}).get("applied_events", [])),
        "branches": branches,
        "causal_scope": {
            "single_fixed_event_prefix": "event0 current FH -> event1 next shot",
            "same_next_recipe_required": True,
            "later_events_excluded": True,
        },
    }
    output = Path(args.json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
