"""Summarize fixed-recipe snapshot whole-action replay audits.

This is deliberately narrower than the off-manifold velocity-transplant analyzer: the
intervention is replaying the source baseline policy action for one post-snapshot control step,
while all branches share the same gate snapshot and next-shot recipe.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _event_rows(payload, recipe_index: int):
    rows = [
        row
        for row in payload.get("rows", [])
        if row.get("snapshot_branch") is not None
        and int(row.get("paired_recipe_index", -1)) == int(recipe_index)
    ]
    return {str(row["snapshot_branch"]): row.get("failure_code") for row in rows}


def analyze(paths: list[Path], recipe_index: int = 1) -> dict:
    sources = []
    aggregate = Counter()
    intervention_modes = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        events = payload.get("snapshot_branch", {}).get("applied_events", [])
        if not events:
            raise RuntimeError(f"{path} has no applied snapshot event")
        event = events[0]
        intervention_modes.add(str(event.get("intervention_mode", "unknown")))
        outcomes = _event_rows(payload, recipe_index)
        g0 = outcomes.get("G0")
        if g0 is None:
            raise RuntimeError(f"{path} has no G0 row for recipe index {recipe_index}")
        rescue = []
        damage = []
        classification_changes = []
        for branch, outcome in outcomes.items():
            if branch == "G0":
                continue
            if outcome != g0:
                classification_changes.append(
                    {"branch": branch, "g0": g0, "branch_outcome": outcome}
                )
            if outcome != g0:
                pair = {"branch": branch, "g0": g0, "branch_outcome": outcome}
                if g0 == "LEGAL":
                    damage.append(pair)
                elif outcome == "LEGAL":
                    rescue.append(pair)
        obs_max = max(float(value) for value in event["policy_obs_max_abs_diff_vs_G0"])
        mismatch_count = payload.get("paired_recipe_mismatch_count", 0)
        source = int(payload["snapshot_branch"]["source_env_id"])
        sources.append(
            {
                "source_env_id": source,
                "path": str(path),
                "policy_obs_max_abs_diff_vs_G0": obs_max,
                "paired_recipe_mismatch_count": mismatch_count,
                "event_index": recipe_index,
                "outcomes": outcomes,
                "rescue": rescue,
                "damage": damage,
                "classification_changes": classification_changes,
                "reset_reasons": [
                    {
                        "env_id": row.get("env_id"),
                        "termination_reasons": row.get("termination_reasons", []),
                    }
                    for row in payload.get("reset_events", [])
                ],
            }
        )
        aggregate["sources"] += 1
        aggregate["rescue_pairs"] += len(rescue)
        aggregate["damage_pairs"] += len(damage)
        aggregate["failure_classification_changes"] += len(classification_changes)
        aggregate["all_branches_same_as_G0"] += int(not rescue and not damage)
    return {
        "schema_version": 1,
        "intervention": sorted(intervention_modes),
        "causal_scope": "fixed_recipe_event_1_only",
        "source_count": len(sources),
        "aggregate": dict(aggregate),
        "all_sources_observation_sync_max_abs": max(
            source["policy_obs_max_abs_diff_vs_G0"] for source in sources
        ),
        "sources": sources,
        "interpretation": {
            "validity": (
                "observation-synchronized fixed-recipe branch audit; no velocity-transplant claim"
            ),
            "result": (
                "no rescue or damage relative to G0 on the audited event-1 recipes"
                if aggregate["rescue_pairs"] == 0 and aggregate["damage_pairs"] == 0
                else "at least one branch outcome differed from G0"
            ),
            "not_proven": [
                "root angular velocity is causal",
                "upper-body joint velocity is causal",
                "recovery PPO objective is identified",
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--recipe-index", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.inputs, args.recipe_index)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
