#!/usr/bin/env python3
"""Build a row-filtered, fail-closed WorkspaceExpansion manifest."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evidence")
    args = parser.parse_args()
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = payload.get("motions") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise SystemExit("source manifest has no motions")
    evidence = {}
    if args.evidence:
        evidence = json.loads(Path(args.evidence).expanduser().read_text(encoding="utf-8"))
    qualified, reasons = [], Counter()
    for row in rows:
        if not isinstance(row, dict):
            reasons["row_not_mapping"] += 1
            continue
        ev = evidence.get(str(row.get("motion_id", "")), {}) if evidence else row
        row_reasons = []
        if ev.get("physics_qualified") is not True:
            row_reasons.append("physics_pending")
        if ev.get("training_admission") is not True:
            row_reasons.append("training_pending")
        if "canonical_goal_10d" not in row:
            row_reasons.append("missing_canonical_goal")
        if row_reasons:
            reasons.update(row_reasons)
            continue
        selected = dict(row)
        selected["physics_qualified"] = True
        selected["training_admission"] = True
        qualified.append(selected)
    if not qualified:
        raise SystemExit("no qualified anchors found; refusing to write an admitted manifest: " + str(dict(reasons)))
    result = {
        "schema_version": "v13b_workspace_anchor_manifest/v1",
        "status": "workspace_anchor_qualified_v1",
        "source_manifest": str(source), "source_anchor_count": len(rows),
        "qualified_anchor_count": len(qualified), "rejected_anchor_count": len(rows) - len(qualified),
        "rejection_reason_counts": dict(reasons), "physics_qualified": True, "training_admission": True,
        "qualification_method": "row_level_physics_and_training_admission_evidence",
        "qualification_timestamp": datetime.now(timezone.utc).isoformat(), "motions": qualified,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "motions"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
