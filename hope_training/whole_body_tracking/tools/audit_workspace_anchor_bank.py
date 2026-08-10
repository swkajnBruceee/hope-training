#!/usr/bin/env python3
"""Audit the metadata-only anchor bank before WorkspaceExpansion preflight."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from training.utils.workspace_anchor_bank import WorkspaceStrikeAnchorBank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    bank = WorkspaceStrikeAnchorBank(
        manifest_path,
        "cpu",
        nominal_local=(0.42, -0.18, 0.18),
        support_half_range=(0.08, 0.08, 0.08),
    )
    nominal = (0.42, -0.18, 0.18)
    stages = []
    for progress in (0.0, 0.10, 0.25, 0.40, 0.60, 1.0):
        _, _, _, _, _, eligible = bank.sample(2048, progress, nominal)
        stages.append({"progress": progress, "eligible_anchor_count": eligible, "eligible_fraction": eligible / bank.anchor_count})
    rows = payload.get("motions", [])
    stroke_counts = Counter(str(row.get("stroke_type", "unknown")) for row in rows)
    # Coarse 5 cm cells are sufficient to expose severe density skew without
    # turning this admission audit into another feasibility model.
    cells = Counter()
    for position in bank.position_local.tolist():
        cells[tuple(int(round(float(value) / 0.05)) for value in position)] += 1
    result = {
        "status": "workspace_anchor_metadata_audit_complete",
        "source": "canonical_goal_10d_only",
        "manifest_status": payload.get("status"),
        "manifest_physics_qualified": bool(payload.get("physics_qualified", False)),
        "manifest_teacher_approved": bool(payload.get("teacher_approved", False)),
        "manifest_training_admission": bool(payload.get("training_admission", False)),
        "qualification_note": (
            "metadata-only audit; physical/runtime qualification remains a separate gate"
        ),
        "metadata_audit_loaded_motion_execution": False,
        "metadata_audit_loaded_teacher_checkpoint": False,
        "anchor_count": bank.anchor_count,
        "source_anchor_count": bank.source_anchor_count,
        "qualified_anchor_count": bank.qualified_anchor_count,
        "rejected_anchor_count": bank.rejected_anchor_count,
        "qualified_fraction": bank.qualified_anchor_count / bank.source_anchor_count,
        "qualification_reason_counts": bank.qualification_reason_counts,
        "stroke_counts": dict(stroke_counts),
        "density_5cm_cells": {"occupied": len(cells), "max_cell_count": max(cells.values()), "top_cell_fraction": max(cells.values()) / bank.anchor_count},
        "statistics": bank.statistics(nominal),
        "coverage_stages": stages,
        "global_fraction": 0.0,
        "metadata_audit_nominal_fallback_count": 0,
    }
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
