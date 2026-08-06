#!/usr/bin/env python3
"""Split P5D references by safety transparency, not actual tracking error.

Actual reference->actual error is retained as a dense tracker-training
diagnostic.  It never rejects a geometrically valid reference.  A large
safety projection does reject it as a *clean* tracker reference because the
runtime command is then a different trajectory.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "eval_outputs/strike_goal_p5/p5d2_complete_runtime_reference_bank/p5d2_complete_all_reference_bank_manifest.json"
SUMMARY = ROOT / "eval_outputs/strike_goal_p5/p5d2_complete_runtime_reference_audit/p5d2_physx_reference_only_replay_summary.json"
OUT = ROOT / "eval_outputs/strike_goal_p5/p5d2_complete_runtime_reference_audit"
SAFETY_THRESHOLD_RAD = 0.01
TARGET_REFERENCE_THRESHOLD_M = 0.005


def main() -> None:
    bank = json.loads(BANK.read_text())
    summary = json.loads(SUMMARY.read_text())
    replay = {r["episode_id"]: r for r in summary["rows"]}
    transparent, projected = [], []
    audited = copy.deepcopy(bank)
    for entry in audited["motions"]:
        r = replay.get(entry["episode_id"], {})
        safety = float(r.get("safety_projection_max_rad", np.inf))
        geom = float(r.get("target_reference_error_m", np.inf))
        # This split deliberately ignores reference_actual_error_m.
        transparent_pass = bool(geom <= TARGET_REFERENCE_THRESHOLD_M and safety <= SAFETY_THRESHOLD_RAD)
        entry["runtime_gate"] = {
            "target_reference_error_m": geom,
            "reference_actual_error_m_diagnostic_only": float(r.get("reference_actual_error_m", np.nan)),
            "safety_projection_max_rad": safety,
            "target_reference_geometry_pass": geom <= TARGET_REFERENCE_THRESHOLD_M,
            "safety_transparent_pass": safety <= SAFETY_THRESHOLD_RAD,
            "actual_tracking_error_is_rejection_criterion": False,
        }
        entry["qualification_status"] = "OFFLINE_REJECTED"
        entry["qualification_reasons"] = ["recovery_and_termination_not_audited", "neighbor_continuity_not_audited"]
        if not transparent_pass:
            entry["qualification_reasons"].append("runtime_safety_projection_requires_reoptimization")
            projected.append(entry)
        else:
            entry["qualification_reasons"].append("runtime_recovery_and_continuity_still_pending")
            transparent.append(entry)

    def write(name: str, rows: list[dict], role: str) -> None:
        part = copy.deepcopy(audited)
        part["manifest_name"] = name
        part["runtime_gate_role"] = role
        part["motions"] = rows
        part["motion_count"] = len(rows)
        part["training_started"] = False
        (OUT / name).write_text(json.dumps(part, ensure_ascii=False, indent=2) + "\n")

    write("p5d2_safety_transparent_pending_recovery_manifest.json", transparent, "clean_tracker_reference_candidates")
    write("p5d2_safety_projected_reoptimization_manifest.json", projected, "runtime_safety_aware_reoptimization_required")
    report = {
        "schema_version": "p5d2_safety_reference_split/v1",
        "source_bank": str(BANK.resolve()),
        "actual_tracking_error_used_as_rejection": False,
        "safety_projection_threshold_rad": SAFETY_THRESHOLD_RAD,
        "target_reference_threshold_m": TARGET_REFERENCE_THRESHOLD_M,
        "reference_count": len(audited["motions"]),
        "safety_transparent_count": len(transparent),
        "safety_projected_count": len(projected),
        "transparent_episode_ids": [e["episode_id"] for e in transparent],
        "projected_episode_ids": [e["episode_id"] for e in projected],
        "training_started": False,
        "note": "reference_to_actual error is a tracker loss diagnostic, not a reference rejection gate",
    }
    (OUT / "p5d2_safety_reference_split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
