#!/usr/bin/env python3
"""Summarize the frozen-prior, zero-P5D-residual PhysX replay.

The replay is a qualification audit, not training.  It records safety
projection and reference/actual decomposition for every bank entry and keeps
admission fail-closed until recovery and continuity evidence are complete.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "eval_outputs/strike_goal_p5/p5d2_complete_runtime_reference_bank/p5d2_complete_all_reference_bank_manifest.json"
LOG = ROOT / "eval_outputs/strike_goal_p5/p5d2_complete_runtime_reference_audit/p5d2_physx_reference_only_replay_v2.log"
OUT = ROOT / "eval_outputs/strike_goal_p5/p5d2_complete_runtime_reference_audit"


def parse_xyz(value: str) -> np.ndarray:
    return np.asarray([float(x) for x in value.split("/")], dtype=float)


def main() -> None:
    bank = json.loads(BANK.read_text())
    lines = LOG.read_text(errors="replace").splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("rank,episode_id,target_xyz"))
    rows = {}
    for line in lines[header + 1:]:
        if not re.match(r"^\d+,", line):
            break
        p = line.split(",")
        if len(p) < 13:
            continue
        rows[p[1]] = {
            "rank": int(p[0]),
            "target_xyz_w": parse_xyz(p[2]).tolist(),
            "reference_xyz_w": parse_xyz(p[3]).tolist(),
            "actual_xyz_w": parse_xyz(p[4]).tolist(),
            "target_reference_error_m": float(p[5]),
            "reference_actual_error_m": float(p[6]),
            "raw_action_max": float(p[7]),
            "raw_action_mean": float(p[8]),
            "frozen_prior_contribution_max_rad": float(p[9]),
            "frozen_prior_contribution_mean_rad": float(p[10]),
            "residual_clip_fraction": float(p[11]),
            "safety_projection_max_rad": float(p[12]),
            "captured_at_hit": True,
        }
    by_id = {e["episode_id"]: e for e in bank["motions"]}
    replay_rows = []
    for episode_id, entry in by_id.items():
        replay = rows.get(episode_id, {"captured_at_hit": False})
        # The replay script currently has no recovery/termination certificate;
        # do not promote on hit-only evidence.
        replay["recovery_and_termination_audited"] = False
        replay["reference_only_safety_pass_1e-2rad"] = bool(replay.get("safety_projection_max_rad", np.inf) <= 0.01)
        replay["reference_geometry_pass_5mm"] = bool(replay.get("target_reference_error_m", np.inf) <= 0.005)
        replay_rows.append({"episode_id": episode_id, "category": entry.get("p5d2_bank", {}).get("category"), "source_seed_motion_id": entry.get("p5d2_bank", {}).get("source_seed_motion_id"), **replay})

    captured = [r for r in replay_rows if r.get("captured_at_hit")]
    safety_pass = [r for r in captured if r.get("reference_only_safety_pass_1e-2rad")]
    geom_pass = [r for r in captured if r.get("reference_geometry_pass_5mm")]
    report = {
        "schema_version": "p5d2_physx_reference_only_replay/v1",
        "bank": str(BANK.resolve()),
        "log": str(LOG.resolve()),
        "checkpoint_role": "existing_p5d2_model0_with_zero_public_residual",
        "frozen_prior": ["model_900", "model_3396", "support_state_machine"],
        "p5d_residual_enabled": False,
        "reference_count": len(replay_rows),
        "captured_at_hit_count": len(captured),
        "geometry_pass_count": len(geom_pass),
        "safety_pass_count_threshold_0.01rad": len(safety_pass),
        "safety_projection_max_rad": max((r.get("safety_projection_max_rad", 0.0) for r in captured), default=float("nan")),
        "safety_projection_mean_rad": float(np.mean([r["safety_projection_max_rad"] for r in captured])) if captured else float("nan"),
        "target_reference_error_mean_m": float(np.mean([r["target_reference_error_m"] for r in captured])) if captured else float("nan"),
        "reference_actual_error_mean_m": float(np.mean([r["reference_actual_error_m"] for r in captured])) if captured else float("nan"),
        "frozen_prior_contribution_max_rad": max((r.get("frozen_prior_contribution_max_rad", 0.0) for r in captured), default=float("nan")),
        "continuity_audit_completed": False,
        "recovery_and_termination_audit_completed": False,
        "training_started": False,
        "rows": replay_rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "p5d2_physx_reference_only_replay_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    audited = copy.deepcopy(bank)
    audited["runtime_replay_summary"] = {k: v for k, v in report.items() if k not in ("rows",)}
    for entry in audited["motions"]:
        replay = rows.get(entry["episode_id"], {"captured_at_hit": False})
        entry["reference_replay"] = replay
        entry["qualification_status"] = "OFFLINE_REJECTED"
        entry["qualification_reasons"] = ["recovery_and_termination_not_audited", "neighbor_continuity_not_audited"]
        if replay.get("safety_projection_max_rad", np.inf) > 0.01:
            entry["qualification_reasons"].append("safety_projection_exceeds_0.01rad")
    (OUT / "p5d2_complete_all_reference_bank_physx_audited.json").write_text(json.dumps(audited, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("reference_count", "captured_at_hit_count", "geometry_pass_count", "safety_pass_count_threshold_0.01rad", "safety_projection_max_rad", "safety_projection_mean_rad", "target_reference_error_mean_m", "reference_actual_error_mean_m", "training_started")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
