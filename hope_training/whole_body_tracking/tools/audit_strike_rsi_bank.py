#!/usr/bin/env python3
"""Audit exact RSI state coverage and data-level frame continuity.

This intentionally does not run PPO or claim simulator continuation.  It
creates deterministic cases for the later save/resume vs direct-load audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


KEY_PHASE_OFFSETS = (
    ("preparation", None),
    ("swing_start", -20),
    ("acceleration", -12),
    ("pre_contact", -5),
    ("contact", 0),
    ("deceleration", 5),
    ("follow_through", 20),
    ("ready_recovery", None),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--cases-out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.bank.joinpath("rsi_bank_manifest.json").read_text(encoding="utf-8"))
    required = set(manifest["required_arrays"])
    cases = []
    report = {"stage": "strike_rsi_exact_data_audit_v1", "training_eligible": False, "entries": [], "passed": True}

    for entry in manifest["entries"]:
        path = args.bank / entry["state_file"]
        with np.load(path, allow_pickle=False) as data:
            missing = sorted(required.difference(data.files))
            finite = all(np.isfinite(data[key]).all() for key in data.files if data[key].dtype.kind in "fc")
            n = int(data["joint_pos"].shape[0]) if "joint_pos" in data else 0
            hit = int(entry["hit_frame"])
            phase_ids = data["phase_id"] if "phase_id" in data else np.array([], dtype=np.int8)
            phase_coverage = sorted(int(x) for x in np.unique(phase_ids))
            frame_delta = np.linalg.norm(np.diff(data["joint_pos"], axis=0), axis=1) if n > 1 else np.array([])
            root_delta = np.linalg.norm(np.diff(data["root_pos_w"], axis=0), axis=1) if n > 1 else np.array([])
            entry_ok = not missing and finite and n > 1 and 0 <= hit < n and phase_coverage == list(range(7))
            report["entries"].append({
                "episode_id": entry["episode_id"],
                "num_frames": n,
                "hit_frame": hit,
                "phase_coverage": phase_coverage,
                "missing": missing,
                "finite": finite,
                "max_joint_frame_delta_norm": float(frame_delta.max()) if frame_delta.size else None,
                "max_root_frame_delta_norm": float(root_delta.max()) if root_delta.size else None,
                "passed": entry_ok,
            })
            report["passed"] = report["passed"] and entry_ok
            for phase_name, offset in KEY_PHASE_OFFSETS:
                if phase_name == "preparation":
                    frame = 0
                elif phase_name == "ready_recovery":
                    # The terminal frame has no continuation horizon and the
                    # MotionCommand wraps it to frame zero on the first step.
                    # Keep ten control frames for an actual recovery-tail
                    # continuation test.
                    frame = max(0, n - 10)
                else:
                    frame = min(max(hit + int(offset), 0), n - 1)
                cases.append({"episode_id": entry["episode_id"], "state_file": entry["state_file"], "phase": phase_name, "phase_id": int(phase_ids[frame]), "frame": frame, "hit_frame": hit})

    args.cases_out.parent.mkdir(parents=True, exist_ok=True)
    args.cases_out.write_text(json.dumps({"schema_version": 1, "stage": "strike_rsi_exact_continuation_cases_v1", "training_eligible": False, "cases": cases}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "entries": len(report["entries"]), "cases": len(cases), "missing_context": manifest["continuation_context"]["not_present_and_must_be_captured_before_training"]}, ensure_ascii=False))
    args.cases_out.with_name("rsi_data_audit_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
