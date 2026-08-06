#!/usr/bin/env python3
"""Audit the no-reset prefix-to-policy handoff recorded by the capture tool.

The capture rows describe the *post-step* motion clock, while the action was
processed just before that step.  Therefore a configured handoff at phase H is
expected to first appear on the recorded row with ``motion_step > H``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("--expected-leg-action", type=float, required=True)
    parser.add_argument("--tolerance", type=float, default=1.0e-6)
    args = parser.parse_args()

    records: list[dict[str, object]] = []
    for path in sorted(args.capture_dir.glob("*.npz")):
        with np.load(path) as data:
            required = {"motion_step", "policy_handoff_step", "raw_action"}
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"{path}: missing {sorted(missing)}")
            steps = data["motion_step"].astype(np.int64)
            handoff = int(data["policy_handoff_step"][0])
            raw = data["raw_action"]
            if raw.ndim != 2 or raw.shape[1] != 14:
                raise ValueError(f"{path}: expected [frames, 14] raw_action, got {raw.shape}")

        prefix = raw[steps <= handoff]
        suffix = raw[steps > handoff]
        prefix_max = float(np.abs(prefix).max()) if len(prefix) else 0.0
        suffix_leg_error = (
            float(np.abs(suffix[:, :12] - args.expected_leg_action).max()) if len(suffix) else 0.0
        )
        waist_max = float(np.abs(raw[:, 12:]).max())
        active_rows = np.flatnonzero(np.abs(raw[:, :12]).max(axis=1) > args.tolerance)
        first_active_step = int(steps[active_rows[0]]) if len(active_rows) else None
        expected_first = int(steps[steps > handoff][0]) if np.any(steps > handoff) else None
        passed = (
            prefix_max <= args.tolerance
            and suffix_leg_error <= args.tolerance
            and waist_max <= args.tolerance
            and first_active_step == expected_first
        )
        records.append(
            {
                "episode_id": path.stem,
                "handoff_step": handoff,
                "prefix_action_max": prefix_max,
                "suffix_leg_action_error": suffix_leg_error,
                "waist_action_max": waist_max,
                "first_active_motion_step": first_active_step,
                "expected_first_active_motion_step": expected_first,
                "passed": passed,
            }
        )

    report = {
        "schema_version": 1,
        "capture_dir": str(args.capture_dir),
        "expected_leg_action": args.expected_leg_action,
        "tolerance": args.tolerance,
        "post_step_action_alignment": "action at phase H is first observed with motion_step > H",
        "cases": records,
        "passed": bool(records) and all(record["passed"] for record in records),
    }
    output = args.capture_dir / "handoff_audit.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(records), "passed": report["passed"], "output": str(output)}, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
