#!/usr/bin/env python3
"""Create a local, fail-closed summary for one MuJoCo planner run.

The official ``pp_gate3_ball_evidence.py`` owns the physical verdict.  This
small adapter only joins that JSON with the current project's launcher,
planner, and runner logs so one run has a single machine-readable summary.
It never promotes a planner command or a preflight pass to a physical hit.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.MULTILINE))


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _has_fall_event(runner: str) -> bool:
    """Detect an actual fall/fall-guard event, not the static config line."""
    patterns = (
        r"fall[_ -]?guard\s*(?:triggered|tripped|active|event)",
        r"fall[_ -]?detected",
        r"robot\s+(?:fell|fallen)",
        r"\bFALL(?:ED|EN)?\b",
    )
    return any(re.search(pattern, runner, flags=re.IGNORECASE) for pattern in patterns)


def build_report(log_dir: Path, physical_path: Path) -> dict[str, Any]:
    launcher = _text(log_dir / "gate3_launcher.log")
    runner = _text(log_dir / "runner.log")
    planner = _text(log_dir / "planner.log")
    aimrt = _text(log_dir / "aimrt.log")
    serves = _text(log_dir / "serve_sequence.txt")
    physical = _load(physical_path)
    rows = list(physical.get("rows", []))
    expected_ids = [int(value) for value in physical.get("expected_shot_ids", [])]
    if not expected_ids:
        match = re.search(r"--shots\s+(\d+)", serves)
        expected_ids = list(range(1, int(match.group(1)) + 1)) if match else []
    mixed_random = "--randomize-mixed" in serves

    launched_ids = sorted({
        int(value)
        for value in re.findall(r"launch shot=(\d+)", launcher)
    })
    if not launched_ids:
        launched_ids = sorted({
            int(value)
            for value in re.findall(r"Gate3 ball launch shot_id=(\d+)", aimrt)
        })
    engage_ids = sorted({
        int(value)
        for value in re.findall(r"\[pp engage\].*?flight=(\d+)", runner)
    })
    preflight_ids = sorted({
        int(value)
        for value in re.findall(r"validated shot=(\d+)", launcher)
    })

    row_by_id = {
        int(row.get("shot_id", 0)): row
        for row in rows
        if int(row.get("shot_id", 0) or 0) > 0
    }
    contact_detected = sum(
        int(row.get("racket_contact_count", 0) or 0) > 0
        for row in row_by_id.values()
    )
    contact_pass = sum(bool(row.get("contact_pass", False)) for row in row_by_id.values())
    landing_pass = sum(bool(row.get("landing_pass", False)) for row in row_by_id.values())
    incoming_bounce_pass = sum(
        bool(row.get("incoming_bounce_pass", False)) for row in row_by_id.values()
    )
    telemetry_complete = sum(
        bool(row.get("telemetry_complete", False)) for row in row_by_id.values()
    )
    measured = bool(
        expected_ids
        and physical.get("physical_contact_measured", False)
        and physical.get("landing_measured", False)
        and set(row_by_id) == set(expected_ids)
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "log_dir": str(log_dir.resolve()),
        "physical_evidence": str(physical_path.resolve()),
        "protocol": {
            "closed_loop": "AimRT MuJoCo + HOPE planner + native runner + Gate3 physical ball",
            "plant_pd": "explicit",
            "plant_step": "1ms",
            "random_sequence": "--randomize" in serves or mixed_random,
            "side_policy": (
                "current-contract balanced side-neutral FH/BH random; planner selects side"
                if mixed_random
                else "current launcher is safe-backhand random; planner side is not independently randomized"
            ),
        },
        "coverage": {
            "expected_shots": len(expected_ids),
            "preflight_pass": len(preflight_ids),
            "launched": len(launched_ids),
            "planner_engaged": len(engage_ids),
            "telemetry_complete": telemetry_complete,
            "physical_rows": len(row_by_id),
        },
        "physical": {
            "measured": measured,
            "contact_detected": contact_detected,
            "contact_pass": contact_pass,
            "incoming_bounce_pass": incoming_bounce_pass,
            "legal_landing_pass": landing_pass,
            "contact_rate": _rate(contact_pass, len(expected_ids)),
            "legal_landing_rate": _rate(landing_pass, len(expected_ids)),
            "contact_rate_over_measured": _rate(contact_pass, len(row_by_id)),
            "legal_landing_rate_over_measured": _rate(landing_pass, len(row_by_id)),
            "official_physical_contact_pass": bool(physical.get("physical_contact_pass", False)),
            "official_landing_pass": bool(physical.get("landing_pass", False)),
        },
        "rows": rows,
        "health": {
            "aimrt_started": "MujocoSimModule" in aimrt and "Start succeeded" in aimrt,
            "planner_started": "HOPE planner started" in planner,
            "runner_motion": "mode=MOTION" in runner,
            "fall_guard_or_fall_marker": _has_fall_event(runner),
            "shot_id_join_complete": measured,
        },
        "limitations": [
            "A preflight pass is not a physical success.",
            "Planner engage is not racket contact.",
            "Legal landing requires Gate3BallState contact/table/net counters and is fail-closed when telemetry is incomplete.",
            "The mixed protocol uses current-contract FH/BH lanes and planner-side selection; it is not a side label injected into the policy.",
        ],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--physical-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.log_dir, args.physical_evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    coverage = report["coverage"]
    physical = report["physical"]
    print(
        "[closed-loop-audit] "
        f"preflight/launch/engage={coverage['preflight_pass']}/"
        f"{coverage['launched']}/{coverage['planner_engaged']} "
        f"contact={physical['contact_pass']}/{coverage['expected_shots']} "
        f"legal={physical['legal_landing_pass']}/{coverage['expected_shots']} "
        f"measured={physical['measured']}"
    )
    print(f"[closed-loop-audit] JSON: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
