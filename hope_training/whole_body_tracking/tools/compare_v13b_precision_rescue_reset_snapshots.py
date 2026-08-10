#!/usr/bin/env python3
"""Fail-closed exact comparison of isolated CompletePriors/Rescue snapshots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--completepriors", required=True)
    parser.add_argument("--rescue", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    left = json.loads(Path(args.completepriors).read_text())
    right = json.loads(Path(args.rescue).read_text())
    flat_left = [value for row in left["goal_10d"] for value in row]
    flat_right = [value for row in right["goal_10d"] for value in row]
    if len(flat_left) != len(flat_right):
        raise SystemExit("goal widths differ")
    maximum = max((abs(a - b) for a, b in zip(flat_left, flat_right)), default=0.0)
    report = {
        "status": "pass" if maximum <= 1.0e-6 and left["runtime_progress"] == right["runtime_progress"] else "fail",
        "goal_abs_max_difference": maximum,
        "completepriors": left, "rescue": right,
        "checks": {
            "same_goal": maximum <= 1.0e-6,
            "same_progress": abs(left["runtime_progress"] - right["runtime_progress"]) <= 1.0e-9,
            "same_lower_alpha": abs(left["lower_alpha"] - right["lower_alpha"]) <= 1.0e-6,
            "same_upper_alpha": abs(left["upper_alpha"] - right["upper_alpha"]) <= 1.0e-6,
            "actor_98d": left["actor_obs_dim"] == right["actor_obs_dim"] == 98,
            "action_26d": left["action_dim"] == right["action_dim"] == 26,
        },
    }
    report["status"] = "pass" if all(report["checks"].values()) else "fail"
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit("first-reset equivalence failed")


if __name__ == "__main__":
    main()
