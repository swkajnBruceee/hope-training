#!/usr/bin/env python3
"""Merge the three single-mode F0 reports without starting Isaac Sim."""

from __future__ import annotations

import argparse
import json
import pathlib


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", required=True)
    parser.add_argument("--zero", required=True)
    parser.add_argument("--stage-a", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = {
        "fixed_model900": pathlib.Path(args.fixed),
        "floating_model900_zero_leg": pathlib.Path(args.zero),
        "floating_model900_stageA": pathlib.Path(args.stage_a),
    }
    modes = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    by_mode = {name: {row["motion_id"]: row for row in report["results"]} for name, report in modes.items()}
    motion_ids = sorted(by_mode["fixed_model900"])
    summary = []
    for motion_id in motion_ids:
        fixed = by_mode["fixed_model900"][motion_id]
        zero = by_mode["floating_model900_zero_leg"][motion_id]
        stage = by_mode["floating_model900_stageA"][motion_id]
        ef = fixed["position_error_m"]
        ez = zero["position_error_m"]
        es = stage["position_error_m"]
        summary.append(
            {
                "motion_id": motion_id,
                "fixed_pos_error_m": ef,
                "floating_zero_pos_error_m": ez,
                "floating_stageA_pos_error_m": es,
                "floating_added_error_m": ez - ef,
                "stageA_recovery_m": ez - es,
            }
        )
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged = {
        "upper_checkpoint": modes["fixed_model900"]["upper_checkpoint"],
        "stage_a_checkpoint": modes["fixed_model900"]["stage_a_checkpoint"],
        "seed": modes["fixed_model900"]["seed"],
        "modes": modes,
        "summary": summary,
    }
    output.write_text(json.dumps(merged, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
