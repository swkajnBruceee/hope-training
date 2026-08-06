#!/usr/bin/env python3
"""Generate explicit forehand/backhand candidate trajectories in one run.

The manifest deliberately keeps stroke labels explicit. `auto` is rejected by
this dataset wrapper because auto branch selection is useful online but can
silently mix labels and READY-state bias in an offline training library.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    goals = data.get("goals")
    if not isinstance(goals, list) or not goals:
        raise ValueError("manifest.goals must be a non-empty list")
    return data


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ready-forehand", type=Path, required=True)
    parser.add_argument("--ready-backhand", type=Path, required=True)
    parser.add_argument("--planner-config", type=Path, required=True)
    parser.add_argument("--robot-xml", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--control-hz", type=float, default=100.0)
    parser.add_argument("--csv-to-npz", type=Path)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    for path in (
        args.binary,
        args.manifest,
        args.ready_forehand,
        args.ready_backhand,
        args.planner_config,
        args.robot_xml,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    manifest = load_manifest(args.manifest)
    base = args.manifest.resolve().parent
    args.output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    used_goal_ids: set[str] = set()

    for index, entry in enumerate(manifest["goals"]):
        if not isinstance(entry, dict):
            raise ValueError(f"goals[{index}] must be an object")
        swing = str(entry.get("swing_type", "")).strip().lower()
        if swing not in {"forehand", "backhand"}:
            raise ValueError(
                f"goals[{index}].swing_type must be forehand or backhand; "
                "generate auto-mode diagnostics separately"
            )
        goal_path = Path(str(entry.get("goal_path", "")))
        if not goal_path.is_absolute():
            goal_path = base / goal_path
        if not goal_path.exists():
            raise FileNotFoundError(goal_path)

        # Read only goal_id and explicit label from the simple YAML without a
        # PyYAML dependency. The C++ binary remains the authoritative parser.
        goal_id = goal_path.stem
        declared_swing = None
        for raw_line in goal_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if line.startswith("goal_id:"):
                goal_id = line.split(":", 1)[1].strip().strip("\"'")
            elif line.startswith("swing_type:"):
                declared_swing = line.split(":", 1)[1].strip().strip("\"'").lower()
        if declared_swing != swing:
            raise ValueError(
                f"manifest label {swing!r} disagrees with {goal_path.name} "
                f"swing_type {declared_swing!r}"
            )
        if goal_id in used_goal_ids:
            raise ValueError(f"duplicate goal_id: {goal_id}")
        used_goal_ids.add(goal_id)

        ready = args.ready_forehand if swing == "forehand" else args.ready_backhand
        out_dir = args.output_root / swing / goal_id
        out_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(args.binary),
            "--goal",
            str(goal_path),
            "--ready",
            str(ready),
            "--planner-config",
            str(args.planner_config),
            "--robot-xml",
            str(args.robot_xml),
            "--output-dir",
            str(out_dir),
            "--control-hz",
            str(args.control_hz),
        ]
        result = run_command(command)
        (out_dir / "generator_stdout.txt").write_text(result.stdout, encoding="utf-8")
        (out_dir / "generator_stderr.txt").write_text(result.stderr, encoding="utf-8")

        diagnostics_path = out_dir / "diagnostics.json"
        diagnostics: dict[str, Any] = {}
        if diagnostics_path.exists():
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))

        npz_status = "not_requested"
        if result.returncode == 0 and args.csv_to_npz is not None:
            npz_result = run_command(
                [
                    sys.executable,
                    str(args.csv_to_npz),
                    "--csv",
                    str(out_dir / "trajectory_100hz.csv"),
                    "--goal",
                    str(out_dir / "normalized_goal.json"),
                    "--diagnostics",
                    str(diagnostics_path),
                    "--output",
                    str(out_dir / "trajectory_100hz.npz"),
                ]
            )
            (out_dir / "npz_stdout.txt").write_text(npz_result.stdout, encoding="utf-8")
            (out_dir / "npz_stderr.txt").write_text(npz_result.stderr, encoding="utf-8")
            npz_status = "ok" if npz_result.returncode == 0 else "failed"
            if npz_result.returncode != 0 and not args.continue_on_error:
                raise RuntimeError(f"NPZ conversion failed for {goal_id}: {npz_result.stderr}")

        row = {
            "goal_id": goal_id,
            "requested_swing_type": swing,
            "selected_swing_type": diagnostics.get("selected_swing_type"),
            "ready_id": diagnostics.get("ready_id"),
            "returncode": result.returncode,
            "success": bool(diagnostics.get("success", False)),
            "status": diagnostics.get("status", "NO_DIAGNOSTICS"),
            "reject_reason": diagnostics.get("trajectory_reject_reason"),
            "position_error_m": diagnostics.get("solved_position_error_m"),
            "normal_error_deg": diagnostics.get("solved_normal_error_deg"),
            "planned_strike_time_s": diagnostics.get("planned_strike_time_s"),
            "npz_status": npz_status,
            "output_dir": str(out_dir.resolve()),
        }
        rows.append(row)

        if result.returncode != 0 and not args.continue_on_error:
            raise RuntimeError(
                f"generation failed for {goal_id} ({swing}):\n{result.stderr}\n"
                f"see {out_dir}"
            )

    summary_json = args.output_root / "generation_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "schema_version": "a3_dual_stroke_generation_summary/v1",
                "manifest": str(args.manifest.resolve()),
                "control_hz": args.control_hz,
                "count": len(rows),
                "forehand_count": sum(r["requested_swing_type"] == "forehand" for r in rows),
                "backhand_count": sum(r["requested_swing_type"] == "backhand" for r in rows),
                "success_count": sum(bool(r["success"]) for r in rows),
                "results": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    summary_csv = args.output_root / "generation_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(summary_json)
    print(summary_csv)
    return 0 if all(row["returncode"] == 0 for row in rows) else 3


if __name__ == "__main__":
    raise SystemExit(main())
