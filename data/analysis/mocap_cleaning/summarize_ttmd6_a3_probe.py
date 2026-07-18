#!/usr/bin/env python3
"""Summarize the TTMD6 A3 probe without applying the legacy source gate.

TTMD6 does not provide the source-quality fields expected by the older A3
optimizer.  This report therefore keeps source provenance separate from A3
task, dynamics, and replay diagnostics and never promotes a sample to
training eligibility.
"""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    _ROOT = Path(__file__).resolve().parents[3]
    _DATA = _ROOT / "data"
    if str(_DATA) not in sys.path:
        sys.path.insert(0, str(_DATA))
    del _ROOT, _DATA

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from analysis.mocap_cleaning.a3_metadata import A3_POLICY_JOINT_ORDER
from analysis.mocap_cleaning.a3_refinement_solver import load_a3_joint_limits


CLASS_LABELS = {
    1: "forehand_attack",
    2: "forehand_drive",
    3: "forehand_push",
    4: "backhand_attack",
    5: "backhand_drive",
    6: "backhand_push",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _class_id(source_id: str) -> int:
    return int(source_id.split("_", 1)[0].replace("class", ""))


def _hard_limit_metrics(csv_path: Path, near_margin_rad: float = 0.02) -> dict[str, Any]:
    limits = load_a3_joint_limits()
    data = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]
    q = data[:, 7:]
    violations: list[str] = []
    near: list[str] = []
    margins: dict[str, float] = {}
    for idx, name in enumerate(A3_POLICY_JOINT_ORDER):
        if name not in limits:
            continue
        lower, upper = limits[name]
        values = q[:, idx]
        min_margin = float(np.min(np.minimum(values - lower, upper - values)))
        margins[name] = min_margin
        if np.any(values < lower - 1e-6) or np.any(values > upper + 1e-6):
            violations.append(name)
        if min_margin <= near_margin_rad:
            near.append(name)
    return {
        "hard_limit_violation": bool(violations),
        "hard_limit_violation_joints": violations,
        "near_hard_limit": bool(near),
        "near_hard_limit_joints": near,
        "minimum_joint_limit_margin_rad": min(margins.values()) if margins else None,
        "minimum_joint_limit_margin_by_joint_rad": margins,
    }


def _first_failure(report: dict[str, Any]) -> str:
    checks = (
        ("hit_geometry", ("hit_position_pass", "hit_orientation_pass", "hit_velocity_direction_pass", "hit_velocity_magnitude_pass")),
        ("wrist_naturalness", ("wrist_naturalness_pass", "wrist_pitch_pass", "wrist_yaw_pass", "wrist_roll_pass")),
        ("waist_yaw", ("waist_yaw_pass",)),
        ("dynamics", ("dynamics_pass",)),
        ("replay_precheck", ("replay_ready",)),
    )
    for category, keys in checks:
        if any(key in report and not bool(report[key]) for key in keys):
            return category
    if report.get("status") == "reject":
        return "other_reject"
    return "pass"


def _diagnostic_status(item: dict[str, Any], ik: dict[str, Any], optimized: dict[str, Any] | None) -> str:
    if item.get("ik_status") != "pass" or ik.get("status") != "pass":
        return "ik_reject"
    if optimized is None:
        return "ik_pass_not_optimized"
    if bool(optimized.get("replay_ready")):
        return "optimized_replay_ready_diagnostic"
    return f"optimized_reject_{_first_failure(optimized)}"


def _markdown(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    lines = [
        "# TTMD6 A3 Probe Summary",
        "",
        "> Diagnostic only. No record in this report is training eligible.",
        "",
        "## Scope",
        "",
        f"- Candidate targets: `{counts['candidate_targets']}`",
        f"- Initial IK reports: `{counts['ik_reports']}`",
        f"- Initial IK pass: `{counts['ik_pass']}`",
        f"- Initial IK reject: `{counts['ik_reject']}`",
        f"- Optimized reports available: `{counts['optimized_reports']}`",
        f"- IK-pass candidates without optimization: `{counts['ik_pass_without_optimization']}`",
        f"- IK-rejected candidates not optimized: `{counts['ik_reject_not_optimized']}`",
        f"- Optimized replay-ready diagnostic candidates: `{counts['optimized_replay_ready_diagnostic']}`",
        "",
        "The legacy optimizer's `bad_source_data` label is not used here. It is a",
        "contract mismatch because TTMD6 intentionally lacks the old A3 source",
        "quality flags. Geometry, wrist, waist, dynamics, and replay fields are",
        "reported independently.",
        "",
        "## Status Counts",
        "",
    ]
    for key, value in sorted(summary["status_counts"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## By Inferred Skill Class", "", "| class | label | total | IK pass | optimized | replay-ready diagnostic |", "|---:|---|---:|---:|---:|---:|"])
    for row in summary["by_class"]:
        lines.append(
            f"| {row['class_id']} | {row['class_label']} | {row['total']} | {row['ik_pass']} | "
            f"{row['optimized']} | {row['optimized_replay_ready_diagnostic']} |"
        )
    lines.extend(["", "## Records", "", "| episode | class | IK | optimized | diagnostic status | near hard limit |", "|---|---|---|---|---|---|"])
    for row in summary["records"]:
        near = ", ".join(row["hard_limit"]["near_hard_limit_joints"]) or "-"
        lines.append(
            f"| `{row['episode_id']}` | `{row['class_label']}` | `{row['ik_status']}` | "
            f"`{'yes' if row['optimized_available'] else 'no'}` | `{row['diagnostic_status']}` | `{near}` |"
        )
    lines.extend([
        "",
        "## Admission Decision",
        "",
        "- `training_eligible=false` for every record.",
        "- `optimized_replay_ready_diagnostic` means only that the current A3 fixed-base replay precheck passed.",
        "- It does not certify TTMD6 units, axes, skill labels, impact timing, constructed paddle orientation, or real A3 execution.",
        "- Before admission, the remaining optimizer candidates must be processed and the diagnostic candidates must pass TTMD6-specific visual, actuator, posture, balance, and impact validation.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument(
        "--optimized-root",
        type=Path,
        action="append",
        help="Additional optimizer output roots to merge into the diagnostic summary.",
    )
    args = parser.parse_args()
    root = args.probe_root
    project_root = root.resolve().parents[4]
    init_manifest = _load_json(root / "ik_init_manifest.json")
    config = yaml.safe_load((root / "retarget_config.yaml").read_text(encoding="utf-8"))
    optimized_by_id = {}
    optimized_roots = [root, *(args.optimized_root or [])]
    optimized_report_paths: dict[str, Path] = {}
    for optimized_root in optimized_roots:
        for path in sorted((optimized_root / "optimized_quality_reports").glob("*.json")):
            optimized_report_paths[str(path.stem)] = path
            report = _load_json(path)
            optimized_by_id[str(report["episode_id"])] = report

    records = []
    status_counts: Counter[str] = Counter()
    class_counts: dict[int, Counter[str]] = {}
    thresholds = config.get("quality_thresholds", {})
    wrist_p95_limits = {
        "right_wrist_roll": float(thresholds.get("right_wrist_roll_neutral_delta_p95_reject_deg", 65.0)),
        "right_wrist_pitch": float(thresholds.get("right_wrist_pitch_neutral_delta_p95_reject_deg", 28.0)),
        "right_wrist_yaw": float(thresholds.get("right_wrist_yaw_neutral_delta_p95_reject_deg", 30.0)),
    }
    for item in init_manifest["samples"]:
        episode_id = str(item["episode_id"])
        ik_path = _resolve(project_root, item["ik_quality_report"])
        ik = _load_json(ik_path)
        opt = optimized_by_id.get(episode_id)
        hard = _hard_limit_metrics(_resolve(project_root, item["ik_init_csv"]))
        wrist = {
            name: float(ik.get(f"{name}_neutral_delta_p95_deg", float("nan")))
            for name in wrist_p95_limits
        }
        wrist_threshold_pass = all(np.isfinite(value) and value <= wrist_p95_limits[name] for name, value in wrist.items())
        class_id = _class_id(str(item["source_id"]))
        status = _diagnostic_status(item, ik, opt)
        status_counts[status] += 1
        bucket = class_counts.setdefault(class_id, Counter())
        bucket["total"] += 1
        bucket["ik_pass"] += int(status != "ik_reject")
        bucket["optimized"] += int(opt is not None)
        bucket["optimized_replay_ready_diagnostic"] += int(opt is not None and bool(opt.get("replay_ready")))
        records.append({
            "episode_id": episode_id,
            "source_id": str(item["source_id"]),
            "class_id": class_id,
            "class_label": CLASS_LABELS.get(class_id, "unknown"),
            "candidate_hit_index": item.get("candidate_hit_index"),
            "candidate_hit_index_status": item.get("candidate_hit_index_status"),
            "orientation_status": item.get("orientation_status"),
            "coordinate_status": item.get("coordinate_status"),
            "ik_status": item.get("ik_status"),
            "ik_pose_status": item.get("ik_pose_status"),
            "ik_report": str(ik_path),
            "optimized_available": opt is not None,
            "optimized_report": str(optimized_report_paths[episode_id]) if opt is not None else None,
            "diagnostic_status": status,
            "training_eligible": False,
            "initial_ik_metrics": {
                "position_error_at_hit_m": ik.get("racket_position_error_at_hit_m"),
                "orientation_error_at_hit_deg": ik.get("racket_orientation_error_at_hit_deg"),
                "tangent_error_at_hit_deg": ik.get("racket_tangent_error_at_hit_deg"),
                "wrist_p95_deg": wrist,
                "wrist_threshold_pass": wrist_threshold_pass,
            },
            "optimized_metrics": {
                key: opt.get(key)
                for key in (
                    "status", "replay_ready", "reject_reasons", "hit_position_pass",
                    "hit_orientation_pass", "hit_velocity_direction_pass", "hit_velocity_magnitude_pass",
                    "waist_yaw_pass", "wrist_naturalness_pass", "dynamics_pass",
                    "max_active_joint_velocity_radps", "max_active_joint_acceleration_radps2",
                    "max_active_joint_jerk_radps3", "max_abs_waist_yaw_rad",
                    "right_wrist_bend_pitch_yaw_p95_deg",
                )
            } if opt is not None else None,
            "hard_limit": hard,
        })

    summary = {
        "dataset": "TTMD6",
        "stage": "a3_fixed_base_probe_diagnostic_summary_v0",
        "probe_root": str(root),
        "training_eligible": False,
        "orientation_status": "constructed_heuristic_only",
        "coordinate_status": "hypothesis_only",
        "hit_index_status": "candidate_only",
        "wrist_p95_thresholds_deg": wrist_p95_limits,
        "counts": {
            "candidate_targets": len(init_manifest["samples"]),
            "ik_reports": len(records),
            "ik_pass": sum(r["ik_status"] == "pass" for r in records),
            "ik_reject": sum(r["ik_status"] != "pass" for r in records),
            "optimized_reports": sum(r["optimized_available"] for r in records),
            "ik_pass_without_optimization": sum(r["ik_status"] == "pass" and not r["optimized_available"] for r in records),
            "ik_reject_not_optimized": sum(r["ik_status"] != "pass" and not r["optimized_available"] for r in records),
            "optimized_replay_ready_diagnostic": sum(r["diagnostic_status"] == "optimized_replay_ready_diagnostic" for r in records),
        },
        "status_counts": dict(status_counts),
        "by_class": [
            {
                "class_id": class_id,
                "class_label": CLASS_LABELS.get(class_id, "unknown"),
                **{key: int(bucket[key]) for key in ("total", "ik_pass", "optimized", "optimized_replay_ready_diagnostic")},
            }
            for class_id, bucket in sorted(class_counts.items())
        ],
        "records": records,
    }
    json_path = root / "a3_ik_probe_summary.json"
    md_path = root / "a3_ik_probe_summary.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(summary), encoding="utf-8")
    print(json_path)
    print(md_path)
    print(json.dumps(summary["counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
