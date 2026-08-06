#!/usr/bin/env python3
"""Summarize the v4 root-translation responsibility audit.

This is an audit-only report.  Root translation is only one component of the
TCP error; the report deliberately does not attribute the remaining error to
root orientation or individual upper-body joints without a separate replay.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval_outputs/p5d3a_root_responsibility_v1.json"


def vec(value: str) -> list[float]:
    return [float(x) for x in value.split("/")]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def parse_log(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for idx, line in enumerate(lines):
        if not line.startswith("rank,episode_id,target_xyz,"):
            continue
        header = next(csv.reader([line]))
        columns = {name: pos for pos, name in enumerate(header)}
        root_ref_idx = columns["reference_root_xyz"]
        root_actual_idx = columns["actual_root_xyz"]
        root_err_idx = columns["root_translation_error_m"]
        tcp_err_idx = columns["reference_minus_actual_m"]
        best_err_idx = columns["best_pos_error_m"]
        best_step_idx = columns["best_pos_step"]
        safety_idx = columns["safety_projection_max_rad"]
        residual_idx = columns["residual_max_rad"]
        for raw in lines[idx + 1 :]:
            parts = next(csv.reader([raw]), [])
            # v4 adds reference_root_xyz, actual_root_xyz and the final root
            # translation error after the v3 velocity diagnostics.
            if len(parts) <= root_err_idx or not parts[0].isdigit() or not parts[1].startswith("p5d2_"):
                continue
            rows[parts[1]] = {
                "episode_id": parts[1],
                "reference_actual_error_m": float(parts[tcp_err_idx]),
                "best_pos_error_m": float(parts[best_err_idx]),
                "best_pos_step": int(parts[best_step_idx]),
                "reference_root_xyz": vec(parts[root_ref_idx]),
                "actual_root_xyz": vec(parts[root_actual_idx]),
                "root_translation_error_m": float(parts[root_err_idx]),
                "safety_projection_max_rad": float(parts[safety_idx]),
                "residual_max_rad": float(parts[residual_idx]),
                "prior_contribution_max_rad": float(parts[columns["prior_contribution_max_rad"]]) if "prior_contribution_max_rad" in columns else None,
                "prior_contribution_mean_rad": float(parts[columns["prior_contribution_mean_rad"]]) if "prior_contribution_mean_rad" in columns else None,
                "tracker_residual_max_rad": float(parts[columns["tracker_residual_max_rad"]]) if "tracker_residual_max_rad" in columns else None,
                "tracker_residual_mean_rad": float(parts[columns["tracker_residual_mean_rad"]]) if "tracker_residual_mean_rad" in columns else None,
            }
        break
    return rows


def summarize_pair(learned: dict[str, dict], zero: dict[str, dict]) -> dict:
    rows = []
    for episode_id in sorted(set(learned) & set(zero)):
        l = learned[episode_id]
        z = zero[episode_id]
        rows.append(
            {
                "episode_id": episode_id,
                "learned": l,
                "zero": z,
                "learned_root_improvement_m": z["root_translation_error_m"] - l["root_translation_error_m"],
                "learned_tcp_improvement_m": z["reference_actual_error_m"] - l["reference_actual_error_m"],
            }
        )
    def agg(mode: str) -> dict:
        values = [row[mode] for row in rows]
        tcp = [x["reference_actual_error_m"] for x in values]
        root = [x["root_translation_error_m"] for x in values]
        best = [x["best_pos_error_m"] for x in values]
        prior = [x["prior_contribution_max_rad"] for x in values if x["prior_contribution_max_rad"] is not None]
        tracker = [x["tracker_residual_max_rad"] for x in values if x["tracker_residual_max_rad"] is not None]
        return {
            "count": len(values),
            "mean_tcp_reference_actual_error_m": mean(tcp),
            "median_tcp_reference_actual_error_m": sorted(tcp)[len(tcp) // 2] if tcp else float("nan"),
            "mean_best_pos_error_m": mean(best),
            "mean_root_translation_error_m": mean(root),
            "max_root_translation_error_m": max(root) if root else float("nan"),
            "root_to_tcp_error_ratio_mean": mean([r / t if t > 1e-9 else float("nan") for r, t in zip(root, tcp)]),
            "root_ge_50_percent_tcp_count": sum(r >= 0.5 * t for r, t in zip(root, tcp)),
            "mean_prior_contribution_max_rad": mean(prior),
            "mean_tracker_residual_max_rad": mean(tracker),
        }
    return {"rows": rows, "aggregate": {"learned": agg("learned"), "zero": agg("zero")}}


def main() -> None:
    specs = {
        "train": ("eval_outputs/p5d2_formal_train_learned_diagnostic_v5.log", "eval_outputs/p5d2_formal_train_zero_diagnostic_v5.log"),
        "validation": ("eval_outputs/p5d2_formal_validation_learned_diagnostic_v5.log", "eval_outputs/p5d2_formal_validation_zero_diagnostic_v5.log"),
        "holdout": ("eval_outputs/p5d2_formal_holdout_learned_diagnostic_v5.log", "eval_outputs/p5d2_formal_holdout_zero_diagnostic_v5.log"),
    }
    result = {
        "schema_version": "p5d3a_root_responsibility/v1",
        "status": "AUDIT_ONLY_NO_NEW_TRAINING",
        "source": "P5D-2 formal learned/zero PhysX paired replay, v5",
        "scope": "root translation only; not a full TCP Jacobian decomposition",
        "splits": {},
        "interpretation": {
            "root_translation_is_not_total_base_contribution": True,
            "remaining_error_requires_root_orientation_and_joint_responsibility_replay": True,
            "training_decision": "no new PPO training until responsibility and difficult-reference classes are reviewed",
        },
    }
    all_rows = []
    for split, (learned_path, zero_path) in specs.items():
        pair = summarize_pair(parse_log(ROOT / learned_path), parse_log(ROOT / zero_path))
        result["splits"][split] = pair
        all_rows.extend(pair["rows"])
    result["aggregate"] = {
        "paired_count": len(all_rows),
        "learned_mean_tcp_error_m": mean([r["learned"]["reference_actual_error_m"] for r in all_rows]),
        "zero_mean_tcp_error_m": mean([r["zero"]["reference_actual_error_m"] for r in all_rows]),
        "learned_mean_root_translation_error_m": mean([r["learned"]["root_translation_error_m"] for r in all_rows]),
        "zero_mean_root_translation_error_m": mean([r["zero"]["root_translation_error_m"] for r in all_rows]),
        "learned_mean_root_tcp_ratio": mean([
            r["learned"]["root_translation_error_m"] / r["learned"]["reference_actual_error_m"]
            for r in all_rows if r["learned"]["reference_actual_error_m"] > 1e-9
        ]),
        "learned_mean_prior_contribution_max_rad": mean([
            r["learned"]["prior_contribution_max_rad"] for r in all_rows
            if r["learned"]["prior_contribution_max_rad"] is not None
        ]),
        "learned_mean_tracker_residual_max_rad": mean([
            r["learned"]["tracker_residual_max_rad"] for r in all_rows
            if r["learned"]["tracker_residual_max_rad"] is not None
        ]),
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
