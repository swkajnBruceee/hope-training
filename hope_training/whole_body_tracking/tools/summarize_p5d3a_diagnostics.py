#!/usr/bin/env python3
"""Summarize v3 per-reference PhysX diagnostics without training."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval_outputs/p5d3a_difficulty_audit_v2.json"
CHECKPOINT = "logs/rsl_rl/agibot_a3_p5d_prior_guided_reference_tracker_p5d2/2026-08-04_01-26-45_p5d2_formal_4096x2000/model_2198.pt"


def vec(s: str) -> list[float]:
    return [float(x) for x in s.split("/")]


def norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def sub(a: list[float], b: list[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]


def parse_log(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for idx, line in enumerate(lines):
        if not line.startswith("rank,episode_id,target_xyz,"):
            continue
        for raw in lines[idx + 1 :]:
            parts = next(csv.reader([raw]), [])
            if len(parts) < 22 or not parts[0].isdigit() or not parts[1].startswith("p5d2_"):
                continue
            target_vel = vec(parts[13])
            reference_vel = vec(parts[14])
            actual_vel = vec(parts[15])
            target_speed = norm(target_vel)
            actual_speed = norm(actual_vel)
            reference_speed = norm(reference_vel)
            actual_target_vec_error = norm(sub(actual_vel, target_vel))
            reference_target_vec_error = norm(sub(reference_vel, target_vel))
            rows[parts[1]] = {
                "episode_id": parts[1],
                "target_xyz": vec(parts[2]),
                "reference_xyz": vec(parts[3]),
                "actual_xyz": vec(parts[4]),
                "target_reference_error_m": float(parts[5]),
                "reference_actual_error_m": float(parts[6]),
                "residual_max_rad": float(parts[9]),
                "residual_clip_fraction": float(parts[11]),
                "safety_projection_max_rad": float(parts[12]),
                "target_velocity_mps": target_vel,
                "reference_velocity_mps": reference_vel,
                "actual_velocity_mps": actual_vel,
                "target_speed_mps": target_speed,
                "reference_speed_mps": reference_speed,
                "actual_speed_mps": actual_speed,
                "velocity_vector_error_mps": actual_target_vec_error,
                "reference_velocity_vector_error_mps": reference_target_vec_error,
                "velocity_magnitude_error_mps": float(parts[16]),
                "velocity_direction_error_deg": float(parts[17]),
                "best_pos_error_m": float(parts[18]),
                "best_pos_step": int(parts[19]),
                "best_pos_offset_from_marked_hit_step": int(parts[19]) - 80,
                "best_pos_velocity_magnitude_error_mps": float(parts[20]),
                "best_pos_velocity_direction_error_deg": float(parts[21]),
            }
        break
    return rows


def mean(rows: list[dict], key: str) -> float:
    return sum(float(r[key]) for r in rows) / len(rows) if rows else float("nan")


def main() -> None:
    split_specs = {
        "train": (16, "eval_outputs/p5d2_formal_train_learned_diagnostic_v3.log", "eval_outputs/p5d2_formal_train_zero_diagnostic_v3.log", "strike_goal_p5/p5d2_dataset_v1/p5d2_train_manifest.json"),
        "validation": (4, "eval_outputs/p5d2_formal_validation_learned_diagnostic_v3.log", "eval_outputs/p5d2_formal_validation_zero_diagnostic_v3.log", "strike_goal_p5/p5d2_dataset_v1/p5d2_validation_manifest.json"),
        "holdout": (4, "eval_outputs/p5d2_formal_holdout_learned_diagnostic_v3.log", "eval_outputs/p5d2_formal_holdout_zero_diagnostic_v3.log", "strike_goal_p5/p5d2_dataset_v1/p5d2_holdout_manifest.json"),
    }
    result = {"schema_version": "p5d3a_difficulty_audit/v2", "status": "AUDIT_ONLY_NO_NEW_TRAINING", "checkpoint": CHECKPOINT, "marked_hit_runtime_step": 80, "splits": {}}
    all_learned: list[dict] = []
    all_zero: list[dict] = []
    for split, (_, learned_log, zero_log, manifest_rel) in split_specs.items():
        manifest = json.loads((ROOT / "eval_outputs" / manifest_rel).read_text())
        regions = {m["episode_id"]: m.get("p5d2_dataset", {}).get("region", "unknown") for m in manifest["motions"]}
        learned = parse_log(ROOT / learned_log)
        zero = parse_log(ROOT / zero_log)
        rows = []
        for episode_id in sorted(learned):
            l = learned[episode_id]
            z = zero.get(episode_id)
            if z is None:
                continue
            l["region"] = regions.get(episode_id, "unknown")
            z["region"] = regions.get(episode_id, "unknown")
            rows.append({"episode_id": episode_id, "region": l["region"], "learned": l, "zero": z})
            all_learned.append(l)
            all_zero.append(z)
        result["splits"][split] = {"count": len(rows), "rows": rows}

    # Classification deliberately uses only observable replay fields. It does
    # not claim phase or infeasibility until the vector/time diagnostics support it.
    groups = {"dynamic_hard": [], "speed_phase_candidate": [], "moderate_learnable": [], "action_clip_limited": []}
    for split_data in result["splits"].values():
        for row in split_data["rows"]:
            l = row["learned"]
            if l["residual_clip_fraction"] > 0.0:
                groups["action_clip_limited"].append(row["episode_id"])
            elif l["reference_actual_error_m"] >= 0.30 and l["best_pos_error_m"] >= 0.25:
                groups["dynamic_hard"].append(row["episode_id"])
            elif l["reference_actual_error_m"] < 0.10 and l["velocity_magnitude_error_mps"] >= 1.50:
                groups["speed_phase_candidate"].append(row["episode_id"])
            else:
                groups["moderate_learnable"].append(row["episode_id"])
    result["groups"] = groups
    result["aggregate"] = {
        "learned": {k: mean(all_learned, k) for k in ("reference_actual_error_m", "velocity_vector_error_mps", "velocity_magnitude_error_mps", "velocity_direction_error_deg", "best_pos_error_m", "best_pos_velocity_magnitude_error_mps", "best_pos_velocity_direction_error_deg")},
        "zero": {k: mean(all_zero, k) for k in ("reference_actual_error_m", "velocity_vector_error_mps", "velocity_magnitude_error_mps", "velocity_direction_error_deg", "best_pos_error_m", "best_pos_velocity_magnitude_error_mps", "best_pos_velocity_direction_error_deg")},
    }
    result["interpretation"] = {
        "reference_velocity_is_available": True,
        "actual_velocity_is_available": True,
        "best_position_time_is_available": True,
        "marked_hit_runtime_step": 80,
        "reference_prior_gap": "not directly inferable from the current trace field contract; requires explicit same-coordinate prior export",
        "root_base_tcp_decomposition": "not directly inferable from the current trace field contract; requires explicit base-frozen replay",
        "next_action": "review these diagnostics and reclassify references before any new training; do not globally increase residual authority"
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
