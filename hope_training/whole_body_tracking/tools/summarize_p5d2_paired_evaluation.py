#!/usr/bin/env python3
"""Build per-reference and per-region P5D-2 paired PhysX evaluation evidence."""
import csv
import json
import math
import pathlib
import statistics
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _num(x):
    try:
        v = float(x)
        return None if not math.isfinite(v) else v
    except (TypeError, ValueError):
        return None


def _rows(path):
    lines = pathlib.Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    summary_header = None
    diag_header = None
    posture_header = None
    for line in lines:
        if line.startswith("rank,stroke,episode_id,"):
            summary_header = next(csv.reader([line]))
        elif line.startswith("rank,episode_id,target_xyz,"):
            diag_header = next(csv.reader([line]))
        elif line.startswith("rank,episode_id,physical_terminated,"):
            posture_header = next(csv.reader([line]))
    def collect(header, kind):
        if not header:
            return {}
        out = {}
        start = lines.index(",".join(header)) + 1
        for line in lines[start:]:
            if not line or not line[0].isdigit():
                if out and kind == "summary":
                    break
                continue
            vals = next(csv.reader([line]))
            if len(vals) < len(header):
                continue
            row = dict(zip(header, vals))
            eid = row.get("episode_id")
            if eid:
                out[eid] = row
        return out
    return {
        "summary": collect(summary_header, "summary"),
        "diagnostic": collect(diag_header, "diagnostic"),
        "posture": collect(posture_header, "posture"),
    }


def _stats(values):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return {"count": 0}
    p95_index = min(len(vals) - 1, int(math.ceil(0.95 * len(vals))) - 1)
    return {
        "count": len(vals),
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "p95": vals[p95_index],
        "worst": vals[-1],
    }


def _metric(row, key):
    return _num(row.get(key)) if row else None


def _join(split, learned_path, zero_path, manifest_path):
    manifest = json.loads(pathlib.Path(manifest_path).read_text(encoding="utf-8"))
    entries = manifest.get("motions", manifest.get("entries", []))
    regions = {str(e["episode_id"]): e.get("p5d2_dataset", {}).get("region", e.get("region", "unknown")) for e in entries}
    learned = _rows(learned_path)
    zero = _rows(zero_path)
    ids = sorted(set(regions) & set(learned["summary"]) & set(zero["summary"]))
    rows = []
    for eid in ids:
        ls, zs = learned["summary"][eid], zero["summary"][eid]
        ld, zd = learned["diagnostic"].get(eid, {}), zero["diagnostic"].get(eid, {})
        lp = learned["posture"].get(eid, {})
        rows.append({
            "episode_id": eid,
            "region": regions[eid],
            "learned": {
                "pos_m": _metric(ls, "pos_exact"),
                "vel_mps": _metric(ls, "vel_exact"),
                "normal_deg": _metric(ls, "normal_deg_exact"),
                "pos_window_m": _metric(ls, "pos_window"),
                "composite_pass": int(ls.get("composite_pass", 0)),
                "episode_count": 1,
                "min_joint_margin": _metric(ls, "min_joint_margin"),
                "min_arm_margin": _metric(ls, "min_arm_margin"),
                "reference_tracking_m": _metric(ld, "reference_minus_actual_m"),
                "safety_projection_max_rad": _metric(ld, "safety_projection_max_rad"),
                "residual_max_rad": _metric(ld, "residual_max_rad"),
                "residual_clip_fraction": _metric(ld, "residual_clip_fraction"),
                "physical_terminated": int(lp.get("physical_terminated", 0)),
                "timeout_seen": int(lp.get("timeout_seen", 0)),
                "min_root_height_m": _metric(lp, "min_root_height_m"),
                "min_root_upright": _metric(lp, "min_root_upright"),
            },
            "zero": {
                "pos_m": _metric(zs, "pos_exact"),
                "vel_mps": _metric(zs, "vel_exact"),
                "normal_deg": _metric(zs, "normal_deg_exact"),
                "pos_window_m": _metric(zs, "pos_window"),
                "composite_pass": int(zs.get("composite_pass", 0)),
                "episode_count": 1,
                "min_joint_margin": _metric(zs, "min_joint_margin"),
                "min_arm_margin": _metric(zs, "min_arm_margin"),
                "reference_tracking_m": _metric(zd, "reference_minus_actual_m"),
                "safety_projection_max_rad": _metric(zd, "safety_projection_max_rad"),
                "residual_max_rad": _metric(zd, "residual_max_rad"),
                "residual_clip_fraction": _metric(zd, "residual_clip_fraction"),
                "physical_terminated": int(zero["posture"].get(eid, {}).get("physical_terminated", 0)),
                "timeout_seen": int(zero["posture"].get(eid, {}).get("timeout_seen", 0)),
                "min_root_height_m": _metric(zero["posture"].get(eid, {}), "min_root_height_m"),
                "min_root_upright": _metric(zero["posture"].get(eid, {}), "min_root_upright"),
            },
        })
    def aggregate(items):
        out = {}
        for mode in ("learned", "zero"):
            out[mode] = {}
            for key in ("pos_m", "vel_mps", "normal_deg", "reference_tracking_m", "safety_projection_max_rad", "residual_clip_fraction", "min_root_height_m", "min_joint_margin", "min_arm_margin"):
                out[mode][key] = _stats([r[mode].get(key) for r in items])
            out[mode]["physical_termination_count"] = sum(r[mode]["physical_terminated"] for r in items)
            out[mode]["timeout_count"] = sum(r[mode]["timeout_seen"] for r in items)
        out["improvement_fraction"] = {
            key: sum((r["learned"].get(key) is not None and r["zero"].get(key) is not None and r["learned"][key] < r["zero"][key]) for r in items) / len(items) if items else 0.0
            for key in ("pos_m", "vel_mps", "normal_deg")
        }
        return out
    by_region = {}
    for region in sorted({r["region"] for r in rows}):
        subset = [r for r in rows if r["region"] == region]
        by_region[region] = {"count": len(subset), **aggregate(subset)}
    return {"split": split, "count": len(rows), "aggregate": aggregate(rows), "by_region": by_region, "rows": rows}


def main():
    base = ROOT / "eval_outputs"
    specs = {
        "train": ("p5d2_formal_train_learned_diagnostic_v2.log", "p5d2_formal_train_zero_diagnostic_v2.log", "strike_goal_p5/p5d2_dataset_v1/p5d2_train_manifest.json"),
        "validation": ("p5d2_formal_validation_learned_diagnostic_v2.log", "p5d2_formal_validation_zero_diagnostic_v2.log", "strike_goal_p5/p5d2_dataset_v1/p5d2_validation_manifest.json"),
        "holdout": ("p5d2_formal_holdout_learned_diagnostic_v2.log", "p5d2_formal_holdout_zero_diagnostic_v2.log", "strike_goal_p5/p5d2_dataset_v1/p5d2_holdout_manifest.json"),
    }
    report = {"schema_version": "p5d2_paired_physx_per_reference/v1", "checkpoint": "logs/rsl_rl/agibot_a3_p5d_prior_guided_reference_tracker_p5d2/2026-08-04_01-26-45_p5d2_formal_4096x2000/model_2198.pt", "holdout_evaluated_after_training": True, "splits": {}}
    for split, (lp, zp, mp) in specs.items():
        report["splits"][split] = _join(split, base / lp, base / zp, base / mp)
    out = base / "p5d2_paired_physx_per_reference_v1.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: {"count": v["count"], "aggregate": v["aggregate"]} for k, v in report["splits"].items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
