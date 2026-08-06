#!/usr/bin/env python3
"""Contract-A phase-lag diagnostic over the marked-hit window.

This is deliberately a command/reference diagnostic, not a teacher qualifier:
the action trace has reference and processed-command streams but no hidden
motion/reference ID is exposed to the actor.  It reports the lag in
[-8,+8] control steps with the highest derivative correlation for each upper
joint and group.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


GROUPS = {
    "waist": range(0, 3),
    "shoulder": range(3, 6),
    "elbow": range(6, 7),
    "wrist": range(7, 10),
}


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3 or b.size < 3:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den > 1.0e-9 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--window-before", type=int, default=15)
    ap.add_argument("--window-after", type=int, default=5)
    ap.add_argument("--max-lag", type=int, default=8)
    args = ap.parse_args()
    z = np.load(args.trace, allow_pickle=True)
    trace = np.asarray(z["trace"], dtype=np.float64)
    time_steps = np.asarray(z["time_steps"], dtype=np.int64)
    motion_ids = np.asarray(z["motion_ids"], dtype=np.int64)
    names = [str(x) for x in z["upper_joint_names"].tolist()]
    # [reference, primary, tracker, processed, safety], each 10D.
    ref = trace[..., :10]
    processed = trace[..., 30:40]
    rows = []
    for env in range(trace.shape[1]):
        hit = int(np.max(time_steps[:, env]))
        center = int(np.argmin(np.abs(time_steps[:, env] - hit)))
        lo = max(0, center - args.window_before)
        hi = min(trace.shape[0], center + args.window_after + 1)
        for j, name in enumerate(names):
            rv = np.diff(ref[lo:hi, env, j])
            pv = np.diff(processed[lo:hi, env, j])
            scores = {}
            for lag in range(-args.max_lag, args.max_lag + 1):
                if lag < 0:
                    score = _corr(rv[-lag:], pv[: len(pv) + lag])
                elif lag > 0:
                    score = _corr(rv[: len(rv) - lag], pv[lag:])
                else:
                    score = _corr(rv, pv)
                scores[str(lag)] = score
            best = max(scores, key=scores.get)
            rows.append({
                "env": env,
                "motion_id": int(motion_ids[center, env]),
                "joint": name,
                "group": next(g for g, ix in GROUPS.items() if j in ix),
                "window_trace_indices": [lo, hi - 1],
                "best_lag_steps": int(best),
                "best_correlation": float(scores[best]),
                "scores": scores,
            })
    grouped = {}
    for group in GROUPS:
        vals = [r["best_lag_steps"] for r in rows if r["group"] == group]
        grouped[group] = {
            "count": len(vals),
            "mean_best_lag_steps": float(np.mean(vals)) if vals else 0.0,
            "max_abs_best_lag_steps": int(max(map(abs, vals))) if vals else 0,
        }
    out = {
        "schema_version": "p5u1_phase_lag_scan/v1",
        "status": "DIAGNOSTIC_ONLY",
        "trace": str(args.trace),
        "window": {"before_steps": args.window_before, "after_steps": args.window_after, "lag_range": [-args.max_lag, args.max_lag]},
        "groups": grouped,
        "rows": rows,
        "note": "Command/reference correlation only; not a PhysX actual-lag or teacher-qualification result.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "groups": grouped}, indent=2))


if __name__ == "__main__":
    main()
