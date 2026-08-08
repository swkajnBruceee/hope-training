#!/usr/bin/env python3
"""Read-only integrity and overlap scan for A3 motion libraries."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def scan(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    motions = payload.get("motions", [])
    ids = [str(x.get("motion_id", x.get("episode_id", ""))) for x in motions]
    result = {"manifest": str(path.resolve()), "entries": len(motions), "duplicate_ids": sorted(k for k,v in Counter(ids).items() if v > 1), "missing_payload": 0, "payload_nonfinite": 0, "wrong_shapes": 0, "stroke_counts": dict(Counter(str(x.get("stroke_type", x.get("stroke", "unknown"))).lower() for x in motions))}
    for entry in motions:
        raw = entry.get("motion_npz") or entry.get("library_motion_npz") or entry.get("canonical_motion_npz")
        if not raw or raw is True:
            result["missing_payload"] += 1; continue
        p = Path(raw); p = p if p.is_absolute() else path.parent / p
        if not p.is_file():
            result["missing_payload"] += 1; continue
        try:
            with np.load(p, allow_pickle=False) as z:
                keys = set(z.files)
                if {"joint_pos", "joint_vel"}.issubset(keys):
                    q, qd = z["joint_pos"], z["joint_vel"]
                    if q.ndim != 2 or qd.shape != q.shape or q.shape[1] not in (10, 31): result["wrong_shapes"] += 1
                    if not np.isfinite(q).all() or not np.isfinite(qd).all(): result["payload_nonfinite"] += 1
                if "body_pos_w" in keys and not np.isfinite(z["body_pos_w"]).all(): result["payload_nonfinite"] += 1
        except Exception:
            result["wrong_shapes"] += 1
    return result


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); parser.add_argument("manifests", nargs="+", type=Path); args = parser.parse_args()
    scans = [scan(p.expanduser().resolve()) for p in args.manifests]
    ids = []
    for p in args.manifests:
        payload = json.loads(p.expanduser().resolve().read_text(encoding="utf-8"))
        ids.extend((str(x.get("motion_id", x.get("episode_id", ""))), str(p)) for x in payload.get("motions", []))
    duplicate_cross = sorted(k for k,v in __import__("collections").Counter(x[0] for x in ids).items() if v > 1)
    report = {"schema_version": "a3_motion_library_scan/v1", "status": "completed", "read_only": True, "libraries": scans, "cross_library_duplicate_id_count": len(duplicate_cross), "cross_library_duplicate_ids_sample": duplicate_cross[:50], "total_entries": sum(x["entries"] for x in scans), "total_missing_payload": sum(x["missing_payload"] for x in scans), "total_nonfinite": sum(x["payload_nonfinite"] for x in scans), "total_wrong_shapes": sum(x["wrong_shapes"] for x in scans)}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(json.dumps({k: report[k] for k in ("total_entries", "total_missing_payload", "total_nonfinite", "total_wrong_shapes", "cross_library_duplicate_id_count")}, ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
