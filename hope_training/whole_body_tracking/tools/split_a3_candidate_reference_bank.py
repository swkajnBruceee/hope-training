#!/usr/bin/env python3
"""Create deterministic, grouped train/validation manifests for the A3 bank.

Candidates with the same stroke, strike position, and paddle orientation are
kept in one split.  This prevents the 1.8 s and 2.2 s variants of a target from
leaking across train and validation while retaining the core/boundary sample
weights in both manifests.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def _group_key(row: dict[str, str]) -> str:
    values = (
        str(row["stroke"]).lower(),
        round(float(row["x_m"]), 6),
        round(float(row["y_m"]), 6),
        round(float(row["z_m"]), 6),
        round(float(row["pitch_deg"]), 4),
        round(float(row["yaw_deg"]), 4),
    )
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def _hash_fraction(seed: int, group_id: str) -> float:
    digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--weighted-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be between 0 and 1")

    source_manifest = args.source_manifest.expanduser().resolve()
    weighted_index = args.weighted_index.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    entries = list(source.get("motions", []))
    if not entries:
        raise ValueError("source manifest has no motions")
    with weighted_index.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(entries):
        raise ValueError(f"source/index length mismatch: manifest={len(entries)} index={len(rows)}")

    groups: dict[str, list[int]] = defaultdict(list)
    group_ids: dict[str, str] = {}
    for index, (entry, row) in enumerate(zip(entries, rows)):
        expected_id = f"candidate_{index:05d}"
        if str(entry.get("episode_id")) != expected_id:
            raise ValueError(f"manifest order/id mismatch at row {index}: {entry.get('episode_id')!r} != {expected_id!r}")
        key = _group_key(row)
        group_id = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        groups[group_id].append(index)
        group_ids[expected_id] = group_id

    split_by_group = {
        group_id: "validation" if _hash_fraction(args.seed, group_id) < args.validation_fraction else "training"
        for group_id in groups
    }
    split_entries: dict[str, list[dict]] = {"training": [], "validation": []}
    split_group_ids: dict[str, set[str]] = {"training": set(), "validation": set()}
    for index, (entry, row) in enumerate(zip(entries, rows)):
        episode_id = str(entry["episode_id"])
        group_id = group_ids[episode_id]
        split = split_by_group[group_id]
        copied = copy.deepcopy(entry)
        copied["split"] = split
        copied["split_group_id"] = group_id
        copied["split_group_key"] = _group_key(row)
        copied["split_seed"] = int(args.seed)
        copied["validation_fraction"] = float(args.validation_fraction)
        split_entries[split].append(copied)
        split_group_ids[split].add(group_id)

    overlap = split_group_ids["training"] & split_group_ids["validation"]
    if overlap:
        raise RuntimeError(f"group leakage detected: {len(overlap)} groups")

    output_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "schema_version": "a3_weighted_candidate_reference_bank_split/v1",
        "status": "candidate_reference_pending_training_split",
        "source_manifest": str(source_manifest),
        "source_index": str(weighted_index),
        "notice_contract": "project_manifest",
        "reference_semantics": source.get("reference_semantics"),
        "physics_qualified": False,
        "teacher_approved": False,
        "training_admission": False,
        "floating_base_replay_done": False,
        "self_collision_observable": False,
        "weights": source.get("weights", {"core": 1.0, "boundary": 0.25}),
        "split_contract": {
            "method": "deterministic_sha256_group_hash",
            "group_fields": ["stroke", "x_m", "y_m", "z_m", "pitch_deg", "yaw_deg"],
            "rounded_position_decimals": 6,
            "rounded_orientation_decimals": 4,
            "validation_fraction_target": float(args.validation_fraction),
            "seed": int(args.seed),
            "no_group_overlap": True,
        },
    }

    for split in ("training", "validation"):
        payload = dict(common)
        payload["split"] = split
        payload["motion_count"] = len(split_entries[split])
        payload["group_count"] = len(split_group_ids[split])
        payload["counts"] = {
            "motions": len(split_entries[split]),
            "core": sum(e.get("dataset_role") == "core" for e in split_entries[split]),
            "boundary": sum(e.get("dataset_role") == "boundary" for e in split_entries[split]),
            "backhand": sum(e.get("stroke_type") == "backhand" for e in split_entries[split]),
            "forehand": sum(e.get("stroke_type") == "forehand" for e in split_entries[split]),
        }
        payload["motions"] = split_entries[split]
        (output_dir / f"{split}_manifest.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    report = {
        "schema_version": "a3_candidate_reference_split_report/v1",
        "source_manifest": str(source_manifest),
        "source_index": str(weighted_index),
        "seed": int(args.seed),
        "validation_fraction_target": float(args.validation_fraction),
        "total_motions": len(entries),
        "total_groups": len(groups),
        "group_size_distribution": dict(sorted(Counter(map(len, groups.values())).items())),
        "split_counts": {
            split: {
                "motions": len(split_entries[split]),
                "groups": len(split_group_ids[split]),
                "by_stroke_role": dict(
                    sorted(
                        (
                            f"{stroke}/{role}",
                            count,
                        )
                        for (stroke, role), count in Counter(
                            (e["stroke_type"], e["dataset_role"]) for e in split_entries[split]
                        ).items()
                    )
                ),
            }
            for split in ("training", "validation")
        },
        "group_overlap_count": len(overlap),
        "status": "completed",
    }
    (output_dir / "split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
