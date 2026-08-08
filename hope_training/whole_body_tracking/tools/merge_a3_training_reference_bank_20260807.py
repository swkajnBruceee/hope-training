#!/usr/bin/env python3
"""Merge the audited 2026-08-07 generalization bank into the A3 split bank."""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path


ROOT = Path("/home/bistu/桌面/HOPETableTennis").resolve()
OLD = ROOT / "a3_ik_point_offline_wrapper_v2/training_reference_bank_20260806"
NEW = ROOT / "a3_ik_point_offline_wrapper_v2/generalization_candidate_bank_20260807/audited_fixed_base_20260807"
OUT = ROOT / "a3_ik_point_offline_wrapper_v2/training_reference_bank_merged_20260807"


def load(name: str, directory: Path) -> dict:
    return json.loads((directory / name).read_text(encoding="utf-8"))


def absolutize(entry: dict, manifest_dir: Path, prefix: str | None = None) -> dict:
    item = copy.deepcopy(entry)
    old_id = str(item["episode_id"])
    if prefix:
        item["episode_id"] = f"{prefix}_{old_id}"
        item["motion_id"] = f"{prefix}_{str(item['motion_id'])}"
        item["source_episode_id"] = old_id
    for key in ("motion_npz", "library_motion_npz"):
        value = item.get(key)
        if isinstance(value, str):
            path = Path(value)
            if not path.is_absolute():
                path = manifest_dir / path
            item[key] = str(path.resolve())
    return item


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_groups: set[str] = set()
    outputs: dict[str, dict] = {}
    for split in ("training", "validation"):
        old = load(f"{split}_manifest.json", OLD)
        new = load(f"{split}_manifest.json", NEW)
        old_entries = [absolutize(e, OLD) for e in old["motions"]]
        new_entries = [absolutize(e, NEW, "gen20260807") for e in new["motions"]]
        entries = old_entries + new_entries
        ids = [str(e["episode_id"]) for e in entries]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"duplicate episode_id in {split}")
        groups = [str(e.get("split_group_id", "")) for e in entries]
        if any(not g for g in groups):
            raise RuntimeError(f"missing split_group_id in {split}")
        overlap = all_groups.intersection(groups)
        if overlap:
            raise RuntimeError(f"cross-split group leakage: {sorted(overlap)[:5]}")
        all_groups.update(groups)
        payload = copy.deepcopy(old)
        payload["status"] = "merged_candidate_reference_pending_training_runtime"
        payload["source_manifest"] = [str((OLD / f"{split}_manifest.json").resolve()), str((NEW / f"{split}_manifest.json").resolve())]
        payload["motion_count"] = len(entries)
        payload["group_count"] = len(set(groups))
        payload["counts"] = {
            "motions": len(entries),
            "core": sum(e.get("dataset_role") == "core" for e in entries),
            "boundary": sum(e.get("dataset_role") == "boundary" for e in entries),
            "backhand": sum(e.get("stroke_type") == "backhand" for e in entries),
            "forehand": sum(e.get("stroke_type") == "forehand" for e in entries),
            "original": len(old_entries),
            "generalization_added": len(new_entries),
        }
        payload["motions"] = entries
        (OUT / f"{split}_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        outputs[split] = payload["counts"]

    report = {
        "status": "completed",
        "source_original": str(OLD),
        "source_generalization": str(NEW),
        "output_dir": str(OUT),
        "counts": outputs,
        "total_added": 661,
        "policy": "fixed-base-audited generalization candidates admitted; floating-base replay failures remain intentionally ignored per plan",
    }
    (OUT / "merge_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
