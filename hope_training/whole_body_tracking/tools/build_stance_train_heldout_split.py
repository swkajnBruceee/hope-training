#!/usr/bin/env python3
"""Build explicit train/held-out manifests for the stance-aware experiment.

The input manifests are never modified.  Prepositioned entries retain their
world-hit and base-offset metadata; fixed entries receive explicit fixed-mode
metadata using the same contract as the mixed held-out builder.

This tool exists to prevent reusing the four-motion stance held-out pool as
training data while still calling it independent evidence.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from build_mixed_stance_manifest import _fixed_metadata


def _resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _ids(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("an episode-id list cannot be empty")
    return result


def _load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("motions"), list):
        raise ValueError(f"expected manifest with motions list: {path}")
    return data


def _select(source: dict, ids: list[str], label: str, source_path: Path) -> list[dict]:
    by_id = {str(item["episode_id"]): item for item in source["motions"]}
    missing = [item for item in ids if item not in by_id]
    if missing:
        raise ValueError(f"{label}: ids not found in {source_path}: {missing}")
    return [copy.deepcopy(by_id[item]) for item in ids]


def _with_fixed_metadata(entries: list[dict], source_path: Path) -> list[dict]:
    result = []
    for entry in entries:
        entry["stance_metadata"] = _fixed_metadata(entry, source_path)
        result.append(entry)
    return result


def _make_manifest(
    *,
    name: str,
    role: str,
    entries: list[dict],
    sources: list[Path],
) -> dict:
    ids = [str(item["episode_id"]) for item in entries]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate episode IDs in {name}")
    return {
        "manifest_name": name,
        "status": "active_training_candidate" if role == "train" else "heldout_evidence_only",
        "training_role": "training_input" if role == "train" else "heldout_not_training",
        "dataset_tier": "stance_train" if role == "train" else "stance_heldout",
        "source_manifests": [str(path) for path in sources],
        "replay_ready_count": len(entries),
        "stroke_counts": {
            "forehand": sum(str(e.get("stroke_type", "")).lower() == "forehand" for e in entries),
            "backhand": sum(str(e.get("stroke_type", "")).lower() == "backhand" for e in entries),
        },
        "stance_contract": {
            "mode": "mixed",
            "allowed_modes": ["fixed", "prepositioned"],
            "walking_enabled": False,
        },
        "motions": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepositioned-manifest", type=Path, required=True)
    parser.add_argument("--fixed-train-manifest", type=Path, required=True)
    parser.add_argument("--fixed-heldout-manifest", type=Path, required=True)
    parser.add_argument("--prepositioned-train-ids", required=True)
    parser.add_argument("--prepositioned-heldout-ids", required=True)
    parser.add_argument("--fixed-train-ids", required=True)
    parser.add_argument("--fixed-heldout-ids", required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--heldout-output", type=Path, required=True)
    args = parser.parse_args()

    pre_path = _resolve(args.prepositioned_manifest)
    fixed_train_path = _resolve(args.fixed_train_manifest)
    fixed_holdout_path = _resolve(args.fixed_heldout_manifest)
    pre = _load(pre_path)
    fixed_train = _load(fixed_train_path)
    fixed_holdout = _load(fixed_holdout_path)

    pre_train_ids = _ids(args.prepositioned_train_ids)
    pre_holdout_ids = _ids(args.prepositioned_heldout_ids)
    fixed_train_ids = _ids(args.fixed_train_ids)
    fixed_holdout_ids = _ids(args.fixed_heldout_ids)
    all_ids = pre_train_ids + pre_holdout_ids + fixed_train_ids + fixed_holdout_ids
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("train/held-out episode IDs overlap")

    pre_train = _select(pre, pre_train_ids, "prepositioned_train", pre_path)
    pre_holdout = _select(pre, pre_holdout_ids, "prepositioned_heldout", pre_path)
    fixed_train_entries = _with_fixed_metadata(
        _select(fixed_train, fixed_train_ids, "fixed_train", fixed_train_path), fixed_train_path
    )
    fixed_holdout_entries = _with_fixed_metadata(
        _select(fixed_holdout, fixed_holdout_ids, "fixed_heldout", fixed_holdout_path), fixed_holdout_path
    )

    train_entries = pre_train + fixed_train_entries
    heldout_entries = pre_holdout + fixed_holdout_entries
    train = _make_manifest(
        name="p2_stance_train_k8_v1_20260716",
        role="train",
        entries=train_entries,
        sources=[pre_path, fixed_train_path],
    )
    heldout = _make_manifest(
        name="p2_stance_heldout_k4_v1_20260716",
        role="heldout",
        entries=heldout_entries,
        sources=[pre_path, fixed_holdout_path],
    )
    for output, payload in ((args.train_output, train), (args.heldout_output, heldout)):
        output = _resolve(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {payload['manifest_name']}: {output} ({len(payload['motions'])} motions)")


if __name__ == "__main__":
    main()
