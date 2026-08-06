#!/usr/bin/env python3
"""Select a balanced, provenance-preserving TTMD6 intake batch.

This creates an intake manifest only. It does not claim that the selected
clips are A3 retargetable or training eligible. The numeric class-to-stroke
mapping is retained as inferred metadata and must be confirmed during the
retarget/replay review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PILOT_IDS = {
    "class1_sample1",
    "class1_sample301",
    "class1_sample601",
    "class1_sample901",
    "class1_sample1201",
    "class2_sample51",
    "class2_sample351",
    "class2_sample651",
    "class2_sample951",
    "class2_sample1251",
    "class3_sample101",
    "class3_sample401",
    "class3_sample701",
    "class3_sample1001",
    "class3_sample1301",
    "class4_sample151",
    "class5_sample201",
    "class6_sample251",
}


def source_id(record: dict) -> str:
    return f"class{int(record['class_id'])}_sample{int(record['sample_id'])}"


def evenly_spaced(records: list[dict], count: int) -> list[dict]:
    records = sorted(records, key=lambda item: (int(item["sample_id"]), int(item["group_id"])))
    if len(records) < count:
        raise ValueError(f"class {records[0]['class_id']} has only {len(records)} eligible records")
    if count == 1:
        indices = [len(records) // 2]
    else:
        indices = [round(i * (len(records) - 1) / (count - 1)) for i in range(count)]
    return [records[index] for index in indices]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_index", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=4)
    args = parser.parse_args()

    index = json.loads(args.source_index.read_text(encoding="utf-8"))
    by_class: dict[int, list[dict]] = {class_id: [] for class_id in range(1, 7)}
    excluded = []
    for record in index["records"]:
        sid = source_id(record)
        if not record.get("structurally_eligible", False):
            continue
        if sid in PILOT_IDS:
            excluded.append({"source_id": sid, "reason": "pilot_already_processed"})
            continue
        by_class[int(record["class_id"])].append(record)

    selected = []
    for class_id in range(1, 7):
        for record in evenly_spaced(by_class[class_id], args.per_class):
            selected.append(
                {
                    "source_id": source_id(record),
                    "class_id": int(record["class_id"]),
                    "class_label": record["class_label"],
                    "class_label_status": record.get("class_label_status"),
                    "sample_id": int(record["sample_id"]),
                    "group_id": int(record["group_id"]),
                    "human_path": record["human_path"],
                    "bat_path": record["bat_path"],
                    "fps": int(record["fps"]),
                    "source_length_declared": int(record["source_length_declared"]),
                    "stored_length_declared": int(record["stored_length_declared"]),
                    "structure_status": record["structure_status"],
                    "source_audit_status": "structurally_eligible_inferred_label",
                    "coordinate_status": "locked_contract_not_yet_applied",
                    "hit_frame_status": "candidate_only",
                    "retarget_status": "not_started",
                    "training_eligible": False,
                }
            )

    output = {
        "dataset": "TTMD6",
        "stage": "a3_expansion_source_intake_v1",
        "source_index": str(args.source_index),
        "selection_policy": {
            "per_class": args.per_class,
            "class_balance": "4 clips per numeric class; classes 1-3 inferred forehand, 4-6 inferred backhand",
            "selection": "evenly spaced by sample_id after structural filtering",
            "pilot_excluded": sorted(PILOT_IDS),
        },
        "source_contract": "TTMD6_local_csv_v1",
        "coordinate_contract": "data/analysis/mocap_cleaning/ttmd6_a3_coordinate_contract_v1.yaml",
        "label_policy": "numeric class labels remain inferred until authoritative metadata/manual review",
        "training_eligible": False,
        "record_count": len(selected),
        "class_counts": {str(class_id): sum(item["class_id"] == class_id for item in selected) for class_id in range(1, 7)},
        "excluded_pilot_count": len(excluded),
        "records": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"record_count": len(selected), "class_counts": output["class_counts"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
