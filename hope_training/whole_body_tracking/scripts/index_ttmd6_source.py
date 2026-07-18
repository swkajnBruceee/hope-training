#!/usr/bin/env python3
"""Build a provenance-preserving index for the local TTMD6 source pool.

The output is an audit/index artifact only. It does not normalize coordinates,
assign A3 joints, infer a hit frame, or create training files.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


NAME_RE = re.compile(
    r"^(?P<kind>bat|human)_(?P<fps>\d+)_(?P<sample>\d+)_(?P<class_id>\d+)"
    r"_(?P<group_id>\d+)_(?P<stored_len>\d+)_(?P<source_len>\d+)\.csv$"
)

CLASS_HYPOTHESIS = {
    "1": "forehand_attack",
    "2": "forehand_drive",
    "3": "forehand_push",
    "4": "backhand_attack",
    "5": "backhand_drive",
    "6": "backhand_push",
}


def inspect_csv(path: Path) -> tuple[int, int, int, bool]:
    rows = 0
    columns = None
    nonzero_rows = 0
    trailing_zero_only = True
    seen_zero = False
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream):
            rows += 1
            columns = len(row) if columns is None else columns
            if len(row) != columns:
                raise ValueError(f"inconsistent columns at row {rows}")
            zero = all(float(value) == 0.0 for value in row)
            if zero:
                seen_zero = True
            else:
                nonzero_rows += 1
                if seen_zero:
                    trailing_zero_only = False
    return rows, int(columns or 0), nonzero_rows, trailing_zero_only


def parse(path: Path) -> dict[str, str]:
    match = NAME_RE.match(path.name)
    if not match:
        raise ValueError(f"unexpected filename: {path.name}")
    return match.groupdict()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bat_dir = args.dataset / "TTMD_cut_bat"
    human_dir = args.dataset / "TTMD_cut_hum"
    records: list[dict[str, object]] = []
    status_counts = Counter()
    class_counts = Counter()

    for bat_path in sorted(bat_dir.glob("bat_*.csv")):
        suffix = bat_path.name.removeprefix("bat_")
        human_path = human_dir / f"human_{suffix}"
        record: dict[str, object] = {
            "bat_path": str(bat_path),
            "human_path": str(human_path),
            "paired": human_path.exists(),
            "source_contract": "TTMD6_local_csv_v1",
            "structurally_eligible": False,
            "training_eligible": False,
            "class_label": None,
            "class_label_status": "unconfirmed_local_mapping",
        }
        try:
            meta = parse(bat_path)
            record.update(
                {
                    "fps": int(meta["fps"]),
                    "sample_id": int(meta["sample"]),
                    "class_id": int(meta["class_id"]),
                    "group_id": int(meta["group_id"]),
                    "stored_length_declared": int(meta["stored_len"]),
                    "source_length_declared": int(meta["source_len"]),
                    "class_label": CLASS_HYPOTHESIS.get(meta["class_id"]),
                    "class_label_status": "high_confidence_inferred",
                }
            )
            class_counts[meta["class_id"]] += 1
            bat_shape = inspect_csv(bat_path)
            record["bat_rows"], record["bat_columns"], record["bat_nonzero_rows"], record["bat_trailing_zero_only"] = bat_shape
            if human_path.exists():
                human_shape = inspect_csv(human_path)
                record["human_rows"], record["human_columns"], record["human_nonzero_rows"], record["human_trailing_zero_only"] = human_shape
            else:
                human_shape = None

            expected_rows = int(meta["stored_len"])
            expected_bat = (expected_rows, 3)
            expected_human = (expected_rows, 42)
            shape_ok = (
                human_shape is not None
                and bat_shape[:2] == expected_bat
                and human_shape[:2] == expected_human
                and bat_shape[2] == human_shape[2]
                and bat_shape[3]
                and human_shape[3]
                and bat_shape[2] == min(int(meta["source_len"]), expected_rows)
            )
            record["structure_status"] = "valid" if shape_ok else "quarantine_shape_or_padding"
            record["structurally_eligible"] = bool(shape_ok)
        except (OSError, ValueError) as exc:
            record["structure_status"] = "quarantine_parse_error"
            record["error"] = str(exc)
        status_counts[str(record["structure_status"])] += 1
        records.append(record)

    report = {
        "dataset": str(args.dataset.resolve()),
        "source_contract": "TTMD6_local_csv_v1",
        "training_artifacts_written": False,
        "label_policy": "numeric class labels remain inferred until authoritative metadata/manual review",
        "counts": {
            "records": len(records),
            "class_counts": dict(sorted(class_counts.items())),
            "structure_status": dict(sorted(status_counts.items())),
            "structurally_eligible": sum(bool(record["structurally_eligible"]) for record in records),
            "training_eligible": 0,
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps({"counts": report["counts"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
