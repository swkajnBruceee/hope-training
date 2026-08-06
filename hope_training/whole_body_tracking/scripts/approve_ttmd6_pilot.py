#!/usr/bin/env python3
"""Record manual approval for the bounded TTMD6 pilot set.

Manual approval is deliberately not training admission. The output remains a
source-review manifest until TTMD6-specific retargeting and A3 replay gates
have passed.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewer", default="human")
    args = parser.parse_args()

    payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not records:
        raise ValueError("candidate file contains no records")

    seen: set[tuple[int, int]] = set()
    approved = []
    for record in records:
        key = (int(record["class_id"]), int(record["sample_id"]))
        if key in seen:
            raise ValueError(f"duplicate pilot record: {key}")
        seen.add(key)
        for field in ("human_path", "bat_path"):
            path = Path(record[field])
            if not path.is_file():
                raise FileNotFoundError(path)
        approved_record = dict(record)
        approved_record.update(
            {
                "manual_review_status": "manual_pass",
                "manual_review_reviewer": args.reviewer,
                "manual_review_note": "Pilot clip visually reviewed; no blocking issue observed.",
                "training_eligible": False,
                "retarget_status": "pending",
                "a3_replay_status": "pending",
            }
        )
        approved.append(approved_record)

    output = {
        "dataset": "TTMD6",
        "purpose": "manual_reviewed_source_pilot",
        "review_scope": "30 pilot clips only; not a full-dataset approval",
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "reviewer": args.reviewer,
        "manual_review_status": "complete",
        "training_eligible": False,
        "retarget_status": "pending",
        "a3_replay_status": "pending",
        "records": approved,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(f"approved {len(approved)} manual-review records -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
