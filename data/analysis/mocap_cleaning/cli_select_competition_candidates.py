#!/usr/bin/env python3
"""Select coverage-oriented IK candidates from a cheap global index."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    del _ROOT

import argparse
import json
from collections import Counter
from pathlib import Path

from analysis.mocap_cleaning.competition_candidate_utils import farthest_point_select, load_index, write_selection_markdown


def _balanced_pool(rows: list[dict], preferred: list[str]) -> list[dict]:
    if not preferred:
        return rows
    by_stroke = {stroke: [row for row in rows if row["stroke_type"] == stroke] for stroke in preferred}
    ordered: list[dict] = []
    max_len = max((len(items) for items in by_stroke.values()), default=0)
    for idx in range(max_len):
        for stroke in preferred:
            items = by_stroke.get(stroke, [])
            if idx < len(items):
                ordered.append(items[idx])
    extras = [row for row in rows if row["stroke_type"] not in preferred]
    return ordered + extras


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, default=None)
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--feature-mode", choices=["simple", "full"], default="simple")
    parser.add_argument("--quality-weight", type=float, default=0.15)
    parser.add_argument("--max-per-stroke", type=int, default=0)
    args = parser.parse_args()

    index = load_index(args.index)
    selection_cfg = index.get("selection", {})
    preferred = [str(x) for x in selection_cfg.get("preferred_strokes", [])]
    rows = [row for row in index["candidates"] if row["cheap_quality_pass"]]
    rows.sort(key=lambda row: (-float(row["cheap_quality_score"]), int(row["dataset_index"])))
    if args.max_per_stroke > 0:
        kept = []
        counts = Counter()
        for row in rows:
            stroke = row["stroke_type"]
            if counts[stroke] >= args.max_per_stroke:
                continue
            kept.append(row)
            counts[stroke] += 1
        rows = kept
    selected = farthest_point_select(_balanced_pool(rows, preferred), args.count, args.feature_mode, args.quality_weight)
    report = {
        "stage": "ik_candidate_selection",
        "source_index": str(args.index),
        "selected_count": len(selected),
        "feature_mode": args.feature_mode,
        "quality_weight": args.quality_weight,
        "selected": selected,
        "stroke_counts": dict(Counter(row["stroke_type"] for row in selected)),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = args.output_summary or args.output_json.with_suffix(".md")
    write_selection_markdown(report, summary)
    print(f"Selected {len(selected)} candidates")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {summary}")


if __name__ == "__main__":
    main()
