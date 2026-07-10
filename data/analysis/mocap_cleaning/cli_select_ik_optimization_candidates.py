#!/usr/bin/env python3
"""Select a smaller optimization manifest from IK-pass candidates."""

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


def _ik_score(item: dict) -> float:
    pos = float(item.get("racket_position_error_at_hit_m", 1.0))
    normal = float(item.get("racket_orientation_error_at_hit_deg", 180.0))
    tangent = float(item.get("racket_tangent_error_at_hit_deg", 180.0))
    return max(0.0, 2.0 - 20.0 * pos - 0.02 * normal - 0.005 * tangent)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-index", type=Path, required=True)
    parser.add_argument("--ik-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-selection", type=Path, default=None)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--feature-mode", choices=["simple", "full"], default="simple")
    parser.add_argument("--quality-weight", type=float, default=0.25)
    args = parser.parse_args()

    index = load_index(args.candidate_index)
    row_by_episode = {row["episode_id"]: row for row in index["candidates"]}
    ik_manifest = json.loads(args.ik_manifest.read_text(encoding="utf-8"))
    rows = []
    item_by_episode = {}
    for item in ik_manifest["samples"]:
        if item.get("ik_status") != "pass":
            continue
        row = dict(row_by_episode.get(item["episode_id"], {}))
        if not row:
            continue
        row["ik_quality_score"] = _ik_score(item)
        row["cheap_quality_score"] = float(row.get("cheap_quality_score", 0.0)) + row["ik_quality_score"]
        rows.append(row)
        item_by_episode[item["episode_id"]] = item
    rows.sort(key=lambda row: (-float(row["cheap_quality_score"]), int(row["dataset_index"])))
    selected = farthest_point_select(rows, args.count, args.feature_mode, args.quality_weight)
    selected_ids = {row["episode_id"] for row in selected}
    filtered_samples = [item_by_episode[row["episode_id"]] for row in selected if row["episode_id"] in selected_ids]
    out_manifest = {
        **ik_manifest,
        "source_ik_manifest": str(args.ik_manifest),
        "selection_stage": "optimization_candidate_selection",
        "selected_count": len(filtered_samples),
        "samples": filtered_samples,
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(out_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    selection_report = {
        "stage": "optimization_candidate_selection",
        "source_index": str(args.candidate_index),
        "source_ik_manifest": str(args.ik_manifest),
        "selected_count": len(selected),
        "feature_mode": args.feature_mode,
        "quality_weight": args.quality_weight,
        "stroke_counts": dict(Counter(row["stroke_type"] for row in selected)),
        "selected": selected,
    }
    selection_path = args.output_selection or args.output_manifest.with_name("optimization_candidate_selection.json")
    selection_path.write_text(json.dumps(selection_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_selection_markdown(selection_report, selection_path.with_suffix(".md"))
    print(f"Selected {len(filtered_samples)} optimization candidates")
    print(f"Wrote {args.output_manifest}")
    print(f"Wrote {selection_path}")


if __name__ == "__main__":
    main()
