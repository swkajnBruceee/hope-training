#!/usr/bin/env python3
"""Build tracking/training motion manifests from optimized fixed-base outputs."""

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
from collections import Counter, defaultdict
from pathlib import Path


def _load_strike_metadata(npz_entry: dict, sample: dict) -> dict:
    metadata = {}
    if npz_entry.get("hit_event"):
        metadata["hit_event"] = npz_entry["hit_event"]
    if npz_entry.get("strike_target"):
        metadata["strike_target"] = npz_entry["strike_target"]
    if metadata:
        return metadata

    spec_path = Path(str(sample.get("target_spec_json", "")))
    if not spec_path.exists():
        return {}
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    contract = spec.get("coordinate_contract", {})
    hit_target = spec.get("hit_target", {})
    source_fps = float(contract.get("fps", 0.0) or 0.0)
    hit_index = int(contract.get("hit_index", 0) or 0)
    motion_fps = int(npz_entry.get("fps", 50))
    hit_time_from_start_s = float(hit_index / source_fps) if source_fps > 0 else None
    hit_frame_float = float(hit_time_from_start_s * motion_fps) if hit_time_from_start_s is not None else None
    hit_frame = int(hit_frame_float) if hit_frame_float is not None else None
    return {
        "hit_event": {
            "source_hit_index": hit_index,
            "source_fps": source_fps,
            "hit_time_from_start_s": hit_time_from_start_s,
            "motion_fps": motion_fps,
            "motion_hit_frame": hit_frame,
            "motion_hit_subframe_alpha": float(hit_frame_float - hit_frame) if hit_frame_float is not None and hit_frame is not None else None,
        },
        "strike_target": {
            key: hit_target[key]
            for key in (
                "racket_position_m",
                "racket_quat_xyzw",
                "racket_normal_w",
                "racket_tangent_w",
                "racket_velocity_mps",
                "racket_velocity_direction_w",
                "ball_position_m",
                "ball_in_velocity_mps",
                "ball_out_velocity_mps",
                "ball_to_racket_center_distance_m",
            )
            if key in hit_target
        },
    }


def _write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Fixed-Base Tracking Motion Manifest",
        "",
        f"- source manifest: `{report['source_manifest']}`",
        f"- replay-ready motions: `{report['replay_ready_count']}`",
        "",
        "## Stroke Counts",
        "",
    ]
    for stroke, count in sorted(report["stroke_counts"].items()):
        lines.append(f"- `{stroke}`: {count}")
    lines.extend(["", "## Smoke Picks", ""])
    for stroke, episode_id in sorted(report["smoke_picks"].items()):
        lines.append(f"- `{stroke}`: `{episode_id}`")
    lines.extend(["", "## Outputs", ""])
    lines.append(f"- manifest: `{report['manifest_json']}`")
    lines.append(f"- forehand manifest: `{report['forehand_json']}`")
    lines.append(f"- backhand manifest: `{report['backhand_json']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optimized-manifest",
        type=Path,
        default=Path("data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/retarget_p2_fixed_a3/optimized_manifest.json"),
    )
    parser.add_argument(
        "--motion-npz-manifest",
        type=Path,
        default=Path("data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/retarget_p2_fixed_a3/optimized_motion_npz_manifest.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    optimized = json.loads(args.optimized_manifest.read_text(encoding="utf-8"))
    motion_npz = json.loads(args.motion_npz_manifest.read_text(encoding="utf-8"))
    output_dir = args.output_dir or args.optimized_manifest.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_by_episode = {item["episode_id"]: item for item in motion_npz["entries"]}
    grouped: dict[str, list[dict]] = defaultdict(list)
    entries: list[dict] = []
    for sample in optimized["samples"]:
        if not sample.get("replay_ready", False):
            continue
        episode_id = str(sample["episode_id"])
        npz_entry = npz_by_episode.get(episode_id)
        if npz_entry is None:
            continue
        entry = {
            "episode_id": episode_id,
            "stroke_type": str(sample["stroke_type_rule_v2"]),
            "stroke_confidence": float(sample.get("stroke_confidence_rule_v2", 1.0)),
            "selection_note": str(sample.get("selection_note", "manual_or_probe_manifest")),
            "motion_npz": str(npz_entry["motion_npz"]),
            "fps": int(npz_entry["fps"]),
            "joint_pos_shape": npz_entry["joint_pos_shape"],
            "body_pos_w_shape": npz_entry["body_pos_w_shape"],
            "optimized_csv": str(sample["optimized_csv"]),
            "target_npz": str(sample["target_npz"]),
            "target_spec_json": str(sample["target_spec_json"]),
            **_load_strike_metadata(npz_entry, sample),
        }
        entries.append(entry)
        grouped[entry["stroke_type"]].append(entry)

    for items in grouped.values():
        items.sort(key=lambda item: item["episode_id"])
    entries.sort(key=lambda item: (item["stroke_type"], item["episode_id"]))

    manifest = {
        "source_manifest": str(args.optimized_manifest),
        "source_motion_npz_manifest": str(args.motion_npz_manifest),
        "replay_ready_count": len(entries),
        "stroke_counts": dict(Counter(item["stroke_type"] for item in entries)),
        "smoke_picks": {stroke: items[0]["episode_id"] for stroke, items in grouped.items() if items},
        "motions": entries,
    }
    forehand_manifest = {
        **manifest,
        "stroke": "forehand",
        "motions": grouped.get("forehand", []),
    }
    backhand_manifest = {
        **manifest,
        "stroke": "backhand",
        "motions": grouped.get("backhand", []),
    }

    manifest_json = output_dir / "tracking_motion_manifest.json"
    forehand_json = output_dir / "tracking_motion_manifest_forehand.json"
    backhand_json = output_dir / "tracking_motion_manifest_backhand.json"
    summary_md = output_dir / "tracking_motion_manifest.md"
    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    forehand_json.write_text(json.dumps(forehand_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    backhand_json.write_text(json.dumps(backhand_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_markdown(
        {
            **manifest,
            "manifest_json": str(manifest_json),
            "forehand_json": str(forehand_json),
            "backhand_json": str(backhand_json),
        },
        summary_md,
    )
    print(f"Wrote {manifest_json}")
    print(f"Wrote {forehand_json}")
    print(f"Wrote {backhand_json}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
