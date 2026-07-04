#!/usr/bin/env python3
"""Detect hit frames in cleaned Tennis windows."""

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
from pathlib import Path

import numpy as np

from analysis.mocap_cleaning.config import load_config
from analysis.mocap_cleaning.hit_detection import detect_hit_index


def _cleaning_usability_by_npz(report_path: Path) -> dict[str, bool]:
    if not report_path.exists():
        return {}
    report = json.loads(report_path.read_text())
    return {item["npz_path"]: bool(item["cleaning"]["usable"]) for item in report.get("windows", [])}


def _write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# DATA260703 Hit Detection Report",
        "",
        f"Input: `{report['cleaned_windows_dir']}`",
        "",
        "| Episode | Racket | Clean Usable | Valid Hit | Usable For Hit | Hit t_rel (s) | Dist (m) | Racket Speed | Ball dV | Score | Reason |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in report["hits"]:
        md = item["hit"]
        hit_time = "nan" if md["hit_time_rel"] is None else f"{md['hit_time_rel']:.3f}"
        dist = "nan" if md["dist_at_hit_m"] is None else f"{md['dist_at_hit_m']:.3f}"
        rs = "nan" if md["racket_speed_at_hit_mps"] is None else f"{md['racket_speed_at_hit_mps']:.3f}"
        dv = "nan" if md["ball_dv_at_hit_mps"] is None else f"{md['ball_dv_at_hit_mps']:.3f}"
        score = "nan" if md["score_at_hit"] is None else f"{md['score_at_hit']:.3f}"
        lines.append(
            f"| `{item['episode_id']}` | {item['racket']} | {item['cleaning_usable']} | "
            f"{md['valid_hit']} | {item['usable_for_hit_analysis']} | {hit_time} | {dist} | {rs} | {dv} | {score} | {md['reason']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("data/analysis/mocap_cleaning/configs/DATA260703.yaml"))
    parser.add_argument(
        "--cleaned-windows-dir",
        type=Path,
        default=Path("data/analysis/mocap_cleaning_outputs/DATA260703/cleaned_windows"),
    )
    parser.add_argument(
        "--cleaned-report",
        type=Path,
        default=Path("data/analysis/mocap_cleaning_outputs/DATA260703/cleaned_tennis_windows_report.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/analysis/mocap_cleaning_outputs/DATA260703"))
    args = parser.parse_args()

    config = load_config(args.config)
    hit_cfg = config["hit_detection"]
    cleaning_usable = _cleaning_usability_by_npz(args.cleaned_report)

    hit_dir = args.output_dir / "hit_debug"
    hit_dir.mkdir(parents=True, exist_ok=True)
    hits = []
    for npz_path in sorted(args.cleaned_windows_dir.glob("*_cleaned.npz")):
        data = np.load(npz_path, allow_pickle=True)
        episode_id = str(data["episode_id"])
        result = detect_hit_index(
            time=data["time"],
            ball_pos=data["ball_pos_clean"],
            ball_vel=data["ball_vel"],
            racket_pos=data["racket_pos"],
            max_distance_m=float(hit_cfg["max_distance_m"]),
            distance_ok_m=float(hit_cfg["distance_ok_m"]),
            min_racket_speed_mps=float(hit_cfg["min_racket_speed_mps"]),
            min_ball_dv_mps=float(hit_cfg["min_ball_dv_mps"]),
            weights=hit_cfg["weights"],
        )
        debug_path = hit_dir / f"{episode_id}_hit_debug.npz"
        np.savez(
            debug_path,
            episode_id=np.asarray(episode_id),
            time=data["time"],
            time_rel=data["time_rel"],
            ball_pos=data["ball_pos_clean"],
            ball_vel=data["ball_vel"],
            racket_pos=data["racket_pos"],
            **result.debug,
        )
        cleaning_ok = cleaning_usable.get(str(npz_path), True)
        hits.append(
            {
                "episode_id": episode_id,
                "source_npz": str(npz_path),
                "debug_npz": str(debug_path),
                "source_csv": str(data["source_csv"]),
                "source_bvh": str(data["source_bvh"]),
                "candidate": str(data["candidate"]),
                "racket": str(data["racket"]),
                "cleaning_usable": cleaning_ok,
                "usable_for_hit_analysis": bool(cleaning_ok and result.valid_hit),
                "hit": result.metadata(),
            }
        )

    report = {
        "config": str(args.config),
        "cleaned_windows_dir": str(args.cleaned_windows_dir),
        "cleaned_report": str(args.cleaned_report),
        "hits": hits,
    }
    json_path = args.output_dir / "hit_detection_report.json"
    md_path = args.output_dir / "hit_detection_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    _write_markdown(report, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {len(hits)} hit debug files to {hit_dir}")


if __name__ == "__main__":
    main()
