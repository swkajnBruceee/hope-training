#!/usr/bin/env python3
"""Validate the DATA260703 Motive CSV loader on one trial."""

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
from statistics import median

import numpy as np

from analysis.mocap_cleaning.config import load_config
from analysis.mocap_cleaning.motive_loader import load_motive_csv, read_motive_header, list_entities


def _dist_m(a: np.ndarray, b: np.ndarray, unit: str) -> np.ndarray:
    scale = {"millimeters": 0.001, "mm": 0.001, "centimeters": 0.01, "cm": 0.01, "meters": 1.0, "m": 1.0}
    return np.linalg.norm(a - b, axis=1) * scale.get(unit.lower(), 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("analysis/mocap_cleaning/configs/DATA260703.yaml"))
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("/workspace/DATA260703/Csv/Rige Body/Table Tennis_01_005.csv"),
    )
    parser.add_argument("--racket", default="TennisBats01")
    parser.add_argument("--skeleton", default="Skeleton 001")
    parser.add_argument("--output", type=Path, default=Path("analysis/mocap_cleaning_outputs/DATA260703/loader_validation.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    hand = f"{args.skeleton}:{config['entities']['rackets'][args.racket]['expected_hand']}"
    bones = [hand, f"{args.skeleton}:Hip", f"{args.skeleton}:LHand"]
    rigid_bodies = list(config["entities"]["rackets"].keys()) + list(config["entities"]["ball_candidates"])

    header = read_motive_header(args.csv)
    entities = list_entities(header)
    trial = load_motive_csv(args.csv, rigid_bodies=rigid_bodies, bones=bones)

    time = trial.time
    time_ok = bool(len(time) > 10 and np.all(np.diff(time) > 0))
    racket = trial.rigid_bodies[args.racket]
    hand_pose = trial.bones[hand]
    distances = _dist_m(racket.pos, hand_pose.pos, trial.position_unit)

    result = {
        "config": str(args.config),
        "csv": str(args.csv),
        "take_name": trial.take_name,
        "fps": trial.fps,
        "frames": int(len(time)),
        "time_monotonic": time_ok,
        "position_unit": trial.position_unit,
        "coordinate_space": trial.coordinate_space,
        "quat_order": trial.metadata["quat_order"],
        "available_entity_counts": {k: len(v) for k, v in entities.items()},
        "loaded_rigid_bodies": {
            name: {
                "pos_shape": list(pose.pos.shape),
                "quat_shape": list(pose.quat_xyzw.shape) if pose.quat_xyzw is not None else None,
            }
            for name, pose in trial.rigid_bodies.items()
        },
        "loaded_bones": {
            name: {
                "pos_shape": list(pose.pos.shape),
                "quat_shape": list(pose.quat_xyzw.shape) if pose.quat_xyzw is not None else None,
            }
            for name, pose in trial.bones.items()
        },
        "racket_hand_match": {
            "racket": args.racket,
            "hand": hand,
            "median_distance_m": float(median(distances)),
            "min_distance_m": float(np.nanmin(distances)),
            "max_distance_m": float(np.nanmax(distances)),
            "ok": bool(float(median(distances)) < 0.35),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
