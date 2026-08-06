#!/usr/bin/env python3
"""Materialize an isolated actuator-aware pilot without relabelling its task target.

Older native-calibrated manifests preserve the original ball-contact target in
``native_calibration.original_strike_target`` but replace the active
``strike_target`` with the state reached by one local Isaac actuator profile.
This tool restores that retained target into a new, one-motion pilot manifest
and copies the NPZ.  It never changes the source manifest or source NPZ.

The output is intentionally *not* training-ready.  It is an input to a
zero-residual actuator executability test, followed by trajectory compensation
only when that test proves it necessary.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def _load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("motions"), list):
        raise ValueError(f"{path}: expected a top-level motions list")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_manifest = args.source_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    source = _load_manifest(source_manifest)
    matches = [entry for entry in source["motions"] if str(entry.get("episode_id")) == args.episode_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one entry for {args.episode_id!r}, found {len(matches)}")
    entry = dict(matches[0])
    calibration = entry.get("native_calibration")
    if not isinstance(calibration, dict) or not isinstance(calibration.get("original_strike_target"), dict):
        raise ValueError(
            f"{args.episode_id}: missing native_calibration.original_strike_target; "
            "this source cannot be safely restored by this tool"
        )
    source_npz = Path(str(entry.get("motion_npz", ""))).expanduser()
    if not source_npz.is_file():
        raise FileNotFoundError(f"{args.episode_id}: missing source NPZ {source_npz}")

    npz_dir = output_dir / "motion_npz"
    npz_dir.mkdir(parents=True, exist_ok=True)
    copied_npz = npz_dir / source_npz.name
    shutil.copy2(source_npz, copied_npz)

    relabelled_target = dict(entry.get("strike_target", {}))
    retained_target = dict(calibration["original_strike_target"])
    entry["motion_npz"] = str(copied_npz)
    entry["strike_target"] = retained_target
    entry["actuator_aware_pilot"] = {
        "status": "unvalidated_retained_task_target",
        "source_manifest": str(source_manifest),
        "source_motion_npz": str(source_npz),
        "relabelled_target_archived_for_traceability": relabelled_target,
        "retained_target_source": "native_calibration.original_strike_target",
        "rule": (
            "Do not train or promote this pilot until zero-residual replay under the declared "
            "actuator contract reaches the retained target with physical safety gates."
        ),
    }

    output = {
        "dataset_status": "actuator_aware_pilot_not_for_training",
        "purpose": "Retained original task target vs declared actuator contract",
        "source_manifest": str(source_manifest),
        "motions": [entry],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Retained-Target Actuator Pilot\n\n"
        f"- episode: `{args.episode_id}`\n"
        f"- source manifest: `{source_manifest}`\n"
        "- active target: original ball-contact/retarget target recovered from provenance\n"
        "- status: not for PPO or data-pool promotion\n\n"
        "Run zero-residual evaluation under an explicit actuator profile first. "
        "If it fails, compensate the command trajectory while retaining this target; "
        "never relabel the target to the failed replay state.\n",
        encoding="utf-8",
    )
    print(f"[pilot] wrote {manifest_path}")
    print(f"[pilot] copied {source_npz} -> {copied_npz}")


if __name__ == "__main__":
    main()
