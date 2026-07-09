#!/usr/bin/env python3
"""Publish replay-ready motion NPZ files into an ASCII-only training library."""

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
import os
from pathlib import Path


def _safe_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    rel = os.path.relpath(src, start=dst.parent)
    dst.symlink_to(rel)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/retarget_p2_fixed_a3/tracking_motion_manifest.json"),
    )
    parser.add_argument(
        "--library-root",
        type=Path,
        default=Path("hope_training/whole_body_tracking/sample_motions/p2_fixed_competition"),
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    library_root = args.library_root
    entries = []
    for item in manifest["motions"]:
        stroke = str(item["stroke_type"])
        episode_id = str(item["episode_id"])
        src = Path(item["motion_npz"]).resolve()
        dst = library_root / stroke / f"{episode_id}.npz"
        _safe_link(src, dst)
        entries.append(
            {
                **item,
                "library_motion_npz": str(dst),
            }
        )

    out_manifest = {
        **manifest,
        "library_root": str(library_root),
        "motions": entries,
    }
    out_json = library_root / "manifest.json"
    out_md = library_root / "manifest.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Tracking Motion Library",
        "",
        f"- source manifest: `{args.manifest}`",
        f"- library root: `{library_root}`",
        f"- motion count: `{len(entries)}`",
        "",
        "## Motions",
        "",
    ]
    for item in entries:
        lines.append(f"- `{item['stroke_type']}`: `{item['episode_id']}` -> `{item['library_motion_npz']}`")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
