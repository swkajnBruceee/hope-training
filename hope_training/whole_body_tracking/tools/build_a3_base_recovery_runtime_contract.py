#!/usr/bin/env python3
"""Freeze source, asset, trace, and git hashes for a Recovery-A evidence run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEVANT_SOURCES = (
    "tools/run_a3_base_stand_recovery_a_calibration.py",
    "tools/analyze_a3_base_recovery_calibration.py",
    "training/robots/agibot_a3.py",
    "training/tasks/base_locomotion/base_env_cfg.py",
    "training/tasks/base_locomotion/config/a3/stand_env_cfg.py",
    "training/tasks/base_locomotion/mdp/actions.py",
    "training/tasks/base_locomotion/mdp/events.py",
    "training/tasks/base_locomotion/mdp/observations.py",
    "training/tasks/base_locomotion/mdp/rewards.py",
    "training/tasks/base_locomotion/mdp/terminations.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str, binary: bool = False) -> str | bytes:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=not binary,
    ).stdout


def _tree_hash(root: Path) -> tuple[str, list[dict[str, object]]]:
    records = []
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        file_sha = _sha256(path)
        size = path.stat().st_size
        records.append({"path": relative, "sha256": file_sha, "size_bytes": size})
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest(), records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--trace-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    trace = args.trace.expanduser().resolve()
    trace_manifest = args.trace_manifest.expanduser().resolve()
    asset_root = ROOT / "training/assets/agibot_a3"
    required = [trace, trace_manifest, asset_root, *(ROOT / path for path in RELEVANT_SOURCES)]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Recovery runtime contract inputs are missing: {missing}")

    asset_tree_sha, asset_files = _tree_hash(asset_root)
    tracked_diff = _git("diff", "--binary", binary=True)
    staged_diff = _git("diff", "--binary", "--cached", binary=True)
    status = _git("status", "--porcelain=v1", "-z", binary=True)
    result = {
        "schema_version": 1,
        "contract_id": "a3_base_recovery_runtime_contract_v1",
        "git": {
            "commit": _git("rev-parse", "HEAD").strip(),
            "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
            "staged_diff_sha256": hashlib.sha256(staged_diff).hexdigest(),
            "porcelain_status_sha256": hashlib.sha256(status).hexdigest(),
            "worktree_clean": not bool(status),
        },
        "trace": {
            "path": str(trace),
            "sha256": _sha256(trace),
            "manifest_path": str(trace_manifest),
            "manifest_sha256": _sha256(trace_manifest),
        },
        "source_files": {
            path: _sha256(ROOT / path)
            for path in RELEVANT_SOURCES
        },
        "robot_asset": {
            "root": str(asset_root),
            "tree_sha256": asset_tree_sha,
            "files": asset_files,
        },
        "approval_mutated": False,
        "training_approved": False,
        "deployment_approved": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
