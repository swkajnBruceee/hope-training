#!/usr/bin/env python3
"""Create the read-only Phase 0 archive inventory and official-file hashes.

The command never moves or deletes an input.  Phase 0E must consume this
manifest rather than globbing a directory during cleanup.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from a3_strike_contract import sha256_file


DEFAULT_CANDIDATES = (
    "docs/eval_reports",
    "eval_outputs",
    "sample_motions",
    "outputs",
    "replay_logs",
)
OFFICIAL_FILES = (
    "agibot/code_deployment/a3_deploy_example/README.md",
    "agibot/code_deployment/a3_deploy_example/README_robot_io_backend.md",
    "agibot/code_deployment/a3_deploy_example/src/a3/a3_deploy_onnx_ref/config/a3_runtime_config.yaml",
    "agibot/code_deployment/a3_deploy_example/src/a3/a3_deploy_onnx_ref/include/a3_policy_parameters.hpp",
    "agibot/code_deployment/a3_deploy_example/src/a3/a3_deploy_onnx_ref/src/a3_deploy/main.cpp",
    "agibot/code_deployment/a3_deploy_example/mujoco_sim_standalone/bin/cfg/a3_t2d5_cfg.yaml",
)


def _entry(path: Path, workspace: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.relative_to(workspace)),
        "kind": "directory" if path.is_dir() else "file",
        "size_bytes": stat.st_size if path.is_file() else None,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "sha256": sha256_file(path) if path.is_file() else None,
        "classification": "DIAGNOSTIC_ARCHIVE_CANDIDATE",
        "decision": "retain_until_phase_0d_evaluator_qualification",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.expanduser().resolve()
    tracking = workspace / "hope_training/whole_body_tracking"
    candidates = [_entry(path, workspace) for item in DEFAULT_CANDIDATES if (path := tracking / item).exists()]
    official = []
    for item in OFFICIAL_FILES:
        path = workspace / item
        official.append({"path": item, "exists": path.is_file(), "sha256": sha256_file(path) if path.is_file() else None})
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "0A_read_only_inventory",
        "delete_authorized": False,
        "archive_root": "artifacts/_archive_not_for_training/20260718_pre_executor_contract",
        "candidates": candidates,
        "official_file_hashes": official,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
