#!/usr/bin/env python3
"""Check the official A3 x86 deployment package without starting control.

This is deliberately a read-only preflight.  It does not load ONNX, start
AimRT, subscribe to ROS2, or publish robot commands.  Its purpose is to avoid
confusing a transport-only dry-run with a policy-ready deployment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


MODEL_KEYS = (
    "model_path",
    "smpl_model_path",
    "a3_fast_model_path",
    "rknn_model_path",
    "smpl_rknn_model_path",
    "a3_fast_rknn_model_path",
)
REQUIRED_RUNTIME_FILES = (
    "a3_deploy_onnx_ref",
    "libaimrt_ros2_plugin.so",
    "libirobot_events_executor.so",
    "libonnxruntime.so.1",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_model_paths(config: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    pattern = re.compile(r"^\s*([A-Za-z0-9_]+):\s*([^#\s]+)")
    for line in config.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match and match.group(1) in MODEL_KEYS:
            values[match.group(1)] = match.group(2).strip('"\'')
    return values


def _file_record(path: Path, *, include_hash: bool = False) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
    }
    if path.is_file():
        record["bytes"] = path.stat().st_size
        if include_hash:
            record["sha256"] = _sha256(path)
    return record


def check(*, project_root: Path, config: Path, dist: Path) -> dict[str, Any]:
    config = config.resolve()
    dist = dist.resolve()
    model_values = _parse_model_paths(config) if config.is_file() else {}

    models: dict[str, Any] = {}
    for key, value in model_values.items():
        path = Path(value)
        if not path.is_absolute():
            path = project_root / path
        models[key] = _file_record(path, include_hash=True)

    runtime_files = {
        name: _file_record(dist / name)
        for name in REQUIRED_RUNTIME_FILES
    }
    required_model = models.get("model_path", {"exists": False})
    return {
        "scope": "official_a3_x86_deployment_asset_preflight",
        "publishes_commands": False,
        "project_root": str(project_root),
        "runtime_config": _file_record(config),
        "runtime_dist": str(dist),
        "runtime_files": runtime_files,
        "models": models,
        "dry_run_ready": all(item["exists"] for item in runtime_files.values()),
        "policy_ready": bool(required_model.get("exists"))
        and all(item["exists"] for item in runtime_files.values()),
        "interpretation": (
            "Transport-only dry-run is possible. The official A3 policy model "
            "is still missing."
            if not required_model.get("exists")
            else "The configured A3 policy model path is present."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="deployment project root used to resolve relative model paths",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "src/a3/a3_deploy_onnx_ref/config/a3_runtime_config.yaml",
    )
    parser.add_argument(
        "--dist",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dist/a3_deploy_x86_64",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument(
        "--require-model",
        action="store_true",
        help="return non-zero unless the configured policy model exists",
    )
    args = parser.parse_args()
    report = check(
        project_root=args.project_root.resolve(),
        config=args.config,
        dist=args.dist,
    )
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"dry_run_ready: {report['dry_run_ready']}")
        print(f"policy_ready: {report['policy_ready']}")
        print(report["interpretation"])
        for key, item in report["models"].items():
            print(f"model {key}: {item['exists']} {item['path']}")
    if not report["dry_run_ready"]:
        return 1
    if args.require_model and not report["policy_ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
