#!/usr/bin/env python3
"""Audit explicit motion payload paths before any PhysX replay.

The runtime loader gives ``library_motion_npz`` precedence over
``motion_npz``.  A generated candidate must therefore point every explicit
payload field at the same NPZ.  This audit is intentionally dependency-free
so it can run before IsaacLab is imported and can mark historical manifests
as invalid instead of allowing a silent source-clip replay.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def resolve(value: str, manifest_dir: Path) -> Path | None:
    raw = Path(os.path.expanduser(str(value)))
    candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw, manifest_dir / raw]
    candidates.append(manifest_dir / "motion_npz" / raw.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def audit_manifest(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("motions", [])
    strict_manifest = bool(
        data.get("payload_contract_strict", False)
        or str(data.get("schema_version", "")).startswith("p5d3a_")
    )
    rows = []
    for entry in entries:
        explicit = {}
        missing = []
        for key in ("motion_npz", "library_motion_npz"):
            value = entry.get(key)
            if not value:
                continue
            resolved = resolve(str(value), path.parent)
            if resolved is None:
                missing.append(key)
            else:
                explicit[key] = str(resolved)
        # ``canonical_motion_npz`` is a boolean marker in the A3 candidate
        # bank, not a third path.  Older manifests may use it as an explicit
        # filename, so only resolve it when the value is a non-empty string.
        canonical_value = entry.get("canonical_motion_npz")
        if isinstance(canonical_value, str) and canonical_value:
            resolved = resolve(canonical_value, path.parent)
            if resolved is None:
                missing.append("canonical_motion_npz")
            else:
                explicit["canonical_motion_npz"] = str(resolved)
        payload_values = {v for k, v in explicit.items() if k in {"motion_npz", "library_motion_npz"}}
        strict_entry = strict_manifest or bool(entry.get("canonical_motion_npz"))
        conflict = len(payload_values) > 1 and strict_entry
        candidate = entry.get("p5d2_dataset", {}).get("reference_id", entry.get("episode_id"))
        rows.append(
            {
                "episode_id": entry.get("episode_id"),
                "reference_id": candidate,
                "resolved_motion_npz": explicit.get("motion_npz"),
                "resolved_library_motion_npz": explicit.get("library_motion_npz"),
                "resolved_canonical_motion_npz": explicit.get("canonical_motion_npz"),
                "missing_fields": missing,
                "payload_conflict": conflict,
                "status": "CONFLICT" if conflict else ("MISSING" if missing else ("PASS_LEGACY_PRECEDENCE" if len(payload_values) > 1 else "PASS")),
            }
        )
    conflicts = [r for r in rows if r["payload_conflict"]]
    missing = [r for r in rows if r["missing_fields"]]
    return {
        "manifest": str(path.resolve()),
        "entry_count": len(rows),
        "conflict_count": len(conflicts),
        "missing_count": len(missing),
        "status": "FAIL_CLOSED_CONFLICT" if conflicts else ("FAIL_MISSING_PAYLOAD" if missing else "PASS"),
        "strict_manifest": strict_manifest,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reports = [audit_manifest(p.resolve()) for p in args.manifests]
    result = {
        "schema_version": "motion_manifest_payload_contract_audit/v1",
        "status": "PASS" if all(r["status"] == "PASS" for r in reports) else "FAIL_CLOSED",
        "loader_rule": "library_motion_npz and motion_npz must resolve to the same payload when both are present",
        "manifests": reports,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
