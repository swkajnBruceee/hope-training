#!/usr/bin/env python3
"""Combine isolated Recovery-A paired safety runs into one formal audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REQUIRED_PROFILES = ("clean", "candidate", "medium")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = [
        (path.expanduser().resolve(), json.loads(path.read_text(encoding="utf-8")))
        for path in args.input
    ]
    if len(sources) != len(REQUIRED_PROFILES):
        raise ValueError(f"Expected {len(REQUIRED_PROFILES)} isolated audits")

    common_fields = (
        "audit_id",
        "task",
        "trace_sha256",
        "envelope_decision_sha256",
        "approved_envelope_name",
        "approved_dwell_s",
        "approved_hysteresis_ratio",
        "noise_std",
        "policy_steps",
    )
    reference = sources[0][1]
    for _path, source in sources:
        for field in common_fields:
            if source[field] != reference[field]:
                raise ValueError(f"Isolated audit mismatch for {field}")
        if source["runtime_smoke_only"] or not source["runtime_integrity_passed"]:
            raise ValueError("Isolated audit is not complete formal evidence")
        if len(source["profiles"]) != 1:
            raise ValueError("Each isolated audit must contain exactly one profile")

    by_name = {
        source["profiles"][0]["profile"]: source["profiles"][0]
        for _path, source in sources
    }
    if set(by_name) != set(REQUIRED_PROFILES):
        raise ValueError(f"Profile set mismatch: {sorted(by_name)}")
    profiles = [by_name[name] for name in REQUIRED_PROFILES]
    safety_verified = all(
        profile["untrained_stochastic_profile_safe"] for profile in profiles
    )
    result = {
        "schema_version": 1,
        "audit_id": reference["audit_id"],
        "task": reference["task"],
        "simulation_only": True,
        "runtime_smoke_only": False,
        "isolated_profile_processes": True,
        "source_audits": [
            {"path": str(path), "sha256": _sha256(path)}
            for path, _source in sources
        ],
        "trace_path": reference["trace_path"],
        "trace_sha256": reference["trace_sha256"],
        "envelope_decision_path": reference["envelope_decision_path"],
        "envelope_decision_sha256": reference["envelope_decision_sha256"],
        "approved_envelope_name": reference["approved_envelope_name"],
        "approved_dwell_s": reference["approved_dwell_s"],
        "approved_hysteresis_ratio": reference["approved_hysteresis_ratio"],
        "noise_std": reference["noise_std"],
        "policy_steps": reference["policy_steps"],
        "profiles": profiles,
        "runtime_integrity_passed": True,
        "untrained_stochastic_policy_safety_verified": safety_verified,
        "training_distribution_approved": False,
        "bounded_recovery_smoke_approved": False,
        "deployment_approved": False,
        "gate_mutated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "profiles": list(REQUIRED_PROFILES),
                "untrained_stochastic_policy_safety_verified": safety_verified,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
