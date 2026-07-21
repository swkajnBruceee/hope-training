#!/usr/bin/env python3
"""Verify three deterministic repeats of the bounded Stand fixture scope."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

import a3_base_contract as contract
from analyze_a3_base_low_zoh_bundle import _result_index


REPEAT_SUFFIX = re.compile(r"__r0(?P<repeat>[123])$")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for engine in ("isaac", "mujoco"):
        for repeat in (1, 2, 3):
            parser.add_argument(
                f"--{engine}-repeat-{repeat}-results-dir", type=Path, required=True
            )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _logical_id(case_id: str, expected_repeat: int) -> str:
    match = REPEAT_SUFFIX.search(case_id)
    if not match or int(match.group("repeat")) != expected_repeat:
        raise ValueError(f"case repeat suffix mismatch: {case_id}")
    return case_id[: match.start()]


def _canonical_array_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as archive:
        for name in sorted(archive.files):
            array = np.ascontiguousarray(archive[name])
            digest.update(name.encode("utf-8") + b"\0")
            digest.update(array.dtype.str.encode("ascii") + b"\0")
            digest.update(json.dumps(array.shape).encode("ascii") + b"\0")
            digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _logical_index(
    directory: Path, repeat: int
) -> dict[str, tuple[dict[str, Any], Path, Path]]:
    raw = _result_index(directory)
    logical = {}
    for case_id, value in raw.items():
        logical_id = _logical_id(case_id, repeat)
        if logical_id in logical:
            raise ValueError(f"duplicate logical case: {logical_id}")
        logical[logical_id] = value
    return logical


def _engine_report(
    engine: str,
    directories: dict[int, Path],
) -> tuple[dict[str, Any], set[Path]]:
    repeats = {
        repeat: _logical_index(directory, repeat)
        for repeat, directory in directories.items()
    }
    logical_sets = {frozenset(index) for index in repeats.values()}
    if len(logical_sets) != 1:
        raise ValueError(f"{engine} logical repeat coverage differs")
    logical_ids = sorted(next(iter(logical_sets)))
    if len(logical_ids) != 89:
        raise ValueError(f"{engine} Stand fixture scope must contain 89 logical cases")
    matrix_hashes = {
        item[0]["matrix_sha256"]
        for index in repeats.values()
        for item in index.values()
    }
    if len(matrix_hashes) != 1:
        raise ValueError(f"{engine} repeats use different matrices")

    metrics_mismatches = []
    evidence_mismatches = []
    safety_failures = []
    source_paths: set[Path] = set()
    per_case = []
    for logical_id in logical_ids:
        items = [repeats[repeat][logical_id] for repeat in (1, 2, 3)]
        metric_hashes = [_canonical_json_hash(item[0]["metrics"]) for item in items]
        evidence_hashes = [_canonical_array_hash(item[1]) for item in items]
        metrics_exact = len(set(metric_hashes)) == 1
        evidence_exact = len(set(evidence_hashes)) == 1
        if not metrics_exact:
            metrics_mismatches.append(logical_id)
        if not evidence_exact:
            evidence_mismatches.append(logical_id)
        for repeat, item in zip((1, 2, 3), items, strict=True):
            if not bool(item[0]["case_validation"]["safety_envelope_passed"]):
                safety_failures.append(f"{logical_id}:repeat{repeat}")
            source_paths.update({item[1], item[2]})
        per_case.append(
            {
                "logical_case_id": logical_id,
                "metrics_exact_across_repeats": metrics_exact,
                "evidence_arrays_exact_across_repeats": evidence_exact,
                "metrics_sha256_by_repeat": metric_hashes,
                "evidence_array_sha256_by_repeat": evidence_hashes,
            }
        )
    runner_source_hashes = sorted(
        {
            item[0].get("runner_source_sha256")
            for index in repeats.values()
            for item in index.values()
            if item[0].get("runner_source_sha256") is not None
        }
    )
    return (
        {
            "engine": engine,
            "matrix_sha256": next(iter(matrix_hashes)),
            "repeat_count": 3,
            "logical_case_count": len(logical_ids),
            "executed_case_count": len(logical_ids) * 3,
            "all_safety_envelopes_passed": not safety_failures,
            "metrics_exact_across_repeats": not metrics_mismatches,
            "evidence_arrays_exact_across_repeats": not evidence_mismatches,
            "metrics_mismatch_logical_case_ids": metrics_mismatches,
            "evidence_mismatch_logical_case_ids": evidence_mismatches,
            "safety_failure_case_ids": safety_failures,
            "runner_source_sha256_values": runner_source_hashes,
            "runner_source_hash_consistent_when_recorded": len(runner_source_hashes) <= 1,
            "cases": per_case,
        },
        source_paths,
    )


def build_report(
    *, isaac_dirs: dict[int, Path], mujoco_dirs: dict[int, Path]
) -> dict[str, Any]:
    isaac, isaac_sources = _engine_report("isaac", isaac_dirs)
    mujoco, mujoco_sources = _engine_report("mujoco", mujoco_dirs)
    if isaac["matrix_sha256"] != mujoco["matrix_sha256"]:
        raise ValueError("Isaac and MuJoCo repeatability matrices differ")
    deterministic = all(
        engine["all_safety_envelopes_passed"]
        and engine["metrics_exact_across_repeats"]
        and engine["evidence_arrays_exact_across_repeats"]
        and engine["runner_source_hash_consistent_when_recorded"]
        for engine in (isaac, mujoco)
    )
    source_paths = isaac_sources | mujoco_sources
    report = {
        "schema_version": 1,
        "artifact_status": "phase0_fixture_repeatability_not_automatic_promotion",
        "scope": "stand_fixture_approval",
        "matrix_sha256": isaac["matrix_sha256"],
        "repeatability_passed": deterministic,
        "engines": {"isaac": isaac, "mujoco": mujoco},
        "gate_interpretation": {
            "stand_fixture_scope_has_three_repeats": deterministic,
            "legacy_full_339_case_promotion_rule_satisfied": False,
            "reason": (
                "The legacy rule also includes post-Stand command-basis and broader "
                "1000 Hz/medium cases; a versioned gate decision is required instead "
                "of mutating the evidence contract."
            ),
        },
        "qualification_status": {
            "fixture_runner_qualified": True,
            "fixture_matrix_approved": False,
            "stand_task_approved": False,
            "locomotion_command_approved": False,
            "deployment_approved": False,
        },
        "automatic_promotion": False,
        "source_sha256": {
            str(path): contract.file_sha256(path) for path in sorted(source_paths)
        },
    }
    report["source_sha256"][str(Path(__file__).resolve())] = contract.file_sha256(
        Path(__file__).resolve()
    )
    return report


def main() -> None:
    args = _parser().parse_args()
    report = build_report(
        isaac_dirs={
            repeat: getattr(args, f"isaac_repeat_{repeat}_results_dir")
            for repeat in (1, 2, 3)
        },
        mujoco_dirs={
            repeat: getattr(args, f"mujoco_repeat_{repeat}_results_dir")
            for repeat in (1, 2, 3)
        },
    )
    output = args.output.expanduser().resolve()
    if output.suffix != ".json":
        raise ValueError("--output must end in .json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "repeatability_passed": report["repeatability_passed"],
                "engines": {
                    name: {
                        "executed_case_count": value["executed_case_count"],
                        "all_safety_envelopes_passed": value[
                            "all_safety_envelopes_passed"
                        ],
                        "metrics_exact_across_repeats": value[
                            "metrics_exact_across_repeats"
                        ],
                        "evidence_arrays_exact_across_repeats": value[
                            "evidence_arrays_exact_across_repeats"
                        ],
                        "runner_source_hash_consistent_when_recorded": value[
                            "runner_source_hash_consistent_when_recorded"
                        ],
                    }
                    for name, value in report["engines"].items()
                },
                "gate_interpretation": report["gate_interpretation"],
                "qualification_status": report["qualification_status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
