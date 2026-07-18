#!/usr/bin/env python3
"""Verify A3 standalone replay qualification evidence without claiming a pass.

The actual replay must be performed by a project-side executable that uses the
official ``a3_deploy_example`` RobotIOBackend.  This tool is intentionally an
evidence checker: it refuses to upgrade a run without repeated raw state,
actual racket task samples, target integrity, and timing metadata.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from a3_strike_contract import command_sha256_from_npz, sha256_file, verify_target_spec


def _single_text_field(data: Any, name: str, path: Path) -> str:
    if name not in data.files:
        raise ValueError(f"{path}: missing task-sample provenance field {name}")
    value = np.asarray(data[name])
    if value.size != 1:
        raise ValueError(f"{path}: {name} must contain exactly one value")
    result = str(value.reshape(-1)[0])
    if not result:
        raise ValueError(f"{path}: {name} must be non-empty")
    return result


def _task_sample(
    path: Path,
    hit_time_s: float,
    expected_target_hash: str | None = None,
    expected_mount_contract_id: str | None = None,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        required = {"timestamp_s", "racket_position_b_m", "racket_velocity_b_mps", "racket_normal_b", "stand_gate_passed"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"{path}: missing actual task fields {', '.join(missing)}")
        time_s = np.asarray(data["timestamp_s"], dtype=np.float64)
        if time_s.ndim != 1 or not len(time_s) or not np.all(np.diff(time_s) > 0.0):
            raise ValueError(f"{path}: timestamp_s must be non-empty and strictly increasing")
        index = int(np.argmin(np.abs(time_s - hit_time_s)))
        task_fields = {"racket_position_b_m", "racket_velocity_b_mps", "racket_normal_b"}
        result = {name: np.asarray(data[name], dtype=np.float64)[index] for name in task_fields}
        result["actual_hit_time_s"] = float(time_s[index])
        result["hit_time_alignment_error_s"] = abs(float(time_s[index]) - hit_time_s)
        if "command_publish_time_s" in data.files and "state_receive_time_s" in data.files:
            publish = np.asarray(data["command_publish_time_s"], dtype=np.float64)
            receive = np.asarray(data["state_receive_time_s"], dtype=np.float64)
            n = min(len(publish), len(receive))
            result["command_to_state_delay_s"] = float(np.median(receive[:n] - publish[:n])) if n else math.nan
        if expected_target_hash is not None:
            observed_target_hash = _single_text_field(data, "source_target_sha256", path)
            if observed_target_hash != expected_target_hash:
                raise ValueError(f"{path}: source_target_sha256 does not match --target-spec")
            observed_mount_contract = _single_text_field(data, "racket_mount_contract_id", path)
            if observed_mount_contract != expected_mount_contract_id:
                raise ValueError(f"{path}: racket_mount_contract_id does not match --target-spec")
            result["source_target_sha256"] = observed_target_hash
            result["racket_mount_contract_id"] = observed_mount_contract
        stand_gate = np.asarray(data["stand_gate_passed"])
        if stand_gate.size != 1 or not bool(stand_gate.reshape(-1)[0]):
            raise ValueError(f"{path}: PD-STAND gate did not pass; task evidence is invalid")
    normal = result["racket_normal_b"]
    norm = float(np.linalg.norm(normal))
    if not all(np.all(np.isfinite(value)) for value in result.values() if isinstance(value, np.ndarray)) or norm <= 1e-9:
        raise ValueError(f"{path}: actual task sample is invalid")
    result["racket_normal_b"] = normal / norm
    return result


def _state_evidence(path: Path) -> dict[str, float]:
    """Require distinct raw state, backend-sync, and command clocks."""

    with np.load(path, allow_pickle=False) as data:
        required = {"raw_state_timestamp_s", "backend_sync_timestamp_s", "command_timestamp_s", "q_actual", "dq_actual", "tau_est"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"{path}: missing raw state evidence fields {', '.join(missing)}")
        raw = np.asarray(data["raw_state_timestamp_s"], dtype=np.float64)
        sync = np.asarray(data["backend_sync_timestamp_s"], dtype=np.float64)
        command = np.asarray(data["command_timestamp_s"], dtype=np.float64)
        for name, values in (("raw_state_timestamp_s", raw), ("backend_sync_timestamp_s", sync), ("command_timestamp_s", command)):
            if values.ndim != 1 or len(values) < 2 or not np.all(np.diff(values) > 0.0):
                raise ValueError(f"{path}: {name} must be strictly increasing with at least two samples")
        for name in ("q_actual", "dq_actual", "tau_est"):
            values = np.asarray(data[name], dtype=np.float64)
            if values.ndim != 2 or values.shape[1] != 31 or not np.all(np.isfinite(values)):
                raise ValueError(f"{path}: {name} must be finite [T,31]")
    def rate(values: np.ndarray) -> float:
        return float(1.0 / np.median(np.diff(values)))
    return {"raw_state_rate_hz": rate(raw), "backend_sync_rate_hz": rate(sync), "command_rate_hz": rate(command)}


def _metrics(sample: dict[str, Any], target: dict[str, Any]) -> dict[str, float]:
    pos = float(np.linalg.norm(sample["racket_position_b_m"] - np.asarray(target["racket_position_b_m"])))
    vel = float(np.linalg.norm(sample["racket_velocity_b_mps"] - np.asarray(target["racket_velocity_b_mps"])))
    dot = float(np.clip(np.dot(sample["racket_normal_b"], np.asarray(target["racket_normal_b"])), -1.0, 1.0))
    return {"position_error_m": pos, "velocity_vector_error_mps": vel, "normal_angle_deg": float(np.degrees(np.arccos(dot)))}


def _std_max(rows: list[dict[str, float]], field: str) -> dict[str, float]:
    values = np.asarray([row[field] for row in rows], dtype=np.float64)
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "max_abs_deviation": float(np.max(np.abs(values - np.mean(values))))}


def _qualification_flags(metrics: dict[str, dict[str, float]], thresholds: dict[str, float], repeats: int) -> tuple[bool, bool]:
    """Separate deterministic repeatability from actual target attainment."""

    repeatability_noise_ok = repeats >= 10 and all(
        metrics[name]["std"] <= thresholds[name] * 0.10 for name in thresholds
    )
    absolute_target_ok = all(metrics[name]["mean"] <= thresholds[name] for name in thresholds)
    return repeatability_noise_ok, absolute_target_ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executor-contract", type=Path, required=True)
    parser.add_argument("--target-spec", type=Path, required=True)
    parser.add_argument("--command", type=Path, required=True)
    parser.add_argument("--rollout-task-samples", type=Path, action="append", required=True,
                        help="One actual_task_samples.npz per deterministic standalone replay; at least 10 required.")
    parser.add_argument("--rollout-state-samples", type=Path, action="append", required=True,
                        help="One raw_state_sidecar.npz per replay, containing raw/backend/command clocks and 31-DOF q/dq/tau.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.executor_contract.read_text(encoding="utf-8"))
    target_spec = json.loads(args.target_spec.read_text(encoding="utf-8"))
    target_hash = verify_target_spec(target_spec)
    target = {key: target_spec[key] for key in ("racket_position_b_m", "racket_velocity_b_mps", "racket_normal_b")}
    hit_time = float(target_spec["hit_time_s"])
    if len(args.rollout_state_samples) != len(args.rollout_task_samples):
        raise ValueError("every task sample file must have a matching raw-state evidence file")
    samples = [
        _task_sample(
            path.expanduser().resolve(),
            hit_time,
            target_hash,
            str(target_spec["racket_mount_contract_id"]),
        )
        for path in args.rollout_task_samples
    ]
    state_rates = [_state_evidence(path.expanduser().resolve()) for path in args.rollout_state_samples]
    rows = [{**_metrics(sample, target), "actual_hit_time_s": sample["actual_hit_time_s"], "command_to_state_delay_s": sample.get("command_to_state_delay_s", math.nan)} for sample in samples]
    metrics = {field: _std_max(rows, field) for field in ("position_error_m", "velocity_vector_error_mps", "normal_angle_deg", "actual_hit_time_s")}
    delays = np.asarray([row["command_to_state_delay_s"] for row in rows], dtype=np.float64)
    if np.any(np.isfinite(delays)):
        metrics["command_to_state_delay_s"] = {"mean": float(np.nanmean(delays)), "std": float(np.nanstd(delays)), "max_abs_deviation": float(np.nanmax(np.abs(delays - np.nanmean(delays))))}
    rates = {name: _std_max(state_rates, name) for name in ("raw_state_rate_hz", "backend_sync_rate_hz", "command_rate_hz")}
    thresholds = {"position_error_m": 0.075, "velocity_vector_error_mps": 0.5, "normal_angle_deg": 15.0}
    repeatability_noise_ok, absolute_target_ok = _qualification_flags(metrics, thresholds, len(rows))
    passed = repeatability_noise_ok and absolute_target_ok
    status = (
        "qualified_target_and_repeatability" if passed
        else "repeatable_but_target_miss" if repeatability_noise_ok
        else "not_qualified"
    )
    report = {
        "schema_version": 1,
        "qualification": "a3_standalone_evaluator_v1",
        "executor_contract_id": contract.get("executor_contract_id"),
        "executor_contract_sha256": sha256_file(args.executor_contract),
        "source_target_sha256": target_hash,
        "command_sha256": command_sha256_from_npz(args.command),
        "required_repeats": 10,
        "observed_repeats": len(rows),
        "bootstrap_task_thresholds": thresholds,
        "repeatability": metrics,
        "repeatability_noise_pass": repeatability_noise_ok,
        "absolute_target_match": absolute_target_ok,
        "observed_rates_hz": rates,
        "pass": bool(passed),
        "status": status,
        "limitations": [
            "This report checks target-bound actual task samples supplied by the standalone evaluator; it is not a ball-contact or deployment-balance admission.",
            "Zero-compensation identity and +/- epsilon sensitivity are separate required reports.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
