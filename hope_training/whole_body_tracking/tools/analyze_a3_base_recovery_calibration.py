#!/usr/bin/env python3
"""Derive a provisional clean-tail recovery envelope from passive calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


FLOORS = {
    "abs_pelvis_roll_rad": 0.010,
    "abs_pelvis_pitch_rad": 0.010,
    "abs_root_angular_velocity_x_rad_s": 0.050,
    "abs_root_angular_velocity_y_rad_s": 0.050,
    "abs_root_linear_velocity_x_m_s": 0.030,
    "abs_root_linear_velocity_y_m_s": 0.030,
    "abs_base_height_error_m": 0.010,
}


def _safe(profile: dict) -> bool:
    non_timeout = sum(
        count
        for name, count in profile["termination_term_counts"].items()
        if name != "time_out"
    )
    return bool(
        profile["runtime_integrity_passed"]
        and profile["disturbed_timeout_fraction"] >= 0.98
        and non_timeout == 0
        and profile["disturbed_max_tilt_rad"]["max"] < 0.35
    )


def _learning_headroom(profile: dict) -> bool:
    strict = profile["settled_envelopes"]["strict"]
    p95_time = strict["disturbed_settle_time_s"]["p95"]
    return bool(
        strict["disturbed_settled_fraction"] < 0.98
        or (p95_time is not None and p95_time > 0.50)
    )


def _stats(values: np.ndarray) -> dict:
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def _recompute_recovery(profile: dict, envelope: dict, dwell_steps: int) -> dict:
    trajectory_path = Path(profile["trajectory_path"])
    actual_sha = hashlib.sha256(trajectory_path.read_bytes()).hexdigest()
    if actual_sha != profile["trajectory_sha256"]:
        raise ValueError(
            f"trajectory SHA mismatch for {profile['profile']}: "
            f"{actual_sha} != {profile['trajectory_sha256']}"
        )
    with np.load(trajectory_path, allow_pickle=False) as payload:
        active = payload["active"].astype(bool)
        disturbed = payload["disturbed"].astype(bool)
        policy_dt_s = float(payload["policy_dt_s"][0])
        inside_enter = active.copy()
        inside_exit = active.copy()
        channel_values = {}
        for channel, thresholds in envelope.items():
            values = payload[channel].copy()
            channel_values[channel] = values
            inside_enter &= values <= thresholds["enter_threshold"]
            inside_exit &= values <= thresholds["exit_threshold"]

    consecutive = np.zeros(active.shape[1], dtype=np.int32)
    recovery_step = np.full(active.shape[1], -1, dtype=np.int32)
    for step in range(active.shape[0]):
        consecutive = np.where(inside_enter[step], consecutive + 1, 0)
        newly_recovered = (recovery_step < 0) & (consecutive >= dwell_steps)
        recovery_step[newly_recovered] = step - dwell_steps + 1

    recovered = disturbed & (recovery_step >= 0)
    recovery_times = recovery_step[recovered].astype(np.float64) * policy_dt_s
    post_recovery_exit_count = np.zeros(active.shape[1], dtype=np.int32)
    maximum_overshoot_ratio = np.zeros(active.shape[1], dtype=np.float64)
    final_envelope_error_ratio = np.zeros(active.shape[1], dtype=np.float64)
    for env_id in np.flatnonzero(recovered):
        start = recovery_step[env_id] + dwell_steps
        outside = active[start:, env_id] & ~inside_exit[start:, env_id]
        previous_outside = np.concatenate(([False], outside[:-1]))
        post_recovery_exit_count[env_id] = int(
            np.count_nonzero(outside & ~previous_outside)
        )
        active_steps = np.flatnonzero(active[:, env_id])
        if active_steps.size:
            final_step = int(active_steps[-1])
            for channel, thresholds in envelope.items():
                values = channel_values[channel][:, env_id]
                post_values = values[start:][active[start:, env_id]]
                if post_values.size:
                    maximum_overshoot_ratio[env_id] = max(
                        maximum_overshoot_ratio[env_id],
                        float(np.max(post_values / thresholds["exit_threshold"])),
                    )
                final_envelope_error_ratio[env_id] = max(
                    final_envelope_error_ratio[env_id],
                    max(
                        0.0,
                        float(values[final_step] / thresholds["enter_threshold"] - 1.0),
                    ),
                )
    disturbed_count = int(np.count_nonzero(disturbed))
    non_timeout_terminations = sum(
        count
        for name, count in profile["termination_term_counts"].items()
        if name != "time_out"
    )
    return {
        "disturbed_count": disturbed_count,
        "recovered_count": int(np.count_nonzero(recovered)),
        "strict_recovery_fraction": (
            float(np.count_nonzero(recovered) / disturbed_count)
            if disturbed_count
            else None
        ),
        "recovery_time_s": _stats(recovery_times),
        "unrecovered_long_tail_count": int(np.count_nonzero(disturbed & ~recovered)),
        "post_recovery_exit_count": int(np.sum(post_recovery_exit_count[recovered])),
        "maximum_overshoot_ratio": _stats(maximum_overshoot_ratio[recovered]),
        "final_envelope_error_ratio": _stats(final_envelope_error_ratio[recovered]),
        "non_timeout_termination_count": int(non_timeout_terminations),
        "trajectory_integrity_passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--p99-margin", type=float, default=1.25)
    parser.add_argument("--hysteresis-ratio", type=float, default=1.25)
    parser.add_argument("--dwell-steps", type=int, default=10)
    args = parser.parse_args()
    if args.p99_margin < 1.0 or args.hysteresis_ratio <= 1.0 or args.dwell_steps < 1:
        parser.error("invalid envelope parameters")

    calibrations = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in args.input
    ]
    if any(not item.get("calibration_measured") for item in calibrations):
        raise ValueError("calibration input is incomplete")
    trace_hashes = {item["disturbance_trace_sha256"] for item in calibrations}
    contract_hashes = {item.get("runtime_contract_sha256") for item in calibrations}
    if len(trace_hashes) != 1 or len(contract_hashes) != 1 or None in contract_hashes:
        raise ValueError("calibration inputs do not share one trace and runtime contract")
    profiles = [
        profile
        for calibration in calibrations
        for profile in calibration["profiles"]
    ]
    by_name = {profile["profile"]: profile for profile in profiles}
    if len(by_name) != len(profiles):
        raise ValueError("duplicate calibration profiles")
    required = {
        "recovery_a_clean",
        "recovery_a_candidate",
        "recovery_a_medium",
        "recovery_a_upper_probe",
    }
    if required - by_name.keys():
        raise ValueError(f"missing profiles: {sorted(required - by_name.keys())}")

    clean_profile = by_name["recovery_a_clean"]
    channel_names = set(FLOORS)
    channel_names.update(
        name
        for name in clean_profile["clean_tail_statistics"]
        if name.startswith("abs_joint_velocity_rad_s/")
    )
    envelope = {}
    for channel in sorted(channel_names):
        clean_stats = clean_profile["clean_tail_statistics"][channel]
        if clean_stats["p99"] is None:
            raise ValueError(f"clean tail is empty for channel {channel}")
        p99 = float(clean_stats["p99"])
        floor = FLOORS.get(channel, 0.05)
        enter = max(p99 * args.p99_margin, floor)
        exit_threshold = enter * args.hysteresis_ratio
        envelope[channel] = {
            "clean_tail_p90": float(clean_stats["p90"]),
            "clean_tail_p95": float(clean_stats["p95"]),
            "clean_tail_p99": p99,
            "floor": floor,
            "enter_threshold": enter,
            "exit_threshold": exit_threshold,
        }

    profile_decisions = {}
    recomputed_recovery = {}
    for name in (
        "recovery_a_candidate",
        "recovery_a_medium",
        "recovery_a_upper_probe",
    ):
        profile = by_name[name]
        profile_decisions[name] = {
            "passive_safe": _safe(profile),
            "has_learning_headroom": _learning_headroom(profile),
            "first_smoke_training_eligible": False,
        }
        recomputed_recovery[name] = _recompute_recovery(
            profile, envelope, args.dwell_steps
        )
    mixture_candidate = bool(
        profile_decisions["recovery_a_candidate"]["passive_safe"]
        and profile_decisions["recovery_a_candidate"]["has_learning_headroom"]
        and profile_decisions["recovery_a_medium"]["passive_safe"]
        and profile_decisions["recovery_a_medium"]["has_learning_headroom"]
    )

    result = {
        "schema_version": 1,
        "analysis_id": "a3_base_recovery_clean_tail_envelope_candidate_v1",
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_calibrations": [str(path) for path in args.input],
        "source_trace_sha256": next(iter(trace_hashes)),
        "source_runtime_contract_sha256": next(iter(contract_hashes)),
        "clean_tail_window_s": 2.0,
        "quantile": 0.99,
        "p99_margin": args.p99_margin,
        "hysteresis_ratio": args.hysteresis_ratio,
        "dwell_steps": args.dwell_steps,
        "policy_dt_s": calibrations[0]["policy_dt_s"],
        "envelope": envelope,
        "clean_tail_source_profile": "recovery_a_clean",
        "recovery_definition": {
            "entry": "all primary channels <= enter_threshold for dwell_steps consecutive policy steps",
            "post_recovery_exit": "any primary channel > exit_threshold",
            "recovery_time": "first step of the first completed dwell interval",
            "post_recovery_metrics": [
                "exit_count",
                "maximum_overshoot_ratio",
                "final_envelope_error",
            ],
            "root_linear_velocity_is_privileged_evaluation_only": True,
        },
        "profile_decisions": profile_decisions,
        "recomputed_recovery": recomputed_recovery,
        "provisional_first_smoke_mixture": {
            "clean": 0.30,
            "candidate": 0.40,
            "medium": 0.30,
            "upper": 0.0,
            "supported_by_passive_calibration": mixture_candidate,
        },
        "analysis_passed": True,
        "recovery_envelope_approved": False,
        "training_distribution_approved": False,
        "bounded_recovery_smoke_approved": False,
        "deployment_approved": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
