#!/usr/bin/env python3
"""Build a read-only, reproducible Recovery-A manual-review evidence package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts/a3_base_stand"
CORE = (
    "abs_pelvis_roll_rad",
    "abs_pelvis_pitch_rad",
    "abs_root_angular_velocity_x_rad_s",
    "abs_root_angular_velocity_y_rad_s",
    "abs_root_linear_velocity_x_m_s",
    "abs_root_linear_velocity_y_m_s",
    "abs_base_height_error_m",
)
AUX = (
    "abs_joint_velocity_rad_s/left_ankle_pitch_joint",
    "abs_joint_velocity_rad_s/left_ankle_roll_joint",
    "abs_joint_velocity_rad_s/right_ankle_pitch_joint",
    "abs_joint_velocity_rad_s/right_ankle_roll_joint",
    "abs_joint_velocity_rad_s/waist_roll_joint",
    "abs_joint_velocity_rad_s/waist_pitch_joint",
)
ANKLE = tuple(name for name in AUX if "ankle" in name)
WAIST = tuple(name for name in AUX if "waist" in name)
ALL_CHANNELS = CORE + AUX
FLOORS = {
    "abs_pelvis_roll_rad": 0.010,
    "abs_pelvis_pitch_rad": 0.010,
    "abs_root_angular_velocity_x_rad_s": 0.050,
    "abs_root_angular_velocity_y_rad_s": 0.050,
    "abs_root_linear_velocity_x_m_s": 0.030,
    "abs_root_linear_velocity_y_m_s": 0.030,
    "abs_base_height_error_m": 0.010,
}
PROFILE_FILES = {
    "clean": "recovery_clean_calibration_v1.json",
    "candidate": "recovery_candidate_calibration_v1.json",
    "medium": "recovery_medium_calibration_v1.json",
    "upper": "recovery_upper_calibration_v1.json",
}
PROFILE_IDS = {"clean": 0, "candidate": 1, "medium": 2, "upper": 3}
EXPECTED_NAMES = {
    "clean": "recovery_a_clean",
    "candidate": "recovery_a_candidate",
    "medium": "recovery_a_medium",
    "upper": "recovery_a_upper_probe",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values: np.ndarray) -> dict:
    """Return robust descriptive statistics for finite one-dimensional values."""
    values = np.asarray(values).reshape(-1)
    if not values.size:
        return {
            key: None
            for key in ("mean", "std", "median", "p90", "p95", "p99", "max", "mad", "iqr")
        } | {"count": 0, "p95_p99_gap": None}
    # Compute separately to reproduce the source analyzer's float32 quantiles.
    q25 = np.quantile(values, 0.25)
    median = np.quantile(values, 0.50)
    q75 = np.quantile(values, 0.75)
    p90 = np.quantile(values, 0.90)
    p95 = np.quantile(values, 0.95)
    p99 = np.quantile(values, 0.99)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(median),
        "p90": float(p90),
        "p95": float(p95),
        "p99": float(p99),
        "max": float(np.max(values)),
        "mad": float(np.median(np.abs(values - median))),
        "iqr": float(q75 - q25),
        "p95_p99_gap": float(p99 - p95),
    }


def rolling_rms(values: np.ndarray, window_steps: int) -> np.ndarray:
    """Trailing RMS with a shortened window at the beginning."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or window_steps < 1:
        raise ValueError("rolling_rms expects [steps, envs] and a positive window")
    squared = values * values
    prefix = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(squared, axis=0)))
    result = np.empty_like(values)
    for end in range(1, values.shape[0] + 1):
        start = max(0, end - window_steps)
        result[end - 1] = np.sqrt((prefix[end] - prefix[start]) / (end - start))
    return result


def rolling_quantile(values: np.ndarray, window_steps: int, quantile: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    for step in range(values.shape[0]):
        result[step] = np.quantile(values[max(0, step - window_steps + 1) : step + 1], quantile, axis=0)
    return result


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _first_dwell(inside: np.ndarray, dwell_steps: int) -> tuple[int | None, int | None]:
    for start, end in _runs(inside):
        if end - start >= dwell_steps:
            return start, start + dwell_steps - 1
    return None, None


def _threshold_masks(
    arrays: dict[str, np.ndarray], envelope: dict, scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    shape = next(iter(arrays.values())).shape
    enter = np.ones(shape, dtype=bool)
    exit_inside = np.ones(shape, dtype=bool)
    triggers = {}
    for channel, threshold in envelope.items():
        values = arrays[channel]
        enter &= values <= threshold["enter_threshold"] * scale
        channel_exit = values > threshold["exit_threshold"] * scale
        triggers[channel] = channel_exit
        exit_inside &= ~channel_exit
    return enter, exit_inside, triggers


def episode_events(
    arrays: dict[str, np.ndarray],
    active: np.ndarray,
    envelope: dict,
    dt: float,
    dwell_s: float = 0.20,
    scale: float = 1.0,
) -> list[dict]:
    """Compute recovery with a RECOVERING/RECOVERED/OUTSIDE state machine."""
    dwell_steps = max(1, int(round(dwell_s / dt)))
    enter, exit_inside, triggers = _threshold_masks(arrays, envelope, scale)
    core_channels = [name for name in envelope if name in CORE]
    aux_channels = [name for name in envelope if name not in CORE]

    def is_ankle(name: str) -> bool:
        return name.split("__", 1)[0] in ANKLE

    def is_waist(name: str) -> bool:
        return name.split("__", 1)[0] in WAIST

    events = []
    for env in range(active.shape[1]):
        valid = active[:, env]
        entry_candidates = np.flatnonzero(valid & enter[:, env])
        first_entry = int(entry_candidates[0]) if entry_candidates.size else None
        state = "RECOVERING"
        dwell_count = 0
        dwell_start = None
        first_dwell_start = None
        first_confirmed = None
        recovery_confirmations = []
        exit_events = []
        current_exit = None

        def classify_exit(trigger_names: list[str], violation_steps: int) -> str:
            core_exit = any(name in core_channels for name in trigger_names)
            aux_only = bool(trigger_names) and all(name in aux_channels for name in trigger_names)
            if aux_only and violation_steps <= 2:
                return "numerical_or_contact_spike_contact_unverified"
            if core_exit and "abs_base_height_error_m" in trigger_names:
                return "height_instability" if len(trigger_names) == 1 else "multi_metric_exit"
            if core_exit:
                return "core_body_reinstability" if len(trigger_names) == 1 else "multi_metric_exit"
            if trigger_names and all(is_waist(name) for name in trigger_names):
                return "waist_velocity_only"
            if trigger_names and all(is_ankle(name) for name in trigger_names):
                return "ankle_velocity_only"
            return "multi_metric_exit" if len(trigger_names) > 1 else "unclassified"

        def finish_exit(confirmed_step: int | None, last_step: int) -> None:
            assert current_exit is not None
            if confirmed_step is None:
                outside_steps = last_step - current_exit["exit_step"] + 1
            else:
                outside_steps = confirmed_step - current_exit["exit_step"]
            trigger_names = sorted(current_exit.pop("_trigger_metrics"))
            current_exit.update(
                {
                    "trigger_metrics": trigger_names,
                    "outside_state_steps": int(outside_steps),
                    "outside_state_duration_s": float(outside_steps * dt),
                    "exit_threshold_violation_steps": int(
                        current_exit.pop("_violation_steps")
                    ),
                    "recovered_again": confirmed_step is not None,
                    "recovery_confirmed_step": confirmed_step,
                    "recovery_confirmed_s": (
                        None if confirmed_step is None else confirmed_step * dt
                    ),
                }
            )
            current_exit["threshold_violation_duration_s"] = (
                current_exit["exit_threshold_violation_steps"] * dt
            )
            current_exit["classification"] = classify_exit(
                trigger_names, current_exit["exit_threshold_violation_steps"]
            )
            exit_events.append(current_exit.copy())

        active_steps = np.flatnonzero(valid)
        for step in active_steps:
            step = int(step)
            step_triggers = {
                name for name, mask in triggers.items() if mask[step, env]
            }
            if state == "RECOVERED":
                if step_triggers:
                    current_exit = {
                        "exit_event_id": len(exit_events),
                        "exit_step": step,
                        "exit_time_s": step * dt,
                        "initial_trigger_metrics": sorted(step_triggers),
                        "_trigger_metrics": set(step_triggers),
                        "_violation_steps": 1,
                    }
                    state = "OUTSIDE"
                    dwell_count = 0
                    dwell_start = None
                continue
            if state == "OUTSIDE":
                if step_triggers:
                    current_exit["_trigger_metrics"].update(step_triggers)
                    current_exit["_violation_steps"] += 1
                if enter[step, env]:
                    if dwell_count == 0:
                        dwell_start = step
                    dwell_count += 1
                    if dwell_count >= dwell_steps:
                        recovery_confirmations.append(step)
                        finish_exit(step, step)
                        current_exit = None
                        state = "RECOVERED"
                        dwell_count = 0
                        dwell_start = None
                else:
                    dwell_count = 0
                    dwell_start = None
                continue
            # RECOVERING: the first complete enter-envelope dwell confirms recovery.
            if enter[step, env]:
                if dwell_count == 0:
                    dwell_start = step
                dwell_count += 1
                if dwell_count >= dwell_steps:
                    first_dwell_start = dwell_start
                    first_confirmed = step
                    recovery_confirmations.append(step)
                    state = "RECOVERED"
                    dwell_count = 0
                    dwell_start = None
            else:
                dwell_count = 0
                dwell_start = None
        last_active_step = int(active_steps[-1]) if active_steps.size else -1
        if state == "OUTSIDE" and current_exit is not None:
            finish_exit(None, last_active_step)

        trigger_names = sorted(
            {name for exit_event in exit_events for name in exit_event["trigger_metrics"]}
        )
        categories = sorted({exit_event["classification"] for exit_event in exit_events})
        if not categories:
            category = "no_post_recovery_exit"
        elif len(categories) == 1:
            category = categories[0]
        else:
            category = "multiple_exit_categories"
        core_exit = any(name in core_channels for name in trigger_names)
        ankle_only = bool(exit_events) and all(
            event["trigger_metrics"]
            and all(is_ankle(name) for name in event["trigger_metrics"])
            for event in exit_events
        )
        longest_steps = max(
            (event["outside_state_steps"] for event in exit_events), default=0
        )
        total_exit_steps = sum(
            event["outside_state_steps"] for event in exit_events
        )
        final_steps = max(1, int(round(1.0 / dt)))
        final_slice = active_steps[-final_steps:] if active_steps.size else np.array([], dtype=int)
        final_inside_fraction = (
            float(np.mean(enter[final_slice, env])) if final_slice.size else 0.0
        )
        final_inside = bool(final_slice.size and final_inside_fraction >= 0.95)
        durable = bool(recovery_confirmations and state == "RECOVERED")
        durable_step = recovery_confirmations[-1] if durable else None
        recovered_again = any(event["recovered_again"] for event in exit_events)
        events.append(
            {
                "env_id": env,
                "first_envelope_entry_step": first_entry,
                "first_envelope_entry_s": None if first_entry is None else first_entry * dt,
                "first_entry_s": None if first_entry is None else first_entry * dt,
                "first_dwell_start_step": first_dwell_start,
                "first_dwell_start_s": (
                    None if first_dwell_start is None else first_dwell_start * dt
                ),
                "dwell_start_s": (
                    None if first_dwell_start is None else first_dwell_start * dt
                ),
                "first_dwell_completion_step": first_confirmed,
                "first_recovery_s": (
                    None if first_confirmed is None else first_confirmed * dt
                ),
                "recovery_confirmed_step": first_confirmed,
                "recovery_confirmed_s": (
                    None if first_confirmed is None else first_confirmed * dt
                ),
                "recovery_time_s": (
                    None if first_confirmed is None else first_confirmed * dt
                ),
                "recovered": first_confirmed is not None,
                "transient_recovery": first_confirmed is not None,
                "durable_recovery": durable,
                "durable_recovery_step": durable_step,
                "durable_recovery_time_s": (
                    None if durable_step is None else durable_step * dt
                ),
                "exit_events": exit_events,
                "post_recovery_exit_count": len(exit_events),
                "first_exit_step": None if not exit_events else exit_events[0]["exit_step"],
                "first_exit_s": None if not exit_events else exit_events[0]["exit_time_s"],
                "longest_exit_steps": longest_steps,
                "longest_exit_s": longest_steps * dt,
                "longest_exit_duration_s": longest_steps * dt,
                "longest_threshold_violation_steps": max(
                    (
                        event["exit_threshold_violation_steps"]
                        for event in exit_events
                    ),
                    default=0,
                ),
                "longest_threshold_violation_duration_s": max(
                    (
                        event["threshold_violation_duration_s"]
                        for event in exit_events
                    ),
                    default=0.0,
                ),
                "total_exit_steps": int(total_exit_steps),
                "total_exit_s": float(total_exit_steps * dt),
                "final_1s_inside_fraction": final_inside_fraction,
                "final_1s_inside": final_inside,
                "final_1s_stable": final_inside,
                "exit_trigger_metrics": trigger_names,
                "exit_triggers": trigger_names,
                "core_exit": core_exit,
                "ankle_only_exit": ankle_only,
                "recovered_again": recovered_again,
                "classification": category,
                "classification_label": category,
            }
        )
    return events


def summarize_events(events: list[dict], safety_terminations: int) -> dict:
    recovered = [event for event in events if event["recovered"]]
    times = np.asarray([event["recovery_time_s"] for event in recovered], dtype=float)
    durable = [event for event in events if event["durable_recovery"]]
    durable_times = np.asarray(
        [event["durable_recovery_time_s"] for event in durable], dtype=float
    )
    exit_events = [
        exit_event for episode in events for exit_event in episode["exit_events"]
    ]
    category_names = (
        "core_body_reinstability",
        "height_instability",
        "waist_velocity_only",
        "ankle_velocity_only",
        "multi_metric_exit",
        "numerical_or_contact_spike_contact_unverified",
        "unclassified",
    )
    counts = {
        name: sum(event["classification"] == name for event in exit_events)
        for name in category_names
    }
    return {
        "episode_count": len(events),
        "recovered_count": len(recovered),
        "recovery_rate": len(recovered) / len(events) if events else None,
        "transient_recovered_count": len(recovered),
        "transient_recovery_rate": len(recovered) / len(events) if events else None,
        "recovery_time_s": stats(times),
        "durable_recovered_count": len(durable),
        "durable_recovery_rate": len(durable) / len(events) if events else None,
        "durable_recovery_time_s": stats(durable_times),
        "post_recovery_exits": int(sum(event["post_recovery_exit_count"] for event in events)),
        "exit_cycle_count": len(exit_events),
        "episodes_with_post_exit": int(sum(event["post_recovery_exit_count"] > 0 for event in events)),
        "ankle_only_exit_episodes": int(sum(event["ankle_only_exit"] for event in events)),
        "core_exit_episodes": int(sum(event["core_exit"] for event in events)),
        "recovered_again_episodes": int(sum(event["recovered_again"] for event in events)),
        "final_1s_stable_count": int(sum(event["final_1s_inside"] for event in events)),
        "final_1s_stable_rate": (
            float(np.mean([event["final_1s_stable"] for event in events]))
            if events
            else None
        ),
        "mean_final_1s_inside_fraction": float(
            np.mean([event["final_1s_inside_fraction"] for event in events])
        ),
        "safety_terminations": int(safety_terminations),
        "classification": {
            name: {
                "count": count,
                "fraction_of_exit_events": count / len(exit_events) if exit_events else None,
                "total_exit_duration_s": float(
                    sum(
                        event["outside_state_duration_s"]
                        for event in exit_events
                        if event["classification"] == name
                    )
                ),
                "maximum_exit_duration_s": max(
                    (
                        event["outside_state_duration_s"]
                        for event in exit_events
                        if event["classification"] == name
                    ),
                    default=0.0,
                ),
                "recovered_again_count": int(
                    sum(
                        event["recovered_again"]
                        for event in exit_events
                        if event["classification"] == name
                    )
                ),
            }
            for name, count in counts.items()
        },
    }


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    candidate = ROOT / path
    if candidate.exists():
        return candidate
    raise ValueError(f"referenced path does not exist: {path_text}")


def load_and_validate(
    calibration_paths: dict[str, Path], envelope_path: Path, contract_path: Path, trace_path: Path
) -> tuple[dict, dict, dict, dict]:
    """Strictly validate all immutable source artifacts and load trajectory arrays."""
    calibrations = {name: json.loads(path.read_text()) for name, path in calibration_paths.items()}
    envelope_source = json.loads(envelope_path.read_text())
    contract = json.loads(contract_path.read_text())
    contract_hash = sha256(contract_path)
    trace_hash = sha256(trace_path)
    if contract["trace"]["sha256"] != trace_hash:
        raise ValueError("runtime contract trace hash mismatch")
    with np.load(trace_path, allow_pickle=False) as payload:
        trace_indices = payload["trace_index"]
        trace_profile = payload["profile_id"]
    trajectories = {}
    integrity = {}
    shared_trace = set()
    shared_contract = set()
    for short_name, calibration in calibrations.items():
        if not calibration.get("calibration_measured") or not calibration.get("runtime_integrity_passed"):
            raise ValueError(f"{short_name}: incomplete calibration")
        if calibration["num_envs_per_profile"] != 256 or calibration["policy_steps"] != 500:
            raise ValueError(f"{short_name}: expected exactly 256 envs and 500 steps")
        shared_trace.add(calibration["disturbance_trace_sha256"])
        shared_contract.add(calibration["runtime_contract_sha256"])
        profile = calibration["profiles"][0]
        if profile["profile"] != EXPECTED_NAMES[short_name]:
            raise ValueError(f"{short_name}: profile mismatch")
        if sum(profile["termination_term_counts"].values()) != 256:
            raise ValueError(f"{short_name}: termination sum is not 256")
        trajectory_path = _resolve(profile["trajectory_path"])
        actual_trajectory_hash = sha256(trajectory_path)
        if actual_trajectory_hash != profile["trajectory_sha256"]:
            raise ValueError(f"{short_name}: trajectory SHA mismatch")
        with np.load(trajectory_path, allow_pickle=False) as payload:
            missing = set(ALL_CHANNELS) | {"active", "disturbed", "trace_index", "policy_dt_s"}
            missing -= set(payload.files)
            if missing:
                raise ValueError(f"{short_name}: missing arrays {sorted(missing)}")
            # Preserve source dtype: NumPy's percentile interpolation can differ by
            # a few float32 ULPs after eager float64 conversion.
            arrays = {name: payload[name].copy() for name in ALL_CHANNELS}
            active = payload["active"].astype(bool)
            disturbed = payload["disturbed"].astype(bool)
            selected = payload["trace_index"].astype(np.int64)
            dt = float(payload["policy_dt_s"][0])
        if active.shape != (500, 256) or any(array.shape != (500, 256) for array in arrays.values()):
            raise ValueError(f"{short_name}: trajectory shape is not 500x256")
        if disturbed.shape != (256,) or selected.shape != (256,) or len(np.unique(selected)) != 256:
            raise ValueError(f"{short_name}: trace_index/disturbed shape or uniqueness failure")
        if not all(np.all(np.isfinite(array)) for array in arrays.values()) or not np.isfinite(dt):
            raise ValueError(f"{short_name}: non-finite trajectory data")
        if hashlib.sha256(selected.astype(np.int32).tobytes()).hexdigest() != profile["trace_index_sha256"]:
            raise ValueError(f"{short_name}: trace_index SHA mismatch")
        if np.any(selected < 0) or np.any(selected >= trace_indices.size):
            raise ValueError(f"{short_name}: trace_index out of range")
        if not np.array_equal(trace_indices[selected], selected):
            raise ValueError(f"{short_name}: trace_index identity mismatch")
        if not np.all(trace_profile[selected] == PROFILE_IDS[short_name]):
            raise ValueError(f"{short_name}: trace_index profile mismatch")
        expected_disturbed = short_name != "clean"
        if not np.all(disturbed == expected_disturbed):
            raise ValueError(f"{short_name}: disturbed flags mismatch")
        trajectories[short_name] = {
            "arrays": arrays,
            "active": active,
            "disturbed": disturbed,
            "trace_index": selected,
            "dt": dt,
            "profile": profile,
        }
        integrity[short_name] = {
            "calibration_sha256": sha256(calibration_paths[short_name]),
            "trajectory_sha256": actual_trajectory_hash,
            "trace_index_sha256": profile["trace_index_sha256"],
            "termination_sum": 256,
            "shape": [500, 256],
            "finite": True,
            "trace_profile_correspondence": True,
            "trace_index_unique": True,
        }
    if shared_trace != {trace_hash} or shared_contract != {contract_hash}:
        raise ValueError("calibrations do not share verified trace/runtime contract hashes")
    if envelope_source["source_trace_sha256"] != trace_hash:
        raise ValueError("envelope source trace hash mismatch")
    if envelope_source["source_runtime_contract_sha256"] != contract_hash:
        raise ValueError("envelope source runtime contract hash mismatch")
    for source in envelope_source["source_calibrations"]:
        source_path = _resolve(source)
        if source_path.name not in {path.name for path in calibration_paths.values()}:
            raise ValueError("envelope calibration source set mismatch")
    return calibrations, trajectories, envelope_source, {
        "passed": True,
        "trace_sha256": trace_hash,
        "runtime_contract_sha256": contract_hash,
        "envelope_source_trace_and_contract_hashes_verified": True,
        "envelope_source_calibration_hashes_available": False,
        "envelope_source_calibration_hash_limitation": (
            "source envelope records calibration paths but not calibration SHA256 values"
        ),
        "profiles": integrity,
    }


def build_envelopes(
    clean_arrays: dict[str, np.ndarray], active: np.ndarray, dt: float, current: dict
) -> tuple[dict, dict, dict[str, np.ndarray]]:
    tail_steps = int(round(2.0 / dt))
    tail_active = active[-tail_steps:]
    clean_stats = {}
    for channel, values in clean_arrays.items():
        samples = values[-tail_steps:][tail_active]
        channel_stats = stats(samples)
        p99 = channel_stats["p99"]
        channel_stats["above_p99_sample_count"] = int(np.count_nonzero(samples > p99))
        above = (values[-tail_steps:] > p99) & tail_active
        channel_stats["above_p99_environment_count"] = int(np.count_nonzero(np.any(above, axis=0)))
        max_run = max((end - start for env in range(above.shape[1]) for start, end in _runs(above[:, env])), default=0)
        channel_stats["longest_above_p99_steps"] = max_run
        channel_stats["tail_shape_assessment"] = (
            "sustained_anomaly" if max_run > max(2, int(round(0.10 / dt))) else "short_spikes"
        )
        clean_stats[channel] = channel_stats
    recomputed_a = {}
    for channel in ALL_CHANNELS:
        p99 = clean_stats[channel]["p99"]
        enter = max(p99 * 1.25, FLOORS.get(channel, 0.05))
        recomputed_a[channel] = {
            "enter_threshold": enter,
            "exit_threshold": enter * 1.25,
            "source": "clean_tail_raw_p99_x_1.25_or_floor",
        }
        if not np.isclose(enter, current[channel]["enter_threshold"], rtol=2e-6, atol=1e-9):
            raise ValueError(f"current envelope does not reproduce from clean tail: {channel}")
    envelope_b = {name: dict(recomputed_a[name]) for name in CORE}
    transformed = dict(clean_arrays)
    envelope_c = {name: dict(recomputed_a[name]) for name in CORE}
    window = max(1, int(round(0.20 / dt)))
    for channel in AUX:
        transformed_name = f"{channel}__rms_200ms"
        transformed[transformed_name] = rolling_rms(clean_arrays[channel], window)
        samples = transformed[transformed_name][-tail_steps:][tail_active]
        enter = max(float(np.quantile(samples, 0.99)) * 1.25, 0.05)
        envelope_c[transformed_name] = {
            "enter_threshold": enter,
            "exit_threshold": enter * 1.25,
            "source": "clean_tail_200ms_rms_p99_x_1.25_or_floor",
        }
    return {"A_current_full": recomputed_a, "B_core_only": envelope_b, "C_core_plus_aux_200ms_rms": envelope_c}, clean_stats, transformed


def _transform_for_c(arrays: dict[str, np.ndarray], dt: float) -> dict[str, np.ndarray]:
    transformed = dict(arrays)
    window = max(1, int(round(0.20 / dt)))
    for channel in AUX:
        transformed[f"{channel}__rms_200ms"] = rolling_rms(arrays[channel], window)
    return transformed


def clean_episode_metrics(
    arrays: dict[str, np.ndarray], active: np.ndarray, envelope: dict, dt: float
) -> dict:
    enter, exit_inside, _ = _threshold_masks(arrays, envelope)
    episodes = []
    final_steps = int(round(1.0 / dt))
    for env in range(active.shape[1]):
        valid = active[:, env]
        inside = valid & enter[:, env]
        outside = valid & ~exit_inside[:, env]
        outside_runs = _runs(outside)
        transitions = np.diff(np.concatenate(([False], inside, [False])).astype(np.int8))
        active_idx = np.flatnonzero(valid)
        final_idx = active_idx[-final_steps:]
        final_inside_fraction = (
            float(np.mean(enter[final_idx, env])) if final_idx.size else 0.0
        )
        episodes.append(
            {
                "env_id": env,
                "inside_fraction": float(np.mean(inside[valid])),
                "entry_count": int(np.count_nonzero(transitions == 1)),
                "exit_count": len(outside_runs),
                "longest_outside_steps": max((end - start for start, end in outside_runs), default=0),
                "longest_outside_s": max((end - start for start, end in outside_runs), default=0) * dt,
                "final_1s_inside_fraction": final_inside_fraction,
                "final_1s_inside": bool(final_idx.size and final_inside_fraction == 1.0),
            }
        )
    return {
        "episodes": episodes,
        "mean_inside_fraction": float(np.mean([item["inside_fraction"] for item in episodes])),
        "total_exit_count": int(sum(item["exit_count"] for item in episodes)),
        "final_1s_inside_fraction": float(np.mean([item["final_1s_inside"] for item in episodes])),
        "mean_final_1s_inside_fraction": float(
            np.mean([item["final_1s_inside_fraction"] for item in episodes])
        ),
    }


def _safety_count(profile: dict) -> int:
    return int(sum(value for key, value in profile["termination_term_counts"].items() if key != "time_out"))


def _manifest_entry(profile: str, event: dict, trace_index: int, reason: str) -> dict:
    base_time_s = 8.0 if profile == "clean" else 0.0
    focus_time_s = event.get("first_exit_s")
    if focus_time_s is None:
        focus_time_s = event.get("first_recovery_s")
    if focus_time_s is None:
        time_window = [0.0, 10.0]
    else:
        focus_time_s += base_time_s
        time_window = [max(0.0, focus_time_s - 1.0), min(10.0, focus_time_s + 2.0)]
    triggers = event.get("exit_trigger_metrics", event.get("exit_triggers", []))
    if any(name in CORE for name in triggers):
        camera_focus = "pelvis"
    elif any(name in ANKLE for name in triggers) or "ankle" in reason:
        camera_focus = "feet"
    else:
        camera_focus = "full_body"
    return {
        "profile": profile,
        "env_id": event["env_id"],
        "trace_index": int(trace_index),
        "reason": reason,
        "time_window_s": time_window,
        "suggested_camera_focus": camera_focus,
        "recovered": event.get("recovered"),
        "post_recovery_exit_count": event.get("post_recovery_exit_count"),
        "longest_exit_s": event.get("longest_exit_s"),
        "exit_trigger_metrics": triggers,
        "termination_reason": event.get("termination_reason"),
    }


def build_manifest(
    trajectories: dict, events_a: dict, clean_metrics: dict, ankle_scores: np.ndarray
) -> dict:
    groups = {}
    clean_trace = trajectories["clean"]["trace_index"]
    clean_events = events_a["clean"]
    exits_sorted = sorted(clean_metrics["episodes"], key=lambda item: (-item["exit_count"], -item["longest_outside_steps"], item["env_id"]))
    groups["clean_top5_exits"] = [
        _manifest_entry("clean", clean_events[item["env_id"]], clean_trace[item["env_id"]], "clean_exit_rank")
        for item in exits_sorted[:5]
    ]
    ankle_order = np.argsort(-ankle_scores)
    groups["clean_ankle_max_top5"] = [
        _manifest_entry("clean", clean_events[int(env)], clean_trace[int(env)], "clean_ankle_max_rank")
        for env in ankle_order[:5]
    ]
    for profile in ("candidate", "medium"):
        groups[f"{profile}_unrecovered_all"] = [
            _manifest_entry(profile, event, trajectories[profile]["trace_index"][event["env_id"]], "unrecovered")
            for event in events_a[profile] if not event["recovered"]
        ]
    upper_unrecovered = [event for event in events_a["upper"] if not event["recovered"]]
    groups["upper_unrecovered_representative"] = [
        _manifest_entry("upper", event, trajectories["upper"]["trace_index"][event["env_id"]], "unrecovered_representative")
        for event in upper_unrecovered[:10]
    ]
    early = np.flatnonzero(np.sum(trajectories["upper"]["active"], axis=0) < trajectories["upper"]["active"].shape[0])
    safety_entries = []
    for env in early[:9]:
        event = dict(events_a["upper"][int(env)])
        event["termination_reason"] = "recovery_envelope (inferred from early active end and aggregate count; exact per-env reason unknown)"
        safety_entries.append(
            _manifest_entry("upper", event, trajectories["upper"]["trace_index"][env], "safety_termination_inferred")
        )
    groups["upper_safety_terminations_9"] = safety_entries
    all_disturbed = [
        (profile, event)
        for profile in ("candidate", "medium", "upper")
        for event in events_a[profile]
    ]
    selectors = {
        "post_exits_top10": lambda pair: (-pair[1]["post_recovery_exit_count"], -pair[1]["longest_exit_steps"]),
        "ankle_only_top10": lambda pair: (not pair[1]["ankle_only_exit"], -pair[1]["post_recovery_exit_count"]),
        "core_body_top10": lambda pair: (not pair[1]["core_exit"], -pair[1]["longest_exit_steps"]),
    }
    for group, key in selectors.items():
        selected = sorted(all_disturbed, key=key)
        if group == "ankle_only_top10":
            selected = [pair for pair in selected if pair[1]["ankle_only_exit"]]
        if group == "core_body_top10":
            selected = [pair for pair in selected if pair[1]["core_exit"]]
        groups[group] = [
            _manifest_entry(profile, event, trajectories[profile]["trace_index"][event["env_id"]], group)
            for profile, event in selected[:10]
        ]
    return {
        "groups": groups,
        "replay_tooling_available": False,
        "replay_note": "Existing calibration runner does not expose a documented single-trace-index replay command; manifest only.",
    }


def _markdown(result: dict) -> str:
    a = result["envelope_comparison"]["A_current_full"]
    clean_stats = result["clean_tail_statistics"]
    right_pitch = clean_stats[
        "abs_joint_velocity_rad_s/right_ankle_pitch_joint"
    ]
    right_roll = clean_stats[
        "abs_joint_velocity_rad_s/right_ankle_roll_joint"
    ]
    left_pitch = clean_stats[
        "abs_joint_velocity_rad_s/left_ankle_pitch_joint"
    ]
    left_roll = clean_stats[
        "abs_joint_velocity_rad_s/left_ankle_roll_joint"
    ]
    lines = [
        "# A3 Base Recovery-A Envelope Manual Review V1",
        "",
        "## Decision",
        "",
        f"**Recommendation: `{result['manual_decision_recommendation']}`.** "
        "This is a fail-closed manual-review recommendation, not a gate change or approval.",
        "",
        result["decision_rationale"],
        "",
        "All gate, training-environment, reward, and disturbance settings remained untouched. No PPO was run and no source calibration was overwritten.",
        "",
        "## Evidence availability",
        "",
        "Available: seven core body channels, six waist/ankle joint-velocity channels, active, disturbed, trace_index, and policy_dt_s.",
        "",
        "**Optional evidence unavailable:** torso state; ankle target, actual position, and torque; foot load and contact. "
        "Therefore contact causality cannot be verified and `numerical_or_contact_spike` labels are reported as `contact_unverified`.",
        "",
        "## Integrity",
        "",
        f"- Strict source validation: `{result['integrity']['passed']}`; four trajectories are 500×256, finite, uniquely trace-indexed, and profile-matched.",
        f"- Trace SHA: `{result['integrity']['trace_sha256']}`.",
        f"- Runtime contract SHA: `{result['integrity']['runtime_contract_sha256']}`.",
        "- The source envelope contains source calibration paths but not their SHA256 values; current calibration files are hashed in this package and the envelope thresholds are independently reproduced from clean data.",
        "",
        "## Clean steady-state review",
        "",
        f"- Review result: `clean_steady_state_valid={result['clean_steady_state_valid']}` using the final 2 seconds. "
        f"Current full-envelope mean inside fraction is {a['clean']['mean_inside_fraction']:.4f}; "
        f"mean final-1s inside fraction is {a['clean']['mean_final_1s_inside_fraction']:.4f}; "
        f"there are {a['clean']['total_exit_count']} exit runs.",
        f"- Right ankle pitch raw p95/p99/max: {right_pitch['p95']:.4f}/{right_pitch['p99']:.4f}/{right_pitch['max']:.4f} rad/s; "
        f"tail pattern: `{right_pitch['tail_shape_assessment']}`, across {right_pitch['above_p99_environment_count']} environments.",
        f"- Right ankle roll raw p95/p99/max: {right_roll['p95']:.4f}/{right_roll['p99']:.4f}/{right_roll['max']:.4f} rad/s; "
        f"tail pattern: `{right_roll['tail_shape_assessment']}`, across {right_roll['above_p99_environment_count']} environments. "
        "Roll, not pitch, is the larger clean-tail p99 target.",
        f"- Left/right pitch p99: {left_pitch['p99']:.4f}/{right_pitch['p99']:.4f} rad/s; "
        f"left/right roll p99: {left_roll['p99']:.4f}/{right_roll['p99']:.4f} rad/s. "
        "The roll channel is materially asymmetric.",
        "- Only absolute-valued channels were recorded, so signed one-way drift cannot be proven or excluded. "
        "The package reports persistence and environment concentration instead of inventing drift direction.",
        "",
        "## Current-envelope outcome",
        "",
        f"- Clean mean inside fraction: {a['clean']['mean_inside_fraction']:.4f}; exits: {a['clean']['total_exit_count']}.",
    ]
    for profile in ("candidate", "medium", "upper"):
        summary = a["profiles"][profile]
        lines.append(
            f"- {profile}: recovery {summary['recovered_count']}/256 "
            f"({summary['recovery_rate']:.2%}), p50/p90/p95 "
            f"{summary['recovery_time_s']['median']:.2f}/"
            f"{summary['recovery_time_s']['p90']:.2f}/"
            f"{summary['recovery_time_s']['p95']:.2f}s, "
            f"post exits {summary['post_recovery_exits']}, ankle-only episodes "
            f"{summary['ankle_only_exit_episodes']}, safety terminations {summary['safety_terminations']}."
        )
    lines += ["", "## Post-recovery exit classification", ""]
    for profile in ("candidate", "medium", "upper"):
        summary = a["profiles"][profile]
        categories = ", ".join(
            f"{name}={details['count']}"
            for name, details in summary["classification"].items()
            if details["count"]
        )
        lines.append(
            f"- {profile}: episodes with exits={summary['episodes_with_post_exit']}, "
            f"re-entered={summary['recovered_again_episodes']}; {categories}."
        )
    lines += [
        "- `numerical_or_contact_spike` requires an auxiliary-only excursion of at most two policy steps. "
        "Because contact/load data are absent, every such label remains `contact_unverified`; true contact causality is not claimed.",
    ]
    lines += [
        "",
        "## Envelope A/B/C",
        "",
        "A is current full raw-channel logic; B uses only the seven core channels; C uses core plus all waist/ankle 200 ms RMS channels. "
        "C thresholds are independently derived from the clean-tail RMS p99×1.25/floor, with exit=1.25×enter.",
    ]
    for name, comparison in result["envelope_comparison"].items():
        rates = ", ".join(
            f"{profile}={comparison['profiles'][profile]['recovery_rate']:.2%}"
            for profile in ("candidate", "medium", "upper")
        )
        lines.append(
            f"- {name}: clean inside {comparison['clean']['mean_inside_fraction']:.4f}, "
            f"clean exits {comparison['clean']['total_exit_count']}; recovery {rates}."
        )
    lines += [
        "",
        "## Right-ankle review",
        "",
        "Raw absolute velocity, 50/100/200 ms RMS, and 100/200 ms rolling p95 are included in JSON for right ankle pitch. "
        "Right ankle roll is also included because its clean raw p99 is larger, avoiding a preselected target.",
        "",
        f"Recommendation option: **{result['right_ankle_special_review']['recommendation_option']}**. "
        "Evidence is insufficient to attribute spikes to contact or to justify deleting an ankle channel.",
        "The raw, 50/100/200 ms RMS, and 100/200 ms rolling-p95 distributions are preserved in the JSON. "
        "No single-frame/RMS replacement is approved yet; Envelope C is only the preferred revision candidate.",
        "",
        "## Sensitivity and timing logic",
        "",
        "Scales 0.90/0.95/1.00/1.05/1.10 were run for A/B/C. `materially_sensitive` is a review heuristic only: "
        ">10 percentage-point recovery-rate shift or >1 s p90 recovery-time shift versus 1.00. "
        "Dwell 0.20/0.30/0.50 s and hysteresis none/light/current (1.0/1.10/1.25) were also run. Hysteresis changes post-entry exits only, never first entry.",
        f"- Current A envelope robust under this heuristic: `{result['current_envelope_robust']}`.",
        "- A/B/C materially-sensitive flags: "
        + ", ".join(
            f"{name}={value['materially_sensitive']}"
            for name, value in result["sensitivity_results"].items()
        )
        + ".",
        "- Candidate timing recommendation: dwell=0.30 s and hysteresis ratio=1.25, pending replay/contact evidence. "
        "This is a review candidate, not an approved contract.",
        "",
        "## Recommended metric roles",
        "",
        "- Core: pelvis roll/pitch, root angular velocity x/y, root linear velocity x/y, and base-height error.",
        "- Auxiliary candidate: 200 ms RMS waist and ankle velocities.",
        "- Quality-only: raw single-frame waist and ankle velocities.",
        "- Excluded: none. No channel is deleted on the present evidence.",
        "",
        "## Episode review manifest",
        "",
        "The JSON includes clean top exits, clean ankle maxima, all candidate/medium unrecovered episodes, representative upper unrecovered episodes, "
        "nine inferred upper safety terminations, and top post-exit/ankle-only/core-body cases. Exact per-environment termination reasons are unknown; "
        "upper safety rows are inferred from early `active` endings plus the aggregate termination count.",
        "",
        "Replay tooling unavailable: the existing runner does not provide a documented single-trace-index replay command, so the package provides a manifest only.",
        "",
        "## Approval state",
        "",
        "**Do not approve `recovery_envelope_approved` yet.** The recommendation is `revise`.",
        "",
        "Minimum additional evidence: record signed torso/root state, ankle target/actual/torque, and foot load/contact for the selected manifest episodes; "
        "replay the dominant core-body, ankle-only, and short-spike cases; then rerun the same sensitivity package for the revised windowed envelope.",
        "",
        "All approval fields are false. `gate_mutated=false`. This package is analysis evidence only.",
    ]
    return "\n".join(lines) + "\n"


def _markdown_state_machine(result: dict) -> str:
    comparisons = result["envelope_comparison"]
    b_sensitivity = result["sensitivity_results"]["B_core_only"]
    timing = result["recommended_envelope"]["timing_results"]["profiles"]
    difficulty = sorted(
        ("candidate", "medium", "upper"),
        key=lambda profile: comparisons["B_core_only"]["profiles"][profile][
            "recovery_time_s"
        ]["p90"]
        or float("inf"),
    )
    lines = [
        "# A3 Base Recovery-A Envelope Manual Review V1",
        "",
        "## Decision",
        "",
        f"**Recommendation: `{result['manual_decision_recommendation']}`.** "
        "Envelope B may enter manual-approval review as `candidate_not_approved`; this report does not approve it.",
        "",
        result["decision_rationale"],
        "",
        "No gate, training environment, reward, disturbance setting, or source trajectory was changed. No PPO or stochastic audit was run.",
        "",
        "## Corrected state-machine semantics",
        "",
        "The analyzer uses `RECOVERING -> RECOVERED -> OUTSIDE`. An exit cycle starts only on "
        "`RECOVERED -> OUTSIDE`. Threshold crossings while already OUTSIDE do not create more cycles. "
        "Another cycle is possible only after a complete enter-envelope dwell confirms recovery.",
        "",
        "**The old 1563/894/541 values are deprecated and invalid as exit-cycle counts.** "
        "They counted threshold runs while an episode was already outside.",
        "",
        "`recovery_time_s` is dwell completion, not dwell start. `longest_exit_duration_s` is the OUTSIDE-state "
        "duration until confirmed recovery or episode end; threshold-violation duration is separate.",
        "",
        "## A/B/C corrected outcomes",
        "",
    ]
    for set_name, comparison in comparisons.items():
        lines.append(
            f"- **{set_name}**: clean-tail inside={comparison['clean']['mean_inside_fraction']:.4f}, "
            f"clean threshold runs={comparison['clean']['total_exit_count']}."
        )
        for profile in ("candidate", "medium", "upper"):
            summary = comparison["profiles"][profile]
            categories = ", ".join(
                f"{name}={details['count']}"
                for name, details in summary["classification"].items()
                if details["count"]
            ) or "none"
            lines.append(
                f"  - {profile}: transient={summary['transient_recovery_rate']:.2%}, "
                f"durable={summary['durable_recovery_rate']:.2%}, "
                f"final1s={summary['final_1s_stable_rate']:.2%}, "
                f"exit cycles={summary['exit_cycle_count']}; {categories}."
            )
    lines += [
        "",
        "Exit-cycle counts are not expected to decrease monotonically from A to B to C: a less restrictive "
        "envelope can confirm recovery earlier and more often, creating more legitimate opportunities for a "
        "later RECOVERED-to-OUTSIDE transition. Durable recovery and final-1s stability must therefore be "
        "reviewed alongside the cycle count.",
        "",
        "## Recommended candidate structure",
        "",
        "**Envelope B (`B_core_only`) is the recommended hard-recovery candidate, status "
        "`candidate_not_approved`.** Its seven core metrics are the only hard recovery criteria. "
        "All raw waist/ankle velocities and all 200 ms RMS waist/ankle velocities are quality metrics; "
        "they do not veto hard recovery.",
        "",
        "Envelope C remains a `candidate_not_approved` comparison. Dwell=0.30 s and hysteresis=1.25 "
        "are research-candidate settings only.",
        "",
        "B 0.30 s/1.25 corrected research result: "
        + "; ".join(
            f"{profile} transient={timing[profile]['transient_recovery_rate']:.2%}, "
            f"durable={timing[profile]['durable_recovery_rate']:.2%}, "
            f"final1s={timing[profile]['final_1s_stability_fraction']:.2%}, "
            f"exit cycles={timing[profile]['exit_cycle_count']}"
            for profile in ("candidate", "medium", "upper")
        )
        + ".",
        "",
        f"B materially-sensitive={b_sensitivity['materially_sensitive']} under the documented review heuristic; "
        f"difficulty ordering by baseline p90 recovery time is {' < '.join(difficulty)}.",
        "",
        "## Per-exit classification",
        "",
        "Classification is per exit event. No-exit episodes are `no_post_recovery_exit`; episodes with different "
        "event classes are `multiple_exit_categories`. Only auxiliary-only events with at most two actual "
        "exit-threshold violation steps may be `numerical_or_contact_spike_contact_unverified`.",
        "",
        "## Right-ankle decision",
        "",
        "**Option E:** temporarily remove raw and RMS ankle velocity from hard recovery decisions while retaining "
        "both as quality evidence. Missing contact/torque evidence blocks promotion of auxiliary metrics into a "
        "hard gate; it does not block later human approval of the core-only candidate.",
        "",
        "## Evidence and integrity",
        "",
        f"Strict integrity validation passed={result['integrity']['passed']}: four finite 500×256 trajectories, "
        "unique profile-matched trace indices, verified hashes and termination sums, and reproduced source envelope.",
        "",
        "**Optional evidence unavailable:** torso; ankle target/actual/torque; foot load/contact. "
        "No unavailable evidence is fabricated.",
        "",
        "The JSON retains the requested episode manifest. Upper per-environment termination reasons are "
        "inferred/unknown. Replay tooling remains unavailable.",
        "",
        "## Approval state",
        "",
        "All approval fields are false and `gate_mutated=false`. No candidate is automatically approved.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument(
        "--output", type=Path, default=ARTIFACT_DIR / "recovery_envelope_manual_review_package_v1.json"
    )
    parser.add_argument(
        "--markdown", type=Path, default=ROOT / "docs/A3_BASE_RECOVERY_ENVELOPE_MANUAL_REVIEW_V1.md"
    )
    args = parser.parse_args()
    calibration_paths = {
        name: args.artifact_dir / filename for name, filename in PROFILE_FILES.items()
    }
    envelope_path = args.artifact_dir / "recovery_envelope_analysis_v1.json"
    contract_path = args.artifact_dir / "recovery_runtime_contract_v1.json"
    trace_path = args.artifact_dir / "recovery_disturbance_trace_v1.npz"
    calibrations, trajectories, envelope_source, integrity = load_and_validate(
        calibration_paths, envelope_path, contract_path, trace_path
    )
    dt = trajectories["clean"]["dt"]
    if any(not np.isclose(item["dt"], dt) for item in trajectories.values()):
        raise ValueError("policy_dt_s mismatch")
    envelopes, clean_tail_stats, clean_c_arrays = build_envelopes(
        trajectories["clean"]["arrays"],
        trajectories["clean"]["active"],
        dt,
        envelope_source["envelope"],
    )
    arrays_by_envelope = {
        "A_current_full": {name: trajectories[name]["arrays"] for name in trajectories},
        "B_core_only": {name: trajectories[name]["arrays"] for name in trajectories},
        "C_core_plus_aux_200ms_rms": {
            name: clean_c_arrays if name == "clean" else _transform_for_c(trajectories[name]["arrays"], dt)
            for name in trajectories
        },
    }
    clean_tail_steps = int(round(2.0 / dt))

    def clean_tail_arrays(set_name: str) -> dict[str, np.ndarray]:
        return {
            name: values[-clean_tail_steps:]
            for name, values in arrays_by_envelope[set_name]["clean"].items()
        }

    clean_tail_active = trajectories["clean"]["active"][-clean_tail_steps:]
    comparison = {}
    events_by_set = {}
    for set_name, envelope in envelopes.items():
        events_by_set[set_name] = {}
        clean = clean_episode_metrics(
            clean_tail_arrays(set_name), clean_tail_active, envelope, dt
        )
        profiles = {}
        for profile in ("candidate", "medium", "upper"):
            events = episode_events(
                arrays_by_envelope[set_name][profile],
                trajectories[profile]["active"],
                envelope,
                dt,
            )
            events_by_set[set_name][profile] = events
            profiles[profile] = summarize_events(events, _safety_count(trajectories[profile]["profile"]))
        comparison[set_name] = {"definition": envelope, "clean": clean, "profiles": profiles}
    # Clean gets event records too, solely for manifest identity.
    events_by_set["A_current_full"]["clean"] = episode_events(
        clean_tail_arrays("A_current_full"),
        clean_tail_active,
        envelopes["A_current_full"],
        dt,
    )
    sensitivity = {}
    for set_name, envelope in envelopes.items():
        runs = {}
        for scale in (0.90, 0.95, 1.00, 1.05, 1.10):
            clean = clean_episode_metrics(
                clean_tail_arrays(set_name), clean_tail_active,
                {name: {key: value * scale if key.endswith("threshold") else value for key, value in threshold.items()} for name, threshold in envelope.items()},
                dt,
            )
            profiles = {}
            for profile in ("candidate", "medium", "upper"):
                events = episode_events(
                    arrays_by_envelope[set_name][profile], trajectories[profile]["active"],
                    envelope, dt, scale=scale
                )
                summary = summarize_events(events, _safety_count(trajectories[profile]["profile"]))
                profiles[profile] = {
                    "recovery_rate": summary["recovery_rate"],
                    "transient_recovery_rate": summary["transient_recovery_rate"],
                    "durable_recovery_rate": summary["durable_recovery_rate"],
                    "final_1s_stable_rate": summary["final_1s_stable_rate"],
                    "recovery_time_p50_s": summary["recovery_time_s"]["median"],
                    "recovery_time_p90_s": summary["recovery_time_s"]["p90"],
                    "post_recovery_exits": summary["exit_cycle_count"],
                    "exit_cycle_count": summary["exit_cycle_count"],
                    "safety_terminations": summary["safety_terminations"],
                }
            runs[f"{scale:.2f}"] = {
                "clean_inside_fraction": clean["mean_inside_fraction"], "profiles": profiles
            }
        baseline = runs["1.00"]
        materially_sensitive = False
        reasons = []
        for scale, run in runs.items():
            if scale == "1.00":
                continue
            for profile in ("candidate", "medium", "upper"):
                rate_shift = abs(run["profiles"][profile]["recovery_rate"] - baseline["profiles"][profile]["recovery_rate"])
                p90 = run["profiles"][profile]["recovery_time_p90_s"]
                base_p90 = baseline["profiles"][profile]["recovery_time_p90_s"]
                time_shift = abs(p90 - base_p90) if p90 is not None and base_p90 is not None else 0.0
                if rate_shift > 0.10 or time_shift > 1.0:
                    materially_sensitive = True
                    reasons.append({"scale": scale, "profile": profile, "rate_shift": rate_shift, "p90_shift_s": time_shift})
        sensitivity[set_name] = {
            "runs": runs,
            "materially_sensitive": materially_sensitive,
            "heuristic": ">10 percentage-point recovery-rate shift or >1s p90 shift vs scale 1.00",
            "reasons": reasons,
            "recovery_rate_ranking_at_1_00": sorted(
                ("candidate", "medium", "upper"),
                key=lambda profile: baseline["profiles"][profile]["recovery_rate"],
                reverse=True,
            ),
        }
    dwell_hysteresis = {}
    # Timing research is attached to the recommended core-only candidate.
    base_enter = envelopes["B_core_only"]
    for dwell_s in (0.20, 0.30, 0.50):
        for label, ratio in (("none", 1.0), ("light", 1.10), ("current", 1.25)):
            envelope = {
                name: {
                    **threshold,
                    "exit_threshold": threshold["enter_threshold"] * ratio,
                }
                for name, threshold in base_enter.items()
            }
            key = f"dwell_{dwell_s:.2f}s_hysteresis_{label}"
            dwell_hysteresis[key] = {
                "dwell_s": dwell_s,
                "hysteresis_ratio": ratio,
                "first_entry_invariant": True,
                "profiles": {},
            }
            for profile in ("candidate", "medium", "upper"):
                events = episode_events(
                    trajectories[profile]["arrays"], trajectories[profile]["active"],
                    envelope, dt, dwell_s=dwell_s
                )
                summary = summarize_events(events, _safety_count(trajectories[profile]["profile"]))
                dwell_hysteresis[key]["profiles"][profile] = {
                    "recovery_time_s": summary["recovery_time_s"],
                    "durable_recovery_time_s": summary["durable_recovery_time_s"],
                    "transient_recovery_rate": summary["transient_recovery_rate"],
                    "durable_recovery_rate": summary["durable_recovery_rate"],
                    "post_recovery_exits": summary["exit_cycle_count"],
                    "exit_cycle_count": summary["exit_cycle_count"],
                    "reentries": summary["recovered_again_episodes"],
                    "final_1s_stability_fraction": summary["final_1s_stable_rate"],
                }
    ankle_review = {}
    for channel in (
        "abs_joint_velocity_rad_s/right_ankle_pitch_joint",
        "abs_joint_velocity_rad_s/right_ankle_roll_joint",
    ):
        transforms = {}
        for profile, trajectory in trajectories.items():
            raw = trajectory["arrays"][channel]
            valid = trajectory["active"]
            transform_arrays = {
                "raw_abs": raw,
                "rms_50ms": rolling_rms(raw, max(1, int(round(0.05 / dt)))),
                "rms_100ms": rolling_rms(raw, max(1, int(round(0.10 / dt)))),
                "rms_200ms": rolling_rms(raw, max(1, int(round(0.20 / dt)))),
                "rolling_p95_100ms": rolling_quantile(raw, max(1, int(round(0.10 / dt))), 0.95),
                "rolling_p95_200ms": rolling_quantile(raw, max(1, int(round(0.20 / dt))), 0.95),
            }
            raw_threshold = envelopes["A_current_full"][channel]["exit_threshold"]
            core_enter, core_exit_inside, _ = _threshold_masks(
                trajectory["arrays"], {name: envelopes["A_current_full"][name] for name in CORE}
            )
            del core_enter
            above = valid & (raw > raw_threshold)
            sync = above & ~core_exit_inside
            max_run = max((end - start for env in range(256) for start, end in _runs(above[:, env])), default=0)
            transforms[profile] = {
                "distributions": {
                    name: stats(values[valid]) for name, values in transform_arrays.items()
                },
                "raw_exit_threshold": raw_threshold,
                "above_threshold_samples": int(np.count_nonzero(above)),
                "above_threshold_episode_count": int(np.count_nonzero(np.any(above, axis=0))),
                "longest_continuous_above_threshold_s": max_run * dt,
                "core_outside_synchronous_samples": int(np.count_nonzero(sync)),
                "core_sync_fraction_of_above_samples": (
                    float(np.count_nonzero(sync) / np.count_nonzero(above))
                    if np.count_nonzero(above) else None
                ),
            }
        ankle_review[channel] = transforms
    clean_ankle_score = np.max(
        trajectories["clean"]["arrays"]["abs_joint_velocity_rad_s/right_ankle_pitch_joint"], axis=0
    )
    manifest = build_manifest(
        trajectories,
        events_by_set["A_current_full"],
        comparison["A_current_full"]["clean"],
        clean_ankle_score,
    )
    a_summary = comparison["A_current_full"]["profiles"]
    clean_full = comparison["A_current_full"]["clean"]
    clean_steady_state_valid = bool(
        clean_full["mean_inside_fraction"] >= 0.95
        and clean_full["mean_final_1s_inside_fraction"] >= 0.95
    )
    current_envelope_robust = not sensitivity["A_current_full"]["materially_sensitive"]
    blocking_reasons = [
        "the core-only candidate requires explicit human review and approval",
        "upper retains nine aggregate safety terminations and remains diagnostic only",
        "contact/torque evidence is unavailable, so auxiliary velocity metrics cannot be promoted into hard recovery criteria",
    ]
    if not clean_steady_state_valid:
        blocking_reasons.append(
            "clean final-2s full-envelope stability is below the 95% review heuristic"
        )
    post_recovery_classification = {
        profile: a_summary[profile]["classification"]
        for profile in ("candidate", "medium", "upper")
    }
    recommended_roles = {
        "hard_recovery": list(CORE),
        "recovery_quality_only": list(AUX)
        + [f"{name}__rms_200ms" for name in AUX],
        "auxiliary_hard_recovery": [],
        "excluded": [],
    }
    recommended_timing = dwell_hysteresis[
        "dwell_0.30s_hysteresis_current"
    ]
    result = {
        "schema_version": 1,
        "package_id": "a3_base_recovery_envelope_manual_review_package_v1",
        "analysis_mode": "read_only_offline_numpy",
        "analyzer_sha256": sha256(Path(__file__)),
        "source_artifacts": {
            "calibrations": {name: str(path) for name, path in calibration_paths.items()},
            "envelope_analysis": str(envelope_path),
            "runtime_contract": str(contract_path),
            "disturbance_trace": str(trace_path),
        },
        "integrity": integrity,
        "input_evidence_complete": {
            "input_evidence_complete": True,
            "profiles_present": ["clean", "candidate", "medium", "upper"],
            "num_envs_per_profile": 256,
            "steps_per_profile": 500,
            "trace_sha256_match": True,
            "contract_hash_match": True,
            "trajectory_hash_verified": True,
            "all_finite": True,
        },
        "evidence_complete": True,
        "clean_steady_state_valid": clean_steady_state_valid,
        "current_envelope_robust": current_envelope_robust,
        "optional_evidence_unavailable": [
            "torso",
            "ankle_target",
            "ankle_actual_position",
            "ankle_torque",
            "foot_load",
            "foot_contact",
        ],
        "clean_tail_statistics": clean_tail_stats,
        "clean_episode_analysis": {
            "full": comparison["A_current_full"]["clean"],
            "core": comparison["B_core_only"]["clean"],
        },
        "episode_events": {
            profile: events_by_set["A_current_full"][profile]
            for profile in ("candidate", "medium", "upper")
        },
        "event_classification_note": (
            "numerical_or_contact_spike_contact_unverified requires <=2 exit-threshold "
            "violation policy steps and auxiliary-only triggers; "
            "contact is unverified because contact/load evidence is unavailable"
        ),
        "right_ankle_special_review": {
            "channels": ankle_review,
            "actual_larger_clean_p99_target": "abs_joint_velocity_rad_s/right_ankle_roll_joint",
            "recommendation_option": "E_remove_from_hard_recovery_keep_as_quality_metric",
        },
        "right_ankle_velocity_assessment": {
            "recommendation": "E",
            "recommendation_text": "temporarily remove raw/RMS ankle velocity from hard recovery while retaining all forms as quality metrics",
            "pitch": ankle_review[
                "abs_joint_velocity_rad_s/right_ankle_pitch_joint"
            ],
            "roll": ankle_review[
                "abs_joint_velocity_rad_s/right_ankle_roll_joint"
            ],
            "actual_larger_clean_p99_target": "abs_joint_velocity_rad_s/right_ankle_roll_joint",
        },
        "envelope_comparison": comparison,
        "candidate_envelopes": [
            {
                "name": name,
                "definition": value["definition"],
                "clean": value["clean"],
                "profiles": value["profiles"],
            }
            for name, value in comparison.items()
        ],
        "post_recovery_exit_classification": post_recovery_classification,
        "sensitivity_analysis": sensitivity,
        "sensitivity_results": sensitivity,
        "sensitivity_review_heuristic": (
            "materially_sensitive when >10 percentage-point recovery-rate shift or >1s p90 shift; "
            "this is a review heuristic, not an approval gate"
        ),
        "dwell_hysteresis_analysis": dwell_hysteresis,
        "dwell_hysteresis_results": dwell_hysteresis,
        "episode_manifest": manifest,
        "recommended_envelope": {
            "status": "candidate_not_approved",
            "name": "B_core_only",
            "reason": "seven core-body channels define hard recovery; raw and RMS waist/ankle velocities remain quality-only evidence",
            "recommended_dwell_s": 0.30,
            "recommended_hysteresis_ratio": 1.25,
            "timing_status": "research_candidate_only",
            "timing_results": recommended_timing,
        },
        "envelope_status": {
            "A_current_full": "current_comparison_not_approved",
            "B_core_only": "candidate_not_approved",
            "C_core_plus_aux_200ms_rms": "candidate_not_approved_comparison",
        },
        "recommended_role_per_metric": recommended_roles,
        "manual_decision_recommendation": "revise",
        "blocking_reasons": blocking_reasons,
        "decision_rationale": (
            "The corrected state machine supports B_core_only as a candidate for subsequent "
            "human approval review, while preserving all auxiliary velocity channels as quality "
            "evidence. Automatic approval remains prohibited; upper has "
            f"{a_summary['upper']['safety_terminations']} safety terminations. Missing contact/load/torque "
            "evidence blocks only promotion of auxiliary metrics into hard recovery criteria."
        ),
        "gate_mutated": False,
        "recovery_envelope_approved": False,
        "untrained_stochastic_policy_safety_verified": False,
        "training_distribution_approved": False,
        "bounded_recovery_smoke_approved": False,
        "deployment_approved": False,
        "approval_mutated": False,
        "ppo_run": False,
        "source_calibrations_overwritten": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    args.markdown.write_text(_markdown_state_machine(result))
    print(json.dumps({
        "output": str(args.output),
        "markdown": str(args.markdown),
        "recommendation": result["manual_decision_recommendation"],
        "current_full": a_summary,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
