#!/usr/bin/env python3
"""Shared response metrics for A3 single-joint fixture runners."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def _target_band_entry_time(
    time_s: np.ndarray,
    actual: np.ndarray,
    target: float,
    start_s: float,
    end_s: float,
    tolerance: float,
) -> float:
    window = np.flatnonzero((time_s >= start_s) & (time_s < end_s))
    for offset, index in enumerate(window):
        if np.all(np.abs(actual[window[offset:]] - target) <= tolerance):
            return float(time_s[index] - start_s)
    return float(end_s - start_s)


def _window_slope(time_s: np.ndarray, values: np.ndarray) -> float:
    if time_s.size < 2:
        return 0.0
    centered = time_s - float(np.mean(time_s))
    denominator = float(np.dot(centered, centered))
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(centered, values - float(np.mean(values))) / denominator)


def summarize_response(
    *,
    category: str,
    evidence: Mapping[str, np.ndarray],
    trace_metadata: Mapping[str, Any],
    command_delta: float,
    excited_value: float,
    physics_dt: float,
    constraint_reaction_available: bool,
) -> dict[str, Any]:
    """Compute category metrics identically for MuJoCo and Isaac runners."""

    actual = np.asarray(evidence["joint_q_rad"], dtype=np.float64)
    target_trace = np.asarray(evidence["joint_target_rad"], dtype=np.float64)
    time_trace = np.asarray(evidence["time_s"], dtype=np.float64)
    torque_trace = np.asarray(evidence["joint_torque_nm"], dtype=np.float64)
    saturation_trace = np.asarray(evidence["selected_joint_saturated"], dtype=np.bool_)
    velocity_trace = np.asarray(evidence["joint_dq_radps"], dtype=np.float64)
    window = trace_metadata["metric_window"]
    active_start = float(window["active_start_s"])
    active_end = float(window["active_end_s"])
    baseline_indices = np.flatnonzero(time_trace <= float(window["baseline_end_s"]))
    if baseline_indices.size == 0:
        raise ValueError("fixture evidence has no baseline sample")
    baseline_actual = float(actual[baseline_indices[-1]])
    active_mask = (time_trace > active_start) & (time_trace <= active_end)
    if not np.any(active_mask):
        raise ValueError("fixture evidence has no active-window sample")
    active_actual = actual[active_mask]
    active_torque = torque_trace[active_mask]
    active_saturation = saturation_trace[active_mask]
    selected_metrics: dict[str, Any] = {
        "commanded_joint_delta_rad": float(command_delta),
        "selected_joint_peak_torque_nm": float(np.max(np.abs(active_torque))),
        "selected_joint_effort_rms_nm": float(np.sqrt(np.mean(active_torque * active_torque))),
        "selected_joint_saturation_duration_s": float(
            np.count_nonzero(active_saturation) * physics_dt
        ),
        "constraint_reaction_available": bool(constraint_reaction_available),
    }

    end_window_s = float(window.get("end_window_s", 0.1))
    tail_mask = (time_trace > max(active_start, active_end - end_window_s)) & (
        time_trace <= active_end
    )
    end_slope = _window_slope(time_trace[tail_mask], actual[tail_mask])

    if category == "joint_zero_baseline":
        end_mean = float(np.mean(actual[tail_mask]))
        selected_metrics.update(
            {
                "end_window_mean_q_rad": end_mean,
                "end_window_drift_from_baseline_rad": end_mean - baseline_actual,
                "peak_abs_drift_from_baseline_rad": float(
                    np.max(np.abs(active_actual - baseline_actual))
                ),
                "end_window_slope_radps": end_slope,
            }
        )
    elif category in {"base_action_step", "waist_pitch_residual"}:
        sign = 1.0 if command_delta > 0.0 else -1.0
        peak_delta = float(np.max(np.abs(active_actual - baseline_actual)))
        overshoot = max(
            0.0,
            float(np.max(sign * (active_actual - baseline_actual))) - abs(command_delta),
        )
        end_error = float(np.mean(np.abs(actual[tail_mask] - excited_value)))
        end_delta = float(np.mean(actual[tail_mask]) - baseline_actual)
        tolerance = max(
            float(window.get("settling_min_tolerance_rad", 0.002)),
            float(window.get("settling_relative_tolerance", 0.02))
            * abs(command_delta),
        )
        target_band_entry = _target_band_entry_time(
            time_trace,
            actual,
            excited_value,
            active_start,
            active_end,
            tolerance,
        )
        selected_metrics.update(
            {
                "peak_joint_delta_rad": peak_delta,
                "overshoot_rad": overshoot,
                "target_band_entry_time_s": target_band_entry,
                "target_band_reached_and_held": bool(
                    target_band_entry < active_end - active_start - 0.5 * physics_dt
                ),
                "end_window_joint_delta_rad": end_delta,
                "end_window_response_ratio": end_delta / command_delta,
                "end_window_error_rad": end_error,
                "end_window_slope_radps": end_slope,
            }
        )
        if category == "waist_pitch_residual":
            selected_metrics["composer_residual_clip_hit"] = bool(
                trace_metadata["composer_residual_clip_hit"]
            )
    elif category == "target_transport":
        error = actual[active_mask] - target_trace[active_mask]
        acceleration = np.diff(velocity_trace) / physics_dt
        selected_metrics.update(
            {
                "transport_mode": trace_metadata["transport_mode"],
                "tracking_rmse_rad": float(np.sqrt(np.mean(error * error))),
                "peak_tracking_error_rad": float(np.max(np.abs(error))),
                "peak_joint_acceleration_radps2": float(
                    np.max(np.abs(acceleration)) if acceleration.size else 0.0
                ),
            }
        )
    else:
        raise ValueError(f"unsupported fixture category: {category}")
    return selected_metrics
