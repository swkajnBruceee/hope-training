#!/usr/bin/env python3
"""Summarize official AimSim arm-channel probes as a balance shadow report.

The official HTTP state service currently exposes joint states and pelvis/torso
IMU data, but not foot wrench, support-polygon, or native controller torque
data. This report therefore measures command-path and torso-dynamic stability
without claiming a complete whole-body balance proof.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else None


def summarize(path: Path) -> dict:
    data = json.loads(path.read_text())
    samples = data.get("samples", [])
    report = {
        "source": str(path),
        "motion_npz": data.get("motion_npz"),
        "sample_count": len(samples),
        "http_failures": 0,
        "torso": {},
        "pelvis": {},
        "arm_tracking": {},
        "limitations": [
            "No foot wrench or support-polygon data is exposed by this endpoint.",
            "No native controller torque/current data is exposed by this endpoint.",
            "Absolute IMU tilt includes the simulator's static body-frame offset.",
        ],
    }

    for sample in samples:
        if (sample.get("response") or {}).get("status_code") != 200:
            report["http_failures"] += 1

    for body_name in ("pelvis", "torso"):
        orientations = []
        angular_velocities = []
        acceleration_residuals = []
        for sample in samples:
            body = (sample.get("imu") or {}).get(body_name) or {}
            if body.get("orientation"):
                orientations.append(np.asarray(body["orientation"], dtype=float))
            if body.get("angular_velocity"):
                angular_velocities.append(
                    np.asarray(body["angular_velocity"], dtype=float)
                )
            if body.get("linear_acceleration"):
                acceleration = np.asarray(body["linear_acceleration"], dtype=float)
                acceleration_residuals.append(
                    float(np.linalg.norm(acceleration - np.array([0.0, 0.0, 9.81])))
                )

        q = np.asarray(orientations, dtype=float)
        w = np.asarray(angular_velocities, dtype=float)
        a = np.asarray(acceleration_residuals, dtype=float)
        # Official IMU responses are xyzw in the current AimSim service. The
        # first three components are the vector part used for tilt magnitude.
        tilt_deg = (
            2.0
            * np.arcsin(np.clip(np.linalg.norm(q[:, :3], axis=1), 0.0, 1.0))
            * 180.0
            / np.pi
            if q.size
            else np.asarray([])
        )
        angular_speed = np.linalg.norm(w, axis=1) if w.size else np.asarray([])
        report[body_name] = {
            "orientation_tilt_deg_p50": percentile(tilt_deg, 50),
            "orientation_tilt_deg_p95": percentile(tilt_deg, 95),
            "orientation_tilt_deg_max": float(tilt_deg.max()) if tilt_deg.size else None,
            "angular_velocity_rad_s_p50": percentile(angular_speed, 50),
            "angular_velocity_rad_s_p95": percentile(angular_speed, 95),
            "angular_velocity_rad_s_max": float(angular_speed.max()) if angular_speed.size else None,
            "linear_acceleration_residual_m_s2_p95": percentile(a, 95),
        }

    errors = []
    for sample in samples:
        command = sample.get("command")
        actual = sample.get("actual_arm")
        if command is not None and actual is not None:
            errors.append(np.asarray(actual, dtype=float) - np.asarray(command, dtype=float))
    if errors:
        error = np.asarray(errors, dtype=float)
        abs_error = np.abs(error)
        report["arm_tracking"] = {
            "abs_error_rad_mean": float(abs_error.mean()),
            "abs_error_rad_p95": float(np.percentile(abs_error, 95)),
            "abs_error_rad_max": float(abs_error.max()),
            "per_joint_abs_error_rad_p95": [
                float(x) for x in np.percentile(abs_error, 95, axis=0)
            ],
        }

    torso_velocity_p95 = report["torso"].get("angular_velocity_rad_s_p95")
    arm_error_p95 = report["arm_tracking"].get("abs_error_rad_p95")
    report["command_path_shadow_pass"] = bool(
        report["http_failures"] == 0
        and torso_velocity_p95 is not None
        and torso_velocity_p95 < 0.25
        and arm_error_p95 is not None
        and arm_error_p95 < 0.15
    )
    report["pass_definition"] = (
        "Diagnostic only: HTTP success, torso angular-velocity p95 < 0.25 rad/s, "
        "and arm absolute tracking-error p95 < 0.15 rad. This is not a full balance gate."
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    reports = [summarize(path) for path in args.inputs]
    output = {
        "report_type": "official_aimsim_balance_shadow",
        "reports": reports,
        "overall_command_path_shadow_pass": all(
            report["command_path_shadow_pass"] for report in reports
        ),
        "not_a_full_balance_validation": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
