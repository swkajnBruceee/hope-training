#!/usr/bin/env python3
"""Compare causal ZOH and linear target transport in the no-contact fixture."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

import a3_base_contract as contract
from analyze_a3_base_low_zoh_bundle import _distribution, _npz, _result_index


CASE_PATTERN = re.compile(
    r"^transport__200hz__(?P<mode>zero_order_hold|linear_substep_interpolation)"
    r"__(?P<joint>.+)__r01$"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--isaac-results-dir", type=Path, required=True)
    parser.add_argument("--mujoco-results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _trace_index(bundle_dir: Path) -> dict[str, tuple[dict[str, np.ndarray], Path]]:
    index = {}
    for path in sorted((bundle_dir / "traces").glob("*.npz")):
        if path.name.endswith(".trace.npz"):
            continue
        with np.load(path, allow_pickle=False) as archive:
            trace = {name: archive[name] for name in archive.files}
        metadata_path = path.with_suffix(".trace.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        index[metadata["case_id"]] = (trace, path)
    return index


def _transition_delay(
    zoh: dict[str, np.ndarray], linear: dict[str, np.ndarray], joint_name: str
) -> dict[str, float]:
    names = zoh["joint_names"].tolist()
    if names != linear["joint_names"].tolist():
        raise ValueError("transport traces use different joint order")
    index = names.index(joint_name)
    zoh_target = np.asarray(zoh["composed_target_rad"][:, index], dtype=np.float64)
    linear_target = np.asarray(linear["composed_target_rad"][:, index], dtype=np.float64)
    time_s = np.asarray(zoh["metric_timestamp_s"], dtype=np.float64)
    if not np.array_equal(time_s, np.asarray(linear["metric_timestamp_s"])):
        raise ValueError("transport timestamps differ")
    baseline = float(zoh_target[0])
    zoh_change = np.flatnonzero(np.abs(zoh_target - baseline) > 1.0e-12)
    if not zoh_change.size:
        raise ValueError("transport ZOH trace has no transition")
    start = int(zoh_change[0])
    full_value = float(zoh_target[start])
    amplitude = full_value - baseline
    if amplitude == 0.0:
        raise ValueError("transport transition amplitude is zero")
    linear_full = np.flatnonzero(
        np.abs(linear_target - full_value) <= max(abs(amplitude) * 1.0e-9, 1.0e-12)
    )
    linear_full = linear_full[linear_full >= start]
    if not linear_full.size:
        raise ValueError("linear trace never reaches current policy target")
    full_index = int(linear_full[0])
    dt = float(zoh["physics_dt_s"][0])
    normalized_deficit = (zoh_target[start : full_index + 1] - linear_target[start : full_index + 1]) / amplitude
    return {
        "first_effective_command_delay_s": 0.0,
        "full_target_delay_s": float(time_s[full_index] - time_s[start]),
        "equivalent_command_area_lag_s": float(np.sum(normalized_deficit) * dt),
    }


def build_report(
    *, bundle_dir: Path, isaac_results_dir: Path, mujoco_results_dir: Path
) -> dict[str, Any]:
    bundle_dir = bundle_dir.resolve()
    bundle = json.loads(
        (bundle_dir / "bundle_report.json").read_text(encoding="utf-8")
    )
    isaac = _result_index(isaac_results_dir)
    mujoco = _result_index(mujoco_results_dir)
    traces = _trace_index(bundle_dir)
    expected = set(bundle["case_ids"])
    if set(isaac) != expected or set(mujoco) != expected or set(traces) != expected:
        raise ValueError("transport artifact coverage differs from bundle")
    by_joint: dict[str, dict[str, str]] = {}
    for case_id in bundle["case_ids"]:
        match = CASE_PATTERN.match(case_id)
        if not match:
            raise ValueError(f"unexpected transport case: {case_id}")
        by_joint.setdefault(match.group("joint"), {})[match.group("mode")] = case_id
    if len(by_joint) != 7 or any(len(modes) != 2 for modes in by_joint.values()):
        raise ValueError("transport bundle must contain two modes for seven joints")

    rows = []
    source_paths: set[Path] = {bundle_dir / "bundle_report.json"}
    for joint in sorted(by_joint):
        zoh_id = by_joint[joint]["zero_order_hold"]
        linear_id = by_joint[joint]["linear_substep_interpolation"]
        delay = _transition_delay(traces[zoh_id][0], traces[linear_id][0], joint)
        engines = {}
        for engine_name, index in (("isaac", isaac), ("mujoco", mujoco)):
            zoh = index[zoh_id][0]["metrics"]
            linear = index[linear_id][0]["metrics"]
            engines[engine_name] = {
                "zoh_tracking_rmse_rad": zoh["tracking_rmse_rad"],
                "linear_tracking_rmse_rad": linear["tracking_rmse_rad"],
                "linear_to_zoh_tracking_rmse_ratio": (
                    linear["tracking_rmse_rad"] / zoh["tracking_rmse_rad"]
                ),
                "zoh_peak_tracking_error_rad": zoh["peak_tracking_error_rad"],
                "linear_peak_tracking_error_rad": linear["peak_tracking_error_rad"],
                "zoh_peak_joint_acceleration_radps2": zoh[
                    "peak_joint_acceleration_radps2"
                ],
                "linear_peak_joint_acceleration_radps2": linear[
                    "peak_joint_acceleration_radps2"
                ],
                "linear_to_zoh_peak_acceleration_ratio": (
                    linear["peak_joint_acceleration_radps2"]
                    / zoh["peak_joint_acceleration_radps2"]
                ),
                "linear_rmse_improved": (
                    linear["tracking_rmse_rad"] < zoh["tracking_rmse_rad"]
                ),
                "linear_peak_acceleration_not_increased": (
                    linear["peak_joint_acceleration_radps2"]
                    <= zoh["peak_joint_acceleration_radps2"] * (1.0 + 1.0e-9)
                ),
                "both_safety_envelopes_passed": bool(
                    index[zoh_id][0]["case_validation"]["safety_envelope_passed"]
                    and index[linear_id][0]["case_validation"]["safety_envelope_passed"]
                ),
            }
            source_paths.update(
                {
                    index[zoh_id][1], index[zoh_id][2],
                    index[linear_id][1], index[linear_id][2],
                }
            )
        source_paths.update({traces[zoh_id][1], traces[linear_id][1]})
        rows.append(
            {
                "selected_joint_name": joint,
                "zoh_case_id": zoh_id,
                "linear_case_id": linear_id,
                "causal_transport_delay": delay,
                "engines": engines,
            }
        )
    report = {
        "schema_version": 1,
        "artifact_status": "phase0_transport_diagnostic_not_promotion_evidence",
        "scope": bundle["scope"],
        "matrix_sha256": bundle["matrix_sha256"],
        "joint_count": len(rows),
        "causal_linear_transport": {
            "future_policy_target_accessed": False,
            "first_effective_command_delay_s": _distribution(
                [row["causal_transport_delay"]["first_effective_command_delay_s"] for row in rows]
            ),
            "full_target_delay_s": _distribution(
                [row["causal_transport_delay"]["full_target_delay_s"] for row in rows]
            ),
            "equivalent_command_area_lag_s": _distribution(
                [row["causal_transport_delay"]["equivalent_command_area_lag_s"] for row in rows]
            ),
        },
        "engine_summary": {
            engine: {
                "linear_rmse_improved_joint_count": sum(
                    row["engines"][engine]["linear_rmse_improved"] for row in rows
                ),
                "linear_peak_acceleration_not_increased_joint_count": sum(
                    row["engines"][engine]["linear_peak_acceleration_not_increased"]
                    for row in rows
                ),
                "linear_to_zoh_tracking_rmse_ratio": _distribution(
                    [
                        row["engines"][engine]["linear_to_zoh_tracking_rmse_ratio"]
                        for row in rows
                    ]
                ),
                "linear_to_zoh_peak_acceleration_ratio": _distribution(
                    [
                        row["engines"][engine]["linear_to_zoh_peak_acceleration_ratio"]
                        for row in rows
                    ]
                ),
            }
            for engine in ("isaac", "mujoco")
        },
        "transport_decision": {
            "phase0_default": "zero_order_hold",
            "linear_enabled_for_stand_smoke": False,
            "final_transport_frozen": False,
            "reason": (
                "Linear reduces no-contact acceleration but adds causal target delay, "
                "does not improve RMSE for every joint in both engines, and has not "
                "been validated under free-base foot contact."
            ),
            "contact_phase_retest_required": True,
        },
        "qualification_status": {
            "fixture_runner_qualified": True,
            "fixture_matrix_approved": False,
            "stand_task_approved": False,
            "locomotion_command_approved": False,
            "deployment_approved": False,
        },
        "automatic_promotion": False,
        "cases": rows,
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
        bundle_dir=args.bundle_dir,
        isaac_results_dir=args.isaac_results_dir,
        mujoco_results_dir=args.mujoco_results_dir,
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
                "causal_linear_transport": report["causal_linear_transport"],
                "engine_summary": report["engine_summary"],
                "transport_decision": report["transport_decision"],
                "qualification_status": report["qualification_status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
