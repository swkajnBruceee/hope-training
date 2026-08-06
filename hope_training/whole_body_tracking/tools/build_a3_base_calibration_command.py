#!/usr/bin/env python3
"""Build one replayable A3 Base Phase 0 calibration command.

Only cases whose semantics are faithfully represented by the existing 50 Hz
RobotIO replay are accepted. Command-basis cases require a trained policy,
and target-transport cases require simulator-native substep logic; pretending
that a faster external publisher is equivalent would invalidate the evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import a3_base_calibration as calibration
import a3_base_contract as contract
from a3_strike_contract import canonical_command_payload_sha256


def _integer_ticks(duration_s: Any, rate_hz: float, label: str) -> int:
    duration = float(duration_s)
    ticks = round(duration * rate_hz)
    if duration <= 0.0 or not np.isclose(ticks / rate_hz, duration, atol=1.0e-12):
        raise ValueError(f"{label} must be a positive integer multiple of 1/{rate_hz:g} s")
    return int(ticks)


def _compose(
    composer: Mapping[str, Any], base_action: list[float], strike_reference: list[float]
) -> dict[str, Any]:
    command = contract.compose_command(composer, base_action, strike_reference)
    if any(command["debug"]["joint_limit_hit"]):
        raise ValueError("calibration command reaches a configured joint limit")
    return command


def build_case_payload(
    case: Mapping[str, Any], contracts: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return canonical command arrays and generation facts for one case."""

    protocol = contracts["calibration_contract.json"]["command_payload_protocol"]
    category = str(case.get("category", ""))
    supported = set(protocol["direct_robotio_replay_supported_categories"])
    if category not in supported:
        reason = protocol["unsupported_reason_by_category"].get(
            category, "category_is_not_supported_by_direct_robotio_replay"
        )
        raise ValueError(f"{category} cannot use direct RobotIO replay: {reason}")

    inputs = case.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("calibration case inputs must be an object")
    transport = str(inputs.get("target_transport", ""))
    expected_transport = protocol["direct_robotio_replay_supported_transport"]
    if transport != expected_transport:
        raise ValueError(
            f"direct RobotIO replay supports only {expected_transport}, got {transport}"
        )

    action = [float(value) for value in inputs.get("base_action", [])]
    strike_reference = [
        float(value) for value in inputs.get("strike_q_reference", [])
    ]
    rate_hz = float(protocol["joint_command_rate_hz"])
    pre_ticks = _integer_ticks(inputs.get("pre_hold_s"), rate_hz, "pre_hold_s")
    step_ticks = _integer_ticks(inputs.get("step_hold_s"), rate_hz, "step_hold_s")
    post_ticks = _integer_ticks(inputs.get("post_hold_s"), rate_hz, "post_hold_s")

    composer = contracts["command_composer_contract.json"]
    baseline = _compose(composer, [0.0] * 14, strike_reference)
    target = _compose(composer, action, strike_reference)
    q_baseline = np.asarray(baseline["q_des"], dtype=np.float64)
    q_target = np.asarray(target["q_des"], dtype=np.float64)
    q_des = np.concatenate(
        (
            np.tile(q_baseline, (pre_ticks, 1)),
            np.tile(q_target, (step_ticks, 1)),
            np.tile(q_baseline, (post_ticks, 1)),
        ),
        axis=0,
    )
    sample_count = int(q_des.shape[0])
    zeros = np.zeros_like(q_des)
    payload = {
        "timestamps_s": np.arange(sample_count, dtype=np.float64) / rate_hz,
        "q_des": q_des,
        "dq_des": zeros.copy(),
        "tau_ff": zeros.copy(),
        "kp": np.tile(
            np.asarray(baseline["kp"], dtype=np.float64), (sample_count, 1)
        ),
        "kd": np.tile(
            np.asarray(baseline["kd"], dtype=np.float64), (sample_count, 1)
        ),
        # Keep the Unicode array last for the vendored cnpy implementation.
        "joint_names": np.asarray(baseline["joint_names"]),
    }
    generation = {
        "sample_count": sample_count,
        "duration_s": sample_count / rate_hz,
        "pre_hold_ticks": pre_ticks,
        "step_hold_ticks": step_ticks,
        "post_hold_ticks": post_ticks,
        "changed_joint_names": [
            name
            for name, before, after in zip(
                baseline["joint_names"], q_baseline.tolist(), q_target.tolist()
            )
            if before != after
        ],
        "composer_waist_pitch_residual_rad": target["debug"][
            "waist_pitch_residual_rad"
        ],
        "composer_action_was_clipped": target["debug"]["clipped_base_action"]
        != action,
    }
    return payload, generation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract-dir", type=Path, default=contract.contract_dir_from_script()
    )
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract_dir = args.contract_dir.expanduser().resolve()
    contracts = contract.load_contracts(contract_dir)
    contract.validate_contracts(contracts)
    matrix_path = args.matrix.expanduser().resolve()
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    calibration.validate_matrix(matrix, contracts)
    matches = [case for case in matrix["cases"] if case["case_id"] == args.case_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one matrix case named {args.case_id!r}")

    payload, generation = build_case_payload(matches[0], contracts)
    digest = canonical_command_payload_sha256(**payload)
    output = args.output.expanduser().resolve()
    if output.suffix != ".npz":
        raise ValueError("--output must end in .npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **payload)
    metadata = {
        "schema_version": 1,
        "artifact_status": "phase0_candidate_not_execution_approved",
        "case_id": args.case_id,
        "case_category": matches[0]["category"],
        "matrix_sha256": matrix["matrix_sha256"],
        "matrix_file_sha256": contract.file_sha256(matrix_path),
        "calibration_contract_id": contracts["calibration_contract.json"][
            "calibration_contract_id"
        ],
        "command_composer_contract_id": contracts[
            "command_composer_contract.json"
        ]["command_composer_contract_id"],
        "joint_order_sha256": contracts["command_composer_contract.json"][
            "joint_order_sha256"
        ],
        "command_sha256": digest,
        "command_semantics": "canonical_command_payload_v1",
        "npz_container": "stored_uncompressed_npz_v1",
        "target_transport": matches[0]["inputs"]["target_transport"],
        "single_publisher_required": True,
        "isolated_simulator_required": True,
        "hardware_execution_approved": False,
        "generation": generation,
        "note": (
            "Generation does not authorize replay. Run only in a resettable isolated "
            "simulator after verifying sole command-publisher ownership."
        ),
    }
    metadata_path = output.with_suffix(".command.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
