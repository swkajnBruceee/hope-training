"""Dependency-free tests for the Phase 0–3 A3 strike contracts."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import a3_strike_contract as contract  # noqa: E402
import bootstrap_a3_executor_contract as bootstrap  # noqa: E402
import build_a3_body_drive_command as builder  # noqa: E402
import extract_a3_review_target_candidate as target_candidate  # noqa: E402
import record_a3_racket_task_samples as task_samples  # noqa: E402


def _target() -> dict:
    payload = {
        "schema_version": 1,
        "source_dataset": "curated_source",
        "source_episode_id": "fh-001",
        "stroke_type": "forehand",
        "hit_time_s": 0.46,
        "racket_position_b_m": [0.5, -0.1, 0.8],
        "racket_velocity_b_mps": [1.0, 0.0, 0.1],
        "racket_normal_b": [0.0, 0.0, 1.0],
        "racket_mount_contract_id": "a3_racket_mount_v1",
    }
    payload["source_target_sha256"] = contract.target_sha256(payload)
    return payload


def test_target_hash_rejects_any_semantic_change():
    spec = _target()
    assert contract.verify_target_spec(spec) == spec["source_target_sha256"]
    spec["racket_position_b_m"][0] += 0.001
    with pytest.raises(ValueError, match="mismatch"):
        contract.verify_target_spec(spec)


def test_canonical_command_hash_is_independent_of_npz_container(tmp_path: Path):
    names = list(bootstrap.OFFICIAL_31_DOF)
    values = np.zeros((3, 31), dtype=np.float64)
    payload = {
        "joint_names": np.asarray(names), "timestamps_s": np.asarray([0.0, 0.02, 0.04]),
        "q_des": values, "dq_des": values, "tau_ff": values,
        "kp": np.ones_like(values), "kd": np.ones_like(values),
    }
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    np.savez_compressed(first, **payload)
    np.savez_compressed(second, **payload)
    assert contract.command_sha256_from_npz(first) == contract.command_sha256_from_npz(second)
    payload["q_des"] = values.copy()
    payload["q_des"][1, 0] = 0.001
    np.savez_compressed(second, **payload)
    assert contract.command_sha256_from_npz(first) != contract.command_sha256_from_npz(second)


def test_frozen_contract_is_complete():
    frozen = bootstrap.build_contract()
    contract.validate_executor_contract(frozen)
    assert sum(joint["ownership"] == "strike_optimized" for joint in frozen["joints"]) == 10
    assert [joint["sdk_index"] for joint in frozen["joints"]] == list(range(31))


def test_builder_emits_full_contract_order_and_zero_feedforward(tmp_path: Path):
    reference = tmp_path / "reference.npz"
    reference_q = np.zeros((4, 31), dtype=np.float32)
    reference_q[:, builder.ISAAC_31_DOF.index("waist_pitch_joint")] = np.arange(4)
    np.savez_compressed(reference, joint_pos=reference_q, fps=np.asarray([50]))
    payload, digest = builder.build_payload(reference, bootstrap.build_contract())
    assert payload["q_des"].shape == (4, 31)
    assert tuple(payload["joint_names"].tolist()) == bootstrap.OFFICIAL_31_DOF
    assert np.all(payload["dq_des"] == 0.0)
    assert np.all(payload["tau_ff"] == 0.0)
    assert digest == contract.canonical_command_payload_sha256(**payload)


def test_builder_cli_writes_cnp_compatible_stored_numeric_first(tmp_path: Path):
    reference = tmp_path / "reference.npz"
    np.savez(reference, joint_pos=np.zeros((2, 31), dtype=np.float32), fps=np.asarray([50]))
    contract_path = tmp_path / "executor_contract.json"
    contract_path.write_text(json.dumps(bootstrap.build_contract()), encoding="utf-8")
    output = tmp_path / "command.npz"
    subprocess.run(
        [
            sys.executable, str(ROOT / "tools" / "build_a3_body_drive_command.py"),
            "--reference-npz", str(reference), "--executor-contract", str(contract_path),
            "--output", str(output),
        ],
        check=True, capture_output=True, text=True,
    )
    with zipfile.ZipFile(output) as archive:
        members = archive.infolist()
        assert [member.filename for member in members][-1] == "joint_names.npy"
        assert all(member.compress_type == zipfile.ZIP_STORED for member in members)


def test_training_manifest_fails_closed(tmp_path: Path):
    target = _target()
    valid = {
        "dataset_status": "active_training_candidate",
        "source_target_sha256": target["source_target_sha256"],
        "command_sha256": "abc",
        "executor_contract_id": "a3_t2d5_body_drive_fixed_stand_diag_v1",
        "gate_version": "a3_strike_gate_v3",
        "provenance_version": "v1",
        "motions": [{"episode_id": "fh-001"}],
    }
    path = tmp_path / "active_manifest.json"
    path.write_text(json.dumps(valid), encoding="utf-8")
    assert contract.assert_training_manifest(path)["dataset_status"] == "active_training_candidate"
    valid["dataset_status"] = "diagnostic_only_target_relabel"
    path.write_text(json.dumps(valid), encoding="utf-8")
    with pytest.raises(ValueError, match="only 'active_training_candidate'"):
        contract.assert_training_manifest(path)


def test_qualification_requires_and_records_separate_rates(tmp_path: Path):
    command = tmp_path / "command.npz"
    values = np.zeros((3, 31), dtype=np.float64)
    np.savez_compressed(
        command,
        joint_names=np.asarray(bootstrap.OFFICIAL_31_DOF), timestamps_s=np.asarray([0.0, 0.02, 0.04]),
        q_des=values, dq_des=values, tau_ff=values, kp=np.ones_like(values), kd=np.ones_like(values),
    )
    target = _target()
    target_path = tmp_path / "target_spec.json"
    target_path.write_text(json.dumps(target), encoding="utf-8")
    contract_path = tmp_path / "executor_contract.json"
    contract_path.write_text(json.dumps(bootstrap.build_contract()), encoding="utf-8")
    task_paths, state_paths = [], []
    for index in range(10):
        task = tmp_path / f"task_{index}.npz"
        state = tmp_path / f"state_{index}.npz"
        np.savez_compressed(
            task,
            timestamp_s=np.asarray([0.40, 0.46, 0.50]),
            racket_position_b_m=np.asarray([[0.4, -0.1, 0.8], target["racket_position_b_m"], [0.6, -0.1, 0.8]]),
            racket_velocity_b_mps=np.asarray([[0.0, 0.0, 0.0], target["racket_velocity_b_mps"], [0.0, 0.0, 0.0]]),
            racket_normal_b=np.asarray([[0.0, 0.0, 1.0]] * 3),
            stand_gate_passed=np.asarray([True]),
            command_publish_time_s=np.asarray([0.0, 0.02, 0.04]), state_receive_time_s=np.asarray([0.001, 0.021, 0.041]),
            source_target_sha256=np.asarray([target["source_target_sha256"]]),
            racket_mount_contract_id=np.asarray([target["racket_mount_contract_id"]]),
        )
        np.savez_compressed(
            state,
            raw_state_timestamp_s=np.asarray([0.0, 0.005, 0.010]),
            backend_sync_timestamp_s=np.asarray([0.0, 0.01, 0.02]),
            command_timestamp_s=np.asarray([0.0, 0.02, 0.04]),
            q_actual=values, dq_actual=values, tau_est=values,
        )
        task_paths.append(task)
        state_paths.append(state)
    report = tmp_path / "qualification.json"
    invocation = [
        sys.executable, str(ROOT / "tools" / "a3_standalone_qualification.py"),
        "--executor-contract", str(contract_path), "--target-spec", str(target_path), "--command", str(command), "--out", str(report),
    ]
    for task, state in zip(task_paths, state_paths):
        invocation.extend(["--rollout-task-samples", str(task), "--rollout-state-samples", str(state)])
    subprocess.run(invocation, check=True, capture_output=True, text=True)
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["pass"] is True
    assert result["observed_rates_hz"]["raw_state_rate_hz"]["mean"] == 200.0
    assert result["observed_rates_hz"]["backend_sync_rate_hz"]["mean"] == 100.0
    assert result["observed_rates_hz"]["command_rate_hz"]["mean"] == 50.0


def test_qualification_rejects_task_sample_with_other_target_hash(tmp_path: Path):
    path = tmp_path / "task.npz"
    np.savez_compressed(
        path,
        timestamp_s=np.asarray([0.0, 0.01, 0.02]),
        racket_position_b_m=np.zeros((3, 3)),
        racket_velocity_b_mps=np.zeros((3, 3)),
        racket_normal_b=np.asarray([[0.0, 1.0, 0.0]] * 3),
        stand_gate_passed=np.asarray([True]),
        source_target_sha256=np.asarray(["not-the-target"]),
        racket_mount_contract_id=np.asarray(["a3_racket_mount_v1"]),
    )
    with pytest.raises(ValueError, match="source_target_sha256"):
        qualification = __import__("a3_standalone_qualification")
        qualification._task_sample(path, 0.01, _target()["source_target_sha256"], "a3_racket_mount_v1")


def test_qualification_requires_absolute_target_match_not_only_low_noise():
    metrics = {
        "position_error_m": {"mean": 1.0, "std": 0.001},
        "velocity_vector_error_mps": {"mean": 0.0, "std": 0.001},
        "normal_angle_deg": {"mean": 0.0, "std": 0.001},
    }
    noise_ok, target_ok = __import__("a3_standalone_qualification")._qualification_flags(
        metrics, {"position_error_m": 0.075, "velocity_vector_error_mps": 0.5, "normal_angle_deg": 15.0}, 10
    )
    assert noise_ok is True
    assert target_ok is False


def test_official_pose_samples_use_training_base_velocity_semantics():
    # Base is yawed +90 degrees: world +X becomes base -Y.  The velocity must
    # be this rotated world velocity, rather than a derivative in a rotating
    # frame.  The local +Y racket face normal follows the same two rotations.
    yaw_90_xyzw = np.asarray([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])
    position_b, velocity_b, normal_b = task_samples.world_racket_to_base(
        pelvis_position_w_m=np.asarray([1.0, 2.0, 0.0]),
        pelvis_quaternion_w_xyzw=yaw_90_xyzw,
        racket_position_w_m=np.asarray([2.0, 2.0, 0.0]),
        racket_quaternion_w_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0]),
        racket_velocity_w_mps=np.asarray([3.0, 0.0, 0.0]),
        racket_normal_local=np.asarray([0.0, 1.0, 0.0]),
    )
    assert np.allclose(position_b, [0.0, -1.0, 0.0], atol=1.0e-12)
    assert np.allclose(velocity_b, [0.0, -3.0, 0.0], atol=1.0e-12)
    assert np.allclose(normal_b, [1.0, 0.0, 0.0], atol=1.0e-12)


def test_task_sample_payload_is_qualification_compatible():
    identity = np.asarray([0.0, 0.0, 0.0, 1.0])
    pairs = []
    for index in range(3):
        receive = 1_000_000_000 + index * 10_000_000
        base = task_samples.PoseRecord(receive, 12.0 + index * 0.01, np.zeros(3), identity)
        racket = task_samples.PoseRecord(receive, 12.0 + index * 0.01, np.asarray([0.01 * index, 0.0, 1.0]), identity)
        pairs.append(task_samples.PairedPoseRecord(base, racket, 0.0))
    payload = task_samples.build_task_sample_payload(
        pairs, [1_000_000_000, 1_020_000_000], 1_000_000_000, np.asarray([0.0, 1.0, 0.0]), stand_gate_passed=True
    )
    assert np.all(np.diff(payload["timestamp_s"]) > 0.0)
    assert payload["racket_position_b_m"].shape == (3, 3)
    assert np.allclose(payload["racket_velocity_b_mps"], [[1.0, 0.0, 0.0]] * 3)
    assert np.allclose(payload["racket_normal_b"], [[0.0, 1.0, 0.0]] * 3)
    assert bool(payload["stand_gate_passed"][0]) is True


def test_review_target_candidate_is_base_frame_and_not_approved(tmp_path: Path):
    source_target = {
        "episode_id": "fh-001",
        "coordinate_contract": {"position_frame": "table_m"},
        "hit_target": {
            "racket_position_m": [2.0, 1.0, 0.5],
            "racket_velocity_mps": [1.0, 0.0, 0.0],
            "racket_normal_w": [0.0, 1.0, 0.0],
        },
    }
    source_manifest = {
        "motions": [{
            "episode_id": "fh-001", "stroke_type": "forehand",
            "hit_event": {"hit_time_from_start_s": 0.6, "source_hit_index": 120, "source_fps": 200.0},
            "stance_metadata": {"base_pose_target_w": {"position_m": [1.0, 1.0, 0.0], "yaw_rad": np.pi / 2.0}},
        }]
    }
    target_path = tmp_path / "source_target.json"
    manifest_path = tmp_path / "manifest.json"
    target_path.write_text(json.dumps(source_target), encoding="utf-8")
    manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
    candidate = target_candidate.derive_candidate(
        source_target, source_manifest, "fh-001", "unit_source", "right_racket_red_face_y_v1", target_path, manifest_path
    )
    proposed = candidate["proposed_target_input"]
    assert candidate["status"] == "requires_human_review_not_an_immutable_target"
    assert np.allclose(proposed["racket_position_b_m"], [0.0, -1.0, 0.5], atol=1.0e-12)
    assert np.allclose(proposed["racket_velocity_b_mps"], [0.0, -1.0, 0.0], atol=1.0e-12)
    assert np.allclose(proposed["racket_normal_b"], [1.0, 0.0, 0.0], atol=1.0e-12)
