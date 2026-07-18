"""Immutable contracts for the A3 executor-aware strike pipeline.

This module deliberately has no ROS, Isaac, or MuJoCo imports.  It is used by
training entry points and offline tools so a missing simulator cannot turn a
historical/relabelled manifest into a training input.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


GATE_VERSION = "a3_strike_gate_v3"
ACTIVE_DATASET_STATUS = "active_training_candidate"
FORBIDDEN_PATH_TOKENS = ("_archive_not_for_training", "diagnostic", "relabel", "invalid")
REQUIRED_MANIFEST_FIELDS = (
    "dataset_status",
    "source_target_sha256",
    "command_sha256",
    "executor_contract_id",
    "gate_version",
    "provenance_version",
)
TARGET_FIELDS = (
    "schema_version",
    "source_dataset",
    "source_episode_id",
    "stroke_type",
    "hit_time_s",
    "racket_position_b_m",
    "racket_velocity_b_mps",
    "racket_normal_b",
    "racket_mount_contract_id",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_vector(value: Any, name: str, size: int) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{name} must be a {size}-vector")
    result = [float(x) for x in value]
    if not all(math.isfinite(x) for x in result):
        raise ValueError(f"{name} contains a non-finite value")
    return result


def normalized_target_payload(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return the target payload whose hash is immutable across all stages."""

    missing = [field for field in TARGET_FIELDS if field not in spec]
    if missing:
        raise ValueError(f"target_spec missing required fields: {', '.join(missing)}")
    payload = {field: spec[field] for field in TARGET_FIELDS}
    if int(payload["schema_version"]) != 1:
        raise ValueError("target_spec.schema_version must be 1")
    if not isinstance(payload["source_dataset"], str) or not payload["source_dataset"]:
        raise ValueError("target_spec.source_dataset must be a non-empty string")
    if not isinstance(payload["source_episode_id"], str) or not payload["source_episode_id"]:
        raise ValueError("target_spec.source_episode_id must be a non-empty string")
    if str(payload["stroke_type"]).lower() not in {"forehand", "backhand"}:
        raise ValueError("target_spec.stroke_type must be forehand or backhand")
    hit_time_s = float(payload["hit_time_s"])
    if not math.isfinite(hit_time_s) or hit_time_s < 0.0:
        raise ValueError("target_spec.hit_time_s must be finite and non-negative")
    payload["hit_time_s"] = hit_time_s
    payload["racket_position_b_m"] = _finite_vector(payload["racket_position_b_m"], "racket_position_b_m", 3)
    payload["racket_velocity_b_mps"] = _finite_vector(payload["racket_velocity_b_mps"], "racket_velocity_b_mps", 3)
    normal = np.asarray(_finite_vector(payload["racket_normal_b"], "racket_normal_b", 3), dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if norm <= 1.0e-9:
        raise ValueError("target_spec.racket_normal_b must be non-zero")
    # Normalization is allowed only while creating the spec.  A checked-in spec
    # must already contain the exact normalized values to make its hash stable.
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-8):
        raise ValueError("target_spec.racket_normal_b must already be unit length")
    payload["racket_normal_b"] = [float(x) for x in normal]
    if not isinstance(payload["racket_mount_contract_id"], str) or not payload["racket_mount_contract_id"]:
        raise ValueError("target_spec.racket_mount_contract_id must be a non-empty string")
    return payload


def target_sha256(spec: Mapping[str, Any]) -> str:
    return sha256_json(normalized_target_payload(spec))


def verify_target_spec(spec: Mapping[str, Any]) -> str:
    digest = target_sha256(spec)
    declared = spec.get("source_target_sha256")
    if declared is not None and declared != digest:
        raise ValueError(f"target_spec source_target_sha256 mismatch: declared={declared}, computed={digest}")
    return digest


def canonical_command_payload_sha256(
    *,
    joint_names: Sequence[str],
    timestamps_s: np.ndarray,
    q_des: np.ndarray,
    dq_des: np.ndarray,
    tau_ff: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
) -> str:
    """Hash command semantics, not NPZ/ZIP container bytes.

    The dtype and little-endian byte order are pinned deliberately.  Any joint
    order, timestamp, or floating-point value difference changes the digest.
    """

    names = tuple(str(name) for name in joint_names)
    if len(names) != 31 or len(set(names)) != 31:
        raise ValueError("canonical command payload requires 31 unique joint names")
    arrays = {
        "timestamps_s": np.asarray(timestamps_s),
        "q_des": np.asarray(q_des),
        "dq_des": np.asarray(dq_des),
        "tau_ff": np.asarray(tau_ff),
        "kp": np.asarray(kp),
        "kd": np.asarray(kd),
    }
    samples = arrays["timestamps_s"].shape
    if len(samples) != 1 or samples[0] == 0:
        raise ValueError("timestamps_s must be a non-empty rank-1 array")
    for name, array in arrays.items():
        if not np.all(np.isfinite(array)):
            raise ValueError(f"command payload {name} contains non-finite values")
        if name == "timestamps_s":
            if array.shape != samples:
                raise ValueError("timestamps_s shape changed unexpectedly")
        elif array.shape != (samples[0], 31):
            raise ValueError(f"command payload {name} must have shape [T,31], got {array.shape}")
    if not np.all(np.diff(arrays["timestamps_s"].astype(np.float64)) > 0.0):
        raise ValueError("command timestamps must be strictly increasing")
    digest = hashlib.sha256()
    digest.update(b"a3-canonical-command-payload-v1\0")
    digest.update("\n".join(names).encode("utf-8") + b"\0")
    for name in ("timestamps_s", "q_des", "dq_des", "tau_ff", "kp", "kd"):
        array = np.ascontiguousarray(arrays[name], dtype="<f8")
        digest.update(name.encode("ascii") + b"\0")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def command_sha256_from_npz(path: Path) -> str:
    with np.load(path, allow_pickle=False) as archive:
        required = {"joint_names", "timestamps_s", "q_des", "dq_des", "tau_ff", "kp", "kd"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"{path}: command NPZ missing {', '.join(missing)}")
        names = [str(x) for x in archive["joint_names"].tolist()]
        return canonical_command_payload_sha256(
            joint_names=names,
            timestamps_s=archive["timestamps_s"],
            q_des=archive["q_des"],
            dq_des=archive["dq_des"],
            tau_ff=archive["tau_ff"],
            kp=archive["kp"],
            kd=archive["kd"],
        )


def validate_executor_contract(contract: Mapping[str, Any]) -> None:
    if int(contract.get("schema_version", 0)) != 1:
        raise ValueError("executor_contract.schema_version must be 1")
    if not str(contract.get("executor_contract_id", "")):
        raise ValueError("executor_contract_id is required")
    if float(contract.get("policy_hz", 0.0)) != 50.0:
        raise ValueError("first A3 strike contract is pinned to policy_hz=50")
    joints = contract.get("joints")
    if not isinstance(joints, list) or len(joints) != 31:
        raise ValueError("executor contract must declare all 31 joints")
    names = [str(joint.get("joint_name", "")) for joint in joints if isinstance(joint, Mapping)]
    indices = [int(joint.get("sdk_index", -1)) for joint in joints if isinstance(joint, Mapping)]
    if len(names) != 31 or len(set(names)) != 31 or sorted(indices) != list(range(31)):
        raise ValueError("executor contract joint names/indices are not a complete 31-DOF layout")
    for joint in joints:
        for field in ("ownership", "q_source", "dq_source", "tau_ff_source", "kp_source", "kd_source"):
            if not str(joint.get(field, "")):
                raise ValueError(f"executor contract joint {joint.get('joint_name')!r} lacks {field}")


def assert_training_manifest(path: Path) -> dict[str, Any]:
    """Fail closed before PPO can load a manifest."""

    resolved = path.expanduser().resolve()
    path_text = str(resolved).lower()
    forbidden = [token for token in FORBIDDEN_PATH_TOKENS if token in path_text]
    if forbidden:
        raise ValueError(f"training manifest path is forbidden by A3 contract: {forbidden}: {resolved}")
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("training manifest root must be an object")
    missing = [field for field in REQUIRED_MANIFEST_FIELDS if not data.get(field)]
    if missing:
        raise ValueError(f"training manifest missing required provenance fields: {', '.join(missing)}")
    if data["dataset_status"] != ACTIVE_DATASET_STATUS:
        raise ValueError(
            f"training manifest dataset_status={data['dataset_status']!r}; "
            f"only {ACTIVE_DATASET_STATUS!r} is accepted"
        )
    if data["gate_version"] != GATE_VERSION:
        raise ValueError(f"training manifest gate_version must be {GATE_VERSION!r}")
    motions = data.get("motions")
    if not isinstance(motions, list) or not motions:
        raise ValueError("training manifest must contain a non-empty motions list")
    return data
