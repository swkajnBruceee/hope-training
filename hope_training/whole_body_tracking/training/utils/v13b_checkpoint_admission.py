"""Machine-verifiable admission for pure V1.3B actor checkpoints."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

SCHEMA_VERSION = "v13b_pure_actor_admission/v1"
GOAL_CONTRACT = "policy_strike_goal_10d/racket_contact_v1"


def sidecar_candidates(checkpoint: str | Path) -> tuple[Path, ...]:
    path = Path(checkpoint).expanduser().resolve()
    return (path.with_suffix(".v13b_admission.json"), Path(str(path) + ".v13b_admission.json"))


def find_sidecar(checkpoint: str | Path) -> Path | None:
    return next((path for path in sidecar_candidates(checkpoint) if path.is_file()), None)


def _state_dict(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError("checkpoint payload is not a mapping")
    for key in ("model_state_dict", "state_dict", "model"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return payload


def inspect_checkpoint_shapes(checkpoint: str | Path) -> dict[str, Any]:
    path = Path(checkpoint).expanduser().resolve()
    payload = torch.load(path, map_location="cpu")
    state = _state_dict(payload)
    weights = []
    normalizer_shapes = []
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        lower = str(key).lower()
        if "actor" in lower and lower.endswith("weight") and value.ndim == 2:
            weights.append((str(key), tuple(int(x) for x in value.shape)))
        if "obs_norm" in lower or "normalizer" in lower:
            if value.ndim >= 1 and value.numel() > 1:
                normalizer_shapes.append([int(x) for x in value.shape])
    if isinstance(payload, dict):
        for key in ("obs_norm_state_dict", "normalizer", "obs_normalizer"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                for value in nested.values():
                    if isinstance(value, torch.Tensor) and value.numel() > 1:
                        normalizer_shapes.append([int(x) for x in value.shape])
    if not weights:
        raise RuntimeError(f"checkpoint has no actor linear weights: {path}")
    first, last = weights[0][1], weights[-1][1]
    return {
        "checkpoint": str(path),
        "actor_weight_keys": [key for key, _ in weights],
        "actor_first_weight_shape": list(first),
        "actor_last_weight_shape": list(last),
        "actor_obs_dim": int(first[1]),
        "action_dim": int(last[0]),
        "normalizer_shapes": normalizer_shapes,
        "normalizer_has_98d": any(98 in shape for shape in normalizer_shapes),
    }


def validate_pure_v13b_checkpoint(checkpoint: str | Path, *, require_behavioral: bool = False, min_progress: float = 0.70) -> dict[str, Any]:
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"pure V1.3B checkpoint does not exist: {path}")
    sidecar = find_sidecar(path)
    if sidecar is None:
        raise RuntimeError("pure V1.3B checkpoint admission sidecar is missing; expected " + ", ".join(map(str, sidecar_candidates(path))))
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"unsupported V1.3B admission schema in {sidecar}")
    shapes = inspect_checkpoint_shapes(path)
    failures = []
    recorded_checkpoint = metadata.get("checkpoint")
    if recorded_checkpoint and Path(str(recorded_checkpoint)).expanduser().resolve() != path:
        failures.append("sidecar checkpoint path does not match loaded checkpoint")
    if int(metadata.get("actor_obs_dim", -1)) != 98 or shapes["actor_obs_dim"] != 98:
        failures.append(f"actor_obs_dim mismatch: {metadata.get('actor_obs_dim')} / {shapes['actor_obs_dim']}")
    if int(metadata.get("action_dim", -1)) != 26 or shapes["action_dim"] != 26:
        failures.append(f"action_dim mismatch: {metadata.get('action_dim')} / {shapes['action_dim']}")
    if not shapes["normalizer_has_98d"]:
        failures.append("actor observation normalizer is not 98D")
    if metadata.get("goal_contract_version") != GOAL_CONTRACT:
        failures.append("goal contract mismatch")
    if float(metadata.get("training_progress", -1.0)) < min_progress:
        failures.append(f"training_progress<{min_progress}")
    for key in ("upper_prior_alpha", "lower_prior_alpha"):
        if abs(float(metadata.get(key, 1.0))) > 1.0e-8:
            failures.append(f"{key} is not zero")
    for key in ("model900_runtime_enabled", "model3396_runtime_enabled", "reference_action_enabled"):
        if metadata.get(key) is not False:
            failures.append(f"{key} is not false")
    for key in ("public_actor_reference_free", "pure_v13b_phase"):
        if metadata.get(key) is not True:
            failures.append(f"{key} is not true")
    if require_behavioral:
        for key in ("one_strike_runtime_verified", "teacher_kill_test_verified", "target_causality_verified"):
            if metadata.get(key) is not True:
                failures.append(f"behavior gate {key} is not verified")
        if metadata.get("qualification_status") != "qualified":
            failures.append("qualification_status is not qualified")
    if failures:
        raise RuntimeError("pure V1.3B checkpoint admission failed: " + "; ".join(failures))
    return {"requested_source_kind": "pure_v13b_actor", "verified_source_kind": "pure_v13b_actor", "checkpoint_admission_verified": True, "sidecar": str(sidecar), "require_behavioral": bool(require_behavioral), "metadata": metadata, "runtime_shapes": shapes}
