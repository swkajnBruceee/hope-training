"""Fail-closed contract for the scratch-training AMP experiment.

The experiment may resume its own checkpoints, but it must never silently consume an
older A5, Residual, or unrelated PPO checkpoint.  Keeping this contract independent of
Isaac Lab makes the initialization boundary unit-testable on a host Python installation.
"""

from __future__ import annotations

from collections.abc import Mapping


SCRATCH_CONTRACT_VERSION = 1
SCRATCH_POLICY_CLASS = "ActorCritic"


def validate_scratch_algorithm(algo: Mapping, *, enabled: bool) -> None:
    """Reject Residual/other policy classes when scratch mode is enabled."""
    if not enabled:
        return
    policy = algo.get("policy", {})
    policy_class = str(policy.get("class_name", SCRATCH_POLICY_CLASS))
    if policy_class != SCRATCH_POLICY_CLASS:
        raise ValueError(
            "scratch_training requires the standard randomly initialized ActorCritic; "
            f"got policy.class_name={policy_class!r}. Do not use Residual/Frozen-A5 for this experiment."
        )


def build_scratch_contract(*, amp_enabled: bool) -> dict:
    """Return the checkpoint provenance contract for this experiment."""
    return {
        "version": SCRATCH_CONTRACT_VERSION,
        "initialization": "random_policy_and_fresh_optimizer",
        "policy_class": SCRATCH_POLICY_CLASS,
        "base_policy": None,
        "amp_enabled": bool(amp_enabled),
    }


def validate_scratch_checkpoint(
    payload: Mapping, *, amp_enabled: bool | None, path: str
) -> None:
    """Accept only a checkpoint produced by the current scratch contract."""
    contract = payload.get("hope_scratch_contract")
    if not isinstance(contract, Mapping):
        raise ValueError(
            "scratch_training refuses checkpoint without hope_scratch_contract: "
            f"{path}. This prevents loading historical A5/PingPong/Residual checkpoints."
        )
    expected = build_scratch_contract(
        amp_enabled=bool(contract.get("amp_enabled", False))
        if amp_enabled is None
        else bool(amp_enabled)
    )
    mismatches = {
        key: (contract.get(key), value)
        for key, value in expected.items()
        if key != "amp_enabled" or amp_enabled is not None
        if contract.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "scratch checkpoint contract mismatch; refusing to load checkpoint "
            f"{path}: {mismatches}"
        )
