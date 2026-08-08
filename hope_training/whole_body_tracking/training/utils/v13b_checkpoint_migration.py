"""Explicit-name checkpoint migration for the V1.3B observation contract.

This module never copies the first ``N`` columns blindly.  Callers must pass
term-name slices from the source and destination observation managers; unknown
new terms are zero initialized and optimizer state is intentionally discarded.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class ObservationSlice:
    name: str
    start: int
    end: int


def migrate_first_layer(
    old_weight: torch.Tensor,
    *,
    old_terms: tuple[ObservationSlice, ...],
    new_terms: tuple[ObservationSlice, ...],
) -> torch.Tensor:
    if old_weight.ndim != 2:
        raise ValueError("actor first layer must be a rank-2 tensor")
    new_dim = max(term.end for term in new_terms)
    result = torch.zeros((old_weight.shape[0], new_dim), dtype=old_weight.dtype, device=old_weight.device)
    old_by_name = {term.name: term for term in old_terms}
    for target in new_terms:
        source = old_by_name.get(target.name)
        if source is None:
            continue
        width = min(source.end - source.start, target.end - target.start)
        result[:, target.start : target.start + width] = old_weight[:, source.start : source.start + width]
    return result


def migrate_actor_state_dict(
    state: dict[str, torch.Tensor],
    *,
    old_terms: tuple[ObservationSlice, ...],
    new_terms: tuple[ObservationSlice, ...],
) -> dict[str, torch.Tensor]:
    migrated = dict(state)
    if "actor.0.weight" not in state:
        raise KeyError("checkpoint has no actor.0.weight")
    migrated["actor.0.weight"] = migrate_first_layer(
        state["actor.0.weight"], old_terms=old_terms, new_terms=new_terms
    )
    # Hidden layers, bias, and 26-D output head are shape-compatible and may be
    # copied.  Optimizer state is deliberately not returned by this utility.
    return migrated
