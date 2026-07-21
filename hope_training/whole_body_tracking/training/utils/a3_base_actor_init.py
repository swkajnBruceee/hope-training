"""Initialization contract for passive-plant A3 Base residual policies."""

from __future__ import annotations

import torch


def initialize_zero_residual_actor_mean(runner, action_dim: int = 14) -> torch.nn.Linear:
    """Make the constructed Actor mean exactly zero for every observation."""
    actor_output = runner.alg.policy.actor[-1]
    if not isinstance(actor_output, torch.nn.Linear) or actor_output.out_features != action_dim:
        raise RuntimeError(
            "A3 Base zero-residual initialization found an unexpected actor output layer"
        )
    torch.nn.init.zeros_(actor_output.weight)
    torch.nn.init.zeros_(actor_output.bias)
    return actor_output
