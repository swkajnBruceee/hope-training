"""Dependency-free checks for the archived Stage-A yaw compatibility layer."""

from __future__ import annotations

import pytest
import torch

from stage_a_compat import (
    STAGE_A_TERM_DIMS,
    STAGE_A_TERM_NAMES,
    adapt_stage_a_observation_legacy_yaw_pi,
    validate_stage_a_legacy_layout,
)


def test_legacy_yaw_adapter_rotates_only_declared_xy_vector_fields():
    observation = torch.arange(2 * 126, dtype=torch.float32).reshape(2, 126)
    adapted = adapt_stage_a_observation_legacy_yaw_pi(observation)

    changed = {index for start in (51, 54, 57) for index in (start, start + 1)}
    for index in range(126):
        expected = -observation[:, index] if index in changed else observation[:, index]
        assert torch.equal(adapted[:, index], expected)


def test_legacy_yaw_adapter_is_an_involution_and_does_not_mutate_input():
    observation = torch.randn(3, 126)
    original = observation.clone()
    adapted = adapt_stage_a_observation_legacy_yaw_pi(observation)
    assert torch.equal(adapt_stage_a_observation_legacy_yaw_pi(adapted), observation)
    assert torch.equal(observation, original)


@pytest.mark.parametrize("shape", [(125,), (2, 127), (2, 0)])
def test_legacy_yaw_adapter_rejects_an_unknown_observation_schema(shape):
    with pytest.raises(ValueError, match="126"):
        adapt_stage_a_observation_legacy_yaw_pi(torch.zeros(shape))


def test_legacy_yaw_adapter_rejects_a_changed_runtime_term_layout():
    validate_stage_a_legacy_layout(STAGE_A_TERM_NAMES, STAGE_A_TERM_DIMS)
    with pytest.raises(RuntimeError, match="frozen 126-D"):
        validate_stage_a_legacy_layout(tuple(reversed(STAGE_A_TERM_NAMES)), STAGE_A_TERM_DIMS)
    with pytest.raises(RuntimeError, match="frozen 126-D"):
        validate_stage_a_legacy_layout(STAGE_A_TERM_NAMES, (126,))
