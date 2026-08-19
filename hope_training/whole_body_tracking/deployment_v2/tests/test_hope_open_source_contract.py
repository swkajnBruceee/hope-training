from pathlib import Path

import numpy as np
import pytest

from hope_training.whole_body_tracking.deployment_v2.hope_open_source_contract import (
    ContractError,
    MODEL21800_SHA256,
    load_canonical_metadata,
    map_normalized_velocity,
    select_nearest_station_side,
    velocity_inside_native_component_support,
    velocity_inside_planner_box,
)

MODEL = Path("/home/a104/hope_training_repo/hope-deploy-baseline/a3_deploy/a3_deploy_example/models/model_21800/policy/exported/policy.onnx")


@pytest.fixture(scope="module")
def metadata():
    return load_canonical_metadata(MODEL)


def bounds(box):
    return np.array(box[0::2]), np.array(box[1::2])


def test_model_identity_and_exact_metadata(metadata):
    assert metadata.model_sha256 == MODEL21800_SHA256
    assert metadata.planner_velocity_boxes == (
        (1.57, 2.55, 0.10, 0.52, 0.41, 1.35),
        (1.55, 2.52, -0.18, 0.29, 0.40, 1.32),
    )
    assert metadata.reach_offsets == ((0.58, -0.44), (0.58, -0.09))


@pytest.mark.parametrize("side", [1, -1])
def test_normalized_endpoints_and_center(metadata, side):
    box = metadata.planner_velocity_boxes[0 if side == 1 else 1]
    low, high = bounds(box)
    np.testing.assert_allclose(map_normalized_velocity([-1, -1, -1], side, metadata), low)
    np.testing.assert_allclose(map_normalized_velocity([1, 1, 1], side, metadata), high)
    np.testing.assert_allclose(map_normalized_velocity([0, 0, 0], side, metadata), 0.5 * (low + high))


@pytest.mark.parametrize("side", [1, -1])
def test_ten_thousand_actions_stay_in_unexpanded_planner_box(metadata, side):
    rng = np.random.default_rng(20260819 + side)
    for action in rng.uniform(-1.0, 1.0, size=(10_000, 3)):
        velocity = map_normalized_velocity(action, side, metadata)
        assert velocity_inside_planner_box(velocity, side, metadata)
    # A point only admitted through the runtime margin can never be generated.
    low, _ = bounds(metadata.planner_velocity_boxes[0 if side == 1 else 1])
    outside = low - 0.01
    assert velocity_inside_native_component_support(outside, side, metadata, gate_margin=0.30)
    assert not velocity_inside_planner_box(outside, side, metadata)


def test_nearest_station_forehand_backhand_and_tie(metadata):
    target_fh = np.array(metadata.reach_offsets[0])
    target_bh = np.array(metadata.reach_offsets[1])
    assert select_nearest_station_side(target_fh, [0, 0], metadata)[0] == 1
    assert select_nearest_station_side(target_bh, [0, 0], metadata)[0] == -1
    midpoint = 0.5 * (target_fh + target_bh)
    assert select_nearest_station_side(midpoint, [0, 0], metadata)[0] == 1


def test_component_support_is_not_bounding_union(metadata):
    # FH: x=1.30 is core-only, z=0.45 is planner-only. The combination is in
    # the bounding union, but in neither zero-margin component.
    corner = [1.30, 0.20, 0.45]
    assert not velocity_inside_native_component_support(corner, 1, metadata, gate_margin=0.0)


def test_nan_inf_and_action_bounds_rejected(metadata):
    for bad in ([np.nan, 0, 0], [np.inf, 0, 0], [1.01, 0, 0]):
        with pytest.raises(ContractError):
            map_normalized_velocity(bad, 1, metadata)
