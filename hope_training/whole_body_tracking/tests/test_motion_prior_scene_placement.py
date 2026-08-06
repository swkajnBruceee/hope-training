from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.utils.motion_prior_canonical import (  # noqa: E402
    MOTION_PRIOR_CONTRACT_VERSION,
    MotionPriorCanonicalError,
)
from training.utils.motion_prior_scene_placement import (  # noqa: E402
    SCENE_PLACEMENT_CONTRACT_VERSION,
    place_canonical_motion_arrays,
    place_canonical_strike_target,
)


def _canonical_arrays():
    positions = np.asarray(
        [
            [[0.0, 0.0, 0.0], [0.5, 0.2, -0.1]],
            [[0.1, 0.0, 0.0], [0.6, 0.2, -0.1]],
        ],
        dtype=np.float32,
    )
    quaternions = np.zeros((2, 2, 4), dtype=np.float32)
    quaternions[..., 0] = 1.0
    return {
        "contract_version_utf8": np.frombuffer(
            MOTION_PRIOR_CONTRACT_VERSION.encode(), dtype=np.uint8
        ),
        "fps": np.asarray([50]),
        "joint_pos": np.arange(62, dtype=np.float32).reshape(2, 31),
        "joint_vel": np.arange(62, dtype=np.float32).reshape(2, 31) * 0.1,
        "body_pos_b0": positions,
        "body_quat_b0_wxyz": quaternions,
        "body_lin_vel_b0": positions.copy(),
        "body_ang_vel_b0": positions.copy() * 2.0,
    }


def test_scene_placement_is_one_rigid_transform_and_preserves_joint_trajectory():
    canonical = _canonical_arrays()
    placed = place_canonical_motion_arrays(
        canonical,
        root_anchor_w_m=(-0.5, -0.7625, 1.0684),
        root_heading_w_rad=np.pi / 2.0,
    )
    assert placed["joint_pos"] == pytest.approx(canonical["joint_pos"])
    assert placed["joint_vel"] == pytest.approx(canonical["joint_vel"])
    assert placed["body_pos_w"][0, 0] == pytest.approx(
        (-0.5, -0.7625, 1.0684), abs=1.0e-6
    )
    assert placed["body_pos_w"][0, 1] == pytest.approx(
        (-0.7, -0.2625, 0.9684), abs=1.0e-6
    )
    assert placed["body_quat_w"][0, 0] == pytest.approx(
        (np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)), abs=1.0e-6
    )
    assert bytes(placed["scene_placement_contract_utf8"]).decode() == (
        SCENE_PLACEMENT_CONTRACT_VERSION
    )


def test_strike_target_uses_same_rotation_translation_and_preserves_distance_metadata():
    target = place_canonical_strike_target(
        {
            "racket_position_b0_m": [0.5, 0.2, -0.1],
            "racket_velocity_b0_mps": [1.0, 0.0, 0.5],
            "racket_normal_b0": [1.0, 0.0, 0.0],
            "ball_position_b0_m": [0.52, 0.2, -0.1],
        },
        root_anchor_w_m=(-0.5, -0.7625, 1.0684),
        root_heading_w_rad=np.pi / 2.0,
        source_strike_target={
            "racket_quat_xyzw": [0.0, 0.0, 1.0, 0.0],
            "ball_to_racket_center_distance_m": 0.02,
        },
        source_root_heading_w_rad=np.pi,
    )
    assert target["racket_position_m"] == pytest.approx(
        (-0.7, -0.2625, 0.9684), abs=1.0e-6
    )
    assert target["racket_velocity_mps"] == pytest.approx((0.0, 1.0, 0.5), abs=1.0e-6)
    assert target["racket_normal_w"] == pytest.approx((0.0, 1.0, 0.0), abs=1.0e-6)
    assert target["ball_to_racket_center_distance_m"] == pytest.approx(0.02)


def test_scene_placement_fails_closed_on_wrong_canonical_contract():
    canonical = _canonical_arrays()
    canonical["contract_version_utf8"] = np.frombuffer(b"wrong/v1", dtype=np.uint8)
    with pytest.raises(MotionPriorCanonicalError, match="expected"):
        place_canonical_motion_arrays(
            canonical,
            root_anchor_w_m=(0.0, 0.0, 0.0),
            root_heading_w_rad=0.0,
        )
