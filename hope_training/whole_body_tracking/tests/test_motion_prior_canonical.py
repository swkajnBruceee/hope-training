from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.utils.motion_prior_canonical import (  # noqa: E402
    MOTION_PRIOR_CONTRACT_VERSION,
    MotionPriorCanonicalError,
    canonicalize_motion_arrays,
    canonicalize_strike_target,
)


def _source_arrays():
    root = np.asarray([3.15, -0.35, 0.3084])
    positions = np.zeros((2, 2, 3), dtype=np.float32)
    positions[:, 0] = root
    positions[0, 1] = [2.6229154, -0.7281592, 0.2149282]
    positions[1, 1] = [2.60, -0.75, 0.20]
    quaternions = np.zeros((2, 2, 4), dtype=np.float32)
    quaternions[..., 3] = 1.0  # wxyz: 180 degrees about Z.
    linear = np.zeros_like(positions)
    linear[:, 1] = [-1.5473132, -0.0920320, 0.9293890]
    angular = np.zeros_like(positions)
    return {
        "fps": np.asarray([50]),
        "joint_pos": np.zeros((2, 31), dtype=np.float32),
        "joint_vel": np.zeros((2, 31), dtype=np.float32),
        "body_pos_w": positions,
        "body_quat_w": quaternions,
        "body_lin_vel_w": linear,
        "body_ang_vel_w": angular,
    }


def test_canonicalization_removes_source_xy_yaw_without_changing_metric_motion():
    result = canonicalize_motion_arrays(_source_arrays())
    assert bytes(result["contract_version_utf8"]).decode() == MOTION_PRIOR_CONTRACT_VERSION
    assert result["body_pos_b0"][0, 0] == pytest.approx((0.0, 0.0, 0.0))
    assert result["body_pos_b0"][0, 1] == pytest.approx(
        (0.5270846, 0.3781592, -0.0934718), abs=1.0e-6
    )
    assert result["body_lin_vel_b0"][0, 1] == pytest.approx(
        (1.5473132, 0.0920320, 0.9293890), abs=1.0e-6
    )
    assert result["body_quat_b0_wxyz"][0, 0] == pytest.approx(
        (1.0, 0.0, 0.0, 0.0), abs=1.0e-6
    )


def test_strike_metadata_uses_the_exact_same_frozen_heading_frame():
    target = canonicalize_strike_target(
        {
            "racket_position_m": [2.6229154, -0.7281592, 0.2149282],
            "racket_velocity_mps": [-1.5473132, -0.0920320, 0.9293890],
            "racket_normal_w": [0.9928530, -0.0554928, 0.1056573],
            "ball_position_m": [2.6136936, -0.7375996, 0.2377120],
        },
        root_anchor_position_w=np.asarray([3.15, -0.35, 0.3084]),
        root_anchor_yaw_rad=np.pi,
    )
    assert target["racket_position_b0_m"] == pytest.approx(
        (0.5270846, 0.3781592, -0.0934718), abs=1.0e-6
    )
    assert target["racket_velocity_b0_mps"] == pytest.approx(
        (1.5473132, 0.0920320, 0.9293890), abs=1.0e-6
    )
    assert target["racket_normal_b0"] == pytest.approx(
        (-0.9928530, 0.0554928, 0.1056573), abs=1.0e-6
    )


def test_canonicalization_rejects_invalid_quaternions_and_shapes():
    source = _source_arrays()
    source["body_quat_w"][:] = 0.0
    with pytest.raises(MotionPriorCanonicalError, match="zero quaternion"):
        canonicalize_motion_arrays(source)
    source = _source_arrays()
    source["body_lin_vel_w"] = np.zeros((2, 1, 3))
    with pytest.raises(MotionPriorCanonicalError, match="velocity shapes"):
        canonicalize_motion_arrays(source)
    source = _source_arrays()
    source["joint_vel"] = np.zeros((1, 31))
    with pytest.raises(MotionPriorCanonicalError, match="joint position/velocity shapes"):
        canonicalize_motion_arrays(source)
    source = _source_arrays()
    source["fps"] = np.asarray([0])
    with pytest.raises(MotionPriorCanonicalError, match="positive finite scalar"):
        canonicalize_motion_arrays(source)
