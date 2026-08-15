"""Regression test for retrying an initially out-of-envelope ball command."""

from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "mujoco_reference", "reference"))

from a3_deploy_onnx_ref_pingpong.lifecycle import LifecycleConfig, Phase, SwingLifecycle  # noqa: E402
from a3_deploy_onnx_ref_pingpong.observation import RobotState  # noqa: E402
from a3_deploy_onnx_ref_pingpong.racket_command import FOREHAND, RacketCommand  # noqa: E402


def _state() -> RobotState:
    return RobotState(
        base_pos_w=np.array([-0.041, 0.0, 1.068]),
        base_quat_w=np.array([1.0, 0.0, 0.0, 0.0]),
        base_ang_vel_b=np.zeros(3),
        q=np.zeros(31),
        qd=np.zeros(31),
        base_lin_vel_w=np.zeros(3),
    )


def _cmd(task_revision: int, position) -> RacketCommand:
    return RacketCommand(
        task_id=1,
        task_revision=task_revision,
        swing_sign=FOREHAND,
        position=np.asarray(position, dtype=np.float64),
        velocity=np.array([2.0, 0.4, 1.0]),
        time_to_strike=0.5,
    )


def test_rejected_task_is_retried_when_a_later_revision_enters_geometry():
    lifecycle = SwingLifecycle(LifecycleConfig(station_ready_hold_s=0.0))
    lifecycle.configure_geometry(
        pos_boxes=((0.58, 0.58, -0.48, -0.40, 0.85, 1.30),
                   (0.58, 0.58, -0.13, -0.05, 0.85, 1.30)),
        vel_boxes=((0.0, 3.0, -1.0, 1.0, 0.0, 2.0),
                   (0.0, 3.0, -1.0, 1.0, 0.0, 2.0)),
        reach_offsets=(0.630113, -0.441390, 0.524830, -0.081652),
    )
    state = _state()

    # First prediction is out of range and must not consume task_id=1.
    lifecycle.update(_cmd(0, [0.58, -0.44, 0.20]), state)
    assert lifecycle.phase is Phase.READY
    assert lifecycle.pending_station_xy is None

    # A later revision for the same ball is valid and must be allowed to engage.
    lifecycle.update(_cmd(1, [0.58, -0.44, 1.05]), state)
    assert lifecycle.phase is Phase.SWING
    assert lifecycle.active_task_id == 1


def test_streaming_revisions_do_not_restart_station_dwell():
    lifecycle = SwingLifecycle(LifecycleConfig(station_ready_hold_s=0.12))
    lifecycle.configure_geometry(
        pos_boxes=((0.58, 0.58, -0.48, -0.40, 0.85, 1.30),
                   (0.58, 0.58, -0.13, -0.05, 0.85, 1.30)),
        vel_boxes=((0.0, 3.0, -1.0, 1.0, 0.0, 2.0),
                   (0.0, 3.0, -1.0, 1.0, 0.0, 2.0)),
        reach_offsets=(0.630113, -0.441390, 0.524830, -0.081652),
    )
    state = _state()

    # A planner commonly refreshes the same ball every control tick.  The
    # unchanged station must accumulate its dwell time and engage normally.
    for revision in range(7):
        lifecycle.update(_cmd(revision, [0.58, -0.44, 1.05]), state)

    assert lifecycle.phase is Phase.SWING
    assert lifecycle.active_task_id == 1
