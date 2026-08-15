"""Host regression coverage for compact and complete HOPE motion schemas."""

import importlib.util
from pathlib import Path

import numpy as np


_ROOT = Path(__file__).resolve().parents[1]
_MODULE = (
    _ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "utils"
    / "motion_schema.py"
)
_SPEC = importlib.util.spec_from_file_location("hope_motion_schema", _MODULE)
motion_schema = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(motion_schema)


def test_compact_motion_is_already_in_command_order() -> None:
    array = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    selected = motion_schema.select_motion_bodies(
        array, [1, 7, 11], "clip.npz", "body_quat_w"
    )
    assert selected is array


def test_complete_motion_uses_live_articulation_indexes() -> None:
    array = np.arange(2 * 12 * 3).reshape(2, 12, 3)
    selected = motion_schema.select_motion_bodies(
        array, [1, 7, 11], "clip.npz", "body_pos_w", articulation_body_count=12
    )
    np.testing.assert_array_equal(selected, array[:, [1, 7, 11]])


def test_ambiguous_motion_schema_fails_before_cuda() -> None:
    array = np.zeros((2, 6, 3), dtype=np.float32)
    try:
        motion_schema.select_motion_bodies(
            array, [1, 4, 5], "clip.npz", "body_pos_w", articulation_body_count=12
        )
    except ValueError as error:
        assert "stores 6 bodies" in str(error)
    else:
        raise AssertionError("invalid motion-body schema must fail before CUDA")
