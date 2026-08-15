"""Motion-array schema helpers that do not require Isaac Sim."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def select_motion_bodies(
    array: np.ndarray,
    body_indexes: Sequence[int],
    source: str,
    field: str,
    articulation_body_count: int | None = None,
) -> np.ndarray:
    """Select configured bodies from a compact or complete motion artifact.

    The published HOPE clips use the complete 32-body articulation schema.  Compact
    replacement clips may contain only the configured tracked bodies in command order.
    """

    indexes = np.asarray(body_indexes, dtype=np.int64).reshape(-1)
    if indexes.size == 0:
        raise ValueError("MotionLoader resolved no tracked body indexes")
    if int(indexes.min()) < 0:
        raise ValueError(
            f"MotionLoader body indexes must be non-negative, got {int(indexes.min())}"
        )
    if array.ndim < 2:
        raise ValueError(
            f"motion file {source!r} field {field!r} must have a body axis, "
            f"got shape {array.shape}"
        )

    body_count = int(array.shape[1])
    tracked_count = int(indexes.size)
    if body_count == tracked_count:
        return array

    max_index = int(indexes.max())
    if body_count > max_index and (
        articulation_body_count is None or body_count >= articulation_body_count
    ):
        return array[:, indexes]

    raise ValueError(
        f"motion file {source!r} stores {body_count} bodies in {field}, but "
        f"the prepared articulation maps a tracked body to index {max_index}. "
        f"Provide either {tracked_count} tracked bodies in configured order or "
        "the complete articulation-body arrays."
    )
