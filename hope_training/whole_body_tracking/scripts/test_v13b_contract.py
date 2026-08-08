#!/usr/bin/env python3
"""CPU-only V1.3B contract tests (no Isaac/PhysX startup)."""

from __future__ import annotations

from pathlib import Path
import sys
import math

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.utils.strike_goal import (  # noqa: E402
    AxialRacketContactCalibration,
    PolicyStrikeGoalAdapter,
    StrikeGoal10D,
)
from training.utils.v13b_contract import beta_global, reference_reward_multiplier, teacher_alpha  # noqa: E402
from training.utils.v13b_checkpoint_migration import ObservationSlice, migrate_first_layer  # noqa: E402


def main() -> None:
    source = StrikeGoal10D(
        position=(1.0, 2.0, 3.0),
        normal=(1.0, 0.0, 0.0),
        linear_velocity=(2.0, 0.0, 0.0),
        time_to_hit_s=0.4,
        frame_id="world",
        source="synthetic",
    )
    calibration = AxialRacketContactCalibration(
        ball_radius_m=0.020,
        link_origin_to_effective_face_along_normal_m=0.003,
        calibration_version="test_calibration",
        qualified_domain="unit_test_only",
    )
    goal = PolicyStrikeGoalAdapter(calibration).adapt(source, elapsed_s=0.1)
    expected = (1.017, 2.0, 3.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.3)
    assert all(math.isclose(a, b, rel_tol=0.0, abs_tol=1.0e-9) for a, b in zip(goal.to_vector(), expected))
    assert teacher_alpha(0.70) == 0.0 and teacher_alpha(1.0) == 0.0
    assert reference_reward_multiplier(0.70) == 0.0
    assert beta_global(0.10) == 0.0 and beta_global(0.70) == 1.0
    old = (ObservationSlice("joint_pos", 0, 2), ObservationSlice("old_reference", 2, 4))
    new = (ObservationSlice("joint_pos", 0, 2), ObservationSlice("strike_goal_10d", 2, 12))
    migrated = migrate_first_layer(__import__("torch").ones(3, 4), old_terms=old, new_terms=new)
    assert migrated.shape == (3, 12) and float(migrated[:, 2:].abs().sum()) == 0.0
    print("V1.3B CPU contract tests: PASS")


if __name__ == "__main__":
    main()
