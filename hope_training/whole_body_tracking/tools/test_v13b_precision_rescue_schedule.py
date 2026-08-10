#!/usr/bin/env python3
"""Unit tests for continuation, hold, and monotonic performance gating.

This test is deliberately Isaac-free.  Importing ``training`` normally
registers the project Gym tasks, which in turn requires a live SimulationApp;
that is the wrong dependency for a pure schedule contract test.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[1]


def _load_pure_schedule_module():
    """Load the two pure helper modules without importing ``training``."""
    training_pkg = types.ModuleType("training")
    training_pkg.__path__ = [str(ROOT / "training")]
    utils_pkg = types.ModuleType("training.utils")
    utils_pkg.__path__ = [str(ROOT / "training" / "utils")]
    sys.modules.setdefault("training", training_pkg)
    sys.modules.setdefault("training.utils", utils_pkg)
    for name, filename in (
        ("training.utils.v13b_contract", "v13b_contract.py"),
        ("training.utils.v13b_precision_rescue", "v13b_precision_rescue.py"),
    ):
        if name in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(name, ROOT / "training" / "utils" / filename)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules["training.utils.v13b_precision_rescue"]


PrecisionRescuePriorSchedule = _load_pure_schedule_module().PrecisionRescuePriorSchedule


def main() -> None:
    schedule = PrecisionRescuePriorSchedule(
        source_progress=.435, source_lower_alpha=.554, source_upper_alpha=.305,
        total_chain_updates=50000, hold_updates=300, upper_step=.05,
    )
    assert abs(schedule.global_progress - .435) < 1e-12
    assert abs(schedule.lower_alpha() - .554) < 1e-12
    assert abs(schedule.upper_alpha() - .305) < 1e-12
    lower = []
    upper = []
    for update in range(1000):
        schedule.set_update(update)
        if update >= 300:
            schedule.set_readiness(True)
            schedule.advance_upper_once()
        lower.append(schedule.lower_alpha())
        upper.append(schedule.upper_alpha())
    assert all(a >= b - 1e-12 for a, b in zip(lower, lower[1:]))
    assert all(a >= b - 1e-12 for a, b in zip(upper, upper[1:]))
    assert all(abs(value - .305) < 1e-12 for value in upper[:300])
    assert min(upper) == 0.0
    print("PASS: PrecisionRescue schedule continuity/hold/monotonicity")


if __name__ == "__main__":
    main()
