#!/usr/bin/env python3
"""Numerically audit the exact PrecisionRescue kernel/gate conventions.

No Isaac scene and no PPO rollout is created.  This intentionally checks the
same ``exp(-error^2/std^2)`` and signed-tau equations used by the runtime
reward terms before expensive preflights are admitted.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[1]


def _load_pure_reward_module():
    """Load reward math without importing Gym/Isaac task registration."""
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


_reward = _load_pure_reward_module()
normal_kernel = _reward.normal_kernel
strike_temporal_weight = _reward.strike_temporal_weight
velocity_kernel = _reward.velocity_kernel
velocity_position_gate = _reward.velocity_position_gate


OUT = ROOT / "eval_outputs/v13b_complete_priors_precision_rescue"


def _sensitivity(fn, value: float, epsilon: float = 1.0e-4) -> float:
    return (fn(value + epsilon) - fn(max(0.0, value - epsilon))) / (2.0 * epsilon)


def main() -> None:
    normal_weight_exact, normal_weight_wide = 3.0, 1.5
    velocity_weight_exact, velocity_weight_wide = 2.5, 1.25
    normal_rows = []
    for degrees in (0, 5, 10, 15, 20, 30, 34, 40, 57, 70, 90):
        radians = degrees * 3.141592653589793 / 180.0
        exact = normal_kernel(radians, 0.1745329)
        wide = normal_kernel(radians, 0.60)
        normal_rows.append({
            "error_deg": degrees,
            "exact_kernel": exact,
            "wide_kernel": wide,
            "weighted_exact": normal_weight_exact * exact,
            "weighted_wide": normal_weight_wide * wide,
            "exact_finite_difference_per_rad": _sensitivity(lambda x: normal_kernel(x, 0.1745329), radians),
            "wide_finite_difference_per_rad": _sensitivity(lambda x: normal_kernel(x, 0.60), radians),
        })
    velocity_rows = []
    for error in (0, .25, .5, .75, 1., 1.5, 1.96, 2.5, 3.):
        exact = velocity_kernel(error, .5)
        wide = velocity_kernel(error, 2.0)
        velocity_rows.append({
            "error_mps": error,
            "exact_kernel": exact,
            "wide_kernel": wide,
            "weighted_exact": velocity_weight_exact * exact,
            "weighted_wide": velocity_weight_wide * wide,
            "exact_finite_difference_per_mps": _sensitivity(lambda x: velocity_kernel(x, .5), error),
            "wide_finite_difference_per_mps": _sensitivity(lambda x: velocity_kernel(x, 2.0), error),
        })
    gate_rows = [{"position_error_m": cm / 100.0, "position_gate": velocity_position_gate(cm / 100.0, .02, .05)}
                 for cm in (0, 1, 2, 3, 5, 8, 10, 15, 20)]
    temporal_rows = [{"signed_tau_s": tau, "strike_temporal_weight": strike_temporal_weight(tau, .05)}
                     for tau in (.30, .20, .15, .10, .05, .02, 0., -.02, -.05, -.10)]
    payload = {
        "status": "pass",
        "runtime_convention": {
            "normal": "acos(clamp(dot(actual_normal_w,target_normal_w),-1,1)); exp(-angle^2/std^2) * strike_temporal_weight",
            "velocity": "||actual_linear_velocity_w-target_linear_velocity_w||; exp(-error^2/std^2) * position_gate * strike_temporal_weight",
            "position_gate": "exp(-relu(position_error-0.02)^2/0.05^2)",
            "temporal": "exp(-0.5*(signed_tau/0.05)^2)",
        },
        "normal": normal_rows,
        "velocity": velocity_rows,
        "position_gate": gate_rows,
        "temporal": temporal_rows,
        "assertions": {
            "normal_wide_nonzero_at_57deg": normal_rows[8]["wide_kernel"] > 0.05,
            "normal_exact_saturated_at_57deg": normal_rows[8]["exact_kernel"] < 1.0e-8,
            "velocity_wide_nonzero_at_1_96mps": velocity_rows[6]["wide_kernel"] > 0.1,
            "velocity_exact_saturated_at_1_96mps": velocity_rows[6]["exact_kernel"] < 1.0e-5,
            "gate_near_full_0_to_2cm": all(row["position_gate"] > .99 for row in gate_rows[:3]),
            "gate_medium_at_5cm": .4 < gate_rows[4]["position_gate"] < .9,
            "gate_decays_by_10cm": gate_rows[6]["position_gate"] < .15,
            "gate_near_zero_by_15cm": gate_rows[7]["position_gate"] < .02,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "reward_kernel_sweeps.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not all(payload["assertions"].values()):
        raise SystemExit("PrecisionRescue kernel audit failed")
    print(OUT / "reward_kernel_sweeps.json")


if __name__ == "__main__":
    main()
