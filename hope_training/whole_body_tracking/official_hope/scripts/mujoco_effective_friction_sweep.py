#!/usr/bin/env python3
"""Sweep the verified MuJoCo foot-floor effective sliding friction contract.

This is a stance/contact diagnostic, not a claim about the real PVC coefficient.  Each MuJoCo
instance sets both foot and floor sliding friction to the requested scalar and records the
assembled ``mjData.contact[].friction[0]`` values.  A future ball/rally evaluator can reuse the
same ``--mu-values`` contract without silently falling back to the MJCF foot value of 1.5.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from stance_stability_test import (  # noqa: E402
    default_model,
    run_static,
)
from a3_deploy_onnx_ref_pingpong.stance_stability import (  # noqa: E402
    StanceConfig,
    StanceMujoco,
    aggregate_rows,
    write_rows,
)


def _values(text: str) -> list[float]:
    values = [float(item) for item in text.split(",") if item.strip()]
    if not values or any(not math.isfinite(item) or item < 0.0 for item in values):
        raise ValueError("--mu-values must contain finite non-negative numbers")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-xml", default=str(default_model()))
    parser.add_argument("--output-dir", default="outputs/stance_stability/effective_friction_sweep")
    parser.add_argument("--mu-values", default="0.3,0.4,0.5,0.6,0.7,0.8,1.0,1.2,1.5")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hip", type=float, default=15.0)
    parser.add_argument("--knee", type=float, default=25.0)
    parser.add_argument("--width-m", type=float, default=0.50)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--transition-s", type=float, default=1.0)
    parser.add_argument("--initial-noise", action="store_true")
    parser.add_argument("--base-roll-noise-deg", type=float, default=0.0)
    parser.add_argument("--base-pitch-noise-deg", type=float, default=0.0)
    args = parser.parse_args()
    if args.trials <= 0:
        parser.error("--trials must be positive")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for mu in _values(args.mu_values):
        sim = StanceMujoco(args.model_xml, seed=args.seed, mu_contact=mu)
        stance = sim.generator.generate(
            StanceConfig(
                hip_flexion_deg=args.hip,
                knee_flexion_deg=args.knee,
                torso_pitch_deg=0.0,
                stance_width_scale=1.0,
                stance_width_m=args.width_m,
                fore_aft_m=0.0,
                lead_leg="none",
            )
        )
        if not stance.valid:
            raise RuntimeError(f"invalid stance IK at mu={mu}: {stance.diagnostics}")
        run_args = SimpleNamespace(
            pd_profile="official_stand",
            initial_noise=bool(args.initial_noise),
            base_roll_noise=math.radians(args.base_roll_noise_deg),
            base_pitch_noise=math.radians(args.base_pitch_noise_deg),
            transition_s=args.transition_s,
            duration=args.duration,
            trace_dir=None,
        )
        for trial in range(args.trials):
            seed = args.seed + trial
            sim.rng = np.random.default_rng(seed)
            rows.append(run_static(sim, stance, run_args, trial_id=trial, seed=seed))

    write_rows(out_dir / "friction_results.csv", rows)
    write_rows(
        out_dir / "friction_summary.csv",
        aggregate_rows(rows, ("mu_contact_requested", "mu_contact_configured", "test_type")),
    )
    print(f"wrote {len(rows)} rows to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
