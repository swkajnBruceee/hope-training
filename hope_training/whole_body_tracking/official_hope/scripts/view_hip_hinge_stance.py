#!/usr/bin/env python3
"""Visualize the model-backed hip-hinge stance with MuJoCo passive viewer."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REF = REPO / "mujoco_reference" / "reference"
if str(REF) not in sys.path:
    sys.path.insert(0, str(REF))

import mujoco
from mujoco import viewer as mujoco_viewer

from a3_deploy_onnx_ref_pingpong.stance_stability import (  # noqa: E402
    StanceConfig,
    StanceMujoco,
    official_stand_pd_gains,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--model-xml", default=None)
    args = parser.parse_args()

    model_path = args.model_xml or str(
        REPO.parent.parent.parent
        / "agibot/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml"
    )
    sim = StanceMujoco(model_path, control_dt=0.02)
    stance = sim.generator.generate(
        StanceConfig(
            hip_flexion_deg=15.0,
            knee_flexion_deg=25.0,
            torso_pitch_deg=4.0,
            stance_width_m=0.50,
            pelvis_back_m=0.04,
            pelvis_pitch_deg=0.0,
        )
    )
    if not stance.valid:
        raise RuntimeError(f"invalid hip-hinge stance: {stance.diagnostics}")

    sim.reset(stance)
    kp, kd = official_stand_pd_gains()
    viewer = mujoco_viewer.launch_passive(sim.model, sim.data)
    start = time.monotonic()
    try:
        while viewer.is_running() and time.monotonic() - start < args.duration:
            tick_start = time.monotonic()
            sim.set_targets(stance.q, kp, kd)
            sim.step()
            viewer.sync()
            time.sleep(max(0.0, sim.control_dt - (time.monotonic() - tick_start)))
    finally:
        viewer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
