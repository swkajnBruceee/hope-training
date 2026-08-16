#!/usr/bin/env python3
"""Phase-D zero-strike policy test on a selected nominal stance.

This keeps the existing 110-D observation and 31-D action contracts intact.  It
only changes the physical initial stance; the exported actor's default_q remains
the original deploy default, so any instability is attributable to policy/stance
compatibility rather than a silent contract rewrite.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
REF = REPO / "mujoco_reference" / "reference"
sys.path.insert(0, str(REF))

from a3_deploy_onnx_ref_pingpong.config import RuntimeConfig  # noqa: E402
from a3_deploy_onnx_ref_pingpong.observation import ObsTarget, RobotState, build_observation  # noqa: E402
from a3_deploy_onnx_ref_pingpong.onnx_policy import OnnxPolicy  # noqa: E402
from a3_deploy_onnx_ref_pingpong.stance_stability import (  # noqa: E402
    GeneratedStance,
    MetricCollector,
    StanceConfig,
    StanceMujoco,
    official_stand_pd_gains,
    thresholds_for,
)


def selected_stance(sim: StanceMujoco, args):
    if args.hip == 0.0 and args.knee == 0.0 and args.width_scale == 1.0 and args.fore_aft == 0.0 and args.torso == 0.0:
        return GeneratedStance(
            config=StanceConfig(), q=sim.baseline_q(),
            root_qpos=sim.baseline_qpos[sim.root_qadr:sim.root_qadr + 7].copy(),
            left_foot_target=sim.generator.baseline_left_foot.copy(),
            right_foot_target=sim.generator.baseline_right_foot.copy(),
            pelvis_height_m=float(sim.baseline_qpos[sim.root_qadr + 2]),
            width_m=sim.generator.baseline_width_m, valid=True,
            diagnostics={"baseline_source": "MJCF keyframe 0"},
        )
    cfg = StanceConfig(args.hip, args.knee, args.torso, args.width_scale, None, args.fore_aft,
                       "none" if args.fore_aft == 0.0 else args.lead_leg, None)
    return sim.generator.generate(cfg)


def main() -> int:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--runtime-config", default=str(REPO / "mujoco_reference/config/hope_pingpong_runtime.yaml"))
    p.add_argument("--model-xml", default=None)
    p.add_argument("--checkpoint-onnx", default=None)
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--transition-s", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="outputs/stance_stability/policy_idle_results.csv")
    p.add_argument("--hip", type=float, default=0.0); p.add_argument("--knee", type=float, default=0.0)
    p.add_argument("--torso", type=float, default=0.0); p.add_argument("--width-scale", type=float, default=1.0)
    p.add_argument("--fore-aft", type=float, default=0.0); p.add_argument("--lead-leg", choices=("left", "right"), default="left")
    args = p.parse_args()
    cfg = RuntimeConfig.load(args.runtime_config)
    model_path = args.model_xml or str(cfg.model_xml_path)
    policy_path = args.checkpoint_onnx or str(cfg.onnx_path)
    sim = StanceMujoco(model_path, control_dt=cfg.control_dt, seed=args.seed)
    stance = selected_stance(sim, args)
    if not stance.valid:
        raise RuntimeError(f"invalid stance IK: {stance.diagnostics}")
    policy = OnnxPolicy(policy_path)
    adapter = cfg.action_adapter
    sim.reset(stance)
    kp_stand, kd_stand = official_stand_pd_gains()
    start_q = sim.state()["q"].copy()
    for tick in range(max(1, round(args.transition_s / sim.control_dt))):
        a = (tick + 1) / max(1, round(args.transition_s / sim.control_dt))
        a = a * a * (3.0 - 2.0 * a)
        q_des = (1.0 - a) * start_q + a * stance.q
        tau = sim.set_targets(q_des, kp_stand, kd_stand); sim.step()
    collector = MetricCollector(sim, thresholds=thresholds_for(sim), dt=sim.control_dt)
    last_action = np.zeros(31, dtype=np.float64)
    base_target_xy = sim.state()["base_pos"][:2].copy()
    racket_sid = sim.mj.mj_name2id(sim.model, sim.mj.mjtObj.mjOBJ_SENSOR, "right_racket_framepos")
    gyro_sid = sim.mj.mj_name2id(sim.model, sim.mj.mjtObj.mjOBJ_SENSOR, "pelvis_imu_gyro")
    if racket_sid < 0:
        raise RuntimeError("right_racket_framepos sensor is required for policy idle test")
    racket_adr = int(sim.model.sensor_adr[racket_sid])
    gyro_adr = int(sim.model.sensor_adr[gyro_sid]) if gyro_sid >= 0 else None
    for _ in range(max(1, round(args.duration / sim.control_dt))):
        s = sim.state()
        gyro = sim.data.sensordata[gyro_adr:gyro_adr + 3].copy() if gyro_adr is not None else s["base_ang_vel"]
        racket_pos = sim.data.sensordata[racket_adr:racket_adr + 3].copy()
        obs_state = RobotState(s["base_pos"], s["base_quat"], gyro, s["q"], s["qd"], s["base_lin_vel"])
        target = ObsTarget(racket_pos, np.zeros(3), 10.0)
        obs = build_observation(obs_state, target, last_action, adapter.default_q, base_target_xy)
        raw = np.nan_to_num(policy.infer(obs, time_step=0).astype(np.float64), nan=0.0)
        raw[[3, 4]] = 0.0
        last_action = np.clip(raw, -20.0, 20.0)
        q_des = adapter.decode(last_action)
        q_des[[3, 4]] = adapter.default_q[[3, 4]]
        tau = sim.set_targets(q_des, cfg.sim_kp, cfg.sim_kd); sim.step()
        collector.step(tau, q_des)
    result = collector.finalize(survival_time=sim.time)
    result.update({"mode": "policy_idle", "hip_deg": args.hip, "knee_deg": args.knee, "torso_deg": args.torso,
                   "stance_width_scale": args.width_scale, "fore_aft_m": args.fore_aft,
                   "lead_leg": "none" if args.fore_aft == 0 else args.lead_leg,
                   "policy_onnx": policy_path, "runtime_config": args.runtime_config,
                   "default_q_contract": "original_deploy_default"})
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=sorted(result))
        writer.writeheader(); writer.writerow(result)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
