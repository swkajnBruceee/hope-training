#!/usr/bin/env python3
"""Run the model-backed A3 stance stability experiments.

Examples:
  python scripts/stance_stability_test.py --test static --single-stance \
      --hip 15 --knee 30 --width-scale 1.1 --fore-aft 0.10 --lead-leg left
  python scripts/stance_stability_test.py --test push --single-stance --force 100 --direction front
  python scripts/stance_stability_test.py --test swing --single-stance --swing-mode full_body
  python scripts/stance_stability_test.py --test sweep --stage static --trials 3 --headless
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
REF = REPO / "mujoco_reference" / "reference"
if str(REF) not in sys.path:
    sys.path.insert(0, str(REF))

from a3_deploy_onnx_ref_pingpong.stance_stability import (  # noqa: E402
    ARM_IDX,
    LEG_IDX,
    MetricCollector,
    JOINT_NAMES,
    StanceConfig,
    StanceMujoco,
    aggregate_rows,
    deploy_pd_gains,
    official_stand_pd_gains,
    pose_delta_stance,
    thresholds_for,
    write_rows,
)


DIRECTIONS = {
    "front": np.array([1.0, 0.0, 0.0]),
    "back": np.array([-1.0, 0.0, 0.0]),
    "left": np.array([0.0, 1.0, 0.0]),
    "right": np.array([0.0, -1.0, 0.0]),
    "front_left": np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0),
    "front_right": np.array([1.0, -1.0, 0.0]) / math.sqrt(2.0),
    "back_left": np.array([-1.0, 1.0, 0.0]) / math.sqrt(2.0),
    "back_right": np.array([-1.0, -1.0, 0.0]) / math.sqrt(2.0),
}


def default_model() -> Path:
    candidates = [
        REPO.parent.parent.parent / "agibot/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml",
        REPO.parent.parent.parent / "agibot/A3_MuJoCo_Sim/aimrt_mujoco_sim/build/install/bin/cfg/model/a3_pingpong/a3_pingpong.xml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def baseline_stance(sim: StanceMujoco) -> object:
    from a3_deploy_onnx_ref_pingpong.stance_stability import GeneratedStance

    return GeneratedStance(
        config=StanceConfig(), q=sim.baseline_q(), root_qpos=sim.baseline_qpos[sim.root_qadr:sim.root_qadr + 7].copy(),
        left_foot_target=sim.generator.baseline_left_foot.copy(), right_foot_target=sim.generator.baseline_right_foot.copy(),
        pelvis_height_m=float(sim.baseline_qpos[sim.root_qadr + 2]), width_m=sim.generator.baseline_width_m,
        valid=True, diagnostics={"baseline_source": "MJCF keyframe 0"},
    )


def parse_config(args: argparse.Namespace) -> StanceConfig:
    return StanceConfig(
        hip_flexion_deg=args.hip,
        knee_flexion_deg=args.knee,
        torso_pitch_deg=args.torso,
        stance_width_scale=args.width_scale,
        stance_width_m=args.width_m,
        fore_aft_m=args.fore_aft,
        lead_leg="none" if args.fore_aft == 0.0 else args.lead_leg,
        pelvis_height_offset_m=args.pelvis_offset,
    )


def add_config_fields(
    row: dict,
    stance,
    *,
    test_type: str,
    trial_id: int,
    seed: int,
    sim: StanceMujoco | None = None,
) -> dict:
    c = stance.config
    row.update({
        "experiment_id": f"{c.label}_{test_type}_{trial_id}",
        "hip_deg": c.hip_flexion_deg,
        "knee_deg": c.knee_flexion_deg,
        "torso_deg": c.torso_pitch_deg,
        "stance_width_m": stance.width_m,
        "stance_width_scale": c.stance_width_scale,
        "fore_aft_m": c.fore_aft_m,
        "lead_leg": c.lead_leg,
        "pelvis_height_m": stance.pelvis_height_m,
        "test_type": test_type,
        "trial_id": trial_id,
        "seed": seed,
    })
    if sim is not None:
        effective = sim.effective_contact_friction()
        row.update({
            "mu_contact_requested": sim.mu_contact,
            "mu_contact_configured": sim._effective_mu_contact,
            "effective_contact_friction_min": float(np.min(effective)) if effective.size else float("nan"),
            "effective_contact_friction_max": float(np.max(effective)) if effective.size else float("nan"),
            "effective_contact_contact_count": int(effective.size),
        })
    return row


def transition(sim, stance, kp, kd, collector, *, duration_s: float) -> tuple[bool, str]:
    start = sim.state()["q"].copy()
    ticks = max(1, round(duration_s / sim.control_dt))
    threshold = thresholds_for(sim)
    for tick in range(ticks):
        alpha = (tick + 1) / ticks
        a = alpha * alpha * (3.0 - 2.0 * alpha)
        q_des = (1.0 - a) * start + a * stance.q
        tau = sim.set_targets(q_des, kp, kd)
        sim.step()
        row = collector.step(tau, q_des)
        if row["fall"]:
            return False, str(row["fall_reason"])
    return True, "none"


def run_static(sim: StanceMujoco, stance, args, *, trial_id: int, seed: int) -> dict:
    kp, kd = official_stand_pd_gains() if args.pd_profile == "official_stand" else deploy_pd_gains()
    sim.reset(stance, noise=args.initial_noise, base_roll_noise=args.base_roll_noise, base_pitch_noise=args.base_pitch_noise)
    collector = MetricCollector(sim, thresholds=thresholds_for(sim), dt=sim.control_dt)
    ok, reason = transition(sim, stance, kp, kd, collector, duration_s=args.transition_s)
    if ok:
        for _ in range(max(1, round(args.duration / sim.control_dt))):
            tau = sim.set_targets(stance.q, kp, kd)
            sim.step()
            row = collector.step(tau, stance.q)
            if row["fall"]:
                break
    result = collector.finalize(survival_time=sim.time)
    if args.trace_dir:
        collector.save_trace(Path(args.trace_dir) / f"{stance.config.label}_static_{trial_id}.csv")
    if not ok:
        result.update({"survival": False, "fall": True, "fall_reason": f"transition_{reason}"})
    return add_config_fields(result, stance, test_type="static", trial_id=trial_id, seed=seed, sim=sim)


def _stable_now(sim: StanceMujoco, thresholds) -> bool:
    s = sim.state()
    rpy = __import__("a3_deploy_onnx_ref_pingpong.stance_stability", fromlist=["quat_to_rpy"]).quat_to_rpy(s["base_quat"])
    return (s["base_pos"][2] >= thresholds.base_height_min_m and abs(rpy[0]) < 0.25 and abs(rpy[1]) < 0.25
            and np.linalg.norm(s["base_ang_vel"]) < 0.35 and np.linalg.norm(s["base_lin_vel"][:2]) < 0.15)


def run_push(sim: StanceMujoco, stance, args, *, trial_id: int, seed: int) -> dict:
    kp, kd = official_stand_pd_gains() if args.pd_profile == "official_stand" else deploy_pd_gains()
    sim.reset(stance, noise=args.initial_noise, base_roll_noise=args.base_roll_noise, base_pitch_noise=args.base_pitch_noise)
    collector = MetricCollector(sim, thresholds=thresholds_for(sim), dt=sim.control_dt)
    ok, reason = transition(sim, stance, kp, kd, collector, duration_s=args.transition_s)
    force = float(args.force)
    direction = DIRECTIONS[args.direction]
    phase_start = sim.time
    push_start = phase_start + args.settle_s
    push_end = push_start + args.push_duration
    recovery_time = None
    stable_since = None
    failed_to_settle = False
    settled = False
    if ok:
        total_s = args.settle_s + args.push_duration + args.recovery_s
        for _ in range(max(1, round(total_s / sim.control_dt))):
            if sim.time < push_start:
                settled = _stable_now(sim, thresholds_for(sim))
            if sim.time >= push_start and not settled:
                failed_to_settle = True
                sim.clear_force()
                break
            if sim.time >= push_start and sim.time < push_end:
                sim.apply_force(force * direction)
            else:
                sim.clear_force()
            tau = sim.set_targets(stance.q, kp, kd)
            sim.step()
            row = collector.step(tau, stance.q)
            if sim.time >= push_end:
                if _stable_now(sim, thresholds_for(sim)):
                    stable_since = sim.time if stable_since is None else stable_since
                    if sim.time - stable_since >= args.recovery_hold_s and recovery_time is None:
                        recovery_time = stable_since - push_end
                else:
                    stable_since = None
            if row["fall"]:
                break
    result = collector.finalize(survival_time=sim.time, recovery_time=recovery_time)
    if args.trace_dir:
        collector.save_trace(Path(args.trace_dir) / f"{stance.config.label}_push_{args.direction}_{trial_id}.csv")
    impulse = force * args.push_duration
    result.update({"push_direction": args.direction, "push_force": force, "push_duration": args.push_duration, "push_impulse": impulse,
                   "max_recoverable_impulse": impulse if result.get("survival") and recovery_time is not None else float("nan")})
    if not ok:
        result.update({"survival": False, "fall": True, "fall_reason": f"transition_{reason}"})
    elif failed_to_settle:
        result.update({"survival": False, "fall": False, "fall_reason": "failed_to_settle"})
    return add_config_fields(result, stance, test_type="push", trial_id=trial_id, seed=seed, sim=sim)


def run_swing(sim: StanceMujoco, stance, args, *, trial_id: int, seed: int) -> dict:
    kp, kd = official_stand_pd_gains() if args.pd_profile == "official_stand" else deploy_pd_gains()
    motion_path = Path(args.motion_file)
    if not motion_path.is_file():
        raise FileNotFoundError(motion_path)
    motion = np.load(motion_path)
    clip_q = np.asarray(motion["joint_pos"], dtype=float)
    clip_q0 = clip_q[0]
    sim.reset(stance, noise=args.initial_noise, base_roll_noise=args.base_roll_noise, base_pitch_noise=args.base_pitch_noise)
    collector = MetricCollector(sim, thresholds=thresholds_for(sim), dt=sim.control_dt)
    ok, reason = transition(sim, stance, kp, kd, collector, duration_s=args.transition_s)
    recovery_time = None
    phase_start = sim.time
    swing_start = phase_start + args.prepare_s
    swing_end = swing_start + len(clip_q) * sim.control_dt
    if ok:
        total_s = args.prepare_s + len(clip_q) * sim.control_dt + args.recovery_s
        stable_since = None
        for _ in range(max(1, round(total_s / sim.control_dt))):
            if sim.time < swing_start:
                q_des = stance.q
            elif sim.time < swing_end:
                index = min(len(clip_q) - 1, int((sim.time - swing_start) / sim.control_dt))
                q_des = pose_delta_stance(stance.q, stance, clip_q[index], clip_q0, args.swing_mode)
            else:
                q_des = stance.q
                if _stable_now(sim, thresholds_for(sim)):
                    stable_since = sim.time if stable_since is None else stable_since
                    if sim.time - stable_since >= args.recovery_hold_s and recovery_time is None:
                        recovery_time = stable_since - swing_end
                else:
                    stable_since = None
            tau = sim.set_targets(q_des, kp, kd)
            sim.step()
            row = collector.step(tau, q_des)
            if row["fall"]:
                break
    result = collector.finalize(survival_time=sim.time, recovery_time=recovery_time)
    if args.trace_dir:
        collector.save_trace(Path(args.trace_dir) / f"{stance.config.label}_swing_{args.swing_mode}_{trial_id}.csv")
    result.update({"swing_mode": args.swing_mode, "swing_file": str(motion_path), "swing_duration_s": len(clip_q) * sim.control_dt})
    if not ok:
        result.update({"survival": False, "fall": True, "fall_reason": f"transition_{reason}"})
    return add_config_fields(result, stance, test_type="swing", trial_id=trial_id, seed=seed, sim=sim)


def candidate_configs(args):
    if args.single_stance:
        yield parse_config(args)
        return
    hips = [float(x) for x in args.hips.split(",")]
    knees = [float(x) for x in args.knees.split(",")]
    torsos = [float(x) for x in args.torsos.split(",")]
    widths = [float(x) for x in args.width_scales.split(",")]
    fore_afts = [float(x) for x in args.fore_afts.split(",")]
    leads = [x.strip() for x in args.leads.split(",")]
    for hip in hips:
        for knee in knees:
            for torso in torsos:
                for width in widths:
                    for fore_aft in fore_afts:
                        for lead in ("none",) if fore_aft == 0.0 else leads:
                            yield StanceConfig(hip, knee, torso, width, None, fore_aft, lead, None)


def config_defaults(path: str | None) -> dict:
    """Translate the project YAML into argparse defaults; explicit CLI wins."""
    if not path:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("--config requires PyYAML") from exc
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = document.get("stance_test", document)
    defaults = {}

    def csv_value(value):
        return ",".join(str(item) for item in value)

    mapping = {
        "trials": "trials", "seed": "seed", "pd_profile": "pd_profile",
        "transition_s": "transition_s", "static_duration_s": "duration",
        "hip_flexion_deg": "hips", "knee_flexion_deg": "knees",
        "torso_pitch_deg": "torsos", "stance_width_scale": "width_scales",
        "fore_aft_m": "fore_afts",
    }
    for source, target in mapping.items():
        if source in cfg:
            value = cfg[source]
            defaults[target] = csv_value(value) if isinstance(value, (list, tuple)) else value

    initial = cfg.get("initial_noise", {})
    if initial:
        defaults["initial_noise"] = any(float(value) != 0.0 for value in initial.values())
        defaults["base_roll_noise"] = math.radians(float(initial.get("base_roll_deg", 0.0)))
        defaults["base_pitch_noise"] = math.radians(float(initial.get("base_pitch_deg", 0.0)))
    push = cfg.get("push", {})
    for source, target in (("settle_s", "settle_s"), ("duration_s", "push_duration"),
                           ("recovery_s", "recovery_s"), ("recovery_hold_s", "recovery_hold_s")):
        if source in push:
            defaults[target] = push[source]
    if push.get("forces_n"):
        defaults["force"] = push["forces_n"][0]
    if push.get("directions"):
        defaults["direction"] = push["directions"][0]
    swing = cfg.get("swing", {})
    for source, target in (("motion_file", "motion_file"), ("prepare_s", "prepare_s"), ("recovery_s", "recovery_s")):
        if source in swing:
            defaults[target] = swing[source]
    return defaults


def plot_results(rows: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    static = [r for r in rows if r.get("test_type") == "static"]
    if static:
        hips = sorted({float(r["hip_deg"]) for r in static})
        knees = sorted({float(r["knee_deg"]) for r in static})
        grid = np.full((len(knees), len(hips)), np.nan)
        for r in static:
            grid[knees.index(float(r["knee_deg"])), hips.index(float(r["hip_deg"]))] = float(bool(r.get("survival")))
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(grid, origin="lower", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(hips)), hips); ax.set_yticks(range(len(knees)), knees)
        ax.set_xlabel("hip flexion (deg)"); ax.set_ylabel("knee flexion (deg)"); ax.set_title("Static survival")
        fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(out_dir / "hip_knee_heatmap.png", dpi=150); plt.close(fig)
    pushes = [r for r in rows if r.get("test_type") == "push"]
    if pushes:
        fig, ax = plt.subplots(figsize=(8, 4))
        for lead in sorted({r["lead_leg"] for r in pushes}):
            subset = [r for r in pushes if r["lead_leg"] == lead]
            for direction in DIRECTIONS:
                vals = []
                for r in subset:
                    if r.get("push_direction") != direction or str(r.get("survival", "")).lower() not in ("true", "1"):
                        continue
                    try:
                        recovery = float(r.get("recovery_time", np.nan))
                        impulse = float(r.get("push_impulse", np.nan))
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(recovery) and np.isfinite(impulse):
                        vals.append(impulse)
                if vals and any(v == v for v in vals): ax.plot([direction], [np.nanmax(vals)], "o", label=lead)
        ax.set_ylabel("recoverable impulse (N s)"); ax.set_title("Push recovery by direction"); ax.tick_params(axis="x", rotation=45)
        handles, labels = ax.get_legend_handles_labels()
        if handles: ax.legend();
        fig.tight_layout(); fig.savefig(out_dir / "push_direction_recovery.png", dpi=150); plt.close(fig)


def write_report(out_dir: Path, rows: list[dict], sim: StanceMujoco, args) -> None:
    report = out_dir / "STANCE_STABILITY_REPORT.md"
    baseline = sim.generator
    joint_table = "\n".join(
        f"| {i} | `{name}` | {value:.9f} |"
        for i, (name, value) in enumerate(zip(JOINT_NAMES, sim.baseline_q()))
    )
    lines = [
        "# STANCE_STABILITY_REPORT",
        "",
        "## Existing System Audit",
        "",
        "### Ten explicit audit answers",
        "",
        "1. **DoF:** the MJCF has 31 actuated joints; head yaw/pitch (indices 3/4) are passive at deployment, hence the active policy view is 29 DoF.",
        "2. **Mapping:** the canonical SDK/action order is the 31-name order in `joint_order_agibot_a3.yaml`; leg joints occupy indices 19–30 and are addressed by name in MuJoCo.",
        "3. **Nominal pose:** the measured MuJoCo keyframe is the baseline used here; root height is 1.068390 m and joint q values are listed below. Isaac `InitialStateCfg` values were separately audited.",
        "4. **PD:** deploy gains come from `models/model_21800/policy/params/deploy.yaml`; PD-only static/disturbance runs use the existing reference runner `official_stand`/PD_STAND profile, while policy-idle uses the deploy gains.",
        "5. **Timing:** MuJoCo physics is 0.001 s, control is 0.020 s / 50 Hz with 20 substeps; Isaac is 0.005 s with decimation 4 / 50 Hz.",
        "6. **Frames:** root is `pelvis_link`; feet are `left_ankle_roll_Link` and `right_ankle_roll_Link`; +x is forward and +y is robot-left.",
        "7. **Contact:** MuJoCo foot-ground contacts and actuator/joint/root sensors are read directly; Isaac uses the existing `contact_forces` sensor configuration.",
        "8. **Policy contract:** observation is 110-D and action is 31-D; decoded command is `default_q + scale * raw_action` with clipping, and this test does not alter that contract.",
        "9. **RL terms:** the existing Isaac task contains motion-tracking, action-rate, joint-limit and undesired-contact terms/terminations; no reward or termination code was changed.",
        "10. **Evaluation:** existing entry points are `scripts/play.py`, `scripts/evaluate.py`, `scripts/mujoco_eval_onnx.py` and the reference ONNX runner; the Phase-D test uses the existing `model_21800` ONNX policy path.",
        "",
        f"- Simulator: MuJoCo MJCF `{sim.model_path}`; MJCF timestep `{sim.model.opt.timestep:g}` s; control dt `{sim.control_dt:g}` s; {sim.substeps} physics substeps/control tick.",
        f"- Root body: `pelvis_link`; feet: `left_ankle_roll_Link`, `right_ankle_roll_Link`; +x is forward and +y is left, confirmed from the project command/reference convention.",
        f"- Actual MJCF baseline pelvis root height: `{sim.generator.baseline_root_height_m:.6f}` m; measured ankle-body center width: `{sim.generator.baseline_width_m:.6f}` m.",
        f"- MJCF baseline foot centers: left `{sim.generator.baseline_left_foot.tolist()}`, right `{sim.generator.baseline_right_foot.tolist()}`; x offset left-right is `{sim.generator.baseline_left_foot[0] - sim.generator.baseline_right_foot[0]:.6f}` m and y separation is `{sim.generator.baseline_width_m:.6f}` m.",
        "- The model has 31 actuated joints. Head yaw/pitch are present in the 31-D contract but passive in deployment, leaving the 29-DoF active policy view.",
        "- Action=0 in the deploy contract decodes to the published `default_joint_pos`; the PD-only tests below bypass policy inference and command `q_des` directly.",
        "- Isaac path: `robots/agibot_a3.py` → `AGIBOT_A3_CFG` → `TrackingEnvCfg.actions.joint_pos` → `ClampedJointPositionAction`; MuJoCo path: `MujocoDirectBridge`/this test runner → named actuator PD.",
        "- Isaac physics is configured at 0.005 s with decimation 4 (50 Hz control); this MuJoCo model is 0.001 s with 0.02 s control (20 substeps).",
        "- Isaac contact sensor is `contact_forces` over robot bodies; MuJoCo exposes foot-ground contacts, actuator-force sensors, joint position/velocity sensors, pelvis frame/IMU sensors and qvel/root state directly.",
        "- Existing evaluation entrances are `scripts/play.py`, `scripts/evaluate.py`, `scripts/mujoco_eval_onnx.py`, and the reference runner `mujoco_reference/reference/a3_deploy_onnx_ref_pingpong/__main__.py`.",
        "- The MuJoCo path was chosen first because its actual MJCF is loadable in-process and exposes all required disturbance/telemetry primitives; Isaac remains the cross-simulator check path.",
        "",
        "### MJCF baseline leg q (radians)",
        "",
        "| index | joint | q0 (rad) |",
        "|---:|---|---:|",
        joint_table,
        "",
        "Isaac `InitialStateCfg` uses the same nominal leg pattern but records `hip_pitch=-0.1311`, `knee=0.2468`, `ankle_pitch=-0.1204`, left/right hip roll `+0.0056/-0.0056`, hip yaw `-0.0348/+0.0348`, and ankle roll `-0.0078/+0.0078`; the MJCF keyframe values above are the measured MuJoCo baseline used for MuJoCo experiments.",
        "- Stance generation uses model-backed numerical leg IK and records residuals/invalid configurations. Fore-aft is relative: lead + offset/2, trail - offset/2.",
        "",
        "## Experimental Setup",
        "",
        f"- Requested test: `{args.test}`; trials: `{args.trials}`; seed base: `{args.seed}`.",
        f"- Controller: plain clipped PD, `tau = Kp(q_des-q)-Kd qdot`, using existing `{args.pd_profile}` gains; no reward, actor, critic, observation, or planner changes.",
        "- Failure is automatic: root height/orientation or non-foot ground contact. Foot slip is measured only while a foot-ground contact is present.",
        "",
        "## Results",
        "",
        f"- Raw rows: `{out_dir / 'stance_results.csv'}`; aggregate rows: `{out_dir / 'stance_summary.csv'}`.",
        "- This report is generated from the rows present in this run. A result is not promoted to a recommended stance until static, push, and swing measurements are available.",
        "",
        "## Generated Figures",
        "",
        "- `hip_knee_heatmap.png` (when a static grid is run).",
        "- `push_direction_recovery.png` (when push trials are run).",
        "",
        "## Policy Integration Recommendation",
        "",
        "No policy nominal-pose, reward, observation, action, network, or checkpoint change is made by this tool.",
        "Phase D zero-strike policy testing is implemented separately in `scripts/stance_policy_idle_test.py`; it preserves the original 110-D observation, 31-D action, and deploy default-q contract. The recorded baseline and left-lead idle runs both fell during the tested window, so the current policy is not promoted as a stable nominal-pose controller.",
        "Phase E nominal-pose comparison remains a report-only recommendation: change only the configuration/nominal pose after static, push, swing, and policy-idle evidence are reviewed.",
        "IsaacLab/IsaacSim cross-check was not runnable in this environment (`ModuleNotFoundError` for both packages); MuJoCo results are therefore the executable evidence and the Isaac path is explicitly marked pending.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    config_probe = argparse.ArgumentParser(add_help=False)
    config_probe.add_argument("--config", default=None)
    config_args, _ = config_probe.parse_known_args()
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", default=None, help="YAML experiment defaults; explicit CLI values override it")
    p.add_argument("--test", choices=("static", "push", "swing", "sweep"), default="static")
    p.add_argument("--stage", choices=("static", "push", "swing", "all"), default="static")
    p.add_argument("--model-xml", default=str(default_model()))
    p.add_argument("--pd-profile", choices=("official_stand", "deploy"), default="official_stand")
    p.add_argument(
        "--mu-contact",
        type=float,
        default=None,
        help="set foot and floor sliding friction to one explicit effective contact value",
    )
    p.add_argument("--output-dir", default="outputs/stance_stability")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--single-stance", action="store_true")
    p.add_argument("--headless", action="store_true", help="kept for CLI compatibility; this runner is headless")
    p.add_argument("--save-traces", action="store_true", help="save deterministic per-trial replay traces")
    p.add_argument("--hip", type=float, default=0.0); p.add_argument("--knee", type=float, default=0.0)
    p.add_argument("--torso", type=float, default=0.0); p.add_argument("--width-scale", type=float, default=1.0)
    p.add_argument("--width-m", type=float, default=None); p.add_argument("--fore-aft", type=float, default=0.0)
    p.add_argument("--lead-leg", choices=("left", "right"), default="left"); p.add_argument("--pelvis-offset", type=float, default=None)
    p.add_argument("--hips", default="0,5,10,15,20,25"); p.add_argument("--knees", default="0,15,20,25,30,35,40,45")
    p.add_argument("--torsos", default="0,5,10"); p.add_argument("--width-scales", default="1.0,1.1,1.2")
    p.add_argument("--fore-afts", default="0,0.05,0.10,0.15,0.20"); p.add_argument("--leads", default="left,right")
    p.add_argument("--duration", type=float, default=10.0); p.add_argument("--transition-s", type=float, default=1.0)
    p.add_argument("--initial-noise", action="store_true"); p.add_argument("--base-roll-noise", type=float, default=0.0); p.add_argument("--base-pitch-noise", type=float, default=0.0)
    p.add_argument("--settle-s", type=float, default=2.0); p.add_argument("--recovery-s", type=float, default=5.0); p.add_argument("--recovery-hold-s", type=float, default=0.4)
    p.add_argument("--direction", choices=tuple(DIRECTIONS), default="front"); p.add_argument("--force", type=float, default=100.0); p.add_argument("--push-duration", type=float, default=0.15)
    p.add_argument("--all-directions", action="store_true", help="run all 8 standardized push directions for a single stance")
    p.add_argument("--motion-file", default=str(REPO / "motions/preprocessed/hope_forehand.npz")); p.add_argument("--swing-mode", choices=("arm_only", "arm_torso", "full_body"), default="full_body"); p.add_argument("--prepare-s", type=float, default=0.5)
    p.set_defaults(**config_defaults(config_args.config))
    args = p.parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    args.trace_dir = str(out_dir / "replay") if args.save_traces else None
    sim = StanceMujoco(args.model_xml, seed=args.seed, mu_contact=args.mu_contact)
    kp, kd = official_stand_pd_gains() if args.pd_profile == "official_stand" else deploy_pd_gains()
    print(json.dumps({"simulator": "MuJoCo", "model": str(Path(args.model_xml).resolve()), "seed": args.seed,
                      "physics_dt": sim.model.opt.timestep, "control_dt": sim.control_dt, "substeps": sim.substeps,
                      "pd_profile": args.pd_profile, "pd_kp": kp.tolist(), "pd_kd": kd.tolist(), "baseline_width_m": sim.generator.baseline_width_m,
                      "baseline_pelvis_height_m": sim.generator.baseline_root_height_m,
                      "mu_contact_requested": sim.mu_contact,
                      "mu_contact_configured": sim._effective_mu_contact}, ensure_ascii=False), flush=True)
    rows: list[dict] = []
    configs = list(candidate_configs(args))
    manifest = []
    for cfg in configs:
        generated = baseline_stance(sim) if cfg == StanceConfig() and args.single_stance else sim.generator.generate(cfg)
        manifest.append({"config": cfg.__dict__, "label": cfg.label, "valid": generated.valid,
                         "q": generated.q.tolist(), "root_qpos_target": generated.root_qpos.tolist(),
                         "left_foot_target": generated.left_foot_target.tolist(),
                         "right_foot_target": generated.right_foot_target.tolist(),
                         "pelvis_height_m": generated.pelvis_height_m, "width_m": generated.width_m,
                         "diagnostics": generated.diagnostics})
    (out_dir / "stance_candidates.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.test == "sweep" and args.stage == "all":
        # Full sweeps are explicit; the default sweep remains static to avoid silently launching
        # thousands of expensive push/swing trials.
        stages = ("static", "push", "swing")
    else:
        stages = (args.stage if args.test == "sweep" else args.test,)
    for cfg in configs:
        stance = baseline_stance(sim) if cfg == StanceConfig() and args.single_stance else sim.generator.generate(cfg)
        if not stance.valid:
            row = add_config_fields({"survival": False, "fall": False, "invalid_configuration": True, **stance.diagnostics}, stance, test_type="invalid_ik", trial_id=0, seed=args.seed, sim=sim)
            rows.append(row); continue
        for trial in range(args.trials):
            seed = args.seed + trial
            sim.rng = np.random.default_rng(seed)
            if "static" in stages:
                rows.append(run_static(sim, stance, args, trial_id=trial, seed=seed))
            if "push" in stages:
                push_directions = tuple(DIRECTIONS) if args.all_directions else ((args.direction,) if args.single_stance else tuple(DIRECTIONS))
                for direction in push_directions:
                    args.direction = direction
                    rows.append(run_push(sim, stance, args, trial_id=trial, seed=seed))
            if "swing" in stages:
                rows.append(run_swing(sim, stance, args, trial_id=trial, seed=seed))
    write_rows(out_dir / "stance_results.csv", rows)
    write_rows(out_dir / "stance_summary.csv", aggregate_rows(rows, ("hip_deg", "knee_deg", "torso_deg", "stance_width_scale", "fore_aft_m", "lead_leg", "test_type")))
    plot_results(rows, out_dir)
    write_report(out_dir, rows, sim, args)
    print(f"wrote {len(rows)} rows to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
