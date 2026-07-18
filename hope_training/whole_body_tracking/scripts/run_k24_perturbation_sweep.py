"""Run paired perturbation sweeps for the K24 native-strike benchmark.

This script builds a fixed perturbation bank per (level, seed), then evaluates:

1. zero residual
2. learned residual

on the exact same bank. It writes a compact JSON report with pass rates,
forehand/backhand splits, and rescue/harm accounting.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    "sample_motions/p2_fixed_competition_global_funnel_tracking_union55_v2_curated/"
    "k24_posture_balanced_v2/manifest.json"
)
DEFAULT_CHECKPOINT = (
    "logs/rsl_rl/agibot_a3_native_strike_manifest/"
    "2026-07-11_18-58-37_residual_probe_k24_v2_200/model_199.pt"
)


LEVELS = {
    "mild": {
        "pose_range": {"roll": [-0.03, 0.03], "pitch": [-0.03, 0.03], "yaw": [-0.05, 0.05]},
        "velocity_range": {
            "x": [-0.05, 0.05],
            "y": [-0.05, 0.05],
            "z": [-0.02, 0.02],
            "roll": [-0.1, 0.1],
            "pitch": [-0.1, 0.1],
            "yaw": [-0.1, 0.1],
        },
        "joint_position_range": [-0.02, 0.02],
    },
    "medium": {
        "pose_range": {"roll": [-0.06, 0.06], "pitch": [-0.06, 0.06], "yaw": [-0.10, 0.10]},
        "velocity_range": {
            "x": [-0.10, 0.10],
            "y": [-0.10, 0.10],
            "z": [-0.04, 0.04],
            "roll": [-0.2, 0.2],
            "pitch": [-0.2, 0.2],
            "yaw": [-0.2, 0.2],
        },
        "joint_position_range": [-0.04, 0.04],
    },
    "strong": {
        "pose_range": {"roll": [-0.09, 0.09], "pitch": [-0.09, 0.09], "yaw": [-0.15, 0.15]},
        "velocity_range": {
            "x": [-0.15, 0.15],
            "y": [-0.15, 0.15],
            "z": [-0.06, 0.06],
            "roll": [-0.3, 0.3],
            "pitch": [-0.3, 0.3],
            "yaw": [-0.3, 0.3],
        },
        "joint_position_range": [-0.06, 0.06],
    },
}


@dataclass
class MotionEval:
    stroke: str
    episode_id: str
    hit_pass: bool
    posture_pass: bool
    whole_cycle_pass: bool
    pos: float
    vel: float
    normal: float
    pelvis_ref_deg: float
    torso_ref_deg: float
    arm_near_limit_frac: float


def _python_executable() -> str:
    """Use the IsaacLab environment even when this driver is started by python3."""
    configured = os.environ.get("HOPE_ISAAC_PY")
    if configured and Path(configured).exists():
        return configured
    for candidate in (
        "/workspace/anaconda3/envs/hope/bin/python",
        shutil.which("python"),
        sys.executable,
    ):
        if candidate and Path(candidate).exists():
            return str(candidate)
    return sys.executable


def _run_eval(script: str, extra_args: list[str]) -> str:
    cmd = [_python_executable(), str(ROOT / "scripts" / script)] + extra_args
    child_env = os.environ.copy()
    # Headless Isaac subprocesses must not try to read the NVIDIA EULA prompt
    # from stdin. This is a local evaluation runner, not a license bypass: the
    # project environment already records acceptance through sitecustomize.
    child_env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    proc = subprocess.run(cmd, cwd=ROOT, env=child_env, capture_output=True, text=True)
    # Isaac Sim may release Kit/GPU resources a moment after the child exits.
    # A short gap prevents the next headless launch from failing in Kit startup.
    time.sleep(5.0)
    if proc.returncode == -6 and "bad_optional_access" in proc.stderr:
        # This is a Kit startup race, not an evaluation result. Retry once after
        # a longer release window; any second failure remains fatal.
        time.sleep(15.0)
        proc = subprocess.run(cmd, cwd=ROOT, env=child_env, capture_output=True, text=True)
        time.sleep(5.0)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{script} failed with code {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc.stdout


def _fmt_scalar(x: float) -> str:
    return f"{x:.6f}".rstrip("0").rstrip(".")


def _fmt_list(xs: list[float]) -> str:
    return "[" + ",".join(_fmt_scalar(x) for x in xs) + "]"


def _fmt_dict(d: dict[str, list[float]]) -> str:
    return "{" + ",".join(f"{k}:{_fmt_list(v)}" for k, v in d.items()) + "}"


def _hydra_args(
    manifest: str,
    seed: int,
    level_cfg: dict,
    bank_path: str | None,
    checkpoint: str | None,
    subset_size: int,
    num_envs: int,
    actuator_profile: str,
) -> list[str]:
    args = [
        "headless=true",
        f"+seed={seed}",
        f"num_envs={num_envs}",
        "task=HOPEA3NativeStrikeManifest",
        "algo=ppo",
        f"motion_manifest={manifest}",
        f"manifest_subset_size={subset_size}",
        "manifest_frame_z_offset=0.76",
        f"+task.native_actuator_profile={actuator_profile}",
    ]
    if checkpoint is not None:
        args.append(f"checkpoint={checkpoint}")
    if bank_path is None:
        args.append(f"+task.motion.pose_range={_fmt_dict(level_cfg['pose_range'])}")
        args.append(f"+task.motion.velocity_range={_fmt_dict(level_cfg['velocity_range'])}")
        args.append(f"+task.motion.joint_position_range={_fmt_list(level_cfg['joint_position_range'])}")
    else:
        args.append(f"+perturb_bank={bank_path}")
    return args


def _parse_rows(stdout: str, zero: bool) -> dict[str, MotionEval]:
    out: dict[str, MotionEval] = {}
    for raw in stdout.splitlines():
        parts = list(csv.reader([raw]))[0]
        if len(parts) < 18 or not parts[0].isdigit():
            continue
        stroke = parts[1]
        episode_id = parts[2]
        hit_pass_idx = 6
        pelvis_ref_idx = 13 if zero else 13
        torso_ref_idx = 14 if zero else 14
        arm_near_idx = 16 if zero else 16
        posture_idx = 18 if zero else 18
        hit_pass = bool(int(parts[hit_pass_idx]))
        posture_pass = bool(int(parts[posture_idx]))
        # Both evaluators keep the full robot-level whole-cycle gate in the
        # final CSV column. Do not reconstruct it from hit/posture only: that
        # would hide wrist, soft-limit, and robot-posture failures.
        whole_cycle_pass = bool(int(parts[35])) if len(parts) > 35 else (hit_pass and posture_pass)
        out[episode_id] = MotionEval(
            stroke=stroke,
            episode_id=episode_id,
            hit_pass=hit_pass,
            posture_pass=posture_pass,
            whole_cycle_pass=whole_cycle_pass,
            pos=float(parts[3]),
            vel=float(parts[4]),
            normal=float(parts[5]),
            pelvis_ref_deg=float(parts[pelvis_ref_idx]),
            torso_ref_deg=float(parts[torso_ref_idx]),
            arm_near_limit_frac=float(parts[arm_near_idx]),
        )
    return out


def _summarize_group(rows: list[MotionEval]) -> dict:
    if not rows:
        return {}
    return {
        "n": len(rows),
        "hit_composite_pass_rate": sum(r.hit_pass for r in rows) / len(rows),
        "posture_pass_rate": sum(r.posture_pass for r in rows) / len(rows),
        "whole_cycle_pass_rate": sum(r.whole_cycle_pass for r in rows) / len(rows),
        "pos_mean": sum(r.pos for r in rows) / len(rows),
        "vel_mean": sum(r.vel for r in rows) / len(rows),
        "normal_mean": sum(r.normal for r in rows) / len(rows),
        "torso_margin_mean": sum(20.0 - r.torso_ref_deg for r in rows) / len(rows),
        "pelvis_margin_mean": sum(15.0 - r.pelvis_ref_deg for r in rows) / len(rows),
        "arm_margin_mean": sum(0.10 - r.arm_near_limit_frac for r in rows) / len(rows),
        "worst": max(rows, key=lambda r: (r.pos, r.vel, r.normal)).episode_id,
    }


def _summarize(rows: dict[str, MotionEval]) -> dict:
    motions = list(rows.values())
    return {
        "overall": _summarize_group(motions),
        "forehand": _summarize_group([r for r in motions if r.stroke == "forehand"]),
        "backhand": _summarize_group([r for r in motions if r.stroke == "backhand"]),
    }


def _pair_compare(zero_rows: dict[str, MotionEval], learned_rows: dict[str, MotionEval]) -> dict:
    rescue = []
    harm = []
    both_fail = []
    both_pass = []
    for episode_id, z in zero_rows.items():
        l = learned_rows[episode_id]
        if not z.whole_cycle_pass and l.whole_cycle_pass:
            rescue.append(episode_id)
        elif z.whole_cycle_pass and not l.whole_cycle_pass:
            harm.append(episode_id)
        elif not z.whole_cycle_pass and not l.whole_cycle_pass:
            both_fail.append(episode_id)
        else:
            both_pass.append(episode_id)
    return {
        "rescue_count": len(rescue),
        "harm_count": len(harm),
        "net_rescue": len(rescue) - len(harm),
        "rescue": rescue,
        "harm": harm,
        "both_fail": both_fail,
        "both_pass_count": len(both_pass),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--levels", default="mild,medium,strong")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--output", default=str(ROOT / "docs" / "eval_reports" / "k24_perturbation_sweep.json"))
    parser.add_argument("--subset-size", type=int, default=24)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--residual-scale", type=float, default=0.15)
    parser.add_argument("--raw-clip", type=float, default=0.25)
    parser.add_argument(
        "--actuator-profile",
        default="official_pd",
        choices=("official_pd", "calibrated"),
        help="Isaac actuator profile used for both zero and learned evaluations",
    )
    parser.add_argument(
        "--waist-scale-multiplier",
        type=float,
        default=1.0,
        help="multiply waist residual authority during learned evaluation",
    )
    args = parser.parse_args()

    levels = [x.strip() for x in args.levels.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    num_envs = args.num_envs or args.subset_size

    results = {
        "manifest": args.manifest,
        "checkpoint": args.checkpoint,
        "actuator_profile": args.actuator_profile,
        "levels": levels,
        "seeds": seeds,
        "generated_at": datetime.now().isoformat(),
        "runs": [],
    }

    with tempfile.TemporaryDirectory(prefix="hope_k24_banks_") as tmpdir:
        tmp = Path(tmpdir)
        for level in levels:
            if level not in LEVELS:
                raise ValueError(f"unknown level: {level}")
            for seed in seeds:
                bank_path = tmp / f"{level}_seed{seed}.json"
                zero_build_args = _hydra_args(
                    args.manifest, seed, LEVELS[level], None, None,
                    args.subset_size, num_envs, args.actuator_profile
                )
                zero_build_args.append(f"+write_perturb_bank={bank_path}")
                zero_stdout = _run_eval("eval_manifest_zero_action.py", zero_build_args)

                zero_bank_args = _hydra_args(
                    args.manifest, seed, LEVELS[level], str(bank_path), None,
                    args.subset_size, num_envs, args.actuator_profile
                )
                zero_bank_stdout = _run_eval("eval_manifest_zero_action.py", zero_bank_args)

                learned_args = _hydra_args(
                    args.manifest, seed, LEVELS[level], str(bank_path), args.checkpoint,
                    args.subset_size, num_envs, args.actuator_profile
                )
                learned_args.extend([
                    f"task.actions.native_residual_scale={args.residual_scale}",
                    f"task.actions.raw_clip={args.raw_clip}",
                ])
                if args.waist_scale_multiplier != 1.0:
                    learned_args.append(
                        "+task.actions.native_joint_scale_multipliers="
                        "{waist_yaw_joint: "
                        f"{args.waist_scale_multiplier}, "
                        "waist_roll_joint: "
                        f"{args.waist_scale_multiplier}, "
                        "waist_pitch_joint: "
                        f"{args.waist_scale_multiplier}}}"
                    )
                learned_stdout = _run_eval("eval_manifest_policy.py", learned_args)

                zero_rows = _parse_rows(zero_bank_stdout, zero=True)
                learned_rows = _parse_rows(learned_stdout, zero=False)

                results["runs"].append(
                    {
                        "level": level,
                        "seed": seed,
                        "bank_path": str(bank_path),
                        "zero": _summarize(zero_rows),
                        "learned": _summarize(learned_rows),
                        "pairing": _pair_compare(zero_rows, learned_rows),
                    }
                )
                print(
                    f"[sweep] {level} seed={seed}: zero whole={results['runs'][-1]['zero']['overall']['whole_cycle_pass_rate']:.3f} "
                    f"learned whole={results['runs'][-1]['learned']['overall']['whole_cycle_pass_rate']:.3f} "
                    f"rescue={results['runs'][-1]['pairing']['rescue_count']} harm={results['runs'][-1]['pairing']['harm_count']}",
                    flush=True,
                )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"[sweep] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
