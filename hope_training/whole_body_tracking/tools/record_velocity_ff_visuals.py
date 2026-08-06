#!/usr/bin/env python3
"""Record deterministic visual reviews for the selected velocity contract."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess


CHECKPOINT = (
    "logs/rsl_rl/agibot_a3_joint_coordinator_v2_20260726/"
    "2026-07-26_18-10-06_joint_coordinator_v2_256x1000/model_900.pt"
)
VIDEO_RE = re.compile(r"\[INFO\] wrote video -> (?P<path>.+\.mp4)$", re.MULTILINE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--motions", default="0,1,4,5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video-length", type=int, default=150)
    parser.add_argument("--output", default="eval_outputs/velocity_feedforward/v2_b_shoulders_beta075_visual_20260726.json")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    workspace = Path.cwd()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = (workspace / checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = Path(args.output)
    if not output.is_absolute():
        output = (workspace / output).resolve()
    motion_ids = tuple(int(value.strip()) for value in args.motions.split(",") if value.strip())
    if not motion_ids:
        raise ValueError("at least one motion id is required")
    python = os.environ.get("HOPE_ISAAC_PYTHON")
    if not python:
        raise RuntimeError("source setup_train_env.sh before running this tool")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.environ.get("HOPE_WBT_PYTHONPATH", environment.get("PYTHONPATH", ""))

    records = []
    for motion_id in motion_ids:
        video_name = f"ff_b_shoulders_beta075_motion{motion_id:02d}.mp4"
        command = [
            python,
            "scripts/play.py",
            "task=HOPEA3JointCoordinatorV2",
            "algo=ppo_joint_coordinator",
            "headless=true",
            "video=true",
            f"video_length={args.video_length}",
            f"max_steps={args.video_length}",
            f"video_name={video_name}",
            f"device={args.device}",
            "num_envs=1",
            f"seed={args.seed}",
            f"checkpoint={checkpoint}",
            f"motion_id={motion_id}",
            "++task.actions.joint_velocity_feedforward_mode=task_phase",
            "++task.actions.joint_velocity_feedforward_beta=0.75",
            "++task.actions.joint_velocity_feedforward_joint_names=[right_shoulder_pitch_joint,right_shoulder_yaw_joint]",
            "++task.actions.joint_velocity_feedforward_post_hit_decay_steps=6",
        ]
        print(f"[visual] recording motion={motion_id}", flush=True)
        result = subprocess.run(command, cwd=workspace, env=environment, text=True, capture_output=True)
        combined = result.stdout + "\n" + result.stderr
        if result.returncode != 0:
            raise RuntimeError("\n".join(combined.splitlines()[-100:]))
        video_matches = VIDEO_RE.findall(combined)
        if not video_matches:
            raise RuntimeError(f"motion={motion_id}: replay exited without an MP4 path")
        video_path = Path(video_matches[-1])
        if not video_path.is_file() or video_path.stat().st_size == 0:
            raise RuntimeError(f"motion={motion_id}: invalid video {video_path}")
        stability_match = re.search(
            r"\[INFO\] stability metrics: (?P<metrics>.+)$", combined, re.MULTILINE
        )
        records.append(
            {
                "motion_id": motion_id,
                "video": str(video_path),
                "video_size_bytes": video_path.stat().st_size,
                "stability_metrics": stability_match.group("metrics") if stability_match else None,
                "stdout_tail": combined.splitlines()[-30:],
            }
        )
        print(f"[visual] wrote {video_path}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "purpose": "visual-only review; not a substitute for paired rollout metrics",
                "checkpoint": str(checkpoint),
                "seed": args.seed,
                "candidate_contract": {
                    "position_lead_steps": 12,
                    "velocity_phase": "task_phase",
                    "velocity_beta": 0.75,
                    "velocity_joints": ["right_shoulder_pitch_joint", "right_shoulder_yaw_joint"],
                    "post_hit_decay_steps": 6,
                },
                "records": records,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[visual] wrote manifest {output}", flush=True)


if __name__ == "__main__":
    main()
