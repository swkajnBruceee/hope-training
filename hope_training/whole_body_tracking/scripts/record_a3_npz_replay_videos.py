"""Record full A3 reference replays from converted motion NPZ files."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--jobs", type=Path, required=True, help="JSON list of NPZ replay jobs.")
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--episode-id", type=str, default=None)
parser.add_argument("--video-fps", type=int, default=60)
parser.add_argument("--hold-frames", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from training.robots.agibot_a3 import AGIBOT_A3_CFG


@configclass
class ReplaySceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(size=(20.0, 20.0)),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=900.0),
    )
    robot: ArticulationCfg = AGIBOT_A3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    camera = CameraCfg(
        prim_path="/World/ReplayCamera",
        height=720,
        width=960,
        data_types=["rgb"],
        update_period=0,
        offset=CameraCfg.OffsetCfg(
            pos=(5.4, 2.8, 2.1),
            # World convention: camera +X points at the robot and +Z stays up.
            rot=(-0.4532042, -0.1226960, -0.0631416, 0.8806616),
            convention="world",
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 1000.0),
        ),
    )


@dataclass(frozen=True)
class ReplayJob:
    episode_id: str
    motion_file: Path
    output_file: Path


def _load_jobs(path: Path) -> list[ReplayJob]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("jobs", payload) if isinstance(payload, dict) else payload
    jobs = []
    for item in raw:
        jobs.append(
            ReplayJob(
                episode_id=str(item["episode_id"]),
                motion_file=Path(str(item.get("motion_file", item.get("input_file")))).expanduser(),
                output_file=Path(str(item.get("video_file", item.get("output_file")))).expanduser(),
            )
        )
    return jobs


def _look_at_camera(sim: SimulationContext) -> None:
    sim.set_camera_view(eye=[5.4, 2.8, 2.1], target=[3.15, -0.35, 1.0])


def _read_motion(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"{path}: missing {missing}")
        return {key: np.asarray(data[key]) for key in data.files}


def _write_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    import imageio.v2 as imageio

    valid = [np.asarray(frame) for frame in frames if frame is not None and np.asarray(frame).size]
    if not valid:
        raise RuntimeError(f"no valid camera frames captured for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, valid, fps=fps, codec="libx264", macro_block_size=1)


def main() -> None:
    jobs = _load_jobs(args_cli.jobs)
    if args_cli.episode_id is not None:
        jobs = [job for job in jobs if job.episode_id == args_cli.episode_id]
    if args_cli.limit is not None:
        jobs = jobs[: max(0, int(args_cli.limit))]
    if not jobs:
        raise ValueError("no replay jobs")
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / 120.0
    sim_cfg.gravity_enabled = False
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(ReplaySceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    scene.reset()
    _look_at_camera(sim)
    robot: Articulation = scene["robot"]
    camera = scene["camera"]

    results = []
    for job in jobs:
        motion = _read_motion(job.motion_file)
        frames: list[np.ndarray] = []
        root_pos = torch.zeros((1, 13), device=sim.device, dtype=torch.float32)
        joint_pos = torch.from_numpy(motion["joint_pos"]).to(sim.device, dtype=torch.float32)
        joint_vel = torch.from_numpy(motion["joint_vel"]).to(sim.device, dtype=torch.float32)
        body_pos = torch.from_numpy(motion["body_pos_w"]).to(sim.device, dtype=torch.float32)
        body_quat = torch.from_numpy(motion["body_quat_w"]).to(sim.device, dtype=torch.float32)
        body_lin = torch.from_numpy(motion["body_lin_vel_w"]).to(sim.device, dtype=torch.float32)
        body_ang = torch.from_numpy(motion["body_ang_vel_w"]).to(sim.device, dtype=torch.float32)
        root_pos[:, :3] = body_pos[0, 0]
        root_pos[:, 3:7] = body_quat[0, 0]
        root_pos[:, 7:10] = body_lin[0, 0]
        root_pos[:, 10:13] = body_ang[0, 0]

        for _ in range(max(0, int(args_cli.hold_frames))):
            robot.write_root_state_to_sim(root_pos)
            robot.write_joint_state_to_sim(joint_pos[0:1], joint_vel[0:1])
            scene.write_data_to_sim()
            sim.render()
            scene.update(sim_cfg.dt)
            frames.append(camera.data.output["rgb"][0].detach().cpu().numpy())

        for frame_idx in range(joint_pos.shape[0]):
            root_pos[:, :3] = body_pos[frame_idx, 0]
            root_pos[:, 3:7] = body_quat[frame_idx, 0]
            root_pos[:, 7:10] = body_lin[frame_idx, 0]
            root_pos[:, 10:13] = body_ang[frame_idx, 0]
            robot.write_root_state_to_sim(root_pos)
            robot.write_joint_state_to_sim(joint_pos[frame_idx : frame_idx + 1], joint_vel[frame_idx : frame_idx + 1])
            scene.write_data_to_sim()
            sim.render()
            scene.update(sim_cfg.dt)
            frames.append(camera.data.output["rgb"][0].detach().cpu().numpy())

        for _ in range(max(0, int(args_cli.hold_frames))):
            robot.write_root_state_to_sim(root_pos)
            robot.write_joint_state_to_sim(joint_pos[-1:], joint_vel[-1:])
            scene.write_data_to_sim()
            sim.render()
            scene.update(sim_cfg.dt)
            frames.append(camera.data.output["rgb"][0].detach().cpu().numpy())

        out = args_cli.output_dir / f"{job.episode_id}.mp4"
        _write_video(frames, out, int(args_cli.video_fps))
        results.append({"episode_id": job.episode_id, "video": str(out), "frames": len(frames), "fps": int(args_cli.video_fps)})
        print(f"[record] {job.episode_id}: {len(frames)} frames -> {out}", flush=True)

    (args_cli.output_dir / "video_manifest.json").write_text(
        json.dumps({"stage": "ttmd6_a3_replay_video_v0", "training_eligible": False, "videos": results}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[record] completed {len(results)} videos", flush=True)


if __name__ == "__main__":
    main()
    # Isaac's rendering extensions can keep background GPU workers alive after
    # the normal Python shutdown. This tool is run one motion per process, so
    # force the process boundary after the video has been flushed.
    os._exit(0)
