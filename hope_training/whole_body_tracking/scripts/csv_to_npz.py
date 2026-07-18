"""Replay a retargeted motion CSV in Isaac Lab and save it as a local NPZ file.

Example:

    python csv_to_npz.py --input_file LAFAN/dance1_subject2.csv --input_fps 30 --frame_range 122 722 \
        --output_file ./motions/dance1_subject2.npz --output_fps 50
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import os
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay motion from csv file and output to npz file.")
parser.add_argument("--input_file", type=str, default=None, help="The path to the input motion csv file.")
parser.add_argument(
    "--batch_jobs_json",
    type=str,
    default=None,
    help="JSON file containing a list of csv->npz jobs to process in one Isaac session.",
)
parser.add_argument("--input_fps", type=int, default=30, help="The fps of the input motion.")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help=(
        "frame range: START END (both inclusive). The frame index starts from 1. If not provided, all frames will be"
        " loaded."
    ),
)
parser.add_argument(
    "--output_name",
    type=str,
    default=None,
    help="Motion name for logs or optional WandB upload. Defaults to output file stem.",
)
parser.add_argument(
    "--output_file",
    type=str,
    default=None,
    help="Local NPZ output path. Defaults to <output_name>.npz in the current directory.",
)
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")
parser.add_argument("--upload_wandb", action="store_true", help="Also upload the saved NPZ to a WandB registry.")
parser.add_argument("--wandb_registry", type=str, default="motions", help="WandB artifact/registry type for upload.")
parser.add_argument(
    "--robot",
    type=str,
    default="g1",
    choices=["g1", "agibot_a3"],
    help="Which robot model to replay the motion on (selects the articulation cfg + DOF column order).",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
if bool(args_cli.input_file) == bool(args_cli.batch_jobs_json):
    parser.error("exactly one of --input_file or --batch_jobs_json is required")
if args_cli.batch_jobs_json is None:
    if args_cli.output_file is None and args_cli.output_name is None:
        parser.error("at least one of --output_file or --output_name is required")
    if args_cli.output_file is None:
        args_cli.output_file = f"{args_cli.output_name}.npz"
    if args_cli.output_name is None:
        args_cli.output_name = Path(args_cli.output_file).stem

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

##
# Pre-defined configs
##
from training.robots.agibot_a3 import AGIBOT_A3_CFG, AGIBOT_A3_JOINT_NAMES
from training.robots.g1 import G1_CYLINDER_CFG

# G1 retargeting CSV DOF-column order (29 joints).
G1_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# robot name -> (articulation cfg, retargeting CSV DOF-column order)
_ROBOTS = {
    "g1": (G1_CYLINDER_CFG, G1_JOINT_NAMES),
    "agibot_a3": (AGIBOT_A3_CFG, AGIBOT_A3_JOINT_NAMES),
}
_ROBOT_CFG, _JOINT_NAMES = _ROBOTS[args_cli.robot]


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    # Minimal local-only scene for FK replay. Ground/Nucleus assets are not
    # required because this script never steps physics.
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )

    # articulation (selected by --robot)
    robot: ArticulationCfg = _ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@dataclass(frozen=True)
class MotionJob:
    input_file: str
    output_file: str
    output_name: str
    input_fps: int
    output_fps: int
    frame_range: tuple[int, int] | None


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """Loads the motion from the csv file."""
        if self.frame_range is None:
            motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
        else:
            motion = torch.from_numpy(
                np.loadtxt(
                    self.motion_file,
                    delimiter=",",
                    skiprows=self.frame_range[0] - 1,
                    max_rows=self.frame_range[1] - self.frame_range[0] + 1,
                )
            )
        motion = motion.to(torch.float32).to(self.device)
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7]
        self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]  # convert to wxyz
        self.motion_dof_poss_input = motion[:, 7:]

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"Motion loaded ({self.motion_file}), duration: {self.duration} sec, frames: {self.input_frames}")

    def _interpolate_motion(self):
        """Interpolates the motion to the output fps."""
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, output frames:"
            f" {self.output_frames}, output fps: {self.output_fps}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between two tensors."""
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Spherical linear interpolation between two quaternions."""
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> torch.Tensor:
        """Computes the frame blend for the motion."""
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.clamp(index_0 + 1, max=self.input_frames - 1)
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        """Computes the velocities of the motion."""
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """Computes the derivative of a sequence of SO3 rotations.

        Args:
            rotations: shape (B, 4).
            dt: time step.
        Returns:
            shape (B, 3).
        """
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)  # repeat first and last sample
        return omega

    def get_next_state(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Gets the next state of the motion."""
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def _parse_frame_range(value: object) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"invalid frame_range: {value!r}")
    start, end = int(value[0]), int(value[1])
    return start, end


def _load_jobs() -> list[MotionJob]:
    if args_cli.batch_jobs_json is None:
        return [
            MotionJob(
                input_file=str(args_cli.input_file),
                output_file=str(args_cli.output_file),
                output_name=str(args_cli.output_name),
                input_fps=int(args_cli.input_fps),
                output_fps=int(args_cli.output_fps),
                frame_range=tuple(args_cli.frame_range) if args_cli.frame_range is not None else None,
            )
        ]

    payload = json.loads(Path(args_cli.batch_jobs_json).read_text(encoding="utf-8"))
    jobs_raw = payload["jobs"] if isinstance(payload, dict) else payload
    if not isinstance(jobs_raw, list) or not jobs_raw:
        raise ValueError("batch_jobs_json must contain a non-empty jobs list")
    jobs: list[MotionJob] = []
    for idx, item in enumerate(jobs_raw):
        if not isinstance(item, dict):
            raise ValueError(f"job[{idx}] must be an object")
        input_file = str(item["input_file"])
        output_file = item.get("output_file")
        output_name = item.get("output_name")
        if output_file is None and output_name is None:
            raise ValueError(f"job[{idx}] requires output_file or output_name")
        if output_file is None:
            output_file = f"{output_name}.npz"
        if output_name is None:
            output_name = Path(str(output_file)).stem
        job_output_fps = int(item.get("output_fps", args_cli.output_fps))
        if job_output_fps != int(args_cli.output_fps):
            raise ValueError(
                f"job[{idx}] output_fps={job_output_fps} does not match launcher output_fps={args_cli.output_fps}"
            )
        jobs.append(
            MotionJob(
                input_file=input_file,
                output_file=str(output_file),
                output_name=str(output_name),
                input_fps=int(item.get("input_fps", args_cli.input_fps)),
                output_fps=job_output_fps,
                frame_range=_parse_frame_range(item.get("frame_range", args_cli.frame_range)),
            )
        )
    return jobs


def _save_motion_log(log: dict[str, list | np.ndarray], output_path: str | Path) -> Path:
    for key in (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ):
        log[key] = np.stack(log[key], axis=0)
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **log)
    print(f"[INFO]: Motion saved locally: {path}")
    return path


def _maybe_upload_wandb(output_path: Path, output_name: str) -> None:
    if not args_cli.upload_wandb:
        return
    import wandb

    run = wandb.init(project="csv_to_npz", name=output_name)
    print(f"[INFO]: Logging motion to wandb: {output_name}")
    registry = args_cli.wandb_registry
    logged_artifact = run.log_artifact(artifact_or_path=str(output_path), name=output_name, type=registry)
    run.link_artifact(artifact=logged_artifact, target_path=f"wandb-registry-{registry}/{output_name}")
    run.finish()
    print(f"[INFO]: Motion saved to wandb registry: {registry}/{output_name}")


def _run_motion_job(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    robot,
    robot_joint_indexes: torch.Tensor,
    job: MotionJob,
) -> Path:
    """Replay one motion job inside an already-initialized Isaac session."""
    motion = MotionLoader(
        motion_file=job.input_file,
        input_fps=job.input_fps,
        output_fps=job.output_fps,
        device=sim.device,
        frame_range=job.frame_range,
    )

    # ------- data logger -------------------------------------------------------
    log = {
        "fps": np.asarray([job.output_fps], dtype=np.int64),
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    file_saved = False
    # --------------------------------------------------------------------------

    # Simulation loop
    while simulation_app.is_running():
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        # set root state
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        # set joint state
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim.get_physics_dt())

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        if not file_saved:
            log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

        if reset_flag and not file_saved:
            file_saved = True
            output_path = _save_motion_log(log, job.output_file)
            _maybe_upload_wandb(output_path, job.output_name)
            return output_path

    raise RuntimeError(f"simulation app stopped before job completed: {job.input_file}")


def main():
    """Main function."""
    jobs = _load_jobs()
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    # Design scene
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    scene.reset()
    print("[INFO]: Setup complete...")
    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(_JOINT_NAMES, preserve_order=True)[0]
    failures: list[tuple[str, str]] = []
    for index, job in enumerate(jobs, start=1):
        print(f"[INFO]: Processing job {index}/{len(jobs)} -> {job.output_name}")
        try:
            sim.reset()
            scene.reset()
            _run_motion_job(sim, scene, robot, robot_joint_indexes, job)
        except Exception as exc:
            failures.append((job.output_name, str(exc)))
            print(f"[ERROR]: Job failed for {job.output_name}: {exc}")
    if failures:
        print("[ERROR]: Batch completed with failures:")
        for output_name, message in failures:
            print(f"  - {output_name}: {message}")
        raise RuntimeError(f"{len(failures)} csv_to_npz jobs failed")


if __name__ == "__main__":
    # run the main function
    main()
    # Isaac rendering extensions can keep background GPU workers alive after
    # the output file has been flushed. The caller runs one motion per process,
    # so force the process boundary instead of hanging during shutdown.
    os._exit(0)
