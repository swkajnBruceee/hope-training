#!/usr/bin/env python3
"""Replay an A3 NPZ through the local body_drive ROS2 simulator.

The simulator applies the same command contract used by the local body_drive
path:

    torque = effort + Kp * (position - q) + Kd * (velocity - dq)

NPZ joint_pos/joint_vel are in the Isaac articulation order recorded by the
project metadata. They are converted to body_drive groups by joint name, never
by assuming the CSV order is the runtime order.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, Twist
from joint_msgs.msg import Command, JointCommand, JointState
from mujoco_sim_msgs.msg import SimReset
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import Imu
from sensor_msgs.msg import JointState as RosJointState
from std_msgs.msg import Header


GROUPS = {
    "waist": ("/body_drive/waist_joint_command", "/body_drive/waist_joint_state", (
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")),
    "neck": ("/body_drive/neck_joint_command", "/body_drive/neck_joint_state", (
        "head_yaw_joint", "head_pitch_joint")),
    "arm": ("/body_drive/arm_joint_command", "/body_drive/arm_joint_state", (
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint")),
    "leg": ("/body_drive/leg_joint_command", "/body_drive/leg_joint_state", (
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
        "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
        "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint")),
}

JOINT_NAMES = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "head_yaw_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint", "head_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)

KP = {
    "waist_yaw_joint": 85.0, "waist_roll_joint": 50.0, "waist_pitch_joint": 50.0,
    "head_yaw_joint": 40.0, "head_pitch_joint": 40.0,
    "left_shoulder_pitch_joint": 40.0, "right_shoulder_pitch_joint": 40.0,
    "left_shoulder_roll_joint": 40.0, "right_shoulder_roll_joint": 40.0,
    "left_shoulder_yaw_joint": 30.0, "right_shoulder_yaw_joint": 30.0,
    "left_elbow_joint": 30.0, "right_elbow_joint": 30.0,
    "left_wrist_roll_joint": 30.0, "right_wrist_roll_joint": 30.0,
    "left_wrist_pitch_joint": 20.0, "right_wrist_pitch_joint": 20.0,
    "left_wrist_yaw_joint": 20.0, "right_wrist_yaw_joint": 20.0,
    "left_hip_pitch_joint": 80.0, "right_hip_pitch_joint": 80.0,
    "left_hip_roll_joint": 120.0, "right_hip_roll_joint": 120.0,
    "left_hip_yaw_joint": 80.0, "right_hip_yaw_joint": 80.0,
    "left_knee_joint": 250.0, "right_knee_joint": 250.0,
    "left_ankle_pitch_joint": 50.0, "right_ankle_pitch_joint": 50.0,
    "left_ankle_roll_joint": 50.0, "right_ankle_roll_joint": 50.0,
}
KD = {name: 2.0 for name in JOINT_NAMES}
for _name in ("waist_yaw_joint", "left_hip_pitch_joint", "right_hip_pitch_joint", "left_hip_yaw_joint", "right_hip_yaw_joint"):
    KD[_name] = 3.0
for _name in ("left_hip_roll_joint", "right_hip_roll_joint"):
    KD[_name] = 4.0
for _name in ("left_knee_joint", "right_knee_joint"):
    KD[_name] = 8.0
for _name in ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "left_shoulder_roll_joint", "right_shoulder_roll_joint"):
    KD[_name] = 3.0

# Force limits in the checked-in A3 MuJoCo model. These are diagnostics only;
# MuJoCo remains the authority for actual actuator saturation.
EFFORT_LIMIT = {name: 220.0 for name in JOINT_NAMES}
EFFORT_LIMIT.update({name: 320.0 for name in JOINT_NAMES if "knee" in name})
EFFORT_LIMIT.update({name: 118.2 for name in JOINT_NAMES if "ankle_pitch" in name})
EFFORT_LIMIT.update({name: 54.75 for name in JOINT_NAMES if "ankle_roll" in name})
EFFORT_LIMIT.update({"waist_roll_joint": 46.0, "waist_pitch_joint": 118.0})
EFFORT_LIMIT.update({name: 6.0 for name in JOINT_NAMES if name.startswith("head_") or "wrist_pitch" in name or "wrist_yaw" in name})
EFFORT_LIMIT.update({name: 60.0 for name in JOINT_NAMES if "shoulder_pitch" in name or "shoulder_roll" in name})
EFFORT_LIMIT.update({name: 24.0 for name in JOINT_NAMES if "shoulder_yaw" in name or "elbow" in name or "wrist_roll" in name})


def quat_xyzw_up_tilt_deg(quat: np.ndarray) -> float:
    """Angle between an IMU frame's local +Z and world +Z."""
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-9:
        return float("nan")
    x, y, z, w = quat / norm
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return float(np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0))))


STAND_Q = {name: 0.0 for name in JOINT_NAMES}
STAND_Q.update({
    "left_hip_pitch_joint": -0.1311, "right_hip_pitch_joint": -0.1311,
    "left_hip_roll_joint": 0.0056, "right_hip_roll_joint": -0.0056,
    "left_hip_yaw_joint": -0.0348, "right_hip_yaw_joint": 0.0348,
    "left_knee_joint": 0.2468, "right_knee_joint": 0.2468,
    "left_ankle_pitch_joint": -0.1204, "right_ankle_pitch_joint": -0.1204,
    "left_ankle_roll_joint": -0.0078, "right_ankle_roll_joint": 0.0078,
    "left_shoulder_pitch_joint": 0.3, "right_shoulder_pitch_joint": 0.3,
    "left_shoulder_roll_joint": 0.12, "right_shoulder_roll_joint": -0.12,
    "left_elbow_joint": 0.8, "right_elbow_joint": 0.8,
})

# Production PD_STAND values transcribed from the checked-in A3 deployment
# example. Normal motion gains remain the project's a3_kps/a3_kds values.
STAND_KP = {name: KP[name] for name in JOINT_NAMES}
STAND_KD = {name: KD[name] for name in JOINT_NAMES}
STAND_KP.update({"waist_yaw_joint": 400.0, "waist_roll_joint": 500.0, "waist_pitch_joint": 500.0})
STAND_KD.update({"waist_yaw_joint": 4.0, "waist_roll_joint": 4.0, "waist_pitch_joint": 4.0})
for name in JOINT_NAMES:
    if "shoulder_pitch" in name or "shoulder_roll" in name or "elbow_joint" in name:
        STAND_KP[name] = 200.0
        STAND_KD[name] = 2.0 if ("shoulder" in name) else 1.0
    elif "shoulder_yaw" in name or "wrist_roll" in name:
        STAND_KP[name] = 100.0
        STAND_KD[name] = 1.0
    elif "wrist_pitch" in name or "wrist_yaw" in name:
        STAND_KP[name] = 50.0
        STAND_KD[name] = 1.0
    elif "hip_pitch" in name:
        STAND_KP[name] = 1500.0
        STAND_KD[name] = 8.0
    elif "hip_roll" in name or "hip_yaw" in name:
        STAND_KP[name] = 400.0 if "hip_roll" in name else 300.0
        STAND_KD[name] = 7.0
    elif "knee" in name:
        STAND_KP[name] = 2000.0
        STAND_KD[name] = 8.0
    elif "ankle_pitch" in name:
        STAND_KP[name] = 500.0
        STAND_KD[name] = 5.0
    elif "ankle_roll" in name:
        STAND_KP[name] = 500.0
        STAND_KD[name] = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--motion-file", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("eval_outputs/a3_mujoco_body_drive"))
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--tail-s", type=float, default=1.0)
    parser.add_argument("--command-hz", type=float, default=500.0)
    parser.add_argument(
        "--qos-profile",
        choices=("mujoco_sim", "official"),
        default="mujoco_sim",
        help="body-drive QoS: local MuJoCo uses best_effort; real deployment uses reliable",
    )
    parser.add_argument("--kp-scale", type=float, default=1.0)
    parser.add_argument("--kd-scale", type=float, default=1.0)
    parser.add_argument("--base-z", type=float, default=1.3, help="MuJoCo A3 pelvis reset height in meters")
    parser.add_argument(
        "--skip-reset",
        action="store_true",
        help="do not wait for the project /sim/a3/reset topic; use for official AimSim SIL",
    )
    parser.add_argument(
        "--stand-s",
        type=float,
        default=3.0,
        help="PD_STAND pre-roll; official deploy default is 150 ticks at 50 Hz (3 s)",
    )
    parser.add_argument("--max-runtime-s", type=float, default=20.0)
    return parser.parse_args()


def resolve_motion(args: argparse.Namespace) -> tuple[Path, dict[str, object]]:
    if args.motion_file is not None:
        return args.motion_file.expanduser().resolve(), {}
    data = json.loads(args.manifest.expanduser().read_text())
    motions = data.get("motions", [])
    if not 0 <= args.index < len(motions):
        raise IndexError(f"manifest index {args.index} outside 0..{len(motions) - 1}")
    item = motions[args.index]
    return Path(item["motion_npz"]).expanduser().resolve(), item


def make_qos(reliable: bool = False) -> QoSProfile:
    reliability = QoSReliabilityPolicy.RELIABLE if reliable else QoSReliabilityPolicy.BEST_EFFORT
    durability = (
        QoSDurabilityPolicy.TRANSIENT_LOCAL
        if reliable
        else QoSDurabilityPolicy.VOLATILE
    )
    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=reliability,
        durability=durability,
    )


class BodyDriveReplay(Node):
    def __init__(self, motion: np.lib.npyio.NpzFile, meta: dict[str, object], args: argparse.Namespace):
        super().__init__("hope_a3_body_drive_npz_replay")
        self.args = args
        self.meta = meta
        self.q = np.asarray(motion["joint_pos"], dtype=np.float64)
        self.dq = np.asarray(motion["joint_vel"], dtype=np.float64)
        self.fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
        if self.q.ndim != 2 or self.q.shape[1] != len(JOINT_NAMES):
            raise ValueError(f"expected joint_pos [T, 31], got {self.q.shape}")
        if self.dq.shape != self.q.shape:
            raise ValueError(f"joint_vel shape {self.dq.shape} does not match joint_pos {self.q.shape}")
        self.joint_index = {name: i for i, name in enumerate(JOINT_NAMES)}
        self.command_publishers = {}
        self.states: dict[str, dict[str, float]] = {}
        self.state_samples: list[dict[str, dict[str, float]]] = []
        self.imu: dict[str, dict[str, np.ndarray]] = {}
        self.imu_samples: list[dict[str, dict[str, np.ndarray]]] = []
        # Keep the transport contract explicit. The checked-in local MuJoCo
        # simulator uses best_effort for body-drive topics; the real deployment
        # profile uses reliable. A silent QoS mismatch leaves topics visible but
        # produces zero matched publishers/subscribers.
        qos = make_qos(reliable=args.qos_profile == "official")
        for group, (command_topic, state_topic, names) in GROUPS.items():
            self.command_publishers[group] = self.create_publisher(JointCommand, command_topic, qos)
            self.create_subscription(JointState, state_topic, lambda msg, g=group: self.on_state(g, msg), qos)
        self.create_subscription(Imu, "/body_drive/pelvis_imu/data", lambda msg: self.on_imu("pelvis", msg), qos)
        self.create_subscription(Imu, "/body_drive/torso_imu/data", lambda msg: self.on_imu("torso", msg), qos)
        self.reset_pub = self.create_publisher(SimReset, "/sim/a3/reset", make_qos(reliable=True))
        self.start = time.monotonic()
        # The project A3 MuJoCo example exposes /sim/a3/reset. The official
        # AimSim SIL viewer does not, so its current state must be used as-is.
        self.reset_sent = bool(args.skip_reset)
        self.reset_attempts = 0
        self.finished = False
        self.sequence = 0
        self.q_ref_samples: list[np.ndarray] = []
        self.dq_ref_samples: list[np.ndarray] = []
        self.sim_time_samples: list[float] = []
        self.wall_time_samples: list[float] = []
        self.publish_period_samples: list[float] = []
        self.last_tick_time: float | None = None
        self.timer = self.create_timer(1.0 / args.command_hz, self.tick)

    def on_state(self, group: str, msg: JointState) -> None:
        for state in msg.joints:
            self.states[state.name] = {
                "position": float(state.position),
                "velocity": float(state.velocity),
                "effort": float(state.effort),
            }

    def on_imu(self, name: str, msg: Imu) -> None:
        self.imu[name] = {
            "orientation": np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w], dtype=np.float64),
            "angular_velocity": np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z], dtype=np.float64),
            "linear_acceleration": np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z], dtype=np.float64),
        }

    def interpolated(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        if t <= 0:
            return self.q[0], np.zeros(31, dtype=np.float64)
        duration = (len(self.q) - 1) / self.fps
        if t >= duration:
            return self.q[-1], np.zeros(31, dtype=np.float64)
        x = t * self.fps
        i = min(int(math.floor(x)), len(self.q) - 2)
        a = x - i
        return (1.0 - a) * self.q[i] + a * self.q[i + 1], (1.0 - a) * self.dq[i] + a * self.dq[i + 1]

    def send_reset(self) -> None:
        msg = SimReset()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.mode = SimReset.MODE_ABSOLUTE
        msg.set_base = True
        msg.pelvis_pose = Pose()
        msg.pelvis_pose.position.z = self.args.base_z
        msg.pelvis_pose.orientation.w = 1.0
        msg.set_base_twist = True
        msg.pelvis_twist = Twist()
        msg.set_joints = True
        msg.zero_all_velocities = True
        msg.clear_ctrl = True
        msg.joint_state = RosJointState()
        msg.joint_state.name = list(JOINT_NAMES)
        msg.joint_state.position = [STAND_Q[name] if self.args.stand_s > 0.0 else self.q[0, i] for i, name in enumerate(JOINT_NAMES)]
        msg.joint_state.velocity = [0.0] * len(JOINT_NAMES)
        self.reset_pub.publish(msg)
        self.reset_sent = True
        self.reset_attempts += 1

    def publish_group(self, group: str, q: np.ndarray, dq: np.ndarray, kp_map: dict[str, float], kd_map: dict[str, float]) -> None:
        _, _, names = GROUPS[group]
        msg = JointCommand()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joints = []
        for name in names:
            i = self.joint_index[name]
            command = Command()
            command.name = name
            command.sequence = self.sequence
            command.position = float(q[i])
            command.velocity = float(dq[i])
            command.effort = 0.0
            command.stiffness = self.args.kp_scale * kp_map[name]
            command.damping = self.args.kd_scale * kd_map[name]
            msg.joints.append(command)
        self.command_publishers[group].publish(msg)

    def tick(self) -> None:
        now = time.monotonic()
        elapsed = now - self.start
        if elapsed > self.args.max_runtime_s:
            self.get_logger().error("runtime safety timeout")
            self.finish()
            return
        if not self.reset_sent:
            # Do not publish the one-shot reset before ROS2 discovery has
            # matched the simulator subscriber. This otherwise leaves a
            # floating-base MuJoCo instance in its previous fallen state.
            if self.reset_pub.get_subscription_count() == 0:
                return
            self.send_reset()
        phase_t = elapsed - self.args.settle_s
        sim_t = phase_t - self.args.stand_s
        if phase_t < 0.0 or sim_t < 0.0:
            q_ref = np.array([STAND_Q[name] for name in JOINT_NAMES], dtype=np.float64)
            dq_ref = np.zeros(31, dtype=np.float64)
            kp_map, kd_map = STAND_KP, STAND_KD
        else:
            q_ref, dq_ref = self.interpolated(sim_t)
            kp_map, kd_map = KP, KD
        for group in GROUPS:
            self.publish_group(group, q_ref, dq_ref, kp_map, kd_map)
        self.sequence = (self.sequence + 1) & 0xFFFFFFFF
        sample = {name: self.states.get(name, {"position": np.nan, "velocity": np.nan, "effort": np.nan}) for name in JOINT_NAMES}
        self.state_samples.append(sample)
        self.imu_samples.append({name: {key: value.copy() for key, value in data.items()} for name, data in self.imu.items()})
        self.q_ref_samples.append(q_ref.copy())
        self.dq_ref_samples.append(dq_ref.copy())
        self.sim_time_samples.append(float(sim_t))
        self.wall_time_samples.append(float(elapsed))
        if self.last_tick_time is None:
            self.publish_period_samples.append(float("nan"))
        else:
            self.publish_period_samples.append(float(now - self.last_tick_time))
        self.last_tick_time = now
        duration = (len(self.q) - 1) / self.fps
        if sim_t >= duration + self.args.tail_s:
            self.finish()

    def finish(self) -> None:
        if self.finished:
            return
        self.finished = True
        self.timer.cancel()

    def write_outputs(self, out_dir: Path, motion_path: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        samples = self.state_samples
        actual = np.full((len(samples), len(JOINT_NAMES)), np.nan, dtype=np.float64)
        actual_dq = np.full_like(actual, np.nan)
        effort = np.full_like(actual, np.nan)
        for row, sample in enumerate(samples):
            for col, name in enumerate(JOINT_NAMES):
                actual[row, col] = sample[name]["position"]
                actual_dq[row, col] = sample[name]["velocity"]
                effort[row, col] = sample[name]["effort"]
        q_ref = np.asarray(self.q_ref_samples, dtype=np.float64)
        dq_ref = np.asarray(self.dq_ref_samples, dtype=np.float64)
        sim_time = np.asarray(self.sim_time_samples, dtype=np.float64)
        wall_time = np.asarray(self.wall_time_samples, dtype=np.float64)
        publish_period = np.asarray(self.publish_period_samples, dtype=np.float64)
        imu_orientation = np.full((len(samples), 2, 4), np.nan, dtype=np.float64)
        imu_angular_velocity = np.full((len(samples), 2, 3), np.nan, dtype=np.float64)
        imu_linear_acceleration = np.full((len(samples), 2, 3), np.nan, dtype=np.float64)
        imu_tilt_deg = np.full((len(samples), 2), np.nan, dtype=np.float64)
        for row, sample in enumerate(self.imu_samples):
            for col, name in enumerate(("pelvis", "torso")):
                if name not in sample:
                    continue
                imu_orientation[row, col] = sample[name]["orientation"]
                imu_angular_velocity[row, col] = sample[name]["angular_velocity"]
                imu_linear_acceleration[row, col] = sample[name]["linear_acceleration"]
                imu_tilt_deg[row, col] = quat_xyzw_up_tilt_deg(sample[name]["orientation"])
        np.savez_compressed(
            out_dir / "body_drive_states.npz",
            q_ref=q_ref,
            dq_ref=dq_ref,
            actual=actual,
            actual_velocity=actual_dq,
            effort=effort,
            sim_time=sim_time,
            wall_time=wall_time,
            publish_period=publish_period,
            imu_orientation=imu_orientation,
            imu_angular_velocity=imu_angular_velocity,
            imu_linear_acceleration=imu_linear_acceleration,
            imu_tilt_deg=imu_tilt_deg,
        )
        fields = ["sample", "wall_time_s", "sim_time_s", "publish_period_s"]
        for name in JOINT_NAMES:
            fields += [f"{name}.q_ref", f"{name}.q_actual", f"{name}.dq_actual", f"{name}.effort"]
        with (out_dir / "body_drive_replay.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row, sample in enumerate(samples):
                values = {
                    "sample": row,
                    "wall_time_s": wall_time[row],
                    "sim_time_s": sim_time[row],
                    "publish_period_s": publish_period[row],
                }
                for name in JOINT_NAMES:
                    values[f"{name}.q_ref"] = q_ref[row, self.joint_index[name]]
                    values[f"{name}.q_actual"] = sample[name]["position"]
                    values[f"{name}.dq_actual"] = sample[name]["velocity"]
                    values[f"{name}.effort"] = sample[name]["effort"]
                writer.writerow(values)
        finite = np.isfinite(actual) & np.isfinite(q_ref)
        error = np.abs(actual - q_ref)
        valid_errors = error[finite]
        valid_periods = publish_period[np.isfinite(publish_period)]
        active = (sim_time >= 0.0) & (sim_time <= (len(self.q) - 1) / self.fps)
        active_finite = finite & active[:, None]
        active_errors = error[active_finite]
        active_effort = np.abs(effort[active_finite])
        active_velocity = np.abs(actual_dq[active_finite])
        active_tilt = imu_tilt_deg[active]
        active_gyro = np.abs(imu_angular_velocity[active])
        def percentile(values: np.ndarray, p: float) -> float | None:
            return float(np.percentile(values, p)) if values.size else None

        summary = {
            "motion_file": str(motion_path),
            "episode_id": self.meta.get("episode_id"),
            "stroke_type": self.meta.get("stroke_type"),
            "fps": self.fps,
            "command_hz_requested": self.args.command_hz,
            "qos_profile": self.args.qos_profile,
            "sample_count": len(samples),
            "state_coverage_fraction": float(np.mean(finite)) if finite.size else 0.0,
            "q_tracking_mae_rad": float(np.mean(valid_errors)) if valid_errors.size else None,
            "q_tracking_p95_rad": percentile(valid_errors, 95.0),
            "q_tracking_max_rad": float(np.max(valid_errors)) if valid_errors.size else None,
            "active_motion_sample_count": int(np.sum(active)),
            "active_q_tracking_mae_rad": float(np.mean(active_errors)) if active_errors.size else None,
            "active_q_tracking_p95_rad": percentile(active_errors, 95.0),
            "active_q_tracking_max_rad": float(np.max(active_errors)) if active_errors.size else None,
            "actual_position_abs_max_rad": float(np.nanmax(np.abs(actual))) if np.any(finite) else None,
            "actual_velocity_abs_max_rad_s": float(np.nanmax(np.abs(actual_dq))) if np.any(np.isfinite(actual_dq)) else None,
            "effort_abs_max_nm": float(np.nanmax(np.abs(effort))) if np.any(np.isfinite(effort)) else None,
            "active_velocity_abs_p95_rad_s": percentile(active_velocity, 95.0),
            "active_effort_abs_p95_nm": percentile(active_effort, 95.0),
            "active_torso_tilt_p95_deg": percentile(active_tilt[:, 1][np.isfinite(active_tilt[:, 1])], 95.0) if active_tilt.size else None,
            "active_torso_tilt_max_deg": float(np.nanmax(active_tilt[:, 1])) if active_tilt.size and np.any(np.isfinite(active_tilt[:, 1])) else None,
            "active_torso_gyro_p95_rad_s": percentile(active_gyro[:, 1][np.isfinite(active_gyro[:, 1])], 95.0) if active_gyro.size else None,
            "active_torso_gyro_max_rad_s": float(np.nanmax(active_gyro[:, 1])) if active_gyro.size and np.any(np.isfinite(active_gyro[:, 1])) else None,
            "publish_period_mean_s": float(np.mean(valid_periods)) if valid_periods.size else None,
            "publish_period_p95_s": percentile(valid_periods, 95.0),
            "publish_rate_hz_estimate": float(1.0 / np.mean(valid_periods)) if valid_periods.size and np.mean(valid_periods) > 0 else None,
            "gains": {
                "kp_scale": self.args.kp_scale,
                "kd_scale": self.args.kd_scale,
                "motion_kp_source": "AGIBOT_A3_CFG a3_kps",
                "motion_kd_source": "AGIBOT_A3_CFG a3_kds",
                "stand_kp_source": "a3_policy_parameters.hpp a3_pd_stand_kps",
                "stand_kd_source": "a3_policy_parameters.hpp a3_pd_stand_kds",
                "stand_duration_s": self.args.stand_s,
                "stand_reference": "official a3_default_angles / PD_STAND gains",
                "official_default_ticks": 150,
                "official_default_tick_hz": 50.0,
            },
            "note": "Actual state/effort are from local MuJoCo body_drive; this is actuator-contract validation, not official native MC balance validation.",
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    motion_path, meta = resolve_motion(args)
    motion = np.load(motion_path, allow_pickle=False)
    rclpy.init()
    node = BodyDriveReplay(motion, meta, args)
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.write_outputs(args.out_dir.expanduser().resolve(), motion_path)
        node.destroy_node()
        rclpy.shutdown()
        motion.close()
    print(f"[body-drive-replay] outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
