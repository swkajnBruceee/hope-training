#!/usr/bin/env python3
"""Build a training motion clip from the fixed serve CSV.

The training MotionLoader consumes the complete A3 motion NPZ schema, while the
serve CSV is a controller-side 31-DOF trajectory.  This converter keeps the
official model_21800 ready posture for the legs, waist and head, replaces only
the two arms from the CSV, and computes the complete body FK state with MuJoCo.

The first 0.8 s is a quintic transition from the official ready posture to CSV
frame 0.  The original CSV timing is then kept unchanged, so its event frames
are not slowed down by the transition.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np
import yaml


JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "head_yaw_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "head_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

UPPER_JOINTS = {
    name
    for name in JOINT_NAMES
    if any(
        token in name
        for token in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_")
    )
}

# Isaac's exported complete-articulation arrays keep one entry per DOF in the
# configured joint order.  Waist pitch is a joint on the torso body, so it is
# intentionally duplicated there rather than looking for a nonexistent
# ``waist_pitch_Link``.
BODY_NAMES = [
    "pelvis_link",
    "left_hip_pitch_Link",
    "right_hip_pitch_Link",
    "waist_yaw_Link",
    "left_hip_roll_Link",
    "right_hip_roll_Link",
    "waist_roll_Link",
    "left_hip_yaw_Link",
    "right_hip_yaw_Link",
    "torso_Link",
    "left_knee_Link",
    "right_knee_Link",
    "head_yaw_Link",
    "left_shoulder_pitch_Link",
    "right_shoulder_pitch_Link",
    "left_ankle_pitch_Link",
    "right_ankle_pitch_Link",
    "head_pitch_Link",
    "left_shoulder_roll_Link",
    "right_shoulder_roll_Link",
    "left_ankle_roll_Link",
    "right_ankle_roll_Link",
    "left_shoulder_yaw_Link",
    "right_shoulder_yaw_Link",
    "left_elbow_Link",
    "right_elbow_Link",
    "left_wrist_roll_Link",
    "right_wrist_roll_Link",
    "left_wrist_pitch_Link",
    "right_wrist_pitch_Link",
    "left_wrist_yaw_Link",
    "right_wrist_yaw_Link",
]


def _parse_csv(path: Path) -> tuple[np.ndarray, list[str], dict[str, str]]:
    rows = list(csv.reader(path.open(newline="")))
    if len(rows) < 4:
        raise ValueError(f"serve CSV is too short: {path}")
    metadata = {row[0]: row[1] for row in rows[:2] if len(row) >= 2}
    header = rows[2]
    if header[:4] != ["frame", "time_s", "phase", "event_mask"]:
        raise ValueError(f"unexpected CSV header prefix: {header[:4]}")
    missing = [name for name in JOINT_NAMES if name not in header]
    if missing:
        raise ValueError(f"serve CSV is missing joints: {missing}")
    indexes = [header.index(name) for name in JOINT_NAMES]
    time_index = header.index("time_s")
    data = np.asarray(
        [[float(row[i]) for i in indexes] for row in rows[3:] if row], dtype=np.float64
    )
    times = np.asarray([float(row[time_index]) for row in rows[3:] if row], dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != len(JOINT_NAMES):
        raise ValueError(f"unexpected joint matrix shape: {data.shape}")
    if len(times) != len(data) or not np.all(np.diff(times) > 0):
        raise ValueError("CSV time_s must be strictly increasing")
    dt = np.diff(times)
    if not np.allclose(dt, dt[0], atol=1e-6):
        raise ValueError(f"CSV is not uniformly sampled: dt range {dt.min()}..{dt.max()}")
    return data, JOINT_NAMES, metadata


def _load_default_q(path: Path) -> np.ndarray:
    document = yaml.safe_load(path.read_text())
    values = document["default_joint_pos"]
    if len(values) != len(JOINT_NAMES):
        raise ValueError(f"deploy default_joint_pos has {len(values)} values, expected 31")
    return np.asarray(values, dtype=np.float64)


def _smoothstep5(values: np.ndarray) -> np.ndarray:
    """Quintic smoothstep, with zero velocity and acceleration at both ends."""

    return values * values * values * (values * (values * 6.0 - 15.0) + 10.0)


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q, axis=-1, keepdims=True).clip(min=1e-12)


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def _angular_velocity(quat: np.ndarray, dt: float) -> np.ndarray:
    """World-frame angular velocity from body orientation samples."""

    q = _quat_normalize(quat.copy())
    for i in range(1, len(q)):
        flip = np.sum(q[i - 1] * q[i], axis=-1) < 0.0
        q[i, flip] *= -1.0
    velocity = np.zeros((len(q), q.shape[1], 3), dtype=np.float64)
    pairs = [(0, 1, dt), (len(q) - 2, len(q) - 1, dt)]
    for index in range(1, len(q) - 1):
        pairs.append((index - 1, index + 1, 2.0 * dt))
    for left, right, denominator in pairs:
        delta = _quat_multiply(q[right], _quat_conjugate(q[left]))
        delta = _quat_normalize(delta)
        w = delta[..., 0].clip(-1.0, 1.0)
        angle = 2.0 * np.arccos(w)
        sin_half = np.sqrt(np.maximum(1.0 - w * w, 0.0))
        axis = np.zeros_like(delta[..., 1:])
        np.divide(delta[..., 1:], sin_half[..., None], out=axis, where=sin_half[..., None] > 1e-8)
        # For tiny rotations, axis is irrelevant and the angular velocity is zero.
        velocity[(left + right) // 2] = axis * angle[..., None] / denominator
    return velocity


def _finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    return np.gradient(values, dt, axis=0, edge_order=1)


def _fk(model_path: Path, joint_pos: np.ndarray, root_z: float) -> tuple[np.ndarray, ...]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    body_ids = []
    for body_name in BODY_NAMES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"MuJoCo model has no body {body_name!r}")
        body_ids.append(body_id)
    joint_ids = []
    for joint_name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo model has no joint {joint_name!r}")
        joint_ids.append(joint_id)

    data.qpos[:] = 0.0
    data.qpos[2] = root_z
    data.qpos[3] = 1.0
    body_pos = np.empty((len(joint_pos), len(body_ids), 3), dtype=np.float64)
    body_quat = np.empty((len(joint_pos), len(body_ids), 4), dtype=np.float64)
    for frame_index, frame in enumerate(joint_pos):
        for joint_id, value in zip(joint_ids, frame):
            data.qpos[model.jnt_qposadr[joint_id]] = value
        mujoco.mj_forward(model, data)
        # Copy immediately: MuJoCo reuses the same data buffers every frame.
        body_pos[frame_index] = data.xpos[body_ids]
        body_quat[frame_index] = data.xquat[body_ids]
    return body_pos, body_quat


def build_clip(csv_path: Path, deploy_path: Path, model_path: Path, output_path: Path, transition_s: float) -> dict:
    csv_q, _, csv_metadata = _parse_csv(csv_path)
    default_q = _load_default_q(deploy_path)
    hz = float(csv_metadata.get("policy_hz", "50"))
    dt = 1.0 / hz
    transition_frames = int(round(transition_s * hz))
    if transition_frames < 1:
        raise ValueError("transition_s must be at least one control step")

    q = np.repeat(default_q[None, :], len(csv_q), axis=0)
    for index, name in enumerate(JOINT_NAMES):
        if name in UPPER_JOINTS:
            q[:, index] = csv_q[:, index]

    # Include the endpoint in the prelude and drop duplicate CSV frame 0.  This
    # makes the serve's original event timing start exactly at transition_s.
    u = np.linspace(0.0, 1.0, transition_frames + 1)
    prelude = default_q[None, :] + _smoothstep5(u)[:, None] * (q[0] - default_q)[None, :]
    q = np.concatenate((prelude, q[1:]), axis=0)
    body_pos, body_quat = _fk(model_path, q, root_z=1.0684)
    body_lin_vel = _finite_difference(body_pos, dt)
    body_ang_vel = _angular_velocity(body_quat, dt)
    joint_vel = _finite_difference(q, dt)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": "hope_motion_complete_articulation_v1",
        "source_csv": str(csv_path),
        "source_csv_schema": csv_metadata.get("schema", "unknown"),
        "fps": int(round(hz)),
        "dt_s": dt,
        "transition_s": transition_s,
        "transition_frames": transition_frames,
        "serve_frame_start": transition_frames,
        "upper_body_source": "CSV shoulders/elbows/wrists",
        "lower_body_source": "model_21800 deploy default_joint_pos",
        "head_and_waist_source": "model_21800 deploy default_joint_pos",
        "joint_order": JOINT_NAMES,
        "body_order": BODY_NAMES,
    }
    np.savez_compressed(
        output_path,
        fps=np.asarray([int(round(hz))], dtype=np.int64),
        joint_pos=q.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        body_pos_w=body_pos.astype(np.float32),
        body_quat_w=body_quat.astype(np.float32),
        body_lin_vel_w=body_lin_vel.astype(np.float32),
        body_ang_vel_w=body_ang_vel.astype(np.float32),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    output_path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return metadata | {
        "frame_count": int(len(q)),
        "duration_s": float((len(q) - 1) * dt),
        "max_joint_speed_rad_s": float(np.abs(joint_vel).max()),
        "max_joint_acc_rad_s2": float(np.abs(_finite_difference(joint_vel, dt)).max()),
        "max_body_speed_m_s": float(np.linalg.norm(body_lin_vel, axis=-1).max()),
    }


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("/home/bistu/桌面/pp_serve_v1_fixed.csv"))
    parser.add_argument("--deploy", type=Path, default=repo / "mujoco_reference/models/model_21800/policy/params/deploy.yaml")
    parser.add_argument("--model", type=Path, default=Path("/home/bistu/桌面/mc_ltl/runtime/mujoco_v13b/model/mjcf/a3_pingpong_grounded.xml"))
    parser.add_argument("--output", type=Path, default=repo / "motions/preprocessed/hope_serve_upper.npz")
    parser.add_argument("--transition-s", type=float, default=0.8)
    args = parser.parse_args()
    report = build_clip(args.csv, args.deploy, args.model, args.output, args.transition_s)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
