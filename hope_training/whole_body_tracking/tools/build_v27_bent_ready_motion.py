#!/usr/bin/env python3
"""Audit and build the V27 bent-ready motion contract.

The tool deliberately keeps the validated strike core byte-for-byte identical.
It searches an entry frame in the short pre-hit prefix, builds an analytic
quintic bridge from a shared ready pose, and only slices existing NPZ arrays
when exporting a candidate package.  Runtime bridge/return handling is kept
separate from the motion data so V25/V26 remain reproducible.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    ROOT
    / "sample_motions"
    / "p2_data260708_backhand_strike_only_v2_momentum"
    / "manifest.json"
)
DEFAULT_METADATA = ROOT / "docs" / "a3_articulation_metadata.json"
DEFAULT_MJCF = (
    ROOT.parents[1]
    / "agibot"
    / "A3_MuJoCo_Sim"
    / "aimrt_mujoco_sim"
    / "src"
    / "models"
    / "bin"
    / "cfg"
    / "model"
    / "a3_pingpong"
    / "a3_pingpong.xml"
)
DEFAULT_OUTPUT = ROOT / "eval_outputs" / "v27_bent_ready"

RIGHT_ARM = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
LEFT_ARM = tuple(name.replace("right_", "left_") for name in RIGHT_ARM)
UPPER_FOR_BRIDGE_SCORE = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    *RIGHT_ARM,
)

# Explicit collision geoms only. Adjacent right-arm geoms are excluded because
# their fitted primitives intentionally overlap around physical joints.
RIGHT_DISTAL_COLLISION_GEOMS = (
    "right_shoulder_yaw_collision",
    "right_elbow_collision",
    "right_wrist_roll_collision",
    "right_wrist_pitch_collision",
    "right_wrist_yaw_collision",
    "right_hand_palm_collision",
    "right_hand_finger_collision",
    "right_hand_thumb_collision",
    "right_racket_collision",
    "right_racket_handle_collision",
)
OBSTACLE_COLLISION_GEOMS = (
    "pelvis_collision",
    "torso_collision",
    "head_yaw_collision",
    "head_pitch_collision",
    "left_shoulder_pitch_collision",
    "left_shoulder_roll_collision",
    "left_shoulder_yaw_collision",
    "left_elbow_collision",
    "left_wrist_roll_collision_0",
    "left_wrist_roll_collision_1",
    "left_wrist_pitch_collision",
    "left_wrist_yaw_collision",
    "left_hand_collision",
)


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    ready_q: np.ndarray
    right_blend: float
    right_elbow_rad: float
    left_elbow_rad: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _joint_index(joint_names: list[str]) -> dict[str, int]:
    if len(joint_names) != len(set(joint_names)):
        raise ValueError("Articulation metadata contains duplicate joint names")
    return {name: index for index, name in enumerate(joint_names)}


def _ready_pose(joint_names: list[str]) -> np.ndarray:
    """Return the verified V25/V26 flexed support pose in articulation order."""
    q = np.zeros(len(joint_names), dtype=np.float64)
    index = _joint_index(joint_names)
    for side in ("left", "right"):
        q[index[f"{side}_hip_pitch_joint"]] = -0.1600
        q[index[f"{side}_knee_joint"]] = 0.3200
        q[index[f"{side}_ankle_pitch_joint"]] = -0.1550
    q[index["left_hip_roll_joint"]] = 0.0800
    q[index["right_hip_roll_joint"]] = -0.0800
    for side in ("left", "right"):
        q[index[f"{side}_shoulder_pitch_joint"]] = 0.3
        q[index[f"{side}_elbow_joint"]] = 0.8
    q[index["left_shoulder_roll_joint"]] = 0.12
    q[index["right_shoulder_roll_joint"]] = -0.12
    return q


def _load_motions(
    manifest_path: Path, metadata_path: Path
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    joint_names = list(metadata["joint_names"])
    if len(joint_names) != 31:
        raise ValueError(f"Expected 31 articulation joints, got {len(joint_names)}")

    motions: list[dict[str, Any]] = []
    for row in manifest["motions"]:
        path = Path(row["motion_npz"])
        if not path.is_file():
            local = manifest_path.parent / "motion_npz" / path.name
            if not local.is_file():
                raise FileNotFoundError(path)
            path = local
        with np.load(path) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        fps = int(np.asarray(arrays["fps"]).reshape(-1)[0])
        hit = int(row["hit_event"]["motion_hit_frame"])
        frames = int(arrays["joint_pos"].shape[0])
        if fps != 50 or frames != 39 or hit != 30:
            raise ValueError(
                f"{row['episode_id']}: V27 expects frozen 50 Hz / 39 frame / "
                f"hit=30 contract, got fps={fps}, frames={frames}, hit={hit}"
            )
        if arrays["joint_pos"].shape[1] != len(joint_names):
            raise ValueError(f"{row['episode_id']}: joint order width mismatch")
        motions.append({"row": row, "path": path, "arrays": arrays, "hit": hit})
    if len(motions) != 6:
        raise ValueError(f"Expected the frozen six-motion pool, got {len(motions)}")
    return manifest, joint_names, motions


def _robust_pre_hit_center(
    motions: list[dict[str, Any]], joint_names: list[str]
) -> np.ndarray:
    """Use the only legal common entry window: old frames hit-30..hit-20."""
    samples = []
    for motion in motions:
        hit = motion["hit"]
        samples.append(motion["arrays"]["joint_pos"][hit - 30 : hit - 19])
    return np.median(np.concatenate(samples, axis=0), axis=0).astype(np.float64)


class MujocoAudit:
    def __init__(self, model_path: Path, joint_names: list[str]):
        self.model_path = model_path
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.joint_names = joint_names
        self.npz_index = _joint_index(joint_names)
        self.qpos_address: dict[str, int] = {}
        self.joint_ranges: dict[str, tuple[float, float]] = {}
        for name in joint_names:
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if joint_id < 0:
                raise ValueError(f"MuJoCo model does not contain {name}")
            self.qpos_address[name] = int(self.model.jnt_qposadr[joint_id])
            lower, upper = self.model.jnt_range[joint_id]
            self.joint_ranges[name] = (float(lower), float(upper))
        self.torso_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_Link"
        )
        self.racket_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_racket_collision"
        )
        self.elbow_geometry_joint_ids = {
            side: {
                name: mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    f"{side}_{name}_joint",
                )
                for name in ("shoulder_yaw", "elbow", "wrist_roll")
            }
            for side in ("left", "right")
        }
        self.upper_body_ids = [
            body_id
            for body_id in range(self.model.nbody)
            if 2 <= body_id <= 20
        ]
        self.collision_pairs = self._collision_pairs()

    def _collision_pairs(self) -> list[tuple[int, int, str, str]]:
        pairs = []
        for right_name in RIGHT_DISTAL_COLLISION_GEOMS:
            right_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, right_name
            )
            if right_id < 0:
                raise ValueError(f"Missing MuJoCo geom {right_name}")
            for obstacle_name in OBSTACLE_COLLISION_GEOMS:
                obstacle_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, obstacle_name
                )
                if obstacle_id < 0:
                    raise ValueError(f"Missing MuJoCo geom {obstacle_name}")
                pairs.append((right_id, obstacle_id, right_name, obstacle_name))
        return pairs

    def set_pose(self, q: np.ndarray) -> None:
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[0:3] = (0.0, 0.0, 1.0400)
        self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        for name, source_index in self.npz_index.items():
            self.data.qpos[self.qpos_address[name]] = float(q[source_index])
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def kinematics(self, q: np.ndarray) -> dict[str, Any]:
        self.set_pose(q)
        torso_position = self.data.xpos[self.torso_id]
        torso_rotation = self.data.xmat[self.torso_id].reshape(3, 3)
        racket_position = self.data.geom_xpos[self.racket_geom_id]
        racket_local = torso_rotation.T @ (racket_position - torso_position)

        masses = self.model.body_mass[self.upper_body_ids]
        upper_com = np.average(
            self.data.xipos[self.upper_body_ids], axis=0, weights=masses
        )
        upper_com_local = torso_rotation.T @ (upper_com - torso_position)
        total_mass = float(np.sum(self.model.body_mass[1:]))
        total_com = np.average(
            self.data.xipos[1:], axis=0, weights=self.model.body_mass[1:]
        )
        total_com_local = torso_rotation.T @ (total_com - torso_position)
        min_distance, pair = self.minimum_collision_distance()
        return {
            "racket_position_torso_m": racket_local,
            "racket_distance_torso_m": float(np.linalg.norm(racket_local)),
            "upper_com_torso_m": upper_com_local,
            "total_com_torso_m": total_com_local,
            "total_mass_kg": total_mass,
            "minimum_collision_distance_m": min_distance,
            "minimum_collision_pair": pair,
            "soft_joint_margin_rad": self.soft_joint_margin(q),
            "right_elbow_geometry": self.elbow_geometry("right"),
            "left_elbow_geometry": self.elbow_geometry("left"),
        }

    def elbow_geometry(self, side: str) -> dict[str, float]:
        """Measure physical elbow bend; joint radians are not human elbow angle."""
        ids = self.elbow_geometry_joint_ids[side]
        shoulder = self.data.xanchor[ids["shoulder_yaw"]]
        elbow = self.data.xanchor[ids["elbow"]]
        wrist = self.data.xanchor[ids["wrist_roll"]]
        upper = shoulder - elbow
        forearm = wrist - elbow
        cosine = float(
            np.dot(upper, forearm)
            / (np.linalg.norm(upper) * np.linalg.norm(forearm))
        )
        inner_angle_deg = math.degrees(math.acos(np.clip(cosine, -1.0, 1.0)))
        return {
            "inner_angle_deg": inner_angle_deg,
            "flexion_from_straight_deg": 180.0 - inner_angle_deg,
        }

    def soft_joint_margin(self, q: np.ndarray) -> float:
        return self.soft_joint_margin_detail(q)[0]

    def soft_joint_margin_detail(self, q: np.ndarray) -> tuple[float, str]:
        margins = []
        for name, source_index in self.npz_index.items():
            lower, upper = self.joint_ranges[name]
            center = 0.5 * (lower + upper)
            half = 0.45 * (upper - lower)  # Isaac soft limit factor = 0.9.
            soft_lower, soft_upper = center - half, center + half
            margins.append(
                (
                    min(
                        q[source_index] - soft_lower,
                        soft_upper - q[source_index],
                    ),
                    name,
                )
            )
        margin, name = min(margins)
        return float(margin), name

    def minimum_collision_distance(self) -> tuple[float, tuple[str, str]]:
        minimum = math.inf
        minimum_pair = ("", "")
        from_to = np.zeros(6, dtype=np.float64)
        for first, second, first_name, second_name in self.collision_pairs:
            distance = float(
                mujoco.mj_geomDistance(
                    self.model, self.data, first, second, 2.0, from_to
                )
            )
            if distance < minimum:
                minimum = distance
                minimum_pair = (first_name, second_name)
        return minimum, minimum_pair

    def bridge_clearance(self, q_bridge: np.ndarray) -> dict[str, Any]:
        minimum = math.inf
        minimum_sample = -1
        minimum_pair = ("", "")
        for sample, q in enumerate(q_bridge):
            self.set_pose(q)
            distance, pair = self.minimum_collision_distance()
            if distance < minimum:
                minimum = distance
                minimum_sample = sample
                minimum_pair = pair
        return {
            "minimum_collision_distance_m": minimum,
            "minimum_collision_sample": minimum_sample,
            "minimum_collision_pair": minimum_pair,
        }


def _workspace_penalty(position: np.ndarray) -> float:
    """Penalty outside a deliberately broad compact torso-local work box."""
    lower = np.array((0.12, -0.42, -0.34))
    upper = np.array((0.42, -0.12, 0.06))
    below = np.maximum(lower - position, 0.0)
    above = np.maximum(position - upper, 0.0)
    outside = np.linalg.norm(below + above)
    # Prefer a compact waist/lower-chest point without forcing one exact pose.
    preferred = np.array((0.28, -0.27, -0.16))
    return float(20.0 * outside + np.linalg.norm(position - preferred))


def _workspace_outside_distance(position: np.ndarray) -> float:
    lower = np.array((0.12, -0.42, -0.34))
    upper = np.array((0.42, -0.12, 0.06))
    below = np.maximum(lower - position, 0.0)
    above = np.maximum(position - upper, 0.0)
    return float(np.linalg.norm(below + above))


def _candidate_grid(
    base_ready: np.ndarray,
    robust_center: np.ndarray,
    joint_names: list[str],
) -> list[Candidate]:
    index = _joint_index(joint_names)
    candidates = [
        Candidate(
            name="A_legacy_ready",
            family="A",
            ready_q=base_ready.copy(),
            right_blend=0.0,
            right_elbow_rad=float(base_ready[index["right_elbow_joint"]]),
            left_elbow_rad=float(base_ready[index["left_elbow_joint"]]),
        )
    ]
    right_non_elbow = [name for name in RIGHT_ARM if name != "right_elbow_joint"]
    for blend in (0.15, 0.25, 0.35, 0.45):
        # The A3 elbow zero is already about 81 degrees flexed from straight.
        # Positive joint motion straightens the elbow. These values span about
        # 78/70/61 degrees of physical flexion, matching the requested 60--80
        # degree bent-ready range without confusing joint radians with the
        # human geometric elbow angle.
        for elbow in (0.05, 0.20, 0.35):
            q = base_ready.copy()
            for name in right_non_elbow:
                joint_id = index[name]
                q[joint_id] = (
                    (1.0 - blend) * base_ready[joint_id]
                    + blend * robust_center[joint_id]
                )
            q[index["right_elbow_joint"]] = elbow
            candidates.append(
                Candidate(
                    name=f"B_r{blend:.2f}_e{elbow:.2f}",
                    family="B",
                    ready_q=q,
                    right_blend=blend,
                    right_elbow_rad=elbow,
                    left_elbow_rad=float(q[index["left_elbow_joint"]]),
                )
            )
            # The old left arm at 0.8 rad is already physically flexed about
            # 35 degrees. C makes only a minimal in-range change to about
            # 38 degrees; it is not a new dynamic left-arm policy.
            q_left = q.copy()
            q_left[index["left_elbow_joint"]] = 0.75
            candidates.append(
                Candidate(
                    name=f"C_r{blend:.2f}_e{elbow:.2f}_le0.75",
                    family="C",
                    ready_q=q_left,
                    right_blend=blend,
                    right_elbow_rad=elbow,
                    left_elbow_rad=0.75,
                )
            )
    return candidates


def _finite_difference(
    positions: np.ndarray, frame: int, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    if frame <= 0:
        velocity = (positions[1] - positions[0]) / dt
        acceleration = (positions[2] - 2.0 * positions[1] + positions[0]) / dt**2
    elif frame >= len(positions) - 1:
        velocity = (positions[-1] - positions[-2]) / dt
        acceleration = (
            positions[-1] - 2.0 * positions[-2] + positions[-3]
        ) / dt**2
    else:
        velocity = (positions[frame + 1] - positions[frame - 1]) / (2.0 * dt)
        if frame == len(positions) - 2:
            acceleration = (
                positions[-1] - 2.0 * positions[-2] + positions[-3]
            ) / dt**2
        else:
            acceleration = (
                positions[frame + 1]
                - 2.0 * positions[frame]
                + positions[frame - 1]
            ) / dt**2
    return velocity.astype(np.float64), acceleration.astype(np.float64)


def _quintic_bridge(
    q0: np.ndarray,
    q1: np.ndarray,
    v1: np.ndarray,
    a1: np.ndarray,
    steps: int,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if steps < 2:
        raise ValueError("Bridge must contain at least two control intervals")
    duration = steps * dt
    a0 = q0
    a1_coeff = np.zeros_like(q0)
    a2 = np.zeros_like(q0)
    matrix = np.array(
        [
            [duration**3, duration**4, duration**5],
            [3.0 * duration**2, 4.0 * duration**3, 5.0 * duration**4],
            [6.0 * duration, 12.0 * duration**2, 20.0 * duration**3],
        ],
        dtype=np.float64,
    )
    rhs = np.stack(
        (
            q1 - a0,
            v1,
            a1,
        ),
        axis=0,
    )
    a3, a4, a5 = np.linalg.solve(matrix, rhs).reshape(3, -1)
    time = np.linspace(0.0, duration, steps + 1, dtype=np.float64)[:, None]
    q = a0 + a1_coeff * time + a2 * time**2 + a3 * time**3 + a4 * time**4 + a5 * time**5
    qd = (
        a1_coeff
        + 2.0 * a2 * time
        + 3.0 * a3 * time**2
        + 4.0 * a4 * time**3
        + 5.0 * a5 * time**4
    )
    qdd = 2.0 * a2 + 6.0 * a3 * time + 12.0 * a4 * time**2 + 20.0 * a5 * time**3
    jerk = 6.0 * a3 + 24.0 * a4 * time + 60.0 * a5 * time**2
    return q, qd, qdd, jerk


def _bridge_metrics(
    candidate: Candidate,
    motion: dict[str, Any],
    splice: int,
    joint_names: list[str],
    audit: MujocoAudit,
    bridge_steps: int,
) -> dict[str, Any]:
    positions = motion["arrays"]["joint_pos"].astype(np.float64)
    dt = 1.0 / 50.0
    v1, a1 = _finite_difference(positions, splice, dt)
    q, qd, qdd, jerk = _quintic_bridge(
        candidate.ready_q,
        positions[splice],
        v1,
        a1,
        bridge_steps,
        dt,
    )
    index = _joint_index(joint_names)
    upper_indices = [index[name] for name in UPPER_FOR_BRIDGE_SCORE]
    arm_indices = [index[name] for name in RIGHT_ARM]
    q_distance = float(
        np.linalg.norm(positions[splice, arm_indices] - candidate.ready_q[arm_indices])
    )
    old_speed = float(np.linalg.norm(v1[upper_indices]))
    peak_velocity = float(np.max(np.abs(qd[:, upper_indices])))
    peak_acceleration = float(np.max(np.abs(qdd[:, upper_indices])))
    peak_jerk = float(np.max(np.abs(jerk[:, upper_indices])))
    endpoint_q_error = float(np.max(np.abs(q[-1] - positions[splice])))
    endpoint_v_error = float(np.max(np.abs(qd[-1] - v1)))
    endpoint_a_error = float(np.max(np.abs(qdd[-1] - a1)))
    soft_details = [audit.soft_joint_margin_detail(sample) for sample in q]
    minimum_soft_margin, minimum_soft_joint = min(soft_details)
    start_soft_margin, start_soft_joint = audit.soft_joint_margin_detail(q[0])
    endpoint_soft_margin, endpoint_soft_joint = audit.soft_joint_margin_detail(q[-1])
    allowed_soft_floor = min(start_soft_margin, endpoint_soft_margin)
    clearance = audit.bridge_clearance(q)
    audit.set_pose(q[0])
    start_collision_distance, start_collision_pair = audit.minimum_collision_distance()
    audit.set_pose(q[-1])
    endpoint_collision_distance, endpoint_collision_pair = audit.minimum_collision_distance()
    allowed_collision_floor = min(
        start_collision_distance, endpoint_collision_distance
    )
    collision_regression = (
        clearance["minimum_collision_distance_m"] - allowed_collision_floor
    )
    # MuJoCo mesh distance may return exactly zero for touching or a distance
    # query that cannot establish positive separation. Treat only signed
    # negative distance as geometric penetration, and compare that penetration
    # against the two frozen endpoints.
    collision_penetration_regression = min(
        clearance["minimum_collision_distance_m"], 0.0
    ) - min(allowed_collision_floor, 0.0)
    soft_margin_regression = minimum_soft_margin - allowed_soft_floor
    time_to_hit = (motion["hit"] - splice) * dt
    score = (
        2.0 * q_distance
        + 0.08 * old_speed
        + 0.03 * peak_velocity
        + 0.0015 * peak_acceleration
        + 0.00002 * peak_jerk
        - 0.15 * time_to_hit
    )
    if soft_margin_regression < -0.005:
        score += 1000.0 + 100.0 * abs(soft_margin_regression)
    if collision_penetration_regression < -0.002:
        score += 1000.0 + 100.0 * abs(collision_penetration_regression)
    return {
        "splice_frame_old": splice,
        "new_hit_frame": motion["hit"] - splice,
        "time_splice_to_hit_s": time_to_hit,
        "right_arm_joint_distance_rad": q_distance,
        "entry_upper_speed_norm_radps": old_speed,
        "bridge_peak_upper_velocity_radps": peak_velocity,
        "bridge_peak_upper_acceleration_radps2": peak_acceleration,
        "bridge_peak_upper_jerk_radps3": peak_jerk,
        "bridge_minimum_soft_margin_rad": minimum_soft_margin,
        "bridge_minimum_soft_margin_joint": minimum_soft_joint,
        "bridge_start_soft_margin_rad": start_soft_margin,
        "bridge_start_soft_margin_joint": start_soft_joint,
        "bridge_endpoint_soft_margin_rad": endpoint_soft_margin,
        "bridge_endpoint_soft_margin_joint": endpoint_soft_joint,
        "bridge_soft_margin_regression_rad": soft_margin_regression,
        "bridge_start_collision_distance_m": start_collision_distance,
        "bridge_start_collision_pair": start_collision_pair,
        "bridge_endpoint_collision_distance_m": endpoint_collision_distance,
        "bridge_endpoint_collision_pair": endpoint_collision_pair,
        "bridge_collision_regression_m": collision_regression,
        "bridge_collision_penetration_regression_m": (
            collision_penetration_regression
        ),
        **clearance,
        "endpoint_q_error_rad": endpoint_q_error,
        "endpoint_qdot_error_radps": endpoint_v_error,
        "endpoint_qddot_error_radps2": endpoint_a_error,
        "score": score,
    }


def _audit_candidates(
    candidates: list[Candidate],
    motions: list[dict[str, Any]],
    joint_names: list[str],
    audit: MujocoAudit,
    bridge_steps: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    candidate_rows = []
    splice_results: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        static = audit.kinematics(candidate.ready_q)
        static_score = _workspace_penalty(
            np.asarray(static["racket_position_torso_m"])
        )
        outside_distance = _workspace_outside_distance(
            np.asarray(static["racket_position_torso_m"])
        )
        if static["minimum_collision_distance_m"] < -1e-4:
            static_score += 1000.0
        if static["soft_joint_margin_rad"] < 0.02:
            static_score += 1000.0
        if outside_distance > 1e-6:
            static_score += 100.0 * outside_distance
        candidate_rows.append(
            {
                "name": candidate.name,
                "family": candidate.family,
                "right_blend": candidate.right_blend,
                "right_elbow_rad": candidate.right_elbow_rad,
                "right_elbow_deg": math.degrees(candidate.right_elbow_rad),
                "left_elbow_rad": candidate.left_elbow_rad,
                "left_elbow_deg": math.degrees(candidate.left_elbow_rad),
                "ready_joint_positions": dict(zip(joint_names, candidate.ready_q)),
                "workspace_penalty": _workspace_penalty(
                    np.asarray(static["racket_position_torso_m"])
                ),
                "workspace_outside_distance_m": outside_distance,
                "static_score": static_score,
                **static,
            }
        )
        per_motion = {}
        for motion in motions:
            # The frozen clip has only 30 pre-hit frames. Keep at least 20
            # original strike frames, hence the only legal search is 0..10.
            rows = [
                _bridge_metrics(
                    candidate,
                    motion,
                    splice,
                    joint_names,
                    audit,
                    bridge_steps,
                )
                for splice in range(0, motion["hit"] - 19)
            ]
            rows.sort(key=lambda row: row["score"])
            # Do not shorten a validated strike core for a numerically trivial
            # bridge-score gain. Among candidates within 0.05 of the optimum,
            # preserve the earliest source frame and therefore the most
            # model_900 in-distribution context.
            near_optimal = [
                row for row in rows if row["score"] <= rows[0]["score"] + 0.05
            ]
            selected = min(
                near_optimal, key=lambda row: row["splice_frame_old"]
            )
            per_motion[motion["row"]["episode_id"]] = {
                "selected": selected,
                "all_candidates": rows,
            }
        splice_results[candidate.name] = per_motion

    for row in candidate_rows:
        selected = [
            result["selected"]
            for result in splice_results[row["name"]].values()
        ]
        row["mean_selected_bridge_score"] = float(
            np.mean([item["score"] for item in selected])
        )
        row["minimum_bridge_clearance_m"] = float(
            min(item["minimum_collision_distance_m"] for item in selected)
        )
        row["minimum_bridge_soft_margin_rad"] = float(
            min(item["bridge_minimum_soft_margin_rad"] for item in selected)
        )
        row["combined_score"] = (
            row["static_score"] + row["mean_selected_bridge_score"]
        )
    candidate_rows.sort(key=lambda row: row["combined_score"])
    return candidate_rows, splice_results


def _family_winners(candidate_rows: list[dict[str, Any]]) -> dict[str, str]:
    winners = {}
    for family in ("A", "B", "C"):
        rows = [row for row in candidate_rows if row["family"] == family]
        eligible = [
            row
            for row in rows
            if row["minimum_collision_distance_m"] >= -1e-4
            and row["soft_joint_margin_rad"] >= 0.02
            and row["workspace_outside_distance_m"] <= 1e-6
        ]
        pool = eligible if eligible else rows
        if pool:
            winners[family] = min(pool, key=lambda row: row["combined_score"])[
                "name"
            ]
    return winners


def _hard_gate(
    candidate_row: dict[str, Any], splices: dict[str, dict[str, Any]]
) -> tuple[bool, list[str]]:
    failures = []
    if candidate_row["soft_joint_margin_rad"] < 0.02:
        failures.append("ready pose is within 0.02 rad of a soft joint limit")
    if candidate_row["minimum_collision_distance_m"] < -1e-4:
        failures.append("ready pose has geometric penetration")
    if candidate_row["workspace_outside_distance_m"] > 1e-6:
        failures.append("racket center is outside the compact torso-local READY box")
    for episode_id, result in splices.items():
        selected = result["selected"]
        if selected["bridge_collision_penetration_regression_m"] < -0.002:
            failures.append(
                f"{episode_id}: bridge adds more than 2 mm collision penetration "
                "relative to its frozen endpoints"
            )
        if selected["bridge_soft_margin_regression_rad"] < -0.005:
            failures.append(
                f"{episode_id}: bridge loses more than 0.005 rad soft-limit "
                "margin relative to its frozen endpoints"
            )
        if selected["new_hit_frame"] < 20:
            failures.append(f"{episode_id}: fewer than 20 strike-core frames remain")
        if max(
            selected["endpoint_q_error_rad"],
            selected["endpoint_qdot_error_radps"],
            selected["endpoint_qddot_error_radps2"],
        ) > 1e-8:
            failures.append(f"{episode_id}: quintic endpoint contract mismatch")
    return not failures, failures


def _export_motion_package(
    output_dir: Path,
    source_manifest: dict[str, Any],
    source_manifest_path: Path,
    motions: list[dict[str, Any]],
    candidate: Candidate,
    candidate_row: dict[str, Any],
    splices: dict[str, dict[str, Any]],
    joint_names: list[str],
    bridge_steps: int,
) -> Path:
    package = output_dir / f"motion_package_{candidate.family.lower()}"
    motion_dir = package / "motion_npz"
    if package.exists():
        shutil.rmtree(package)
    motion_dir.mkdir(parents=True)

    manifest = copy.deepcopy(source_manifest)
    manifest["manifest_name"] = (
        f"{source_manifest['manifest_name']}_v27_{candidate.family.lower()}_bent_ready"
    )
    manifest["status"] = "v27_offline_candidate_not_training_approved"
    manifest["source_manifest"] = str(source_manifest_path)
    manifest["v27_bent_ready_contract"] = {
        "version": 1,
        "candidate": candidate.name,
        "candidate_family": candidate.family,
        "joint_names": joint_names,
        "ready_joint_positions": dict(zip(joint_names, candidate.ready_q)),
        "bridge_steps": bridge_steps,
        "bridge_frequency_hz": 50,
        "bridge_method": "quintic_hermite_q_qdot_qddot",
        "source_strike_core_preserved": True,
        "source_zero_velocity_tail_preserved": True,
        "runtime_return_target": "same_bent_ready_joint_positions",
        "v25_v26_stage_a_contract_unchanged": True,
        "training_approved": False,
        "candidate_audit": candidate_row,
    }
    rows_by_id = {row["episode_id"]: row for row in manifest["motions"]}
    for motion in motions:
        episode_id = motion["row"]["episode_id"]
        selected = splices[episode_id]["selected"]
        splice = int(selected["splice_frame_old"])
        destination = motion_dir / motion["path"].name
        arrays = {
            key: value if value.ndim == 0 else value[splice:].copy()
            for key, value in motion["arrays"].items()
        }
        # Scalar metadata arrays such as fps/mass must not be sliced.
        for key, value in motion["arrays"].items():
            if value.shape[0] != motion["arrays"]["joint_pos"].shape[0]:
                arrays[key] = value.copy()
        np.savez_compressed(destination, **arrays)
        row = rows_by_id[episode_id]
        row["motion_npz"] = str(destination.resolve())
        row["joint_pos_shape"] = list(arrays["joint_pos"].shape)
        row["body_pos_w_shape"] = list(arrays["body_pos_w"].shape)
        row["hit_event"]["motion_hit_frame"] = int(selected["new_hit_frame"])
        old_contract = row["strike_only_contract"]
        old_contract["hit_frame"] = int(selected["new_hit_frame"])
        old_contract["last_source_frame_kept"] = int(selected["new_hit_frame"])
        old_contract["zero_velocity_tail_start"] = int(selected["new_hit_frame"] + 1)
        row["v27_splice_contract"] = {
            "source_motion_npz": str(motion["path"]),
            "source_motion_sha256": _sha256(motion["path"]),
            "source_splice_frame": splice,
            "source_hit_frame": motion["hit"],
            "candidate_hit_frame": int(selected["new_hit_frame"]),
            "source_frames_after_splice_copied_verbatim": True,
            "bridge_runtime_metadata": selected,
        }
    manifest_path = package / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bridge-steps", type=int, default=50)
    parser.add_argument(
        "--export-family",
        choices=("B", "C"),
        help="Export the winning family only if every offline hard gate passes.",
    )
    args = parser.parse_args()
    if args.bridge_steps not in (30, 40, 50):
        raise ValueError("V27 bridge scan supports 30/40/50 control steps")

    manifest_path = args.manifest.resolve()
    metadata_path = args.metadata.resolve()
    mjcf_path = args.mjcf.resolve()
    output_dir = args.output_dir.resolve()
    manifest, joint_names, motions = _load_motions(manifest_path, metadata_path)
    base_ready = _ready_pose(joint_names)
    robust_center = _robust_pre_hit_center(motions, joint_names)
    candidates = _candidate_grid(base_ready, robust_center, joint_names)
    audit = MujocoAudit(mjcf_path, joint_names)
    candidate_rows, splice_results = _audit_candidates(
        candidates, motions, joint_names, audit, args.bridge_steps
    )
    winners = _family_winners(candidate_rows)
    by_name = {candidate.name: candidate for candidate in candidates}
    row_by_name = {row["name"]: row for row in candidate_rows}
    gates = {}
    for family, name in winners.items():
        passed, failures = _hard_gate(row_by_name[name], splice_results[name])
        gates[family] = {"passed": passed, "failures": failures}

    report = {
        "contract": "V27BentReadyOfflineAuditV1",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "articulation_metadata": str(metadata_path),
        "articulation_metadata_sha256": _sha256(metadata_path),
        "mujoco_model": str(mjcf_path),
        "mujoco_model_sha256": _sha256(mjcf_path),
        "policy_frequency_hz": 50,
        "physics_frequency_note": (
            "Offline geometry only. Runtime must retain the frozen V25/V26 "
            "200 Hz physics / 50 Hz policy contract; do not switch to 500 Hz."
        ),
        "source_clip_constraint": {
            "frames": 39,
            "hit_frame": 30,
            "legal_splice_frames": [0, 10],
            "reason": (
                "The active strike-only clips contain only hit-30..tail. "
                "Prompt range hit-35..hit-20 is unavailable; preserving at "
                "least 20 source strike frames limits search to old 0..10."
            ),
        },
        "bridge_steps": args.bridge_steps,
        "robust_pre_hit_center": dict(zip(joint_names, robust_center)),
        "family_winners": winners,
        "family_hard_gates": gates,
        "candidate_ranking": candidate_rows,
        "splice_results": splice_results,
        "left_arm_note": (
            "A3 elbow joint radians are not physical flexion angles. Legacy "
            "left_elbow_joint=0.8 rad is about 35.4 deg flexed from straight, "
            "already inside the requested 20--40 deg range. C uses a minimal "
            "0.75 rad ablation (about 38.3 deg flexion) and must beat B on "
            "data; it is not assumed superior."
        ),
        "elbow_zero_definition": {
            "right_elbow_joint_zero_inner_angle_deg": 98.733,
            "right_elbow_joint_zero_flexion_from_straight_deg": 81.267,
            "positive_joint_direction": "straightens until approximately 1.42 rad",
            "candidate_joint_values_rad": [0.05, 0.20, 0.35],
            "candidate_physical_flexion_approx_deg": [78.4, 69.8, 61.2],
        },
        "go_no_go": (
            "OFFLINE_GO"
            if all(gates.get(family, {}).get("passed", False) for family in ("B", "C"))
            else "NO_GO"
        ),
    }
    report_path = output_dir / f"offline_audit_bridge{args.bridge_steps}.json"
    _write_json(report_path, report)

    exported = None
    if args.export_family:
        family = args.export_family
        winner_name = winners[family]
        if not gates[family]["passed"]:
            raise RuntimeError(
                f"Refusing to export family {family}: {gates[family]['failures']}"
            )
        exported = _export_motion_package(
            output_dir,
            manifest,
            manifest_path,
            motions,
            by_name[winner_name],
            row_by_name[winner_name],
            splice_results[winner_name],
            joint_names,
            args.bridge_steps,
        )
    summary = {
        "report": str(report_path),
        "go_no_go": report["go_no_go"],
        "family_winners": winners,
        "family_hard_gates": gates,
        "exported_manifest": str(exported) if exported else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
