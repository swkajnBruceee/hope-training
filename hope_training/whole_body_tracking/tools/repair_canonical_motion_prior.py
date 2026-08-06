#!/usr/bin/env python3
"""Build a deterministic, low-dimensional P4B motion-prior repair candidate.

Run this tool with the AimSim Python environment because it owns the MuJoCo
bindings used by the deployment model:

  .venv/aimsim/bin/python tools/repair_canonical_motion_prior.py --motion-id 0

The tool does not create a training-approved motion package.  It first moves
the structurally violating waist references inside the Isaac 0.9 soft range,
then compensates the policy wrist-offset TCP with right-arm/waist-yaw IK, and
finally projects the per-frame solution onto a Bernstein basis.  The result is
an offline repair candidate plus a complete geometry/limit audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from build_v27_bent_ready_motion import MujocoAudit


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "eval_outputs/strike_goal_p3/canonical_motion_prior_v1/manifest.json"
DEFAULT_METADATA = ROOT / "docs/a3_articulation_metadata.json"
DEFAULT_MJCF = (
    ROOT.parents[1]
    / "agibot/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml"
)
DEFAULT_OUTPUT = ROOT / "eval_outputs/strike_goal_p4/p4b_repair_candidates"

ADAPTER_CONTRACT = "strike_trajectory_bernstein_adapter/v1"
A3_MOUNT_OFFSET = np.array(
    (0.210211399202899, 0.0320784994676765, 0.0320358706296689), dtype=np.float64
)
WAIST_REPAIR_JOINTS = ("waist_roll_joint", "waist_pitch_joint")
RIGHT_ARM = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
IK_JOINTS = ("waist_yaw_joint", *RIGHT_ARM)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    def ready(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): ready(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [ready(child) for child in item]
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        return item

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ready(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _bernstein_basis(num_frames: int, degree: int) -> np.ndarray:
    phase = np.linspace(0.0, 1.0, num_frames, dtype=np.float64)
    return np.stack(
        [math.comb(degree, k) * phase**k * (1.0 - phase) ** (degree - k) for k in range(degree + 1)],
        axis=1,
    )


def _fit_deformation(deformation: np.ndarray, degree: int, ridge: float = 1.0e-8) -> tuple[np.ndarray, np.ndarray]:
    basis = _bernstein_basis(deformation.shape[0], degree)
    coefficients = np.linalg.solve(
        basis.T @ basis + ridge * np.eye(basis.shape[1]),
        basis.T @ deformation,
    )
    return coefficients, basis @ coefficients


def _finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    derivative = np.empty_like(values)
    derivative[0] = (values[1] - values[0]) * fps
    derivative[-1] = (values[-1] - values[-2]) * fps
    derivative[1:-1] = (values[2:] - values[:-2]) * (0.5 * fps)
    return derivative


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=np.float64)


class PolicyTcpKinematics:
    """MuJoCo FK/Jacobian for the exact wrist-offset policy TCP contract."""

    def __init__(self, model_path: Path, joint_names: list[str]):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.joint_names = joint_names
        self.index = {name: i for i, name in enumerate(joint_names)}
        self.qpos_address: dict[str, int] = {}
        self.dof_address: dict[str, int] = {}
        self.hard_ranges: dict[str, tuple[float, float]] = {}
        self.soft_ranges: dict[str, tuple[float, float]] = {}
        for name in joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"MuJoCo model is missing joint {name}")
            self.qpos_address[name] = int(self.model.jnt_qposadr[joint_id])
            self.dof_address[name] = int(self.model.jnt_dofadr[joint_id])
            lower, upper = (float(value) for value in self.model.jnt_range[joint_id])
            center = 0.5 * (lower + upper)
            soft_half = 0.45 * (upper - lower)
            self.hard_ranges[name] = (lower, upper)
            self.soft_ranges[name] = (center - soft_half, center + soft_half)
        self.wrist_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_Link"
        )
        if self.wrist_body_id < 0:
            raise ValueError("MuJoCo model is missing right_wrist_yaw_Link")

    def set_pose(self, q: np.ndarray) -> None:
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[0:3] = (0.0, 0.0, 1.04)
        self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        for name, index in self.index.items():
            self.data.qpos[self.qpos_address[name]] = float(q[index])
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def state(self, q: np.ndarray, jacobian_joints: tuple[str, ...] = ()) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        self.set_pose(q)
        wrist_position = self.data.xpos[self.wrist_body_id]
        wrist_rotation = self.data.xmat[self.wrist_body_id].reshape(3, 3)
        tcp_position = wrist_position + wrist_rotation @ A3_MOUNT_OFFSET
        # Project geometry and policy use the wrist/racket local +Y face normal.
        normal = wrist_rotation[:, 1].copy()
        if not jacobian_joints:
            return tcp_position.copy(), normal, None
        jac_position = np.zeros((3, self.model.nv), dtype=np.float64)
        jac_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jac(
            self.model,
            self.data,
            jac_position,
            jac_rotation,
            tcp_position,
            self.wrist_body_id,
        )
        dofs = [self.dof_address[name] for name in jacobian_joints]
        normal_jacobian = -_skew(normal) @ jac_rotation[:, dofs]
        return tcp_position.copy(), normal, np.vstack((jac_position[:, dofs], normal_jacobian))

    def soft_margin_detail(self, q: np.ndarray) -> tuple[float, str]:
        rows = []
        for name, index in self.index.items():
            lower, upper = self.soft_ranges[name]
            rows.append((min(float(q[index]) - lower, upper - float(q[index])), name))
        return min(rows)

    def soft_margin_for(self, q: np.ndarray, names: tuple[str, ...]) -> tuple[float, str]:
        rows = []
        for name in names:
            index = self.index[name]
            lower, upper = self.soft_ranges[name]
            rows.append((min(float(q[index]) - lower, upper - float(q[index])), name))
        return min(rows)

    def project_soft(self, q: np.ndarray, names: tuple[str, ...], margin: float) -> None:
        for name in names:
            index = self.index[name]
            lower, upper = self.soft_ranges[name]
            q[index] = np.clip(q[index], lower + margin, upper - margin)


def _repair_frame(
    kinematics: PolicyTcpKinematics,
    original_q: np.ndarray,
    waist_repaired_q: np.ndarray,
    *,
    joint_margin: float,
    normal_length_scale_m: float,
    max_iterations: int,
) -> tuple[np.ndarray, dict[str, float]]:
    target_position, target_normal, _ = kinematics.state(original_q)
    q = waist_repaired_q.copy()
    compensation_indices = [kinematics.index[name] for name in IK_JOINTS]
    initial_q = q.copy()
    for _ in range(max_iterations):
        position, normal, jacobian = kinematics.state(q, IK_JOINTS)
        assert jacobian is not None
        residual = np.concatenate(
            (target_position - position, normal_length_scale_m * (target_normal - normal))
        )
        if np.linalg.norm(target_position - position) < 2.0e-5 and np.linalg.norm(target_normal - normal) < 2.0e-4:
            break
        weighted_jacobian = jacobian.copy()
        weighted_jacobian[3:] *= normal_length_scale_m
        damping = 2.0e-4
        delta = weighted_jacobian.T @ np.linalg.solve(
            weighted_jacobian @ weighted_jacobian.T + damping * np.eye(6), residual
        )
        delta = np.clip(delta, -0.05, 0.05)
        q[compensation_indices] += delta
        kinematics.project_soft(q, IK_JOINTS, joint_margin)
        # The structural waist repair is immutable during TCP compensation.
        for name in WAIST_REPAIR_JOINTS:
            q[kinematics.index[name]] = waist_repaired_q[kinematics.index[name]]
    final_position, final_normal, _ = kinematics.state(q)
    return q, {
        "position_error_m": float(np.linalg.norm(final_position - target_position)),
        "normal_error_deg": float(
            np.degrees(np.arccos(np.clip(np.dot(final_normal, target_normal), -1.0, 1.0)))
        ),
        "ik_compensation_linf_rad": float(np.max(np.abs(q - initial_q))),
    }


def _trajectory_task_states(
    kinematics: PolicyTcpKinematics,
    trajectory: np.ndarray,
    joint_velocity: np.ndarray,
) -> dict[str, np.ndarray]:
    positions, normals, velocities = [], [], []
    all_joints = tuple(kinematics.joint_names)
    for q, qd in zip(trajectory, joint_velocity, strict=True):
        position, normal, jacobian = kinematics.state(q, all_joints)
        assert jacobian is not None
        positions.append(position)
        normals.append(normal)
        velocities.append(jacobian[:3] @ qd)
    return {
        "position": np.asarray(positions),
        "normal": np.asarray(normals),
        "velocity": np.asarray(velocities),
    }


def _task_delta(original: dict[str, np.ndarray], candidate: dict[str, np.ndarray], hit: int) -> dict[str, Any]:
    position_error = np.linalg.norm(candidate["position"] - original["position"], axis=1)
    normal_dot = np.sum(candidate["normal"] * original["normal"], axis=1)
    normal_error = np.degrees(np.arccos(np.clip(normal_dot, -1.0, 1.0)))
    velocity_error = np.linalg.norm(candidate["velocity"] - original["velocity"], axis=1)
    return {
        "hit_position_drift_m": float(position_error[hit]),
        "hit_normal_drift_deg": float(normal_error[hit]),
        "hit_velocity_drift_mps": float(velocity_error[hit]),
        "max_position_drift_m": float(position_error.max()),
        "max_normal_drift_deg": float(normal_error.max()),
        "max_velocity_drift_mps": float(velocity_error.max()),
    }


def _collision_audit(audit: MujocoAudit, trajectory: np.ndarray) -> dict[str, Any]:
    rows = []
    for frame, q in enumerate(trajectory):
        audit.set_pose(q)
        distance, pair = audit.minimum_collision_distance()
        rows.append((float(distance), frame, pair))
    distance, frame, pair = min(rows, key=lambda item: item[0])
    return {"minimum_distance_m": distance, "frame": frame, "pair": pair}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--motion-id", type=int, default=0)
    parser.add_argument("--basis-degree", type=int, default=10)
    parser.add_argument("--soft-margin-rad", type=float, default=0.02)
    parser.add_argument(
        "--global-soft-margin-rad",
        type=float,
        default=None,
        help="minimum margin for every joint; defaults to --soft-margin-rad for legacy P4B behavior",
    )
    parser.add_argument("--basis-guard-margin-rad", type=float, default=0.003)
    parser.add_argument("--normal-length-scale-m", type=float, default=0.15)
    parser.add_argument("--max-ik-iterations", type=int, default=30)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    matching = [row for row in manifest["motions"] if int(row["motion_id"]) == args.motion_id]
    if len(matching) != 1:
        raise ValueError(f"Expected one motion_id={args.motion_id}, got {len(matching)}")
    row = matching[0]
    motion_path = args.manifest.parent / row["canonical_motion_npz"]
    with np.load(motion_path) as archive:
        original_q = np.asarray(archive["joint_pos"], dtype=np.float64)
        original_qd = np.asarray(archive["joint_vel"], dtype=np.float64)
        fps = float(np.asarray(archive["fps"]).reshape(-1)[0])
    joint_names = list(metadata["joint_names"])
    hit = int(row.get("hit_frame", row.get("hit_event", {}).get("motion_hit_frame", 30)))
    kinematics = PolicyTcpKinematics(args.mjcf, joint_names)
    structural_margin = args.soft_margin_rad + args.basis_guard_margin_rad
    global_soft_margin = (
        args.soft_margin_rad if args.global_soft_margin_rad is None else args.global_soft_margin_rad
    )
    if global_soft_margin < 0.0:
        raise ValueError("global-soft-margin-rad must be non-negative")

    waist_target = original_q.copy()
    structurally_repaired = []
    for name in WAIST_REPAIR_JOINTS:
        index = kinematics.index[name]
        lower, upper = kinematics.soft_ranges[name]
        repaired = np.clip(original_q[:, index], lower + structural_margin, upper - structural_margin)
        if not np.allclose(repaired, original_q[:, index]):
            waist_target[:, index] = repaired
            structurally_repaired.append(name)

    direct_q = np.empty_like(original_q)
    frame_ik = []
    for frame in range(original_q.shape[0]):
        direct_q[frame], result = _repair_frame(
            kinematics,
            original_q[frame],
            waist_target[frame],
            joint_margin=args.soft_margin_rad,
            normal_length_scale_m=args.normal_length_scale_m,
            max_iterations=args.max_ik_iterations,
        )
        frame_ik.append({"frame": frame, **result})

    coefficients, fitted_delta = _fit_deformation(direct_q - original_q, args.basis_degree)
    fitted_q = original_q + fitted_delta
    # One final conservative projection only on the structural waist joints.
    # If this changes the fitted basis materially, the candidate is rejected by
    # the representation-consistency gate below instead of silently approved.
    projected_q = fitted_q.copy()
    for q in projected_q:
        kinematics.project_soft(q, WAIST_REPAIR_JOINTS, args.soft_margin_rad)

    nominal_position_derivative = _finite_difference(original_q, fps)
    derivative_denominator = float(np.sum(nominal_position_derivative**2))
    if derivative_denominator < 1.0e-12:
        raise RuntimeError("nominal trajectory has no measurable position derivative")
    velocity_time_scale = float(np.sum(original_qd * nominal_position_derivative) / derivative_denominator)

    def deformed_velocity(candidate_q: np.ndarray) -> np.ndarray:
        return original_qd + velocity_time_scale * _finite_difference(candidate_q - original_q, fps)

    direct_qd = deformed_velocity(direct_q)
    fitted_qd = deformed_velocity(fitted_q)
    projected_qd = deformed_velocity(projected_q)

    original_states = _trajectory_task_states(kinematics, original_q, original_qd)
    direct_states = _trajectory_task_states(kinematics, direct_q, direct_qd)
    fitted_states = _trajectory_task_states(kinematics, fitted_q, fitted_qd)
    projected_states = _trajectory_task_states(kinematics, projected_q, projected_qd)

    min_original = min(kinematics.soft_margin_detail(q) for q in original_q)
    min_direct = min(kinematics.soft_margin_detail(q) for q in direct_q)
    min_fitted = min(kinematics.soft_margin_detail(q) for q in fitted_q)
    min_projected = min(kinematics.soft_margin_detail(q) for q in projected_q)
    min_projected_waist = min(
        kinematics.soft_margin_for(q, WAIST_REPAIR_JOINTS) for q in projected_q
    )
    collision = MujocoAudit(args.mjcf, joint_names)
    collision_original = _collision_audit(collision, original_q)
    collision_projected = _collision_audit(collision, projected_q)
    projected_minus_basis_linf = float(np.max(np.abs(projected_q - fitted_q)))

    output_dir = args.output_dir / f"motion_{args.motion_id:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "repair_candidate.npz"
    np.savez_compressed(
        candidate_path,
        contract_version_utf8=np.frombuffer(ADAPTER_CONTRACT.encode("utf-8"), dtype=np.uint8),
        fps=np.array([fps], dtype=np.float64),
        hit_frame=np.array([hit], dtype=np.int64),
        joint_names_utf8=np.frombuffer("\n".join(joint_names).encode("utf-8"), dtype=np.uint8),
        nominal_joint_pos=original_q.astype(np.float32),
        direct_repair_joint_pos=direct_q.astype(np.float32),
        basis_coefficients=coefficients.astype(np.float32),
        basis_fitted_joint_pos=fitted_q.astype(np.float32),
        projected_joint_pos=projected_q.astype(np.float32),
        projected_joint_vel=projected_qd.astype(np.float32),
        source_velocity_time_scale=np.asarray((velocity_time_scale,), dtype=np.float64),
    )

    direct_delta = _task_delta(original_states, direct_states, hit)
    fitted_delta_report = _task_delta(original_states, fitted_states, hit)
    projected_delta = _task_delta(original_states, projected_states, hit)
    gates = {
        "positive_global_soft_margin": min_projected[0] >= global_soft_margin - 1.0e-6,
        "waist_inner_soft_margin": min_projected_waist[0] >= args.soft_margin_rad - 1.0e-6,
        "basis_requires_no_projection": projected_minus_basis_linf <= 1.0e-5,
        "hit_position_drift_le_3mm": projected_delta["hit_position_drift_m"] <= 0.003,
        "hit_normal_drift_le_2deg": projected_delta["hit_normal_drift_deg"] <= 2.0,
        "hit_velocity_drift_le_0p2mps": projected_delta["hit_velocity_drift_mps"] <= 0.2,
        "collision_distance_nonnegative": collision_projected["minimum_distance_m"] >= 0.0,
        "collision_clearance_loss_le_1mm": (
            collision_projected["minimum_distance_m"] >= collision_original["minimum_distance_m"] - 0.001
        ),
    }
    report = {
        "schema_version": "p4b_deterministic_prior_repair/v1",
        "motion_id": args.motion_id,
        "episode_id": row["episode_id"],
        "training_started": False,
        "ppo_allowed": False,
        "training_approved": False,
        "source_manifest": args.manifest,
        "source_manifest_sha256": _sha256(args.manifest),
        "source_motion": motion_path,
        "source_motion_sha256": _sha256(motion_path),
        "model": args.mjcf,
        "model_sha256": _sha256(args.mjcf),
        "candidate_npz": candidate_path,
        "candidate_sha256": _sha256(candidate_path),
        "adapter_contract": {
            "version": ADAPTER_CONTRACT,
            "basis": "Bernstein partition-of-unity",
            "degree": args.basis_degree,
            "coefficients": list(coefficients.shape),
            "frames": int(original_q.shape[0]),
            "controlled_joints": [
                name for index, name in enumerate(joint_names) if np.max(np.abs(coefficients[:, index])) > 1.0e-8
            ],
        },
        "tcp_contract": {
            "body": "right_wrist_yaw_Link",
            "mount_offset_local_m": A3_MOUNT_OFFSET,
            "normal_axis": "+Y",
            "note": "Same policy wrist-offset point used by RacketTargetCommand; not the MuJoCo racket geom origin.",
        },
        "repair": {
            "structurally_repaired_joints": structurally_repaired,
            "ik_compensation_joints": IK_JOINTS,
            "requested_soft_margin_rad": args.soft_margin_rad,
            "requested_global_soft_margin_rad": global_soft_margin,
            "basis_guard_margin_rad": args.basis_guard_margin_rad,
            "structural_repair_margin_rad": structural_margin,
            "normal_length_scale_m": args.normal_length_scale_m,
            "max_ik_iterations": args.max_ik_iterations,
            "source_velocity_time_scale_vs_50hz_position_derivative": velocity_time_scale,
            "velocity_repair_contract": "source_joint_vel + time_scale * d(delta_q)/dt",
            "maximum_direct_frame_position_error_m": max(item["position_error_m"] for item in frame_ik),
            "maximum_direct_frame_normal_error_deg": max(item["normal_error_deg"] for item in frame_ik),
            "maximum_direct_ik_compensation_linf_rad": max(item["ik_compensation_linf_rad"] for item in frame_ik),
        },
        "soft_limit_margin": {
            "nominal_min_rad": min_original[0],
            "nominal_min_joint": min_original[1],
            "direct_min_rad": min_direct[0],
            "direct_min_joint": min_direct[1],
            "basis_fitted_min_rad": min_fitted[0],
            "basis_fitted_min_joint": min_fitted[1],
            "projected_min_rad": min_projected[0],
            "projected_min_joint": min_projected[1],
            "projected_waist_min_rad": min_projected_waist[0],
            "projected_waist_min_joint": min_projected_waist[1],
        },
        "task_state_drift": {
            "direct_repair": direct_delta,
            "basis_fitted": fitted_delta_report,
            "projected": projected_delta,
        },
        "representation": {
            "projection_minus_basis_linf_rad": projected_minus_basis_linf,
            "basis_fit_to_direct_linf_rad": float(np.max(np.abs(fitted_q - direct_q))),
        },
        "collision": {"nominal": collision_original, "projected": collision_projected},
        "gates": gates,
        "all_offline_gates_pass": all(gates.values()),
        "next_required_gate": "formal P1 Isaac dynamic replay of a regenerated, internally consistent motion package",
    }
    report_path = output_dir / "repair_audit.json"
    _write_json(report_path, report)
    print(json.dumps({"gates": gates, "task_state_drift": projected_delta}, indent=2))
    print(report_path)


if __name__ == "__main__":
    main()
