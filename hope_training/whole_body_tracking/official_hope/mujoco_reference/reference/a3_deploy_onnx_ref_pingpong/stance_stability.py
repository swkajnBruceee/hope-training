"""Quantitative A3 nominal-stance stability experiments.

This module is deliberately independent of the learned policy.  It uses the
same MuJoCo MJCF and actuator limits as the existing reference runner, but
commands a fixed joint target through plain PD.  It is therefore suitable for
answering the geometry/controller question before changing a policy or reward.

Coordinate convention is verified from the project model: +x is the robot's
forward direction and +y is the robot's left direction.  ``fore_aft_m`` is the
*relative* left/right foot separation in x.  To preserve the foot midpoint,
the lead foot is moved +fore_aft/2 and the trailing foot -fore_aft/2.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .joint_order import JOINT_NAMES, NUM_JOINTS


LEG_NAMES = (
    "hip_pitch",
    "hip_roll",
    "hip_yaw",
    "knee",
    "ankle_pitch",
    "ankle_roll",
)
LEFT_LEG_IDX = np.array([19, 20, 21, 22, 23, 24], dtype=int)
RIGHT_LEG_IDX = np.array([25, 26, 27, 28, 29, 30], dtype=int)
WAIST_IDX = np.array([0, 1, 2], dtype=int)
ARM_IDX = np.arange(5, 19, dtype=int)
LEG_IDX = np.arange(19, 31, dtype=int)


@dataclass(frozen=True)
class StanceConfig:
    """Human-readable stance parameters; angles are in degrees at the API."""

    hip_flexion_deg: float = 0.0
    knee_flexion_deg: float = 0.0
    torso_pitch_deg: float = 0.0
    stance_width_scale: float = 1.0
    stance_width_m: float | None = None
    fore_aft_m: float = 0.0
    lead_leg: str = "none"
    pelvis_height_offset_m: float | None = None

    def __post_init__(self) -> None:
        if self.lead_leg not in ("none", "left", "right"):
            raise ValueError("lead_leg must be 'none', 'left', or 'right'")
        if self.fore_aft_m < 0.0:
            raise ValueError("fore_aft_m must be non-negative")
        if self.stance_width_scale <= 0.0:
            raise ValueError("stance_width_scale must be positive")

    @property
    def label(self) -> str:
        lead = "parallel" if self.fore_aft_m == 0.0 else f"{self.lead_leg}_lead"
        return (
            f"hip{self.hip_flexion_deg:g}_knee{self.knee_flexion_deg:g}_"
            f"torso{self.torso_pitch_deg:g}_w{self.stance_width_scale:g}_"
            f"fa{self.fore_aft_m:g}_{lead}"
        )


@dataclass
class GeneratedStance:
    config: StanceConfig
    q: np.ndarray
    root_qpos: np.ndarray
    left_foot_target: np.ndarray
    right_foot_target: np.ndarray
    pelvis_height_m: float
    width_m: float
    valid: bool
    diagnostics: dict[str, float | str | bool]


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=float,
    )


def _orientation_error(target: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Small-angle rotation vector taking actual orientation to target."""
    q = _quat_mul(target, _quat_conjugate(actual))
    if q[0] < 0.0:
        q = -q
    norm = float(np.linalg.norm(q[1:]))
    if norm < 1.0e-10:
        return 2.0 * q[1:]
    angle = 2.0 * math.atan2(norm, max(float(q[0]), 1.0e-12))
    return q[1:] / norm * angle


def _smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


class StanceGenerator:
    """Generate leg joint targets with a model-backed numerical IK."""

    def __init__(self, model, data=None, *, max_iterations: int = 160):
        import mujoco

        self.mj = mujoco
        self.model = model
        self.data = data or mujoco.MjData(model)
        self.max_iterations = int(max_iterations)
        self.root_jid = self._joint_id("pelvis_free_joint")
        self.root_qadr = int(model.jnt_qposadr[self.root_jid])
        self.left_foot_bid = self._body_id("left_ankle_roll_Link")
        self.right_foot_bid = self._body_id("right_ankle_roll_Link")
        self.pelvis_bid = self._body_id("pelvis_link")
        self.leg_qadr = np.array(
            [
                self._joint_qadr(f"left_{name}_joint")
                for name in LEG_NAMES
            ]
            + [self._joint_qadr(f"right_{name}_joint") for name in LEG_NAMES],
            dtype=int,
        )
        self.leg_jids = np.array(
            [
                self._joint_id(f"left_{name}_joint")
                for name in LEG_NAMES
            ]
            + [self._joint_id(f"right_{name}_joint") for name in LEG_NAMES],
            dtype=int,
        )
        self.q_min = model.jnt_range[self.leg_jids, 0].copy()
        self.q_max = model.jnt_range[self.leg_jids, 1].copy()
        self._reset_keyframe()
        self.baseline_qpos = self.data.qpos.copy()
        self.baseline_qvel = self.data.qvel.copy()
        self.baseline_left_foot = self.data.xpos[self.left_foot_bid].copy()
        self.baseline_right_foot = self.data.xpos[self.right_foot_bid].copy()
        self.baseline_left_quat = self.data.xquat[self.left_foot_bid].copy()
        self.baseline_right_quat = self.data.xquat[self.right_foot_bid].copy()
        self.baseline_pelvis = self.data.xpos[self.pelvis_bid].copy()
        self.baseline_width_m = float(abs(self.baseline_left_foot[1] - self.baseline_right_foot[1]))
        self.baseline_root_height_m = float(self.baseline_qpos[self.root_qadr + 2])

    def _joint_id(self, name: str) -> int:
        jid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"joint not found in model: {name}")
        return int(jid)

    def _joint_qadr(self, name: str) -> int:
        return int(self.model.jnt_qposadr[self._joint_id(name)])

    def _body_id(self, name: str) -> int:
        bid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"body not found in model: {name}")
        return int(bid)

    def _reset_keyframe(self) -> None:
        if self.model.nkey:
            self.mj.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            self.mj.mj_resetData(self.model, self.data)
        self.mj.mj_forward(self.model, self.data)

    def _leg_residual(self, q: np.ndarray, indices: np.ndarray, target_pos: np.ndarray,
                      target_quat: np.ndarray) -> np.ndarray:
        self.data.qpos[self.leg_qadr] = q
        self.mj.mj_forward(self.model, self.data)
        bid = self.left_foot_bid if indices[0] == 0 else self.right_foot_bid
        return np.concatenate((
            target_pos - self.data.xpos[bid],
            _orientation_error(target_quat, self.data.xquat[bid]),
        ))

    def _solve_leg(self, q: np.ndarray, side: str, target_pos: np.ndarray,
                   target_quat: np.ndarray) -> tuple[np.ndarray, float, int]:
        indices = np.arange(6) if side == "left" else np.arange(6, 12)
        q_local = q[indices].copy()
        bid = self.left_foot_bid if side == "left" else self.right_foot_bid
        try:
            from scipy.optimize import least_squares

            def residual_fn(values: np.ndarray) -> np.ndarray:
                self.data.qpos[self.leg_qadr[indices]] = values
                self.mj.mj_forward(self.model, self.data)
                # Position is weighted so the feet are placed first; orientation then solves
                # ankle pitch/roll without making a millimetre of placement error acceptable.
                return np.concatenate((
                    10.0 * (target_pos - self.data.xpos[bid]),
                    _orientation_error(target_quat, self.data.xquat[bid]),
                ))

            solved = least_squares(
                residual_fn,
                q_local,
                bounds=(self.q_min[indices], self.q_max[indices]),
                max_nfev=self.max_iterations,
                diff_step=1.0e-5,
                xtol=1.0e-10,
                ftol=1.0e-10,
                gtol=1.0e-10,
            )
            q_local = solved.x
            iteration = int(solved.nfev)
        except ImportError:
            # The project workstation includes SciPy; retain a clear failure mode for a minimal
            # deployment environment rather than silently using a different IK implementation.
            raise RuntimeError("stance IK requires scipy.optimize.least_squares")
        self.data.qpos[self.leg_qadr[indices]] = q_local
        self.mj.mj_forward(self.model, self.data)
        final_residual = np.concatenate((
            target_pos - self.data.xpos[bid],
            _orientation_error(target_quat, self.data.xquat[bid]),
        ))
        return q_local, float(np.linalg.norm(final_residual)), iteration + 1

    def generate(self, config: StanceConfig) -> GeneratedStance:
        self._reset_keyframe()
        root = self.baseline_qpos[self.root_qadr:self.root_qadr + 7].copy()
        q_seed = self.baseline_qpos[self.leg_qadr].copy()
        q_seed[[0, 6]] -= math.radians(config.hip_flexion_deg)
        q_seed[[3, 9]] += math.radians(config.knee_flexion_deg)
        if config.pelvis_height_offset_m is None:
            # Derive the pelvis drop from the requested hip/knee seed while keeping the foot
            # centres at the measured ground height.  This makes hip/knee fields real geometric
            # stance parameters instead of transient IK hints.
            self.data.qpos[self.leg_qadr] = q_seed
            self.mj.mj_forward(self.model, self.data)
            derived_heights = np.array([
                self.baseline_root_height_m + self.baseline_left_foot[2] - self.data.xpos[self.left_foot_bid, 2],
                self.baseline_root_height_m + self.baseline_right_foot[2] - self.data.xpos[self.right_foot_bid, 2],
            ])
            root[2] = float(np.mean(derived_heights))
        else:
            root[2] += float(config.pelvis_height_offset_m)
        self.data.qpos[self.root_qadr:self.root_qadr + 7] = root
        self.mj.mj_forward(self.model, self.data)
        width = float(config.stance_width_m if config.stance_width_m is not None
                      else self.baseline_width_m * config.stance_width_scale)
        midpoint_y = 0.5 * (self.baseline_left_foot[1] + self.baseline_right_foot[1])
        # Preserve the model's small left/right x asymmetry; the requested stagger is an
        # additional relative offset, not an instruction to erase link/mesh asymmetry.
        left_x = float(self.baseline_left_foot[0])
        right_x = float(self.baseline_right_foot[0])
        half = 0.5 * config.fore_aft_m
        if config.lead_leg == "left":
            left_x += half
            right_x -= half
        elif config.lead_leg == "right":
            right_x += half
            left_x -= half
        left_target = np.array([left_x, midpoint_y + 0.5 * width, self.baseline_left_foot[2]])
        right_target = np.array([right_x, midpoint_y - 0.5 * width, self.baseline_right_foot[2]])
        # The API exposes flexion parameters explicitly.  In this model the positive knee range
        # is flexion and hip pitch decreases toward the forward/crouched direction; the numerical
        # IK solves the actual ankle compensation, while these fields seed a natural knee/hip bend.
        q_leg = q_seed.copy()
        self.data.qpos[self.leg_qadr] = q_leg
        self.mj.mj_forward(self.model, self.data)
        left_q, left_error, left_iters = self._solve_leg(q_leg, "left", left_target, self.baseline_left_quat)
        q_leg[:6] = left_q
        self.data.qpos[self.leg_qadr] = q_leg
        self.mj.mj_forward(self.model, self.data)
        right_q, right_error, right_iters = self._solve_leg(q_leg, "right", right_target, self.baseline_right_quat)
        q_leg[6:] = right_q
        self.data.qpos[self.leg_qadr] = q_leg
        self.mj.mj_forward(self.model, self.data)
        q = self.baseline_qpos.copy()
        q[self.leg_qadr] = q_leg
        # Torso pitch is a waist-pitch target.  It is retained separately from leg IK so that a
        # requested forward lean is visible in the generated contract and easy to audit.
        q[self._joint_qadr("waist_pitch_joint")] += math.radians(config.torso_pitch_deg)
        self.data.qpos[:] = q
        self.mj.mj_forward(self.model, self.data)
        residual = max(left_error, right_error)
        valid = bool(np.isfinite(residual) and residual < 2.0e-3)
        diagnostics: dict[str, float | str | bool] = {
            "ik_left_residual": left_error,
            "ik_right_residual": right_error,
            "ik_left_iterations": left_iters,
            "ik_right_iterations": right_iters,
            "coordinate_forward": "+x",
            "coordinate_left": "+y",
            "fore_aft_definition": "lead + fore_aft/2, trail - fore_aft/2",
            "valid": valid,
        }
        return GeneratedStance(
            config=config,
            q=q[self._joint_qpos_addresses()],
            root_qpos=root,
            left_foot_target=left_target,
            right_foot_target=right_target,
            pelvis_height_m=float(root[2]),
            width_m=width,
            valid=valid,
            diagnostics=diagnostics,
        )

    def _joint_qpos_addresses(self) -> np.ndarray:
        return np.array([self._joint_qadr(name) for name in JOINT_NAMES], dtype=int)


@dataclass(frozen=True)
class FallThresholds:
    base_height_min_m: float
    roll_abs_max_rad: float = 0.9
    pitch_abs_max_rad: float = 0.9
    nonfoot_ground_margin_m: float = 0.015


def quat_to_rpy(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return np.array([math.atan2(sinr, cosr), math.asin(sinp), math.atan2(siny, cosy)])


class MetricCollector:
    """Collect raw time series and derive all requested standing metrics."""

    def __init__(self, sim, *, thresholds: FallThresholds, dt: float):
        self.sim = sim
        self.thresholds = thresholds
        self.dt = float(dt)
        self.rows: list[dict[str, np.ndarray | float | bool]] = []
        self.max_slip = {"left": 0.0, "right": 0.0}

    def step(self, tau: np.ndarray, q_des: np.ndarray) -> dict[str, np.ndarray | float | bool]:
        s = self.sim.state()
        roll, pitch, yaw = quat_to_rpy(s["base_quat"])
        com = s["com"]
        foot_contact, foot_force = self.sim.foot_contact_and_force()
        stability_margin = self.sim.stability_margin(com, foot_contact)
        for side, pos, contact in (("left", s["left_foot"], foot_contact[0]), ("right", s["right_foot"], foot_contact[1])):
            if contact:
                self.max_slip[side] = max(self.max_slip[side], float(np.linalg.norm(pos[:2] - self.sim.foot_start[side][:2])))
        fall, reason = self.sim.fall_status(self.thresholds, roll, pitch)
        row = {
            "time": self.sim.time,
            "base_pos": s["base_pos"].copy(),
            "base_quat": s["base_quat"].copy(),
            "base_lin_vel": s["base_lin_vel"].copy(),
            "base_ang_vel": s["base_ang_vel"].copy(),
            "com": com.copy(),
            "left_foot": s["left_foot"].copy(),
            "right_foot": s["right_foot"].copy(),
            "foot_contact": foot_contact.copy(),
            "foot_force": foot_force.copy(),
            "stability_margin": stability_margin,
            "q": s["q"].copy(),
            "qd": s["qd"].copy(),
            "tau": np.asarray(tau).copy(),
            "q_des": np.asarray(q_des).copy(),
            "rpy": np.array([roll, pitch, yaw]),
            "fall": fall,
            "fall_reason": reason,
        }
        self.rows.append(row)
        return row

    def finalize(self, *, survival_time: float, recovery_time: float | None = None) -> dict[str, float | bool | str]:
        if not self.rows:
            return {"survival": False, "fall": True, "survival_time": 0.0, "fall_reason": "no_samples"}
        a = self.rows
        rpy = np.stack([x["rpy"] for x in a])
        ang = np.stack([x["base_ang_vel"] for x in a])
        com = np.stack([x["com"] for x in a])
        com_vel = np.vstack((np.zeros((1, 3)), np.diff(com, axis=0) / self.dt))
        qd = np.stack([x["qd"] for x in a])
        tau = np.stack([x["tau"] for x in a])
        forces = np.stack([x["foot_force"] for x in a])
        contacts = np.stack([x["foot_contact"] for x in a])
        com0 = com[0]
        first_fall = next((x for x in a if x["fall"]), None)
        grf_total = np.maximum(forces.sum(axis=1), 1.0e-8)
        load_ratio = forces[:, 0] / grf_total
        joint_margin = np.minimum(
            self.sim.q_range[:, 1] - np.stack([x["q"] for x in a]),
            np.stack([x["q"] for x in a]) - self.sim.q_range[:, 0],
        )
        result: dict[str, float | bool | str] = {
            "survival": first_fall is None,
            "fall": first_fall is not None,
            "survival_time": float(survival_time),
            "fall_reason": "none" if first_fall is None else str(first_fall["fall_reason"]),
            "roll_rms": float(np.sqrt(np.mean(rpy[:, 0] ** 2))),
            "pitch_rms": float(np.sqrt(np.mean(rpy[:, 1] ** 2))),
            "max_abs_roll": float(np.max(np.abs(rpy[:, 0]))),
            "max_abs_pitch": float(np.max(np.abs(rpy[:, 1]))),
            "angular_velocity_rms": float(np.sqrt(np.mean(np.sum(ang ** 2, axis=1)))),
            "com_xy_velocity_rms": float(np.sqrt(np.mean(np.sum(com_vel[:, :2] ** 2, axis=1)))),
            "peak_com_xy_velocity": float(np.max(np.linalg.norm(com_vel[:, :2], axis=1))),
            "max_com_xy_displacement": float(np.max(np.linalg.norm(com[:, :2] - com0[:2], axis=1))),
            "joint_velocity_rms": float(np.sqrt(np.mean(qd ** 2))),
            "mean_abs_torque": float(np.mean(np.abs(tau))),
            "torque_rms": float(np.sqrt(np.mean(tau ** 2))),
            "torque_peak": float(np.max(np.abs(tau))),
            "left_right_grf_difference": float(np.mean(np.abs(forces[:, 0] - forces[:, 1]))),
            "grf_left_mean": float(np.mean(forces[:, 0])),
            "grf_right_mean": float(np.mean(forces[:, 1])),
            "load_ratio_mean": float(np.mean(load_ratio)),
            "foot_slip_left": float(self.max_slip["left"]),
            "foot_slip_right": float(self.max_slip["right"]),
            "max_foot_slip": float(max(self.max_slip.values())),
            "joint_margin_min_rad": float(np.min(joint_margin)),
            "stability_margin_min_m": float(np.min([float(x["stability_margin"]) for x in a])),
            "stability_margin_mean_m": float(np.mean([float(x["stability_margin"]) for x in a])),
            "peak_joint_velocity": float(np.max(np.abs(qd))),
            "peak_joint_torque": float(np.max(np.abs(tau))),
            "recovery_time": float("nan") if recovery_time is None else float(recovery_time),
            "foot_contact_fraction_left": float(np.mean(contacts[:, 0])),
            "foot_contact_fraction_right": float(np.mean(contacts[:, 1])),
        }
        return result

    def save_trace(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.rows:
            return
        scalar_keys = ("time", "fall", "fall_reason")
        vector_keys = ("base_pos", "base_quat", "base_lin_vel", "base_ang_vel", "com",
                       "left_foot", "right_foot", "foot_contact", "foot_force", "stability_margin",
                       "q", "qd", "tau", "q_des", "rpy")
        with path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            header = list(scalar_keys)
            for key in vector_keys:
                value = self.rows[0][key]
                header.extend(f"{key}_{i}" for i in range(np.asarray(value).size))
            writer.writerow(header)
            for row in self.rows:
                values = [row[key] for key in scalar_keys]
                for key in vector_keys:
                    values.extend(np.asarray(row[key]).reshape(-1).tolist())
                writer.writerow(values)


class StanceMujoco:
    """In-process MuJoCo plant used by the test scripts."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        control_dt: float = 0.02,
        seed: int = 0,
        mu_contact: float | None = None,
    ):
        import mujoco

        self.mj = mujoco
        self.model_path = str(model_path)
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.control_dt = float(control_dt)
        self.substeps = max(1, int(round(self.control_dt / self.model.opt.timestep)))
        self.rng = np.random.default_rng(seed)
        self.time = 0.0
        self.generator = StanceGenerator(self.model, self.data)
        self.q_adr = np.array([self.generator._joint_qadr(name) for name in JOINT_NAMES], dtype=int)
        self.v_adr = np.array([self.model.jnt_dofadr[self.generator._joint_id(name)] for name in JOINT_NAMES], dtype=int)
        self.q_range = self.model.jnt_range[self.generator.leg_jids.tolist() + [self.generator._joint_id(n) for n in JOINT_NAMES[:19]], :] if False else self._q_range_in_joint_order()
        self.act_idx = self._resolve_actuators()
        self.ctrl_lo = self.model.actuator_ctrlrange[self.act_idx, 0].copy()
        self.ctrl_hi = self.model.actuator_ctrlrange[self.act_idx, 1].copy()
        self.ctrl_limited = self.model.actuator_ctrllimited[self.act_idx].astype(bool)
        self.foot_bids = (self.generator.left_foot_bid, self.generator.right_foot_bid)
        self.ground_gid = self._find_ground_geom()
        self.foot_collision_gids = self._find_foot_collision_geoms()
        self.mu_contact = None if mu_contact is None else float(mu_contact)
        if self.mu_contact is not None and (
            not math.isfinite(self.mu_contact) or self.mu_contact < 0.0
        ):
            raise ValueError(f"mu_contact must be finite and non-negative, got {mu_contact!r}")
        self._effective_mu_contact = self._configure_effective_contact_friction(self.mu_contact)
        self.root_bid = self.generator.pelvis_bid
        self.root_jid = self.generator.root_jid
        self.root_qadr = self.generator.root_qadr
        self.baseline_qpos = self.generator.baseline_qpos.copy()
        self.baseline_qvel = self.generator.baseline_qvel.copy()
        self.foot_start = {"left": self.generator.baseline_left_foot.copy(), "right": self.generator.baseline_right_foot.copy()}
        self._q_des = self.baseline_q()
        self.last_tau = np.zeros(NUM_JOINTS)

    def _configure_effective_contact_friction(self, mu_contact: float | None) -> float:
        """Set both foot and floor sliding friction to one explicit contact value.

        MuJoCo combines ordinary dynamically generated contacts from the two geom materials.
        Keeping the foot at the MJCF default of 1.5 while sweeping only the floor therefore
        does not implement a floor-friction sweep: with equal geom priorities the effective
        sliding value is the element-wise maximum.  The experiment contract is deliberately
        symmetric (foot == floor == ``mu_contact``), so the effective contact value is
        unambiguous and can be verified from ``data.contact[].friction`` after ``mj_forward``.
        Torsional/rolling terms remain those from the loaded model because this sweep controls
        the scalar sliding coefficient only.
        """
        foot_gids = sorted(self.foot_collision_gids)
        values = [float(self.model.geom_friction[gid, 0]) for gid in foot_gids]
        values.append(float(self.model.geom_friction[self.ground_gid, 0]))
        configured = max(values) if mu_contact is None else float(mu_contact)
        if mu_contact is not None:
            self.model.geom_friction[foot_gids, 0] = configured
            self.model.geom_friction[self.ground_gid, 0] = configured
        self.mj.mj_forward(self.model, self.data)
        return configured

    def effective_contact_friction(self) -> np.ndarray:
        """Return sliding friction values of active foot-ground contacts.

        The returned values come from MuJoCo's assembled contact records, not from the geom
        inputs.  An empty array means no foot-ground contact is active at the instant queried.
        """
        values = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            pair = {int(contact.geom1), int(contact.geom2)}
            if self.ground_gid not in pair:
                continue
            other = int(contact.geom2 if int(contact.geom1) == self.ground_gid else contact.geom1)
            if other in self.foot_collision_gids:
                values.append(float(contact.friction[0]))
        return np.asarray(values, dtype=float)

    def _q_range_in_joint_order(self) -> np.ndarray:
        return np.array([self.model.jnt_range[self.generator._joint_id(name)] for name in JOINT_NAMES], dtype=float)

    def _resolve_actuators(self) -> np.ndarray:
        trn_joint = self.model.actuator_trnid[:, 0]
        ids = []
        for name in JOINT_NAMES:
            jid = self.generator._joint_id(name)
            matches = np.where(trn_joint == jid)[0]
            if len(matches) != 1:
                raise ValueError(f"expected one actuator for {name}, got {matches}")
            ids.append(int(matches[0]))
        return np.asarray(ids, dtype=int)

    def _find_ground_geom(self) -> int:
        for gid in range(self.model.ngeom):
            name = self.mj.mj_id2name(self.model, self.mj.mjtObj.mjOBJ_GEOM, gid) or ""
            if name.lower() in ("floor", "ground", "plane") or int(self.model.geom_bodyid[gid]) == 0:
                if int(self.model.geom_contype[gid]) != 0:
                    return gid
        raise ValueError("ground collision geom not found")

    def _find_foot_collision_geoms(self) -> set[int]:
        return {
            gid for gid in range(self.model.ngeom)
            if int(self.model.geom_bodyid[gid]) in self.foot_bids and int(self.model.geom_contype[gid]) != 0
        }

    def baseline_q(self) -> np.ndarray:
        return self.baseline_qpos[self.q_adr].copy()

    def reset(self, stance: GeneratedStance | None = None, *, noise: bool = False,
              base_roll_noise: float = 0.0, base_pitch_noise: float = 0.0) -> None:
        if self.model.nkey:
            self.mj.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            self.mj.mj_resetData(self.model, self.data)
        self.time = 0.0
        self._q_des = self.baseline_q() if stance is None else stance.q.copy()
        # Keep the physical reset at the measured baseline root pose.  The candidate pelvis height
        # is a target implied by the leg IK, and is reached through the same smooth q_des transition
        # as the joints; teleporting the free root would inject an unfair initial foot-ground impulse.
        if noise:
            self.data.qpos[self.q_adr] += self.rng.uniform(-math.radians(1.0), math.radians(1.0), NUM_JOINTS)
            self.data.qvel[self.v_adr] += self.rng.uniform(-0.05, 0.05, NUM_JOINTS)
        if base_roll_noise or base_pitch_noise:
            roll = self.rng.uniform(-abs(base_roll_noise), abs(base_roll_noise))
            pitch = self.rng.uniform(-abs(base_pitch_noise), abs(base_pitch_noise))
            self.data.qpos[self.root_qadr + 3:self.root_qadr + 7] = self._quat_from_rpy(roll, pitch, 0.0)
        self.mj.mj_forward(self.model, self.data)
        self.foot_start = {"left": self.data.xpos[self.foot_bids[0]].copy(), "right": self.data.xpos[self.foot_bids[1]].copy()}
        self.last_tau[:] = 0.0

    @staticmethod
    def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                         cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])

    def set_targets(self, q_des: np.ndarray, kp: np.ndarray, kd: np.ndarray) -> np.ndarray:
        self._q_des = np.asarray(q_des, dtype=float).reshape(NUM_JOINTS)
        q = self.data.qpos[self.q_adr]
        qd = self.data.qvel[self.v_adr]
        tau = np.asarray(kp) * (self._q_des - q) - np.asarray(kd) * qd
        if np.any(self.ctrl_limited):
            tau = np.where(self.ctrl_limited, np.clip(tau, self.ctrl_lo, self.ctrl_hi), tau)
        self.last_tau = tau.copy()
        self.data.ctrl[self.act_idx] = tau
        return tau

    def step(self) -> None:
        for _ in range(self.substeps):
            self.data.ctrl[self.act_idx] = self.last_tau
            self.mj.mj_step(self.model, self.data)
        self.time += self.control_dt

    def state(self) -> dict[str, np.ndarray]:
        d = self.data
        com = d.subtree_com[0].copy()
        base_pos = d.qpos[self.root_qadr:self.root_qadr + 3].copy()
        base_quat = d.qpos[self.root_qadr + 3:self.root_qadr + 7].copy()
        q = d.qpos[self.q_adr].copy()
        qd = d.qvel[self.v_adr].copy()
        base_lin = d.qvel[self.model.jnt_dofadr[self.root_jid]:self.model.jnt_dofadr[self.root_jid] + 3].copy()
        base_ang = d.qvel[self.model.jnt_dofadr[self.root_jid] + 3:self.model.jnt_dofadr[self.root_jid] + 6].copy()
        return {"base_pos": base_pos, "base_quat": base_quat, "base_lin_vel": base_lin,
                "base_ang_vel": base_ang, "com": com, "q": q, "qd": qd,
                "left_foot": d.xpos[self.foot_bids[0]].copy(), "right_foot": d.xpos[self.foot_bids[1]].copy()}

    def foot_contact_and_force(self) -> tuple[np.ndarray, np.ndarray]:
        contact = np.zeros(2, dtype=bool)
        force = np.zeros(2, dtype=float)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {int(c.geom1), int(c.geom2)}
            foot = next((j for j, bid in enumerate(self.foot_bids)
                         if int(self.model.geom_bodyid[c.geom1]) == bid or int(self.model.geom_bodyid[c.geom2]) == bid), None)
            if foot is None:
                continue
            if self.ground_gid not in pair and int(self.model.geom_bodyid[c.geom1]) != 0 and int(self.model.geom_bodyid[c.geom2]) != 0:
                continue
            wrench = np.zeros(6, dtype=float)
            self.mj.mj_contactForce(self.model, self.data, i, wrench)
            force[foot] += float(np.linalg.norm(wrench[:3]))
            contact[foot] = True
        return contact, force

    def stability_margin(self, com: np.ndarray, contact: np.ndarray) -> float:
        """Distance from COM projection to a conservative rectangular foot support region."""
        points = []
        # Collision capsules are conservative in the loaded MJCF; these measured half extents
        # cover the contact patch without treating the ankle shell as a support foot.
        half_x, half_y = 0.082, 0.058
        for idx, bid in enumerate(self.foot_bids):
            if not contact[idx]:
                continue
            centre = self.data.xpos[bid]
            points.extend(((centre[0] - half_x, centre[1] - half_y),
                           (centre[0] - half_x, centre[1] + half_y),
                           (centre[0] + half_x, centre[1] - half_y),
                           (centre[0] + half_x, centre[1] + half_y)))
        if not points:
            return -1.0
        arr = np.asarray(points)
        x_lo, y_lo = np.min(arr, axis=0)
        x_hi, y_hi = np.max(arr, axis=0)
        dx = min(float(com[0] - x_lo), float(x_hi - com[0]))
        dy = min(float(com[1] - y_lo), float(y_hi - com[1]))
        if dx >= 0.0 and dy >= 0.0:
            return min(dx, dy)
        return -float(np.linalg.norm([min(dx, 0.0), min(dy, 0.0)]))

    def fall_status(self, thresholds: FallThresholds, roll: float, pitch: float) -> tuple[bool, str]:
        s = self.state()
        if s["base_pos"][2] < thresholds.base_height_min_m:
            return True, "base_height"
        if abs(roll) > thresholds.roll_abs_max_rad:
            return True, "roll"
        if abs(pitch) > thresholds.pitch_abs_max_rad:
            return True, "pitch"
        foot_bodies = set(self.foot_bids)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if int(c.geom1) == self.ground_gid or int(c.geom2) == self.ground_gid:
                other = int(c.geom2 if int(c.geom1) == self.ground_gid else c.geom1)
                body = int(self.model.geom_bodyid[other])
                if body not in foot_bodies:
                    return True, "nonfoot_ground_contact"
        return False, "none"

    def apply_force(self, force_world: Sequence[float], body: str = "pelvis_link") -> None:
        bid = self.generator._body_id(body)
        self.data.xfrc_applied[bid, :3] = np.asarray(force_world, dtype=float)

    def clear_force(self) -> None:
        self.data.xfrc_applied[:] = 0.0


def deploy_pd_gains(path: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return the published model_21800 gains without requiring YAML at runtime."""
    kp = np.array([85, 50, 50, 40, 40, 40, 40, 30, 30, 30, 20, 20,
                   40, 40, 30, 30, 30, 20, 20, 80, 120, 80, 250, 50, 50,
                   80, 120, 80, 250, 50, 50], dtype=float)
    kd = np.array([3, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 2,
                   3, 3, 2, 2, 2, 2, 2, 3, 4, 3, 8, 2, 2,
                   3, 4, 3, 8, 2, 2], dtype=float)
    return kp, kd


def official_stand_pd_gains() -> tuple[np.ndarray, np.ndarray]:
    """The existing reference runner's PD_STAND gains, in the 31-D order."""
    kp = np.zeros(NUM_JOINTS, dtype=float)
    kd = np.zeros(NUM_JOINTS, dtype=float)
    kp[0:3] = [400.0, 500.0, 500.0]
    kd[0:3] = [4.0, 4.0, 4.0]
    kp[3:5] = [40.0, 40.0]
    kd[3:5] = [2.0, 2.0]
    kp[5:19] = [200.0, 200.0, 100.0, 200.0, 100.0, 50.0, 50.0] * 2
    kd[5:19] = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0] * 2
    kp[19:31] = [1500.0, 400.0, 300.0, 2000.0, 500.0, 500.0] * 2
    kd[19:31] = [8.0, 7.0, 7.0, 8.0, 5.0, 5.0] * 2
    return kp, kd


def thresholds_for(sim: StanceMujoco) -> FallThresholds:
    # Derived from the loaded model's grounded keyframe rather than copied from another robot.
    return FallThresholds(base_height_min_m=max(0.45, sim.generator.baseline_root_height_m - 0.45))


def pose_delta_stance(base_q: np.ndarray, generated: GeneratedStance, clip_q: np.ndarray,
                      clip_q0: np.ndarray, mode: str) -> np.ndarray:
    """Map an existing no-ball motion reference onto a generated stance."""
    q = generated.q.copy()
    delta = np.asarray(clip_q) - np.asarray(clip_q0)
    if mode == "arm_only":
        q[ARM_IDX] += delta[ARM_IDX]
    elif mode == "arm_torso":
        q[np.concatenate((WAIST_IDX, ARM_IDX))] += delta[np.concatenate((WAIST_IDX, ARM_IDX))]
    elif mode == "full_body":
        q += delta
    else:
        raise ValueError(f"unknown swing mode: {mode}")
    return q


def write_rows(path: str | Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: Sequence[dict], group_keys: Sequence[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(k) for k in group_keys), []).append(row)
    out = []
    for key, group in groups.items():
        base = {k: v for k, v in zip(group_keys, key)}
        numeric = sorted({k for row in group for k, v in row.items()
                          if k not in group_keys and isinstance(v, (int, float)) and math.isfinite(float(v))})
        for name in numeric:
            values = np.array([float(row[name]) for row in group], dtype=float)
            base[f"{name}_mean"] = float(np.mean(values))
            base[f"{name}_std"] = float(np.std(values))
            base[f"{name}_median"] = float(np.median(values))
            base[f"{name}_min"] = float(np.min(values))
            base[f"{name}_max"] = float(np.max(values))
        if "survival" in group[0]:
            base["success_rate"] = float(np.mean([bool(row["survival"]) for row in group]))
        out.append(base)
    return out
