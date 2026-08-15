"""MuJoCo adapter for the portable v13b three-model runtime.

The checkpoint loader deliberately stops at tensors.  This module is the
missing plant-facing layer: it resolves the A3 31-DOF order, computes the
pelvis-local observations, performs the same racket-mount FK as training,
reconstructs the two frozen-prior targets, and applies a bounded torque PD
loop to a MuJoCo motor model.

The live strike path needs the private motion/reference trajectory used by
model 900 and the historical Stage-A model.  ``NpzReferenceProvider`` defines
that boundary explicitly.  ``ReadyHoldReference`` is intentionally provided
for deterministic reset/stand smoke only; it is not a substitute for the
23,118-frame private strike reference.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from .v13b_runtime import ThreeModelOutput, ThreeModelRuntime

try:  # Keep importing the tensor-only runtime possible without MuJoCo.
    import mujoco
except ImportError:  # pragma: no cover - exercised on machines without MuJoCo.
    mujoco = None  # type: ignore[assignment]


BACKEND_JOINTS = (
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "head_yaw_joint", "head_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
)

LOWER_JOINTS = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
)
BASE14_JOINTS = LOWER_JOINTS + ("waist_roll_joint", "waist_pitch_joint")
UPPER_JOINTS = (
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
STRIKE_JOINTS = (
    "waist_yaw_joint", "waist_pitch_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

# These are the physical scales used by the frozen historical prior action
# terms, copied from the training-side A3 config.
LOWER_PRIOR_SCALE = np.asarray(
    (0.03666666666666667, 0.1375, 0.18333333333333332, 0.04, 0.0591, 0.027375,
     0.03666666666666667, 0.1375, 0.18333333333333332, 0.04, 0.0591, 0.027375),
    dtype=np.float32,
)
UPPER_PRIOR_SCALE = np.asarray(
    (0.20, 0.12, 0.14, 0.28, 0.32, 0.28, 0.24, 0.14, 0.12, 0.12),
    dtype=np.float32,
)

MOUNT_OFFSET = np.asarray((0.210211399202899, 0.0320784994676765, 0.0320358706296689), dtype=np.float64)
READY_UPPER_PRELUDE_STEPS = 50
UPPER_RELEASE_STEPS = 12

DEFAULT_KP = {
    "waist_yaw_joint": 85.0, "waist_roll_joint": 50.0, "waist_pitch_joint": 50.0,
    "head_yaw_joint": 40.0, "head_pitch_joint": 40.0,
    "left_shoulder_pitch_joint": 40.0, "left_shoulder_roll_joint": 40.0, "left_shoulder_yaw_joint": 30.0,
    "left_elbow_joint": 30.0, "left_wrist_roll_joint": 30.0, "left_wrist_pitch_joint": 20.0, "left_wrist_yaw_joint": 20.0,
    "right_shoulder_pitch_joint": 40.0, "right_shoulder_roll_joint": 40.0, "right_shoulder_yaw_joint": 30.0,
    "right_elbow_joint": 30.0, "right_wrist_roll_joint": 30.0, "right_wrist_pitch_joint": 20.0, "right_wrist_yaw_joint": 20.0,
    "left_hip_pitch_joint": 80.0, "left_hip_roll_joint": 120.0, "left_hip_yaw_joint": 80.0,
    "left_knee_joint": 250.0, "left_ankle_pitch_joint": 50.0, "left_ankle_roll_joint": 50.0,
    "right_hip_pitch_joint": 80.0, "right_hip_roll_joint": 120.0, "right_hip_yaw_joint": 80.0,
    "right_knee_joint": 250.0, "right_ankle_pitch_joint": 50.0, "right_ankle_roll_joint": 50.0,
}
DEFAULT_KD = {name: 2.0 for name in BACKEND_JOINTS}
DEFAULT_KD.update({
    "waist_yaw_joint": 3.0,
    "waist_roll_joint": 2.0,
    "waist_pitch_joint": 2.0,
    "left_shoulder_pitch_joint": 3.0,
    "left_shoulder_roll_joint": 3.0,
    "right_shoulder_pitch_joint": 3.0,
    "right_shoulder_roll_joint": 3.0,
    "left_hip_pitch_joint": 3.0,
    "left_hip_roll_joint": 4.0,
    "left_hip_yaw_joint": 3.0,
    "right_hip_pitch_joint": 3.0,
    "right_hip_roll_joint": 4.0,
    "right_hip_yaw_joint": 3.0,
    "left_knee_joint": 8.0,
    "right_knee_joint": 8.0,
})


@dataclass(frozen=True)
class MujocoLowLevelConfig:
    """Simulation-only drive/balance settings.

    Isaac's implicit actuators and the A3 native MC stack are not the same
    controller.  This config therefore belongs to the MuJoCo plant adapter;
    it is deliberately not reused by :meth:`hardware_command`.
    """

    name: str
    kp: tuple[float, ...]
    kd: tuple[float, ...]
    balance_enabled: bool = False
    balance_com_kp: float = 0.0
    balance_com_kd: float = 0.0
    balance_tilt_kp: float = 0.0
    balance_tilt_kd: float = 0.0
    balance_lateral_kp: float = 0.0
    balance_lateral_kd: float = 0.0
    balance_max_joint_torque: float = 0.0
    lower_target_envelope_rad: float | None = None
    # MuJoCo-only calibration around the source gains.  These scales do not
    # leak into HardwareCommand; they compensate for the grounded MJCF's
    # contact/solver response when the simulation balance layer is enabled.
    drive_kp_scale: float = 1.0
    drive_kd_scale: float = 1.0

    @staticmethod
    def official() -> "MujocoLowLevelConfig":
        return MujocoLowLevelConfig(
            name="official_pd",
            kp=tuple(DEFAULT_KP[name] for name in BACKEND_JOINTS),
            kd=tuple(DEFAULT_KD[name] for name in BACKEND_JOINTS),
        )

    @staticmethod
    def isaac_passive_stable() -> "MujocoLowLevelConfig":
        """The source-snapshot simulation-only passive-stability candidate.

        Leg/foot values are copied from
        ``A3BaseStandPassiveStableCandidateEnvCfg``.  The balance numbers are
        a bounded MuJoCo support-feedback layer, not a claim about native MC.
        """
        kp = dict(DEFAULT_KP)
        kd = dict(DEFAULT_KD)
        for name in BACKEND_JOINTS:
            if "hip_pitch" in name:
                kp[name], kd[name] = 1500.0, 8.0
            elif "hip_roll" in name:
                kp[name], kd[name] = 400.0, 7.0
            elif "hip_yaw" in name:
                kp[name], kd[name] = 300.0, 7.0
            elif "knee" in name:
                kp[name], kd[name] = 2000.0, 8.0
            elif "ankle" in name:
                kp[name], kd[name] = 500.0, 5.0
        kp["waist_roll_joint"] = 500.0
        kp["waist_pitch_joint"] = 500.0
        kd["waist_roll_joint"] = 4.0
        kd["waist_pitch_joint"] = 4.0
        return MujocoLowLevelConfig(
            name="isaac_passive_stable",
            kp=tuple(kp[name] for name in BACKEND_JOINTS),
            kd=tuple(kd[name] for name in BACKEND_JOINTS),
            balance_enabled=True,
            balance_com_kp=800.0,
            balance_com_kd=100.0,
            balance_tilt_kp=300.0,
            balance_tilt_kd=80.0,
            balance_lateral_kp=500.0,
            balance_lateral_kd=80.0,
            balance_max_joint_torque=150.0,
            # The source strike policy can request a lower-body excursion
            # larger than this MuJoCo contact model can recover from.  Keep a
            # conservative stand envelope for plant qualification; the real
            # robot route must use its native MC balance envelope instead.
            lower_target_envelope_rad=0.12,
            # A/B on the grounded model: slightly softer position drive plus
            # extra damping removes the fast lower-leg chatter while keeping
            # the same Isaac-derived nominal gain ratios.
            drive_kp_scale=0.80,
            drive_kd_scale=1.20,
        )

    @staticmethod
    def resolve(name: str) -> "MujocoLowLevelConfig":
        normalized = str(name).strip().lower()
        if normalized in {"official", "official_pd", "deployment"}:
            return MujocoLowLevelConfig.official()
        if normalized in {"isaac_passive_stable", "passive_stable", "stand_candidate"}:
            return MujocoLowLevelConfig.isaac_passive_stable()
        raise ValueError(
            f"unknown MuJoCo low-level profile {name!r}; expected "
            "official_pd or isaac_passive_stable"
        )


class SimulationBalanceController:
    """Bounded support-feedback layer used only inside the MuJoCo plant.

    The controller observes the simulated COM, the two foot support sites,
    root tilt and angular velocity.  It maps sagittal feedback to the two hip
    pitch motors and lateral feedback to the two ankle-roll motors.  It is
    intentionally conservative and has no hardware-command path.
    """

    def __init__(self, adapter: "MujocoV13BAdapter", config: MujocoLowLevelConfig) -> None:
        if mujoco is None:  # pragma: no cover - adapter construction checks this first.
            raise ImportError("MuJoCo is required for SimulationBalanceController")
        self.config = config
        self.foot_site_ids = tuple(
            int(mujoco.mj_name2id(adapter.model, mujoco.mjtObj.mjOBJ_SITE, name))
            for name in ("left_foot", "right_foot")
        )
        if any(site_id < 0 for site_id in self.foot_site_ids):
            raise RuntimeError("MuJoCo XML must contain left_foot and right_foot sites")
        self._previous_com: np.ndarray | None = None
        self.last_torque = np.zeros(len(BACKEND_JOINTS), dtype=np.float64)

    def reset(self, adapter: "MujocoV13BAdapter") -> None:
        self._previous_com = np.asarray(adapter.data.subtree_com[0], dtype=np.float64).copy()
        self.last_torque.fill(0.0)

    def _has_foot_support(self, adapter: "MujocoV13BAdapter") -> bool:
        foot_bodies = {
            int(adapter.model.site_bodyid[site_id]) for site_id in self.foot_site_ids
        }
        for index in range(adapter.data.ncon):
            contact = adapter.data.contact[index]
            bodies = {
                int(adapter.model.geom_bodyid[int(contact.geom1)]),
                int(adapter.model.geom_bodyid[int(contact.geom2)]),
            }
            if 0 in bodies and bodies.intersection(foot_bodies):
                return True
        return False

    def torque(self, adapter: "MujocoV13BAdapter") -> np.ndarray:
        if not self.config.balance_enabled or not self._has_foot_support(adapter):
            self.last_torque.fill(0.0)
            return self.last_torque.copy()

        com = np.asarray(adapter.data.subtree_com[0], dtype=np.float64)
        if self._previous_com is None:
            self._previous_com = com.copy()
        dt = float(adapter.model.opt.timestep)
        com_velocity = np.clip((com - self._previous_com) / max(dt, 1.0e-6), -5.0, 5.0)
        self._previous_com = com.copy()
        support = np.mean(
            [np.asarray(adapter.data.site_xpos[site_id], dtype=np.float64) for site_id in self.foot_site_ids],
            axis=0,
        )
        _, angular_velocity, gravity_local = adapter._root_state()
        # In the A3 body convention gravity_local[x] is the sagittal pitch
        # error and gravity_local[y] is the lateral roll error near upright.
        sagittal = (
            self.config.balance_com_kp * (com[0] - support[0])
            + self.config.balance_com_kd * com_velocity[0]
            + self.config.balance_tilt_kp * float(gravity_local[0])
            + self.config.balance_tilt_kd * float(angular_velocity[1])
        )
        lateral = -(
            self.config.balance_lateral_kp * (com[1] - support[1])
            + self.config.balance_lateral_kd * com_velocity[1]
            + self.config.balance_lateral_kp * float(gravity_local[1])
            - self.config.balance_lateral_kd * float(angular_velocity[0])
        )
        output = np.zeros(len(BACKEND_JOINTS), dtype=np.float64)
        for name in ("left_hip_pitch_joint", "right_hip_pitch_joint"):
            output[BACKEND_JOINTS.index(name)] = sagittal
        for name in ("left_ankle_roll_joint", "right_ankle_roll_joint"):
            output[BACKEND_JOINTS.index(name)] = lateral
        output = np.clip(
            output,
            -float(self.config.balance_max_joint_torque),
            float(self.config.balance_max_joint_torque),
        )
        self.last_torque = output
        return output.copy()
JOINT_VELOCITY_LIMIT = {
    **{name: 12.0 for name in ("left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint", "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint")},
    "left_knee_joint": 14.6, "right_knee_joint": 14.6,
    "left_ankle_pitch_joint": 10.8, "right_ankle_pitch_joint": 10.8,
    "left_ankle_roll_joint": 19.3, "right_ankle_roll_joint": 19.3,
    "waist_yaw_joint": 12.0, "waist_roll_joint": 22.7, "waist_pitch_joint": 9.2,
    "head_yaw_joint": 12.7, "head_pitch_joint": 12.7,
    **{name: 13.6 for name in ("left_shoulder_pitch_joint", "left_shoulder_roll_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint")},
    **{name: 15.7 for name in ("left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint")},
    "left_wrist_pitch_joint": 12.7, "left_wrist_yaw_joint": 12.7,
    "right_wrist_pitch_joint": 12.7, "right_wrist_yaw_joint": 12.7,
}

# Isaac's default joint pose used by the old Stage-A observation/action
# contract.  v13b READY is deliberately different and is kept separately.
LEGACY_DEFAULT = {
    "left_hip_pitch_joint": -0.1311, "left_hip_roll_joint": 0.0056, "left_hip_yaw_joint": -0.0348,
    "left_knee_joint": 0.2468, "left_ankle_pitch_joint": -0.1204, "left_ankle_roll_joint": -0.0078,
    "right_hip_pitch_joint": -0.1311, "right_hip_roll_joint": -0.0056, "right_hip_yaw_joint": 0.0348,
    "right_knee_joint": 0.2468, "right_ankle_pitch_joint": -0.1204, "right_ankle_roll_joint": 0.0078,
    "waist_roll_joint": 0.0, "waist_pitch_joint": 0.0,
}
STUDENT_DEFAULT_UPPER = {
    "waist_yaw_joint": 0.0, "waist_roll_joint": 0.0, "waist_pitch_joint": 0.0,
    "right_shoulder_pitch_joint": 0.3, "right_shoulder_roll_joint": -0.12,
    "right_shoulder_yaw_joint": 0.0, "right_elbow_joint": 0.8,
    "right_wrist_roll_joint": 0.0, "right_wrist_pitch_joint": 0.0, "right_wrist_yaw_joint": 0.0,
}


def _vec(value: np.ndarray | list[float] | tuple[float, ...], width: int, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float32).reshape(-1)
    if out.size != width:
        raise ValueError(f"{name} must have width {width}, got {out.size}")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return out


def _normalize(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1.0e-8:
        return np.asarray(fallback if fallback is not None else (0.0, 1.0, 0.0), dtype=np.float64)
    return v / n


def quat_rotate(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a vector by an MuJoCo/Isaac wxyz quaternion."""
    w, x, y, z = np.asarray(q_wxyz, dtype=np.float64)
    qv = np.asarray(v, dtype=np.float64)
    t = 2.0 * np.cross(np.array((x, y, z)), qv)
    return qv + w * t + np.cross(np.array((x, y, z)), t)


def quat_rotate_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float64)
    return quat_rotate(np.array((q[0], -q[1], -q[2], -q[3])), v)


def yaw_quat(q_wxyz: np.ndarray) -> np.ndarray:
    """Return the yaw-only quaternion used by Isaac ``*_b`` observations."""
    w, x, y, z = np.asarray(q_wxyz, dtype=np.float64)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.asarray((np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)), dtype=np.float64)


@dataclass(frozen=True)
class StrikeTarget:
    position_world: np.ndarray
    velocity_world: np.ndarray
    normal_world: np.ndarray
    hit_time_s: float
    swing_type: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_world", _vec(self.position_world, 3, "target position"))
        object.__setattr__(self, "velocity_world", _vec(self.velocity_world, 3, "target velocity"))
        normal = _normalize(np.asarray(self.normal_world, dtype=np.float64))
        object.__setattr__(self, "normal_world", normal.astype(np.float32))
        if not np.isfinite(self.hit_time_s):
            raise ValueError("target hit_time_s must be finite")


@dataclass(frozen=True)
class ReferenceFrame:
    """One reference sample in the historical prior coordinate system."""

    lower_reference: np.ndarray
    upper_reference: np.ndarray
    upper_reference_lead: np.ndarray
    upper_velocity: np.ndarray
    strike_joint_pos: np.ndarray
    strike_joint_vel: np.ndarray
    strike_joint_vel_8: np.ndarray
    strike_joint_vel_16: np.ndarray
    phase: float

    def __post_init__(self) -> None:
        for name, width in (
            ("lower_reference", 12), ("upper_reference", 10), ("upper_reference_lead", 10),
            ("upper_velocity", 10),
            ("strike_joint_pos", 9), ("strike_joint_vel", 9),
            ("strike_joint_vel_8", 9), ("strike_joint_vel_16", 9),
        ):
            object.__setattr__(self, name, _vec(getattr(self, name), width, name))
        if not np.isfinite(self.phase):
            raise ValueError("reference phase must be finite")


class ReferenceProvider(Protocol):
    def frame(self, step: int, adapter: "MujocoV13BAdapter") -> ReferenceFrame: ...


class ReadyHoldReference:
    """Reference used by the reset/stand smoke; holds the reviewed READY pose."""

    def frame(self, step: int, adapter: "MujocoV13BAdapter") -> ReferenceFrame:
        q = adapter.joint_vector(STRIKE_JOINTS)
        lower = adapter.joint_vector(LOWER_JOINTS)
        upper = adapter.joint_vector(UPPER_JOINTS)
        zeros9 = np.zeros(9, dtype=np.float32)
        return ReferenceFrame(lower, upper, upper, np.zeros(10, dtype=np.float32), q, zeros9, zeros9, zeros9, 0.0)


class NpzReferenceProvider:
    """Strict adapter for a private motion/reference ``.npz`` file.

    Required arrays are absolute joint positions in the named order:
    ``lower_reference[T,12]``, ``upper_reference[T,10]``,
    ``strike_joint_pos[T,9]``, ``strike_joint_vel[T,9]``, and ``phase[T]``.
    ``strike_joint_vel_8`` and ``strike_joint_vel_16`` are optional and are
    derived from ``strike_joint_vel`` when absent.  Positions for the upper
    prior are sampled 12 steps ahead for the two lookahead joints, matching
    the training action term.  The 50-step prelude is applied outside the
    trajectory and blends the lead in over 12 steps.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        with np.load(self.path, allow_pickle=False) as data:
            required = ("lower_reference", "upper_reference", "strike_joint_pos", "strike_joint_vel", "phase")
            missing = [key for key in required if key not in data]
            if missing:
                raise ValueError(f"{self.path}: missing reference arrays {missing}")
            self.lower = np.asarray(data["lower_reference"], dtype=np.float32)
            self.upper = np.asarray(data["upper_reference"], dtype=np.float32)
            self.strike_q = np.asarray(data["strike_joint_pos"], dtype=np.float32)
            self.strike_dq = np.asarray(data["strike_joint_vel"], dtype=np.float32)
            self.phase = np.asarray(data["phase"], dtype=np.float32).reshape(-1)
            self.strike_dq8 = np.asarray(data["strike_joint_vel_8"], dtype=np.float32) if "strike_joint_vel_8" in data else None
            self.strike_dq16 = np.asarray(data["strike_joint_vel_16"], dtype=np.float32) if "strike_joint_vel_16" in data else None
        T = self.phase.size
        for name, array, width in (("lower_reference", self.lower, 12), ("upper_reference", self.upper, 10),
                                    ("strike_joint_pos", self.strike_q, 9), ("strike_joint_vel", self.strike_dq, 9)):
            if array.shape != (T, width):
                raise ValueError(f"{self.path}: {name} must have shape ({T},{width}), got {array.shape}")
        if self.strike_dq8 is not None and self.strike_dq8.shape != (T, 9):
            raise ValueError(f"{self.path}: strike_joint_vel_8 must have shape ({T},9)")
        if self.strike_dq16 is not None and self.strike_dq16.shape != (T, 9):
            raise ValueError(f"{self.path}: strike_joint_vel_16 must have shape ({T},9)")

    def _sample(self, array: np.ndarray, index: int) -> np.ndarray:
        return array[min(max(int(index), 0), len(array) - 1)]

    def frame(self, step: int, adapter: "MujocoV13BAdapter") -> ReferenceFrame:
        raw = max(0, int(step) - READY_UPPER_PRELUDE_STEPS)
        q = self._sample(self.strike_q, raw)
        dq = self._sample(self.strike_dq, raw)
        if step < READY_UPPER_PRELUDE_STEPS:
            ready = adapter.joint_vector(STRIKE_JOINTS).astype(np.float32)
            alpha = min(1.0, step / float(READY_UPPER_PRELUDE_STEPS))
            q = ready + alpha * (q - ready)
        dq8 = self._sample(self.strike_dq8 if self.strike_dq8 is not None else self.strike_dq, raw + 8)
        dq16 = self._sample(self.strike_dq16 if self.strike_dq16 is not None else self.strike_dq, raw + 16)
        upper = self._sample(self.upper, raw)
        lead = upper.copy()
        # Only shoulder pitch and shoulder yaw used the historical +12-step
        # command lead; the other eight channels stay at the current sample.
        lead[[3, 5]] = self._sample(self.upper, raw + 12)[[3, 5]]
        if step < READY_UPPER_PRELUDE_STEPS:
            ready = adapter.joint_vector(UPPER_JOINTS)
            lead = ready
        elif UPPER_RELEASE_STEPS > 0:
            ready = adapter.joint_vector(UPPER_JOINTS)
            alpha = min(1.0, (step - READY_UPPER_PRELUDE_STEPS + 1) / float(UPPER_RELEASE_STEPS))
            lead = ready + alpha * (lead - ready)
        return ReferenceFrame(
            self._sample(self.lower, raw), upper, lead,
            np.concatenate((np.asarray((dq[0], 0.0), dtype=np.float32), dq[1:])),
            q, dq, dq8, dq16,
            float(self._sample(self.phase, raw)),
        )


class MotionManifestReferenceProvider:
    """Load the project's causal motion-reference bank for the frozen priors.

    The Isaac training command uses the selected manifest motion as a private
    teacher.  This provider mirrors that boundary without importing IsaacLab:
    it selects one already-generated motion from the manifest, starts at its
    frame zero after the 50-step READY prelude, and exposes only the current
    frame plus the finite 8/12/16-step previews used by the historical prior
    contracts.  It never reads future robot measurements.

    The large 23,118-motion bank stays outside the portable checkpoint bundle;
    the manifest path is explicit and each selected payload is loaded lazily.
    ``motion_index`` is useful for deterministic replay.  ``selection`` may
    be ``nearest`` to choose a trajectory from the requested racket target.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        motion_index: int | None = 0,
        selection: str = "fixed",
        stroke_type: str | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.entries = list(payload.get("motions", []))
        if not self.entries:
            raise ValueError(f"{self.manifest_path}: manifest has no motions")
        self.selection = str(selection).lower()
        if self.selection not in {"fixed", "nearest"}:
            raise ValueError("selection must be 'fixed' or 'nearest'")
        self.stroke_type = None if stroke_type is None else str(stroke_type).lower()
        if self.stroke_type not in {None, "forehand", "backhand"}:
            raise ValueError("stroke_type must be forehand, backhand, or None")
        candidates = [
            i for i, entry in enumerate(self.entries)
            if self.stroke_type is None or str(entry.get("stroke_type", "")).lower() == self.stroke_type
        ]
        if not candidates:
            raise ValueError(f"{self.manifest_path}: no entries match stroke_type={self.stroke_type!r}")
        self._candidates = np.asarray(candidates, dtype=np.int64)
        if motion_index is not None:
            motion_index = int(motion_index)
            if motion_index < 0 or motion_index >= len(self.entries):
                raise IndexError(f"motion_index={motion_index} outside [0,{len(self.entries)})")
            if motion_index not in candidates:
                raise ValueError(f"motion_index={motion_index} does not match stroke_type={self.stroke_type!r}")
        self.motion_index = motion_index
        self.selected_index: int | None = None
        self.motion_path: Path | None = None
        self.fps = 50
        self.hit_frame = 0
        self.lower = np.zeros((1, len(LOWER_JOINTS)), dtype=np.float32)
        self.upper = np.zeros((1, len(UPPER_JOINTS)), dtype=np.float32)
        self.strike_q = np.zeros((1, len(STRIKE_JOINTS)), dtype=np.float32)
        self.strike_dq = np.zeros((1, len(STRIKE_JOINTS)), dtype=np.float32)
        self.phase = np.zeros(1, dtype=np.float32)
        self._upper_dq = np.zeros((1, len(UPPER_JOINTS)), dtype=np.float32)

    def reset(self) -> None:
        self.selected_index = None
        self.motion_path = None

    def _entry_path(self, index: int) -> Path:
        entry = self.entries[index]
        values = [entry.get("library_motion_npz"), entry.get("motion_npz")]
        manifest_dir = self.manifest_path.parent
        attempted: list[Path] = []
        for value in values:
            if not value:
                continue
            raw = Path(str(value)).expanduser()
            candidates: list[Path] = []
            stroke = str(entry.get("stroke_type", "")).lower()
            if stroke:
                # The deployment bundle carries the original bank one level
                # above the copied merged manifest.  This fallback lets the
                # unchanged provenance paths remain auditable while making an
                # extracted package self-contained.
                candidates.append(
                    manifest_dir.parent / "training_reference_bank_20260806" / "motions" / stroke / raw.name
                )
            candidates.extend([raw] if raw.is_absolute() else [manifest_dir / raw])
            # Packaged/reference-bank manifests commonly retain the source
            # machine's absolute path while placing the payload beside the
            # manifest under motion_npz/.
            candidates.append(manifest_dir / "motion_npz" / raw.name)
            for candidate in candidates:
                candidate = candidate.resolve()
                attempted.append(candidate)
                if candidate.is_file():
                    return candidate
        raise FileNotFoundError(
            f"manifest entry {index} has no readable motion payload; tried {attempted}"
        )

    @staticmethod
    def _entry_goal(entry: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        target = entry.get("strike_target", {})
        position = target.get("racket_position_m", entry.get("canonical_position", (0.0, 0.0, 0.0)))
        velocity = target.get("racket_velocity_mps", entry.get("canonical_velocity", (0.0, 0.0, 0.0)))
        normal = target.get("racket_normal_w", entry.get("canonical_normal", (0.0, 1.0, 0.0)))
        return (
            _vec(position, 3, "manifest target position").astype(np.float64),
            _vec(velocity, 3, "manifest target velocity").astype(np.float64),
            _normalize(_vec(normal, 3, "manifest target normal")),
        )

    def _choose_index(self, target: StrikeTarget | None, adapter: "MujocoV13BAdapter") -> int:
        if self.motion_index is not None:
            return self.motion_index
        if target is None:
            raise ValueError("nearest manifest selection requires a StrikeTarget")
        requested = adapter._target_local(target).astype(np.float64)
        best_index = int(self._candidates[0])
        best_score = float("inf")
        for candidate in self._candidates.tolist():
            position, velocity, normal = self._entry_goal(self.entries[candidate])
            score = (
                np.linalg.norm((position - requested[:3]) / 0.15)
                + np.linalg.norm((velocity - requested[3:6]) / 2.0)
                + np.linalg.norm(normal - requested[6:9])
            )
            if score < best_score:
                best_index, best_score = int(candidate), float(score)
        return best_index

    def _load(self, index: int) -> None:
        path = self._entry_path(index)
        with np.load(path, allow_pickle=False) as data:
            if "joint_pos" not in data or "joint_vel" not in data:
                raise ValueError(f"{path}: motion payload must contain joint_pos and joint_vel")
            joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
            joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
            if joint_pos.ndim != 2 or joint_pos.shape[1] != len(BACKEND_JOINTS):
                raise ValueError(f"{path}: joint_pos must have shape [T,31], got {joint_pos.shape}")
            if joint_vel.shape != joint_pos.shape:
                raise ValueError(f"{path}: joint_vel shape {joint_vel.shape} != joint_pos {joint_pos.shape}")
            fps = int(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data else 50
            if fps != 50:
                raise ValueError(f"{path}: frozen priors require 50Hz reference, got fps={fps}")
            if "joint_names" in data:
                names = tuple(str(x) for x in np.asarray(data["joint_names"]).reshape(-1))
                if names and names != BACKEND_JOINTS:
                    if len(names) != len(BACKEND_JOINTS) or set(names) != set(BACKEND_JOINTS):
                        raise ValueError(f"{path}: joint_names do not contain the canonical 31-DOF set")
                    # The upright backhand bank is valid but stores joints in
                    # an interleaved left/right/waist order.  Normalize by
                    # name before selecting lower/upper channels; never treat
                    # a different serialized order as a different movement.
                    reorder = [names.index(name) for name in BACKEND_JOINTS]
                    joint_pos = joint_pos[:, reorder]
                    joint_vel = joint_vel[:, reorder]
            hit = int(np.asarray(data["hit_frame"]).reshape(-1)[0]) if "hit_frame" in data else int(
                self.entries[index].get("hit_event", {}).get("motion_hit_frame", round(0.46 * (len(joint_pos) - 1)))
            )
            if not 0 <= hit < len(joint_pos):
                raise ValueError(f"{path}: hit_frame={hit} outside trajectory length {len(joint_pos)}")
        self.selected_index = index
        self.motion_path = path
        self.fps = fps
        self.hit_frame = hit
        self.lower = joint_pos[:, [BACKEND_JOINTS.index(name) for name in LOWER_JOINTS]]
        self.upper = joint_pos[:, [BACKEND_JOINTS.index(name) for name in UPPER_JOINTS]]
        self._upper_dq = joint_vel[:, [BACKEND_JOINTS.index(name) for name in UPPER_JOINTS]]
        self.strike_q = joint_pos[:, [BACKEND_JOINTS.index(name) for name in STRIKE_JOINTS]]
        self.strike_dq = joint_vel[:, [BACKEND_JOINTS.index(name) for name in STRIKE_JOINTS]]
        self.phase = np.linspace(0.0, 1.0, len(joint_pos), dtype=np.float32)

    def begin(self, target: StrikeTarget, adapter: "MujocoV13BAdapter") -> None:
        index = self._choose_index(target, adapter)
        if self.selected_index != index:
            self._load(index)

    def target_for(self, adapter: "MujocoV13BAdapter", index: int | None = None) -> StrikeTarget:
        chosen = self.motion_index if index is None else int(index)
        if chosen is None:
            chosen = int(self._candidates[0])
        self._load(chosen)
        position, velocity, normal = self._entry_goal(self.entries[chosen])
        root = np.asarray(adapter.data.qpos[:3], dtype=np.float64)
        heading = yaw_quat(np.asarray(adapter.data.qpos[3:7], dtype=np.float64))
        return StrikeTarget(
            root + quat_rotate(heading, position),
            quat_rotate(heading, velocity),
            quat_rotate(heading, normal),
            (READY_UPPER_PRELUDE_STEPS + self.hit_frame) / float(self.fps),
            swing_type=(
                1.0 if str(self.entries[chosen].get("stroke_type", "")).lower() == "backhand"
                else -1.0 if str(self.entries[chosen].get("stroke_type", "")).lower() == "forehand"
                else 0.0
            ),
        )

    @staticmethod
    def _sample(array: np.ndarray, index: int) -> np.ndarray:
        return array[min(max(int(index), 0), len(array) - 1)]

    def frame(self, step: int, adapter: "MujocoV13BAdapter") -> ReferenceFrame:
        if self.selected_index is None:
            raise RuntimeError("MotionManifestReferenceProvider.begin() was not called")
        raw = max(0, int(step) - READY_UPPER_PRELUDE_STEPS)
        q = self._sample(self.strike_q, raw)
        dq = self._sample(self.strike_dq, raw)
        if step < READY_UPPER_PRELUDE_STEPS:
            ready = adapter.joint_vector(STRIKE_JOINTS).astype(np.float32)
            alpha = min(1.0, step / float(READY_UPPER_PRELUDE_STEPS))
            q = ready + alpha * (q - ready)
        dq8 = self._sample(self.strike_dq, raw + 8)
        dq16 = self._sample(self.strike_dq, raw + 16)
        upper = self._sample(self.upper, raw)
        lead = upper.copy()
        lead[[3, 5]] = self._sample(self.upper, raw + 12)[[3, 5]]
        if step < READY_UPPER_PRELUDE_STEPS:
            lead = adapter.joint_vector(UPPER_JOINTS).astype(np.float32)
        elif UPPER_RELEASE_STEPS > 0:
            ready = adapter.joint_vector(UPPER_JOINTS).astype(np.float32)
            alpha = min(1.0, (step - READY_UPPER_PRELUDE_STEPS + 1) / float(UPPER_RELEASE_STEPS))
            lead = ready + alpha * (lead - ready)
        return ReferenceFrame(
            self._sample(self.lower, raw), upper, lead,
            self._sample(self._upper_dq, raw),
            q, dq, dq8, dq16,
            float(self._sample(self.phase, raw)),
        )


@dataclass(frozen=True)
class AdapterStep:
    model_output: ThreeModelOutput
    target_joint_positions: np.ndarray
    lower_observation: np.ndarray
    upper_observation: np.ndarray
    student_observation: np.ndarray


@dataclass(frozen=True)
class HardwareCommand:
    """Canonical 31-DOF command before the robot-SDK scatter.

    The real robot path consumes q/dq/Kp/Kd/tau_ff.  ``tau_ff`` is zero by
    default because the MuJoCo-only bias compensation must not be mistaken
    for a calibrated hardware gravity model.  The SDK-specific 31-slot
    scatter remains outside this portable package.
    """

    joint_names: tuple[str, ...]
    q_des: np.ndarray
    dq_des: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    tau_ff: np.ndarray


class MujocoV13BAdapter:
    """State/observation/action adapter for the A3 MuJoCo XML."""

    def __init__(
        self,
        xml_path: str | Path,
        package_root: str | Path,
        *,
        reference: ReferenceProvider | None = None,
        enable_priors: bool | None = None,
        device: str = "cpu",
        kp: float = 1.0,
        kd: float = 1.0,
        policy_hz: float = 50.0,
        low_level_profile: str = "official_pd",
    ) -> None:
        if mujoco is None:
            raise ImportError("mujoco is required for MujocoV13BAdapter; install mujoco>=3.0")
        self.xml_path = Path(xml_path).expanduser().resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.runtime = ThreeModelRuntime(package_root, device=device)
        if enable_priors is True and reference is None:
            raise ValueError("enable_priors=True requires a private ReferenceProvider")
        if enable_priors is True and isinstance(reference, ReadyHoldReference):
            raise ValueError("ReadyHoldReference is wiring-only; use NpzReferenceProvider for complete priors")
        self.reference = reference or ReadyHoldReference()
        self.enable_priors = bool(enable_priors) if enable_priors is not None else reference is not None
        self.kp = float(kp)
        self.kd = float(kd)
        self.low_level_config = MujocoLowLevelConfig.resolve(low_level_profile)
        self.policy_hz = float(policy_hz)
        if self.policy_hz <= 0.0:
            raise ValueError("policy_hz must be > 0")
        self.control_decimation = max(1, int(round(1.0 / (self.policy_hz * float(self.model.opt.timestep)))))
        if self.kp <= 0 or self.kd < 0:
            raise ValueError("kp must be >0 and kd must be >=0")

        self.qpos_addr: dict[str, int] = {}
        self.dof_addr: dict[str, int] = {}
        self.actuator_id: dict[str, int] = {}
        for name in BACKEND_JOINTS:
            jid = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name))
            if jid < 0:
                raise RuntimeError(f"MuJoCo XML is missing canonical joint {name!r}")
            self.qpos_addr[name] = int(self.model.jnt_qposadr[jid])
            self.dof_addr[name] = int(self.model.jnt_dofadr[jid])
            matches = np.flatnonzero(self.model.actuator_trnid[:, 0] == jid)
            if matches.size != 1:
                raise RuntimeError(f"joint {name!r} must map to exactly one motor, got {matches.tolist()}")
            self.actuator_id[name] = int(matches[0])
        self.wrist_body_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_Link"))
        if self.wrist_body_id < 0:
            raise RuntimeError("MuJoCo XML is missing right_wrist_yaw_Link")

        self.ready = self._load_ready_pose(package_root)
        self.ready_by_name = dict(zip(BACKEND_JOINTS, self.ready))
        self.previous_action = np.zeros(26, dtype=np.float32)
        self.previous_lower_action = np.zeros(14, dtype=np.float32)
        self.previous_upper_action = np.zeros(10, dtype=np.float32)
        self.microstep_state = np.zeros(4, dtype=np.float32)
        self.step_index = 0
        self._reference_started = False
        self.target_positions = self.ready.copy()
        self.target_velocities = np.zeros(len(BACKEND_JOINTS), dtype=np.float64)
        self.emergency_stopped = False
        # Keep real-robot gains separate from the MuJoCo-only candidate.  A
        # simulation profile may be much stiffer than the hardware contract,
        # but it must never leak into HardwareCommand.
        self._hardware_kp_by_joint = np.asarray(
            [DEFAULT_KP[name] for name in BACKEND_JOINTS], dtype=np.float64
        ) * self.kp
        self._hardware_kd_by_joint = np.asarray(
            [DEFAULT_KD[name] for name in BACKEND_JOINTS], dtype=np.float64
        ) * self.kd
        self._kp_by_joint = (
            np.asarray(self.low_level_config.kp, dtype=np.float64)
            * self.kp
            * float(self.low_level_config.drive_kp_scale)
        )
        self._kd_by_joint = (
            np.asarray(self.low_level_config.kd, dtype=np.float64)
            * self.kd
            * float(self.low_level_config.drive_kd_scale)
        )
        self._joint_lower = np.asarray([float(self.model.jnt_range[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name), 0]) for name in BACKEND_JOINTS])
        self._joint_upper = np.asarray([float(self.model.jnt_range[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name), 1]) for name in BACKEND_JOINTS])
        self._target_rate_rad_s = 4.0
        self._velocity_limit = np.asarray([JOINT_VELOCITY_LIMIT[name] for name in BACKEND_JOINTS], dtype=np.float64)
        self._force_limit = np.full(len(BACKEND_JOINTS), np.inf, dtype=np.float64)
        for i, name in enumerate(BACKEND_JOINTS):
            aid = self.actuator_id[name]
            if bool(self.model.actuator_forcelimited[aid]):
                self._force_limit[i] = float(abs(self.model.actuator_forcerange[aid, 1]))
        self._balance_controller = SimulationBalanceController(self, self.low_level_config)
        self.last_drive_torque = np.zeros(len(BACKEND_JOINTS), dtype=np.float32)
        self.last_balance_torque = np.zeros(len(BACKEND_JOINTS), dtype=np.float32)

    @staticmethod
    def _load_ready_pose(package_root: str | Path) -> np.ndarray:
        # This parser intentionally handles the fixed contract without adding
        # PyYAML as a hidden runtime dependency.
        path = Path(package_root).expanduser().resolve() / "contracts" / "ready_pose.yaml"
        values: dict[str, float] = {}
        in_backend = False
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped == "backend_joint_positions_rad:":
                in_backend = True
                continue
            if in_backend and line.startswith("  ") and ":" in stripped:
                key, value = stripped.split(":", 1)
                try:
                    values[key.strip()] = float(value.strip())
                except ValueError:
                    pass
        missing = [name for name in BACKEND_JOINTS if name not in values]
        if missing:
            raise ValueError(f"ready_pose.yaml is missing backend joints: {missing}")
        return np.asarray([values[name] for name in BACKEND_JOINTS], dtype=np.float64)

    def reset(self) -> None:
        # Prefer the grounded MJCF READY keyframe whenever the model provides
        # one.  The old hard-coded 1.01953 m root height belongs to the Isaac
        # asset contract; the grounded MuJoCo mesh has a different collision
        # bottom and needs its own calibrated keyframe height.
        if int(self.model.nkey) > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            root = np.asarray((-0.5, -0.7625, 1.019529998075), dtype=np.float64)
            self.data.qpos[:3] = root
            self.data.qpos[3:7] = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
        for name in BACKEND_JOINTS:
            self.data.qpos[self.qpos_addr[name]] = self.ready_by_name[name]
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._balance_controller.reset(self)
        self.previous_action.fill(0.0)
        self.previous_lower_action.fill(0.0)
        self.previous_upper_action.fill(0.0)
        self.microstep_state.fill(0.0)
        self.step_index = 0
        self._reference_started = False
        reset_reference = getattr(self.reference, "reset", None)
        if reset_reference is not None:
            reset_reference()
        self.target_positions = self.ready.copy()
        self.target_velocities.fill(0.0)
        self.emergency_stopped = False
        self.last_drive_torque.fill(0.0)
        self.last_balance_torque.fill(0.0)

    def joint_vector(self, names: tuple[str, ...]) -> np.ndarray:
        return np.asarray([self.data.qpos[self.qpos_addr[name]] for name in names], dtype=np.float32)

    def joint_velocity_vector(self, names: tuple[str, ...]) -> np.ndarray:
        return np.asarray([self.data.qvel[self.dof_addr[name]] for name in names], dtype=np.float32)

    def _root_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = np.asarray(self.data.qpos[3:7], dtype=np.float64)
        lin = quat_rotate_inverse(q, np.asarray(self.data.qvel[:3], dtype=np.float64)).astype(np.float32)
        ang = quat_rotate_inverse(q, np.asarray(self.data.qvel[3:6], dtype=np.float64)).astype(np.float32)
        gravity = quat_rotate_inverse(q, np.asarray((0.0, 0.0, -1.0), dtype=np.float64)).astype(np.float32)
        return lin, ang, gravity

    def racket_state_local(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        bid = self.wrist_body_id
        wrist_pos = np.asarray(self.data.xpos[bid], dtype=np.float64)
        wrist_quat = np.asarray(self.data.xquat[bid], dtype=np.float64)
        # MuJoCo exposes body spatial velocity in ``cvel`` as world-frame
        # angular[0:3] followed by linear[3:6].  There are no xvelp/xvelr
        # arrays in the Python bindings.
        wrist_ang = np.asarray(self.data.cvel[bid, :3], dtype=np.float64)
        wrist_lin = np.asarray(self.data.cvel[bid, 3:6], dtype=np.float64)
        offset_world = quat_rotate(wrist_quat, MOUNT_OFFSET)
        racket_pos = wrist_pos + offset_world
        racket_vel = wrist_lin + np.cross(wrist_ang, offset_world)
        # The A3 mount quaternion is identity and its face normal is +Y.
        racket_normal = quat_rotate(wrist_quat, np.asarray((0.0, 1.0, 0.0), dtype=np.float64))
        root_pos = np.asarray(self.data.qpos[:3], dtype=np.float64)
        root_quat = yaw_quat(np.asarray(self.data.qpos[3:7], dtype=np.float64))
        return (
            quat_rotate_inverse(root_quat, racket_pos - root_pos).astype(np.float32),
            quat_rotate_inverse(root_quat, racket_vel).astype(np.float32),
            quat_rotate_inverse(root_quat, racket_normal).astype(np.float32),
        )

    def _target_local(self, target: StrikeTarget) -> np.ndarray:
        root_pos = np.asarray(self.data.qpos[:3], dtype=np.float64)
        root_quat = yaw_quat(np.asarray(self.data.qpos[3:7], dtype=np.float64))
        return np.concatenate((
            quat_rotate_inverse(root_quat, target.position_world.astype(np.float64) - root_pos),
            quat_rotate_inverse(root_quat, target.velocity_world.astype(np.float64)),
            quat_rotate_inverse(root_quat, target.normal_world.astype(np.float64)),
        )).astype(np.float32)

    def _base14_relative(self) -> tuple[np.ndarray, np.ndarray]:
        q = self.joint_vector(BASE14_JOINTS)
        dq = self.joint_velocity_vector(BASE14_JOINTS)
        default = np.asarray([LEGACY_DEFAULT[name] for name in BASE14_JOINTS], dtype=np.float32)
        return q - default, dq

    def build_student_observation(self, target: StrikeTarget) -> np.ndarray:
        lin, ang, gravity = self._root_state()
        student_joints = LOWER_JOINTS + UPPER_JOINTS
        q = self.joint_vector(student_joints)
        dq = self.joint_velocity_vector(student_joints)
        racket_pos, racket_vel, racket_normal = self.racket_state_local()
        target_local = self._target_local(target)
        # V1.3B does not feed raw goal coordinates to model_5000.  The
        # training function uses this fixed audited target-bank normalization.
        goal = np.concatenate((
            (target_local[:3] - np.asarray((0.44237322, -0.34721070, 0.09162542), dtype=np.float32))
            / np.asarray((0.04256963, 0.29942963, 0.06187854), dtype=np.float32),
            target_local[3:6] / 2.0,
            target_local[6:9],
            np.asarray((np.clip(target.hit_time_s - self.step_index * self.control_dt, -4.0, 4.0),), dtype=np.float32),
        ))
        goal = np.clip(goal, -5.0, 5.0)
        student_default = np.asarray(
            [LEGACY_DEFAULT[n] if n in LEGACY_DEFAULT else STUDENT_DEFAULT_UPPER[n] for n in student_joints],
            dtype=np.float32,
        )
        obs = np.concatenate((lin, ang, gravity, q - student_default,
                              dq, racket_pos, racket_vel, racket_normal, goal, self.previous_action))
        if obs.size != 98 or not np.isfinite(obs).all():
            raise RuntimeError(f"student observation must be finite 98D, got {obs.shape}")
        return obs.astype(np.float32)

    def build_upper_observation(self, target: StrikeTarget, frame: ReferenceFrame) -> np.ndarray:
        _, ang, gravity = self._root_state()
        q = self.joint_vector(UPPER_JOINTS)
        dq = self.joint_velocity_vector(UPPER_JOINTS)
        default = np.asarray([self.ready_by_name[n] for n in UPPER_JOINTS], dtype=np.float32)
        target_local = self._target_local(target)
        actual_pos, actual_vel, actual_normal = self.racket_state_local()
        time_with_prelude = float(target.hit_time_s - self.step_index * self.control_dt +
                                  max(READY_UPPER_PRELUDE_STEPS - self.step_index, 0) * self.control_dt)
        obs = np.concatenate((ang, gravity, q - default, dq, target_local[:3], target_local[3:6], target_local[6:9],
                              actual_pos, actual_vel, actual_normal, np.asarray((time_with_prelude, target.swing_type), dtype=np.float32),
                              self.previous_upper_action))
        if obs.size != 56 or not np.isfinite(obs).all():
            raise RuntimeError(f"upper observation must be finite 56D, got {obs.shape}")
        return obs.astype(np.float32)

    def build_lower_observation(self, target: StrikeTarget, frame: ReferenceFrame) -> np.ndarray:
        lin, ang, gravity = self._root_state()
        base_q, base_dq = self._base14_relative()
        target_local = self._target_local(target)
        actual_pos, actual_vel, actual_normal = self.racket_state_local()
        strike_q = self.joint_vector(STRIKE_JOINTS)
        strike_default = np.asarray([STUDENT_DEFAULT_UPPER[name] for name in STRIKE_JOINTS], dtype=np.float32)
        strike_q = strike_q - strike_default
        strike_dq = self.joint_velocity_vector(STRIKE_JOINTS)
        obs = np.concatenate((
            lin, ang, base_q, base_dq, self.previous_lower_action, gravity,
            target_local[:3], target_local[3:6], target_local[6:9],
            actual_pos, actual_vel, actual_normal,
            np.asarray((target.hit_time_s - self.step_index * self.control_dt, target.swing_type), dtype=np.float32),
            strike_q, strike_dq, frame.strike_joint_pos, frame.strike_joint_vel,
            frame.strike_joint_vel_8, frame.strike_joint_vel_16,
            np.asarray((frame.phase,), dtype=np.float32),
        ))
        if obs.size != 126 or not np.isfinite(obs).all():
            raise RuntimeError(f"lower observation must be finite 126D, got {obs.shape}")
        return obs.astype(np.float32)

    @property
    def control_dt(self) -> float:
        return float(self.control_decimation * self.model.opt.timestep)

    def _microstep_delta(self, action: np.ndarray) -> np.ndarray:
        self.microstep_state = 0.8 * self.microstep_state + 0.2 * np.asarray(action[22:26], dtype=np.float32)
        lx, ly, rx, ry = self.microstep_state
        delta = np.zeros(12, dtype=np.float32)
        delta[[0, 3, 4]] += np.asarray((-0.055, 0.025, 0.030), dtype=np.float32) * lx
        delta[[1, 5]] += np.asarray((0.045, -0.040), dtype=np.float32) * ly
        delta[[6, 9, 10]] += np.asarray((-0.055, 0.025, 0.030), dtype=np.float32) * rx
        delta[[7, 11]] += np.asarray((0.045, -0.040), dtype=np.float32) * ry
        return delta

    def infer_targets(self, target: StrikeTarget) -> AdapterStep:
        frame = self.reference.frame(self.step_index, self)
        student_obs = self.build_student_observation(target)
        lower_obs = self.build_lower_observation(target, frame)
        upper_obs = self.build_upper_observation(target, frame)
        output = self.runtime.infer(torch.from_numpy(student_obs), torch.from_numpy(lower_obs), torch.from_numpy(upper_obs))
        student = output.student_action.detach().cpu().numpy().reshape(-1)
        lower_prior = output.lower_prior_action.detach().cpu().numpy().reshape(-1)
        upper_prior = output.upper_prior_action.detach().cpu().numpy().reshape(-1)
        bounded = np.tanh(student).astype(np.float32)
        ready_lower = np.asarray([self.ready_by_name[n] for n in LOWER_JOINTS], dtype=np.float32)
        ready_upper = np.asarray([self.ready_by_name[n] for n in UPPER_JOINTS], dtype=np.float32)
        legacy_lower = np.asarray([LEGACY_DEFAULT[n] for n in LOWER_JOINTS], dtype=np.float32)
        lower_prior_raw = 0.50 * np.tanh(lower_prior[:12] / 0.50) if self.enable_priors else np.zeros(12, dtype=np.float32)
        upper_prior_raw = np.clip(upper_prior, -0.50, 0.50) if self.enable_priors else np.zeros(10, dtype=np.float32)
        # CompletePriors uses the same READY->teacher bridge as the training
        # action term: zero during the 50-step prelude, then release over 12
        # control steps.  The reference provider supplies the +12 shoulder
        # lead, while the gate prevents a first-frame upper-body kick.
        if not self.enable_priors or self.step_index < READY_UPPER_PRELUDE_STEPS:
            upper_gate = 0.0
        else:
            upper_gate = min(1.0, (self.step_index - READY_UPPER_PRELUDE_STEPS) / float(UPPER_RELEASE_STEPS))
        upper_teacher_target = frame.upper_reference_lead + upper_gate * upper_prior_raw * UPPER_PRIOR_SCALE
        upper_prior_target = ready_upper + upper_gate * (upper_teacher_target - ready_upper)
        lower_prior_target = legacy_lower + lower_prior_raw * LOWER_PRIOR_SCALE if self.enable_priors else ready_lower
        lower_target, upper_target = self.runtime.blend_targets(
            torch.from_numpy(bounded),
            ready_lower=torch.from_numpy(ready_lower),
            lower_prior_target=torch.from_numpy(lower_prior_target),
            ready_upper=torch.from_numpy(ready_upper),
            upper_prior_target=torch.from_numpy(upper_prior_target),
            microstep_delta=torch.from_numpy(self._microstep_delta(bounded)),
        )
        target_positions = self.ready.copy()
        target_velocities = np.zeros(len(BACKEND_JOINTS), dtype=np.float64)
        target_positions[[BACKEND_JOINTS.index(n) for n in LOWER_JOINTS]] = lower_target.detach().cpu().numpy().reshape(-1)
        target_positions[[BACKEND_JOINTS.index(n) for n in UPPER_JOINTS]] = upper_target.detach().cpu().numpy().reshape(-1)
        if upper_gate > 0.0:
            # Training used task-phase feed-forward only for right shoulder
            # pitch and yaw, with beta=0.75.
            target_velocities[BACKEND_JOINTS.index("right_shoulder_pitch_joint")] = upper_gate * 0.75 * frame.upper_velocity[3]
            target_velocities[BACKEND_JOINTS.index("right_shoulder_yaw_joint")] = upper_gate * 0.75 * frame.upper_velocity[5]
        if not np.isfinite(target_positions).all():
            raise RuntimeError("target joint position contains NaN/Inf")
        target_positions = np.clip(target_positions, self._joint_lower * 0.95, self._joint_upper * 0.95)
        max_step = self._target_rate_rad_s * self.control_dt
        target_positions = self.target_positions + np.clip(target_positions - self.target_positions, -max_step, max_step)
        envelope = self.low_level_config.lower_target_envelope_rad
        if envelope is not None:
            lower_indices = np.asarray(
                [BACKEND_JOINTS.index(name) for name in LOWER_JOINTS], dtype=np.int64
            )
            target_positions[lower_indices] = np.clip(
                target_positions[lower_indices],
                self.ready[lower_indices] - float(envelope),
                self.ready[lower_indices] + float(envelope),
            )
        self.target_positions = target_positions
        self.target_velocities = target_velocities
        self.previous_action = bounded.copy()
        self.previous_lower_action = lower_prior.copy() if self.enable_priors else np.zeros(14, dtype=np.float32)
        self.previous_upper_action = upper_prior_raw.astype(np.float32) if self.enable_priors else np.zeros(10, dtype=np.float32)
        return AdapterStep(output, target_positions.copy(), lower_obs, upper_obs, student_obs)

    def apply_target_torque(self) -> np.ndarray:
        if self.emergency_stopped:
            self.data.ctrl[:] = 0.0
            return np.zeros(len(BACKEND_JOINTS), dtype=np.float32)
        q = self.joint_vector(BACKEND_JOINTS).astype(np.float64)
        dq = self.joint_velocity_vector(BACKEND_JOINTS).astype(np.float64)
        # The MuJoCo XML contains torque motors rather than position actuators.
        # Add the current generalized bias term (gravity/Coriolis) so READY
        # does not immediately collapse while the learned residual is still
        # being ramped in.  The final actuator force limit remains authoritative.
        bias = np.asarray([self.data.qfrc_bias[self.dof_addr[name]] for name in BACKEND_JOINTS], dtype=np.float64)
        over_speed = np.abs(dq) > self._velocity_limit
        safe_q_target = np.where(over_speed, q, self.target_positions)
        safe_dq_target = np.where(over_speed, 0.0, self.target_velocities)
        drive_torque = self._kp_by_joint * (safe_q_target - q) + self._kd_by_joint * (safe_dq_target - dq) + bias
        balance_torque = self._balance_controller.torque(self)
        torque = drive_torque + balance_torque
        torque = np.clip(torque, -self._force_limit, self._force_limit)
        self.last_drive_torque = drive_torque.astype(np.float32)
        self.last_balance_torque = balance_torque.astype(np.float32)
        self.data.ctrl[:] = 0.0
        for i, name in enumerate(BACKEND_JOINTS):
            self.data.ctrl[self.actuator_id[name]] = torque[i]
        return torque.astype(np.float32)

    def hardware_command(self) -> HardwareCommand:
        """Return the safe canonical command for a real-robot adapter."""
        return HardwareCommand(
            joint_names=BACKEND_JOINTS,
            q_des=self.target_positions.astype(np.float64, copy=True),
            dq_des=self.target_velocities.astype(np.float64, copy=True),
            kp=self._hardware_kp_by_joint.astype(np.float64, copy=True),
            kd=self._hardware_kd_by_joint.astype(np.float64, copy=True),
            tau_ff=np.zeros(len(BACKEND_JOINTS), dtype=np.float64),
        )

    def safe_halt(self) -> None:
        """Disable all MuJoCo motors and hold no stale learned command."""
        self.emergency_stopped = True
        self.target_positions = self.joint_vector(BACKEND_JOINTS).astype(np.float64)
        self.target_velocities.fill(0.0)
        self.previous_action.fill(0.0)
        self.previous_lower_action.fill(0.0)
        self.previous_upper_action.fill(0.0)
        self.data.ctrl[:] = 0.0

    def safe_halt_command(self) -> HardwareCommand:
        """Build the same zero-gain safe-halt shape used by the C++ driver."""
        q = self.joint_vector(BACKEND_JOINTS).astype(np.float64)
        zeros = np.zeros(len(BACKEND_JOINTS), dtype=np.float64)
        return HardwareCommand(BACKEND_JOINTS, q, zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy())

    def step(self, target: StrikeTarget) -> AdapterStep:
        if self.emergency_stopped:
            raise RuntimeError("MuJoCo adapter is emergency-stopped; call reset() before resuming")
        if not self._reference_started:
            begin_reference = getattr(self.reference, "begin", None)
            if begin_reference is not None:
                begin_reference(target, self)
            self._reference_started = True
        result = self.infer_targets(target)
        for _ in range(self.control_decimation):
            self.apply_target_torque()
            mujoco.mj_step(self.model, self.data)
        self.step_index += 1
        return result
