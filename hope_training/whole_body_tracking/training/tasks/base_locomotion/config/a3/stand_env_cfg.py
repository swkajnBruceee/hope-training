"""Agibot A3 embodiment bindings for deterministic Base Stand smoke."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import training.tasks.base_locomotion.mdp as mdp
from training.robots.agibot_a3 import A3_BASE_ACTION_JOINTS, AGIBOT_A3_CFG
from training.tasks.base_locomotion.base_env_cfg import (
    A3BaseStandEnvCfg as BaseStandEnvCfg,
    EventCfg as BaseEventCfg,
    RewardsCfg as BaseRewardsCfg,
    TerminationsCfg as BaseTerminationsCfg,
)
from training.robots.agibot_a3 import A3_ANCHOR_BODY


# Diagnostic Base14 actuator candidate.  Each scale is 0.25 * effort_limit / Kp
# under the PD_STAND gains selected by the passive-stability ablation.  The raw
# v1 clip remains +/-0.25, so the first smoke can command at most 6.25% of the
# simulated effort limit.  Neither these scales nor the gains are deployment-approved.
A3_PD_STAND_BASE_ACTION_SCALE_RAD = (
    0.03666666666666667,  # left hip pitch: 0.25 * 220 / 1500
    0.1375,               # left hip roll:  0.25 * 220 / 400
    0.18333333333333332,  # left hip yaw:   0.25 * 220 / 300
    0.04,                 # left knee:      0.25 * 320 / 2000
    0.0591,               # left ankle pitch: 0.25 * 118.2 / 500
    0.027375,             # left ankle roll:  0.25 * 54.75 / 500
    0.03666666666666667,
    0.1375,
    0.18333333333333332,
    0.04,
    0.0591,
    0.027375,
    0.023,                # waist roll:  0.25 * 46 / 500
    0.059,                # waist pitch: 0.25 * 118 / 500
)


@configclass
class A3BaseStandEnvCfg(BaseStandEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = AGIBOT_A3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class A3BaseStandAuthorityCandidateEnvCfg(A3BaseStandEnvCfg):
    """Diagnostic Stand variant changing waist pitch authority only.

    This profile is not the deployment contract.  It isolates the existing
    project ``blend_50`` waist-pitch candidate (350/7) after the v1 smoke
    demonstrated that 50/2 cannot support the bounded residual architecture.
    """

    def __post_init__(self):
        super().__post_init__()
        waist = self.scene.robot.actuators["waist"]
        waist.stiffness = {**waist.stiffness, "waist_pitch_joint": 350.0}
        waist.damping = {**waist.damping, "waist_pitch_joint": 7.0}


@configclass
class A3BaseStandClipCandidateEnvCfg(A3BaseStandEnvCfg):
    """Diagnostic Stand variant changing only normalized action authority."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.base.raw_clip = 0.5


@configclass
class A3BaseStandAuthorityClipCandidateEnvCfg(A3BaseStandAuthorityCandidateEnvCfg):
    """Final cell of the bounded 2x2 authority diagnostic."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.base.raw_clip = 0.5


@configclass
class A3BaseStandRewardV2Cfg(BaseRewardsCfg):
    """Reward candidate with DOF-normalized posture and explicit failure cost."""

    joint_posture = RewTerm(
        func=mdp.joint_posture_l2,
        weight=-0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(A3_BASE_ACTION_JOINTS)),
            "normalize_by_dof": True,
        },
    )
    # RewardManager multiplies every weight by policy_dt=0.02 s.  -100 thus
    # produces -2.0 on a non-timeout failure, equal to two seconds of healthy
    # alive reward (100 * 1.0 * 0.02) and zero on normal timeouts.
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)
    # action_rate alone cannot distinguish a small command from a constant
    # saturated command.  Penalize ActionManager's pre-term action so PPO pays
    # for values outside the environment clip instead of hiding behind it.
    action_magnitude = RewTerm(func=mdp.action_l2, weight=-0.05)


@configclass
class A3BaseStandPassiveStableCandidateEnvCfg(A3BaseStandEnvCfg):
    """Simulation-only Base14 PD_STAND candidate selected by gain ablation."""

    rewards: A3BaseStandRewardV2Cfg = A3BaseStandRewardV2Cfg()

    def __post_init__(self):
        super().__post_init__()
        legs = self.scene.robot.actuators["legs"]
        legs.stiffness = {
            ".*_hip_pitch_joint": 1500.0,
            ".*_hip_roll_joint": 400.0,
            ".*_hip_yaw_joint": 300.0,
            ".*_knee_joint": 2000.0,
        }
        legs.damping = {
            ".*_hip_pitch_joint": 8.0,
            ".*_hip_roll_joint": 7.0,
            ".*_hip_yaw_joint": 7.0,
            ".*_knee_joint": 8.0,
        }
        feet = self.scene.robot.actuators["feet"]
        feet.stiffness = 500.0
        feet.damping = 5.0
        waist = self.scene.robot.actuators["waist"]
        waist.stiffness = {
            **waist.stiffness,
            "waist_roll_joint": 500.0,
            "waist_pitch_joint": 500.0,
        }
        waist.damping = {
            **waist.damping,
            "waist_roll_joint": 4.0,
            "waist_pitch_joint": 4.0,
        }
        self.actions.base.action_scale_rad = A3_PD_STAND_BASE_ACTION_SCALE_RAD
        self.actions.base.raw_clip = 0.25


@configclass
class A3BaseStandRecoveryAEventCfg(BaseEventCfg):
    """Recovery-A reset mixture: 50% clean, 35% low, and 15% medium."""

    reset_all = EventTerm(
        func=mdp.reset_scene_with_recovery_a_disturbance,
        mode="reset",
        params={
            "undisturbed_fraction": 0.50,
            "roll_pitch_range_rad": (-0.035, 0.035),
            "angular_velocity_range_rad_s": (-0.20, 0.20),
            # Conditional on the disturbed slice: 30% of 50% = 15% overall.
            "medium_fraction": 0.30,
            "medium_roll_pitch_range_rad": (-0.05, 0.05),
            "medium_angular_velocity_range_rad_s": (-0.30, 0.30),
        },
    )


@configclass
class A3BaseStandRecoveryAV2EventCfg(BaseEventCfg):
    """V2 reset mixture: 60% clean, 30% low, and 10% medium."""

    reset_all = EventTerm(
        func=mdp.reset_scene_with_recovery_a_disturbance,
        mode="reset",
        params={
            "undisturbed_fraction": 0.60,
            "roll_pitch_range_rad": (-0.035, 0.035),
            "angular_velocity_range_rad_s": (-0.20, 0.20),
            "medium_fraction": 0.25,  # 25% of the 40% disturbed slice = 10% overall.
            "medium_roll_pitch_range_rad": (-0.05, 0.05),
            "medium_angular_velocity_range_rad_s": (-0.30, 0.30),
        },
    )


@configclass
class A3BaseStandRecoveryARewardsCfg(A3BaseStandRewardV2Cfg):
    """Unapproved Recovery-A reward candidate."""

    recovery_tilt_progress = RewTerm(func=mdp.RecoveryTiltProgress, weight=2.0)
    undisturbed_action_magnitude = RewTerm(func=mdp.undisturbed_action_l2, weight=-0.20)


@configclass
class A3BaseStandRecoveryAV2RewardsCfg(A3BaseStandRecoveryARewardsCfg):
    """V2 reward: make raw saturation and healthy-state residuals expensive."""

    raw_action_excess = RewTerm(
        func=mdp.raw_action_excess_l2,
        weight=-8.0,
        params={"raw_limit": 0.0625},
    )
    physical_residual = RewTerm(func=mdp.physical_residual_l2, weight=-1.0)
    healthy_action = RewTerm(func=mdp.healthy_action_l2, weight=-0.50)


@configclass
class A3BaseStandRecoveryAV21RewardsCfg(A3BaseStandRecoveryAV2RewardsCfg):
    """V2.1 reward with continuous observable recovery-potential progress."""

    recovery_potential_progress = RewTerm(
        func=mdp.RecoveryPotentialProgress,
        weight=0.50,
        params={
            "tilt_scale_rad": 0.05,
            "angular_velocity_scale_rad_s": 0.20,
            "height_scale_m": 0.02,
        },
    )


@configclass
class A3BaseStandRecoveryATerminationsCfg(BaseTerminationsCfg):
    recovery_envelope = DoneTerm(
        func=mdp.SustainedTorsoTiltExceeded,
        params={
            "torso_body_name": A3_ANCHOR_BODY,
            "max_tilt_rad": 0.55,
            "required_steps": 3,
        },
    )


@configclass
class A3BaseStandRecoveryAEnvCfg(A3BaseStandPassiveStableCandidateEnvCfg):
    """Development-only Recovery-A task; PPO remains externally gated off."""

    events: A3BaseStandRecoveryAEventCfg = A3BaseStandRecoveryAEventCfg()
    rewards: A3BaseStandRecoveryARewardsCfg = A3BaseStandRecoveryARewardsCfg()
    terminations: A3BaseStandRecoveryATerminationsCfg = A3BaseStandRecoveryATerminationsCfg()


_A3_RECOVERY_V2_SCALE = tuple(
    value * (0.35 if joint == "waist_pitch_joint" else 0.50)
    for joint, value in zip(A3_BASE_ACTION_JOINTS, A3_PD_STAND_BASE_ACTION_SCALE_RAD)
)
_A3_RECOVERY_V2_MASK = tuple(
    0.0 if joint == "waist_pitch_joint" else 1.0 for joint in A3_BASE_ACTION_JOINTS
)


@configclass
class A3BaseStandRecoveryAV2EnvCfg(A3BaseStandPassiveStableCandidateEnvCfg):
    """Recovery-A v2: conservative per-joint authority and raw-bound rewards."""

    events: A3BaseStandRecoveryAV2EventCfg = A3BaseStandRecoveryAV2EventCfg()
    rewards: A3BaseStandRecoveryAV2RewardsCfg = A3BaseStandRecoveryAV2RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.actions.base.action_scale_rad = _A3_RECOVERY_V2_SCALE
        self.actions.base.raw_clip = 0.125


@configclass
class A3BaseStandRecoveryAV2WaistMaskEnvCfg(A3BaseStandRecoveryAV2EnvCfg):
    """Recovery-A v2 ablation with waist-pitch residual physically masked."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.base.action_mask = _A3_RECOVERY_V2_MASK


@configclass
class A3BaseStandRecoveryAV21WaistMaskEnvCfg(A3BaseStandRecoveryAV2WaistMaskEnvCfg):
    """Recovery-A v2.1: 70/20/10 curriculum, potential progress, waist masked."""

    events: A3BaseStandRecoveryAV2EventCfg = A3BaseStandRecoveryAV2EventCfg()
    rewards: A3BaseStandRecoveryAV21RewardsCfg = A3BaseStandRecoveryAV21RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_all.params["undisturbed_fraction"] = 0.70
        self.events.reset_all.params["medium_fraction"] = 1.0 / 3.0
