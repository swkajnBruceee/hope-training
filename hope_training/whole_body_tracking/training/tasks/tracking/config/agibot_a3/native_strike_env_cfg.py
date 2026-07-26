"""A3 native-MOTION strike policy task.

This task keeps the existing Isaac A3 body for training, but changes the RL
contract to match the intended real A3 deployment path:

* native MOTION/MC owns standing, balance, legs, head, and the non-paddle arm;
* the learned policy commands only waist + right arm joint targets;
* manifest/reference motion is used as a strike teacher and motion prior, not
  as a full-body actor command.
"""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import training.tasks.tracking.mdp as mdp
from training.tasks.base_locomotion.mdp import (
    A3_PD_STAND_BASE_ACTION_SCALE_RAD,
    A3F1FrozenUpperBaseCompositePositionActionCfg,
    A3F0UpperBaseCompositePositionActionCfg,
    A3FrozenStageAJointCoordinatorActionCfg,
    A3FrozenStageAUpperCorrectionActionCfg,
    A3StrikeConditionedBaseCompositePositionActionCfg,
    RootHeightBelowMinimum,
)
from training.robots.agibot_a3 import (
    A3_BACKEND_JOINTS,
    A3_BASE_ACTION_JOINTS,
    A3_FEET_BODIES,
    A3_NATIVE_STRIKE_JOINTS,
    A3_RIGHT_ARM_JOINTS,
    A3_STRIKE_V2_REFERENCE_JOINTS,
    A3_WAIST_JOINTS,
    AGIBOT_A3_NATIVE_STRIKE_ACTION_SCALE,
)
from training.tasks.tracking.config.agibot_a3.hope_env_cfg import (
    HOPECommandsCfg,
    HOPEEventCfg,
    HOPEPingPongAgibotA3EnvCfg,
)
from training.tasks.tracking.tracking_env_cfg import ActionsCfg, ObservationsCfg, RewardsCfg, TerminationsCfg


def _scale_gain_map(value, scale: float):
    if isinstance(value, dict):
        return {k: float(v) * scale for k, v in value.items()}
    return float(value) * scale


@configclass
class A3NativeStrikeActionsCfg(ActionsCfg):
    joint_pos = mdp.ReferenceResidualJointPositionActionCfg(
        asset_name="robot",
        joint_names=A3_NATIVE_STRIKE_JOINTS,
        scale=AGIBOT_A3_NATIVE_STRIKE_ACTION_SCALE,
        preserve_order=True,
        use_default_offset=True,
        raw_clip=1.0,
        reference_command_name="motion",
    )


@configclass
class A3StrikeConditionedBaseActionsCfg(ActionsCfg):
    """Base14 actor plus phase-indexed Strike reference composer."""

    joint_pos = A3StrikeConditionedBaseCompositePositionActionCfg(
        asset_name="robot",
        base_joint_names=tuple(A3_BASE_ACTION_JOINTS),
        backend_joint_names=tuple(A3_BACKEND_JOINTS),
        strike_joint_names=tuple(A3_STRIKE_V2_REFERENCE_JOINTS),
        action_scale_rad=A3_PD_STAND_BASE_ACTION_SCALE_RAD,
        action_mask=(1.0,) * len(A3_BASE_ACTION_JOINTS),
        raw_clip=0.25,
        clip_to_soft_joint_limits=True,
        reference_command_name="motion",
        base_reference_mode="motion",
    )


@configclass
class A3F0ActionsCfg(ActionsCfg):
    """F0 composite action: external frozen upper actor + Base14 actor."""

    joint_pos = A3F0UpperBaseCompositePositionActionCfg(
        asset_name="robot",
        base_joint_names=tuple(A3_BASE_ACTION_JOINTS),
        backend_joint_names=tuple(A3_BACKEND_JOINTS),
        strike_joint_names=tuple(A3_STRIKE_V2_REFERENCE_JOINTS),
        upper_joint_names=tuple(A3_NATIVE_STRIKE_JOINTS),
        action_scale_rad=A3_PD_STAND_BASE_ACTION_SCALE_RAD,
        action_mask=(1.0,) * len(A3_BASE_ACTION_JOINTS),
        raw_clip=0.25,
        upper_raw_clip=0.50,
        scale=dict(AGIBOT_A3_NATIVE_STRIKE_ACTION_SCALE),
        clip_to_soft_joint_limits=True,
        reference_command_name="motion",
        base_reference_mode="default",
        joint_names=tuple(A3_NATIVE_STRIKE_JOINTS),
        preserve_order=True,
    )


@configclass
class A3F1ActionsCfg(ActionsCfg):
    """F1 composite action with model_900 frozen inside the environment."""

    joint_pos = A3F1FrozenUpperBaseCompositePositionActionCfg(
        asset_name="robot",
        base_joint_names=tuple(A3_BASE_ACTION_JOINTS),
        backend_joint_names=tuple(A3_BACKEND_JOINTS),
        strike_joint_names=tuple(A3_STRIKE_V2_REFERENCE_JOINTS),
        upper_joint_names=tuple(A3_NATIVE_STRIKE_JOINTS),
        action_scale_rad=A3_PD_STAND_BASE_ACTION_SCALE_RAD,
        action_mask=(1.0,) * len(A3_BASE_ACTION_JOINTS),
        raw_clip=0.25,
        upper_raw_clip=0.50,
        scale=dict(AGIBOT_A3_NATIVE_STRIKE_ACTION_SCALE),
        clip_to_soft_joint_limits=True,
        reference_command_name="motion",
        base_reference_mode="default",
        joint_names=tuple(A3_NATIVE_STRIKE_JOINTS),
        preserve_order=True,
        upper_observation_group="upper",
    )


@configclass
class A3UpperCorrectionActionsCfg(ActionsCfg):
    """PPO publishes only a 10-D upper correction; both parent actors are frozen."""

    joint_pos = A3FrozenStageAUpperCorrectionActionCfg(
        asset_name="robot",
        base_joint_names=tuple(A3_BASE_ACTION_JOINTS),
        backend_joint_names=tuple(A3_BACKEND_JOINTS),
        strike_joint_names=tuple(A3_STRIKE_V2_REFERENCE_JOINTS),
        upper_joint_names=tuple(A3_NATIVE_STRIKE_JOINTS),
        action_scale_rad=A3_PD_STAND_BASE_ACTION_SCALE_RAD,
        action_mask=(1.0,) * 12 + (0.0, 0.0),
        raw_clip=1.0,
        upper_raw_clip=0.50,
        scale=dict(AGIBOT_A3_NATIVE_STRIKE_ACTION_SCALE),
        clip_to_soft_joint_limits=True,
        reference_command_name="motion",
        base_reference_mode="default",
        joint_names=tuple(A3_NATIVE_STRIKE_JOINTS),
        preserve_order=True,
        upper_observation_group="upper",
    )


@configclass
class A3JointCoordinatorActionsCfg(ActionsCfg):
    """One PPO action split into leg, waist and right-arm corrections."""

    joint_pos = A3FrozenStageAJointCoordinatorActionCfg(
        asset_name="robot",
        base_joint_names=tuple(A3_BASE_ACTION_JOINTS),
        backend_joint_names=tuple(A3_BACKEND_JOINTS),
        strike_joint_names=tuple(A3_STRIKE_V2_REFERENCE_JOINTS),
        upper_joint_names=tuple(A3_NATIVE_STRIKE_JOINTS),
        action_scale_rad=A3_PD_STAND_BASE_ACTION_SCALE_RAD,
        # This mask applies only to frozen model_3396. Its legacy waist
        # outputs stay disabled while the coordinator owns all waist deltas.
        action_mask=(1.0,) * 12 + (0.0, 0.0),
        raw_clip=1.0,
        smooth_raw_bound=True,
        upper_raw_clip=0.50,
        scale=dict(AGIBOT_A3_NATIVE_STRIKE_ACTION_SCALE),
        clip_to_soft_joint_limits=True,
        reference_command_name="motion",
        base_reference_mode="default",
        joint_names=tuple(A3_NATIVE_STRIKE_JOINTS),
        preserve_order=True,
        upper_observation_group="upper",
    )


@configclass
class A3NativeStrikeObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        # Remove inherited motion-reference actor terms. The deploy actor sees
        # proprioception and a planner/manifest strike command only.
        command = None
        motion_anchor_pos_b = None
        motion_anchor_ori_b = None
        base_lin_vel = None

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=A3_NATIVE_STRIKE_JOINTS, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=A3_NATIVE_STRIKE_JOINTS, preserve_order=True)},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        racket_target_pos_b = ObsTerm(
            func=mdp.racket_target_pos_b,
            params={"command_name": "racket_target"},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        racket_target_vel_b = ObsTerm(func=mdp.racket_target_vel_b, params={"command_name": "racket_target"})
        racket_target_normal_b = ObsTerm(func=mdp.racket_target_normal_b, params={"command_name": "racket_target"})
        racket_pos_b = ObsTerm(func=mdp.racket_pos_b, params={"command_name": "racket_target"})
        racket_lin_vel_b = ObsTerm(func=mdp.racket_lin_vel_b, params={"command_name": "racket_target"})
        racket_normal_b = ObsTerm(func=mdp.racket_normal_b, params={"command_name": "racket_target"})
        time_to_strike = ObsTerm(func=mdp.time_to_strike, params={"command_name": "racket_target"})
        swing_type = ObsTerm(func=mdp.swing_type, params={"command_name": "racket_target"})
        actions = ObsTerm(func=mdp.last_action, params={"action_name": "joint_pos"})

    @configclass
    class CriticCfg(ObservationsCfg.PrivilegedCfg):
        command = None
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=A3_NATIVE_STRIKE_JOINTS, preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=A3_NATIVE_STRIKE_JOINTS, preserve_order=True)},
        )
        racket_target_pos_b = ObsTerm(func=mdp.racket_target_pos_b, params={"command_name": "racket_target"})
        racket_target_vel_b = ObsTerm(func=mdp.racket_target_vel_b, params={"command_name": "racket_target"})
        racket_target_normal_b = ObsTerm(func=mdp.racket_target_normal_b, params={"command_name": "racket_target"})
        racket_target_vel_w = ObsTerm(func=mdp.racket_target_vel_w, params={"command_name": "racket_target"})
        racket_target_normal_w = ObsTerm(func=mdp.racket_target_normal_w, params={"command_name": "racket_target"})
        time_to_strike = ObsTerm(func=mdp.time_to_strike, params={"command_name": "racket_target"})
        racket_pos_b = ObsTerm(func=mdp.racket_pos_b, params={"command_name": "racket_target"})
        racket_lin_vel_w = ObsTerm(func=mdp.racket_lin_vel_w, params={"command_name": "racket_target"})
        racket_normal_w = ObsTerm(func=mdp.racket_normal_w, params={"command_name": "racket_target"})
        actions = ObsTerm(func=mdp.last_action, params={"action_name": "joint_pos"})
        episode_time_left = ObsTerm(func=mdp.episode_time_left)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class A3StrikeConditionedBaseObservationsCfg(A3NativeStrikeObservationsCfg):
    """Deployable Base14 state plus known strike-reference feed-forward."""

    @configclass
    class PolicyCfg(A3NativeStrikeObservationsCfg.PolicyCfg):
        # Forward COM/base velocity is essential for a leg policy to decide
        # whether it must brake a developing post-strike fall.  It is a normal
        # onboard state estimate, not a simulator-only disturbance flag.
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.10, n_max=0.10))
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=A3_BASE_ACTION_JOINTS, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=A3_BASE_ACTION_JOINTS, preserve_order=True)},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        strike_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=A3_STRIKE_V2_REFERENCE_JOINTS, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        strike_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=A3_STRIKE_V2_REFERENCE_JOINTS, preserve_order=True)},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        strike_reference_joint_pos = ObsTerm(
            func=mdp.motion_joint_pos,
            params={
                "command_name": "motion",
                "asset_cfg": SceneEntityCfg("robot", joint_names=A3_STRIKE_V2_REFERENCE_JOINTS, preserve_order=True),
            },
        )
        strike_reference_joint_vel = ObsTerm(
            func=mdp.motion_joint_vel,
            params={
                "command_name": "motion",
                "asset_cfg": SceneEntityCfg("robot", joint_names=A3_STRIKE_V2_REFERENCE_JOINTS, preserve_order=True),
            },
        )
        # Two short-horizon reference velocities provide deployable preview of
        # the arm/waist acceleration and deceleration that will load the legs.
        # This is RL feed-forward, not a simulator-only future-state leak.
        strike_reference_joint_vel_8 = ObsTerm(
            func=mdp.motion_joint_vel,
            params={
                "command_name": "motion",
                "asset_cfg": SceneEntityCfg("robot", joint_names=A3_STRIKE_V2_REFERENCE_JOINTS, preserve_order=True),
                "lookahead_steps": 8,
            },
        )
        strike_reference_joint_vel_16 = ObsTerm(
            func=mdp.motion_joint_vel,
            params={
                "command_name": "motion",
                "asset_cfg": SceneEntityCfg("robot", joint_names=A3_STRIKE_V2_REFERENCE_JOINTS, preserve_order=True),
                "lookahead_steps": 16,
            },
        )
        strike_phase = ObsTerm(func=mdp.motion_phase, params={"command_name": "motion"})

    @configclass
    class CriticCfg(A3NativeStrikeObservationsCfg.CriticCfg):
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=A3_BASE_ACTION_JOINTS, preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=A3_BASE_ACTION_JOINTS, preserve_order=True)},
        )
        strike_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=A3_STRIKE_V2_REFERENCE_JOINTS, preserve_order=True)},
        )
        strike_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=A3_STRIKE_V2_REFERENCE_JOINTS, preserve_order=True)},
        )
        strike_reference_joint_pos = ObsTerm(
            func=mdp.motion_joint_pos,
            params={
                "command_name": "motion",
                "asset_cfg": SceneEntityCfg("robot", joint_names=A3_STRIKE_V2_REFERENCE_JOINTS, preserve_order=True),
            },
        )
        strike_reference_joint_vel = ObsTerm(
            func=mdp.motion_joint_vel,
            params={
                "command_name": "motion",
                "asset_cfg": SceneEntityCfg("robot", joint_names=A3_STRIKE_V2_REFERENCE_JOINTS, preserve_order=True),
            },
        )
        strike_reference_joint_vel_8 = ObsTerm(
            func=mdp.motion_joint_vel,
            params={
                "command_name": "motion",
                "asset_cfg": SceneEntityCfg("robot", joint_names=A3_STRIKE_V2_REFERENCE_JOINTS, preserve_order=True),
                "lookahead_steps": 8,
            },
        )
        strike_reference_joint_vel_16 = ObsTerm(
            func=mdp.motion_joint_vel,
            params={
                "command_name": "motion",
                "asset_cfg": SceneEntityCfg("robot", joint_names=A3_STRIKE_V2_REFERENCE_JOINTS, preserve_order=True),
                "lookahead_steps": 16,
            },
        )
        strike_phase = ObsTerm(func=mdp.motion_phase, params={"command_name": "motion"})

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class A3F0ObservationsCfg(A3NativeStrikeObservationsCfg):
    """Two explicit policy contracts used by the paired F0 evaluator."""

    @configclass
    class PolicyCfg(A3NativeStrikeObservationsCfg.PolicyCfg):
        # The composite action manager is Base14, but model_900 was trained
        # with a 10-D native upper action history.
        actions = ObsTerm(func=mdp.f0_upper_last_action)
        # model_900 is frozen but now runs after the validated 50-step ready
        # prelude; expose wall-clock strike timing so it does not swing early.
        time_to_strike = ObsTerm(
            func=mdp.time_to_strike_with_prelude,
            params={"command_name": "racket_target"},
        )

    @configclass
    class StageACfg(A3StrikeConditionedBaseObservationsCfg.PolicyCfg):
        # The warm-start checkpoint was trained with the explicit manifest
        # stroke label.  Keep F0 paired evaluation on the same lower-policy
        # observation contract instead of inferring it from target geometry.
        swing_type = ObsTerm(func=mdp.manifest_swing_type, params={"command_name": "racket_target"})

    policy: PolicyCfg = PolicyCfg()
    stage_a: StageACfg = StageACfg()


@configclass
class A3F1ObservationsCfg(A3StrikeConditionedBaseObservationsCfg):
    """Stage-A leg observation contract plus a private model_900 group."""

    @configclass
    class UpperCfg(A3NativeStrikeObservationsCfg.PolicyCfg):
        actions = ObsTerm(func=mdp.f0_upper_last_action)
        time_to_strike = ObsTerm(
            func=mdp.time_to_strike_with_prelude,
            params={"command_name": "racket_target"},
        )

    policy: A3StrikeConditionedBaseObservationsCfg.PolicyCfg = A3StrikeConditionedBaseObservationsCfg.PolicyCfg()
    critic: A3StrikeConditionedBaseObservationsCfg.CriticCfg = A3StrikeConditionedBaseObservationsCfg.CriticCfg()
    upper: UpperCfg = UpperCfg()


@configclass
class A3UpperCorrectionObservationsCfg(A3F0ObservationsCfg):
    """Keep independent action histories for frozen model_900 and model_3396."""

    @configclass
    class StageACfg(A3F0ObservationsCfg.StageACfg):
        actions = ObsTerm(func=mdp.legacy_stage_a_last_action)

    policy: A3F0ObservationsCfg.PolicyCfg = A3F0ObservationsCfg.PolicyCfg()
    upper: A3F0ObservationsCfg.PolicyCfg = A3F0ObservationsCfg.PolicyCfg()
    stage_a: StageACfg = StageACfg()


@configclass
class A3JointCoordinatorObservationsCfg(A3UpperCorrectionObservationsCfg):
    """Expose both frozen-policy contracts and the last coordinator action."""

    @configclass
    class CoordinatorCfg(ObservationsCfg.PolicyCfg):
        # The coordinator owns a single explicit 204-D contract below. Do not
        # retain the generic tracking terms inherited from PolicyCfg: they
        # duplicate the frozen-policy inputs and silently enlarge the PPO
        # observation with a third, differently normalized state view.
        command = None
        motion_anchor_pos_b = None
        motion_anchor_ori_b = None
        base_lin_vel = None
        base_ang_vel = None
        joint_pos = None
        joint_vel = None
        actions = None
        coordinator = ObsTerm(func=mdp.joint_coordinator_observation)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: CoordinatorCfg = CoordinatorCfg()
    critic: CoordinatorCfg = CoordinatorCfg()


@configclass
class A3NativeStrikeRewardsCfg(RewardsCfg):
    # Strike objective. These are the main task rewards.
    racket_position = RewTerm(
        func=mdp.racket_position_tracking_exp,
        weight=8.0,
        params={"command_name": "racket_target", "std": 0.05},
    )
    racket_position_y = RewTerm(
        func=mdp.racket_position_axis_tracking_exp,
        weight=4.0,
        params={"command_name": "racket_target", "axis": 1, "std": 0.10},
    )
    # Narrow impact-placement terms used after the coarse K=1 policy can
    # already reach the strike neighborhood. Keeping these separate from the
    # broad terms preserves a useful gradient while rewarding <10 cm precision.
    racket_position_fine = RewTerm(
        func=mdp.racket_position_tracking_exp,
        weight=0.0,
        params={"command_name": "racket_target", "std": 0.08},
    )
    racket_position_y_fine = RewTerm(
        func=mdp.racket_position_axis_tracking_exp,
        weight=0.0,
        params={"command_name": "racket_target", "axis": 1, "std": 0.045},
    )
    racket_velocity = RewTerm(
        func=mdp.racket_velocity_tracking_exp,
        weight=4.0,
        params={"command_name": "racket_target", "std": 0.5},
    )
    racket_normal = RewTerm(
        func=mdp.racket_normal_tracking_exp,
        weight=3.0,
        params={"command_name": "racket_target", "std": 0.262},
    )
    racket_hit_coupled = RewTerm(
        func=mdp.racket_hit_coupled_tracking_exp,
        weight=0.0,
        params={
            "command_name": "racket_target",
            "pos_std": 0.08,
            "vel_std": 2.0,
            "normal_std": 0.7,
            "base": 0.35,
            "vel_coeff": 0.30,
            "normal_coeff": 0.35,
        },
    )

    # Motion prior: only upper strike chain, not legs or whole-body imitation.
    motion_global_anchor_pos = None
    motion_global_anchor_ori = None
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.25, "body_names": ["torso_Link", "right_shoulder_roll_Link", "right_elbow_Link", "right_wrist_yaw_Link"]},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.35, "body_names": ["torso_Link", "right_shoulder_roll_Link", "right_elbow_Link", "right_wrist_yaw_Link"]},
    )
    motion_torso_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.22, "body_names": ["torso_Link"]},
    )
    motion_native_joint_pos = RewTerm(
        func=mdp.motion_joint_position_error_exp,
        weight=0.6,
        params={
            "command_name": "motion",
            "std": 0.35,
            "asset_cfg": SceneEntityCfg("robot", joint_names=A3_NATIVE_STRIKE_JOINTS, preserve_order=True),
        },
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=0.25,
        params={"command_name": "motion", "std": 1.0, "body_names": ["right_elbow_Link", "right_wrist_yaw_Link"]},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=0.25,
        params={"command_name": "motion", "std": 3.14, "body_names": ["torso_Link", "right_elbow_Link", "right_wrist_yaw_Link"]},
    )

    # Native MOTION route should not reward base repositioning in this first
    # single-strike task; the base is assumed stable under native control.
    base_position = None

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    joint_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=A3_NATIVE_STRIKE_JOINTS, preserve_order=True)},
    )
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=A3_NATIVE_STRIKE_JOINTS, preserve_order=True)},
    )
    action_residual_l2 = RewTerm(
        func=mdp.action_raw_l2,
        weight=0.0,
        params={"action_name": "joint_pos"},
    )


@configclass
class A3StrikeStabilizerARewardsCfg(A3NativeStrikeRewardsCfg):
    """Reward only the leg stabilizer's task: dynamic support, not racket aim."""

    racket_position = None
    racket_position_y = None
    racket_position_fine = None
    racket_position_y_fine = None
    racket_velocity = None
    racket_normal = None
    racket_hit_coupled = None

    # The raw kinematic reference remains the controller feed-forward.  Its
    # realized zero-residual trace is the state reference used for stability.
    motion_body_pos = None
    motion_body_ori = None
    motion_torso_ori = None
    motion_native_joint_pos = None
    motion_body_lin_vel = None
    motion_body_ang_vel = None

    # The pre-swing bridge starts from a new, physically settled ready pose.
    # Old floating-base realized traces are therefore not an appropriate
    # target; stability is judged from contact, velocity, height and survival.
    realized_torso_orientation = None
    realized_torso_angular_velocity = None
    realized_root_height = None

    post_strike_root_linear_velocity = RewTerm(
        func=mdp.post_strike_root_linear_velocity_error_exp,
        # Kept as a small bounded preference near the ready state.  The dense
        # tail terms below provide learning signal while velocity is still
        # large, where a narrow exponential would otherwise be saturated.
        weight=0.5,
        params={"command_name": "motion", "std": 1.0},
    )
    post_strike_root_linear_velocity_l2 = RewTerm(
        func=mdp.post_strike_root_velocity_l2,
        weight=-0.50,
        params={"command_name": "motion", "angular": False},
    )
    post_strike_root_angular_velocity_l2 = RewTerm(
        func=mdp.post_strike_root_velocity_l2,
        weight=-0.30,
        params={"command_name": "motion", "angular": True},
    )
    post_strike_root_height_deficit = RewTerm(
        func=mdp.post_strike_root_height_deficit_l2,
        weight=-10.0,
        params={"command_name": "motion", "minimum_height": 0.70},
    )
    post_strike_both_feet_contact = RewTerm(
        func=mdp.post_strike_both_feet_contact,
        weight=1.0,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=A3_FEET_BODIES),
            "threshold": 10.0,
        },
    )
    post_strike_ready = RewTerm(
        func=mdp.post_strike_ready_score,
        weight=1.0,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=A3_FEET_BODIES),
            "linear_velocity_std": 0.35,
            "angular_velocity_std": 0.75,
            "minimum_height": 0.70,
            "height_std": 0.08,
            "contact_threshold": 10.0,
        },
    )
    # A true fall must dominate the incentive to keep residuals small.  This
    # remains a result-level signal: it carries no preferred leg posture.
    fall = RewTerm(func=mdp.is_terminated, weight=-25.0)
    alive = RewTerm(func=mdp.is_alive, weight=0.50)
    feet_contact = RewTerm(
        func=mdp.feet_contact_fraction,
        weight=0.50,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=A3_FEET_BODIES),
            "threshold": 10.0,
        },
    )
    joint_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=A3_BASE_ACTION_JOINTS, preserve_order=True)},
    )
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-15.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=A3_BASE_ACTION_JOINTS, preserve_order=True)},
    )
    action_residual_l2 = RewTerm(
        func=mdp.action_raw_l2,
        # Keep only a weak regularizer during the catch curriculum.  A strong
        # magnitude cost otherwise teaches the known-bad zero-action solution.
        weight=-0.05,
        params={"action_name": "joint_pos"},
    )
    raw_action_excess = RewTerm(
        func=mdp.action_unbounded_excess_l2,
        # The execution envelope remains +/-0.25.  Start discouraging latent
        # policy output above 80% of it, without prescribing a leg motion.
        weight=-2.0,
        params={
            "action_name": "joint_pos",
            "raw_limit": 0.20,
            "action_indices": tuple(range(12)),
        },
    )
    action_execution_gap = RewTerm(
        func=mdp.action_execution_gap_l2,
        # Disabled for the reference branch.  The paired robustness branch
        # enables only this term, leaving the successful Stage-A reward intact.
        weight=0.0,
        params={
            "action_name": "joint_pos",
            "action_indices": tuple(range(12)),
            "deadband": 0.02,
        },
    )


@configclass
class A3F1StrikeAwareRewardsCfg(A3StrikeStabilizerARewardsCfg):
    """Stage-A stability rewards augmented with frozen-upper strike preservation."""

    racket_position = RewTerm(
        func=mdp.racket_position_tracking_exp,
        weight=2.0,
        params={"command_name": "racket_target", "std": 0.08},
    )
    racket_position_y = RewTerm(
        func=mdp.racket_position_axis_tracking_exp,
        weight=1.0,
        params={"command_name": "racket_target", "axis": 1, "std": 0.10},
    )
    racket_velocity = RewTerm(
        func=mdp.racket_velocity_tracking_exp,
        weight=0.75,
        params={"command_name": "racket_target", "std": 0.75},
    )
    racket_normal = RewTerm(
        func=mdp.racket_normal_tracking_exp,
        weight=0.75,
        params={"command_name": "racket_target", "std": 0.35},
    )
    racket_hit_coupled = RewTerm(
        func=mdp.racket_hit_coupled_tracking_exp,
        weight=0.25,
        params={
            "command_name": "racket_target",
            "pos_std": 0.10,
            "vel_std": 2.0,
            "normal_std": 0.7,
            "base": 0.35,
            "vel_coeff": 0.30,
            "normal_coeff": 0.35,
        },
    )
    upper_execution_gap = RewTerm(
        func=mdp.upper_execution_gap_l2,
        weight=-0.20,
        params={"action_name": "joint_pos", "deadband": 0.02},
    )
    root_position_drift = RewTerm(
        func=mdp.root_position_drift_l2,
        weight=-1.0,
    )
    feet_slip = RewTerm(
        func=mdp.feet_slip_l2,
        weight=-0.20,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=A3_FEET_BODIES),
            "threshold": 10.0,
        },
    )


@configclass
class A3JointCoordinatorRewardsCfg(A3F1StrikeAwareRewardsCfg):
    """Strike-aware stability rewards with separate correction trust regions."""

    racket_velocity_position_gated = RewTerm(
        func=mdp.racket_velocity_tracking_position_gated_exp,
        weight=0.0,
        params={
            "command_name": "racket_target",
            "velocity_std": 2.0,
            "position_threshold": 0.10,
            "position_excess_std": 0.025,
        },
    )
    coordinator_leg_l2 = RewTerm(
        func=mdp.action_subset_raw_l2,
        weight=-0.03,
        params={"action_name": "joint_pos", "action_indices": tuple(range(12))},
    )
    coordinator_waist_l2 = RewTerm(
        func=mdp.action_subset_raw_l2,
        weight=-0.12,
        params={"action_name": "joint_pos", "action_indices": (12, 13, 14)},
    )
    coordinator_arm_l2 = RewTerm(
        func=mdp.action_subset_raw_l2,
        weight=-0.05,
        params={"action_name": "joint_pos", "action_indices": tuple(range(15, 22))},
    )


@configclass
class A3NativeStrikeTerminationsCfg(TerminationsCfg):
    pass


@configclass
class A3StrikeStabilizerATerminationsCfg(A3NativeStrikeTerminationsCfg):
    """Fail closed when the fixed upper-body strike destabilizes the base.

    A finite post-strike tail is only meaningful if falling terminates the
    episode.  The native tracking anchor terms alone are reference-relative
    and can stay numerically valid after the physical robot has collapsed.
    """

    base_height = DoneTerm(
        func=RootHeightBelowMinimum,
        params={"minimum_height": 0.65},
    )
    non_foot_ground_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_Link$)(?!right_ankle_roll_Link$)"
                    r"(?!left_wrist_yaw_Link$)(?!right_wrist_yaw_Link$)"
                    r"(?!right_hand_pingpang_Link$)(?!pingpang_red_Link$)"
                    r"(?!pingpang_black_Link$)(?!pingbang_ball_Link$).+$"
                ],
            ),
            "threshold": 10.0,
        },
    )


@configclass
class A3StrikeStabilizerAEventsCfg(HOPEEventCfg):
    sample_leg_policy_handoff = EventTerm(
        func=mdp.sample_strike_stabilizer_handoff_step,
        mode="reset",
        params={
            # The production Stage-A contract is predictive, not an end-of-
            # swing rescue: legs observe the known future strike reference and
            # may make bounded corrections from the first reference frame.
            "full_swing_probability": 1.0,
            "candidate_steps": (0,),
        },
    )


@configclass
class A3NativeStrikeEnvCfg(HOPEPingPongAgibotA3EnvCfg):
    # ``official_pd`` preserves the historical A3 starter/deployment gain map.
    # ``official_pd_stand_approx`` maps the official PD_STAND waist/arm arrays
    # into Isaac implicit actuators for a controlled diagnostic. Neither is a
    # substitute for the native hierarchical MOTION balance controller.
    # ``calibrated`` is an Isaac-only high-gain comparison profile.
    native_actuator_profile: str = "official_pd"

    commands: HOPECommandsCfg = HOPECommandsCfg()
    actions: A3NativeStrikeActionsCfg = A3NativeStrikeActionsCfg()
    observations: A3NativeStrikeObservationsCfg = A3NativeStrikeObservationsCfg()
    rewards: A3NativeStrikeRewardsCfg = A3NativeStrikeRewardsCfg()
    terminations: A3NativeStrikeTerminationsCfg = A3NativeStrikeTerminationsCfg()
    events: HOPEEventCfg = HOPEEventCfg()

    def __post_init__(self):
        super().__post_init__()
        # Stage-1 strike executor training should not spend capacity on balance.
        # Real deployment relies on A3 native MOTION/MC for balance; this Isaac
        # task fixes the base to isolate waist + paddle-arm strike learning.
        self.scene.robot.spawn.fix_base = True
        self.actions.joint_pos.joint_names = A3_NATIVE_STRIKE_JOINTS
        self.actions.joint_pos.scale = AGIBOT_A3_NATIVE_STRIKE_ACTION_SCALE
        self.actions.joint_pos.preserve_order = True
        # Isaac can collapse the fixed ping-pong paddle chain into a body whose
        # reported origin is not the manifest racket center. The validated
        # manifest target matches right_wrist_yaw_Link + A3_MOUNT_OFFSET within
        # millimeters, so force the FK path to use that adapter.
        self.commands.racket_target.racket_body_name = "__force_wrist_offset_racket_fk__"
        # Keep manifest world-frame semantics by default. A base-aligned target
        # mode exists for experiments, but the first root/pelvis-aligned probe
        # made backhand executability worse, so do not enable it globally.
        self.commands.racket_target.manifest_base_aligned = False
        self.commands.motion.body_names = [
            "torso_Link",
            "right_shoulder_roll_Link",
            "right_elbow_Link",
            "right_wrist_yaw_Link",
        ]
        # Reference-residual actions assume the reset state starts exactly on
        # the selected manifest frame. The inherited whole-body tracker adds
        # pose, velocity, and joint perturbations at reset, which is useful for
        # robustness later but corrupts the first K=1 executability test.
        self.commands.motion.pose_range = {}
        self.commands.motion.velocity_range = {}
        self.commands.motion.joint_position_range = (0.0, 0.0)
        self.commands.motion.sample_random_start_phase = False
        self.actions.joint_pos.reference_lookahead_steps = 0
        self.events.randomize_pd_gains.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=A3_WAIST_JOINTS + A3_RIGHT_ARM_JOINTS, preserve_order=True
        )
        # The real A3 route uses MC standing / waist / arm servo behavior, not a
        # weak randomized Isaac bare-PD executor. The default A3 URDF gains were
        # too soft for zero-residual physical tracking: waist_pitch drifted
        # toward the soft limit and produced a false forward-lean failure even
        # when the kinematic reference was upright. Keep native-strike execution
        # deterministic and closer to the real servo contract.
        self.events.randomize_pd_gains = None
        self.apply_native_actuator_profile(self.native_actuator_profile)
        self.events.add_joint_default_pos = None
        self.events.base_com = None
        # External pushes are useful for whole-body balance policies, but this
        # task assumes native MC owns balance. Keep the first strike executor
        # smoke path deterministic.
        self.events.push_robot = None
        self.events.randomize_link_mass = None

        # Disable full-body imitation failures after the A3 base cfg finishes
        # retargeting its default contact/termination body names.
        self.terminations.anchor_pos = None
        self.terminations.anchor_ori = None
        self.terminations.ee_body_pos = None
        self.rewards.undesired_contacts.weight = 0.0

    def apply_native_actuator_profile(self, profile: str) -> None:
        """Apply the selected A3 waist/arm servo gains after cfg construction.

        Hydra task overrides are applied after ``parse_env_cfg`` has already
        constructed this object, so changing only ``native_actuator_profile``
        would otherwise be a no-op. Keep the gain application in one method so
        training and evaluation use the same contract.
        """
        self.native_actuator_profile = str(profile)
        waist_actuator = self.scene.robot.actuators.get("waist")
        arm_actuator = self.scene.robot.actuators.get("arms")
        if self.native_actuator_profile == "official_pd":
            # Historical starter/deployment map retained for reproducibility.
            # It is materially softer than official PD_STAND and must not be
            # described as a one-to-one native-controller reproduction.
            if waist_actuator is not None:
                waist_actuator.stiffness = {
                    "waist_yaw_joint": 85.0,
                    "waist_roll_joint": 50.0,
                    "waist_pitch_joint": 50.0,
                }
                waist_actuator.damping = {
                    "waist_yaw_joint": 3.0,
                    "waist_roll_joint": 2.0,
                    "waist_pitch_joint": 2.0,
                }
            if arm_actuator is not None:
                arm_actuator.stiffness = {
                    ".*_shoulder_pitch_joint": 40.0,
                    ".*_shoulder_roll_joint": 40.0,
                    ".*_shoulder_yaw_joint": 30.0,
                    ".*_elbow_joint": 30.0,
                    ".*_wrist_roll_joint": 30.0,
                    ".*_wrist_pitch_joint": 20.0,
                    ".*_wrist_yaw_joint": 20.0,
                }
                arm_actuator.damping = {
                    ".*_shoulder_pitch_joint": 3.0,
                    ".*_shoulder_roll_joint": 3.0,
                    ".*_shoulder_yaw_joint": 2.0,
                    ".*_elbow_joint": 2.0,
                    ".*_wrist_roll_joint": 2.0,
                    ".*_wrist_pitch_joint": 2.0,
                    ".*_wrist_yaw_joint": 2.0,
                }
        elif self.native_actuator_profile == "official_pd_stand_approx":
            # Direct numerical mapping of the official A3 T2D5 PD_STAND
            # waist/right-arm arrays. Isaac's implicit actuator has different
            # timing and hierarchy, so this is an auditable approximation for
            # tracking diagnostics and training experiments, not a claim that
            # native A3 standing/balance is reproduced.
            if waist_actuator is not None:
                waist_actuator.stiffness = {
                    "waist_yaw_joint": 400.0,
                    "waist_roll_joint": 500.0,
                    "waist_pitch_joint": 500.0,
                }
                waist_actuator.damping = {
                    "waist_yaw_joint": 4.0,
                    "waist_roll_joint": 4.0,
                    "waist_pitch_joint": 4.0,
                }
            if arm_actuator is not None:
                arm_actuator.stiffness = {
                    ".*_shoulder_pitch_joint": 200.0,
                    ".*_shoulder_roll_joint": 200.0,
                    ".*_shoulder_yaw_joint": 100.0,
                    ".*_elbow_joint": 200.0,
                    ".*_wrist_roll_joint": 100.0,
                    ".*_wrist_pitch_joint": 50.0,
                    ".*_wrist_yaw_joint": 50.0,
                }
                arm_actuator.damping = {
                    ".*_shoulder_pitch_joint": 2.0,
                    ".*_shoulder_roll_joint": 2.0,
                    ".*_shoulder_yaw_joint": 1.0,
                    ".*_elbow_joint": 1.0,
                    ".*_wrist_roll_joint": 1.0,
                    ".*_wrist_pitch_joint": 1.0,
                    ".*_wrist_yaw_joint": 1.0,
                }
        elif self.native_actuator_profile == "calibrated":
            # Isaac-only comparison profile. This is intentionally not the
            # official A3 controller setting.
            if waist_actuator is not None:
                waist_actuator.stiffness = {
                    "waist_yaw_joint": 120.0,
                    "waist_roll_joint": 160.0,
                    "waist_pitch_joint": 200.0,
                }
                waist_actuator.damping = {
                    "waist_yaw_joint": 6.0,
                    "waist_roll_joint": 8.0,
                    "waist_pitch_joint": 10.0,
                }
            if arm_actuator is not None:
                # Absolute values keep this method idempotent when a runtime
                # task override switches calibrated -> calibrated again.
                arm_actuator.stiffness = {
                    ".*_shoulder_pitch_joint": 80.0,
                    ".*_shoulder_roll_joint": 80.0,
                    ".*_shoulder_yaw_joint": 60.0,
                    ".*_elbow_joint": 60.0,
                    ".*_wrist_roll_joint": 60.0,
                    ".*_wrist_pitch_joint": 40.0,
                    ".*_wrist_yaw_joint": 40.0,
                }
                arm_actuator.damping = {
                    ".*_shoulder_pitch_joint": 6.0,
                    ".*_shoulder_roll_joint": 6.0,
                    ".*_shoulder_yaw_joint": 4.0,
                    ".*_elbow_joint": 4.0,
                    ".*_wrist_roll_joint": 4.0,
                    ".*_wrist_pitch_joint": 4.0,
                    ".*_wrist_yaw_joint": 4.0,
                }
        else:
            raise ValueError(
                "native_actuator_profile must be 'official_pd', "
                "'official_pd_stand_approx', or 'calibrated', "
                f"got {self.native_actuator_profile!r}"
            )
        print(f"[A3NativeStrikeEnvCfg] native_actuator_profile={self.native_actuator_profile}", flush=True)


@configclass
class A3StrikeConditionedBaseEnvCfg(A3NativeStrikeEnvCfg):
    """Diagnostic floating-base strike plant; not yet a training task.

    Zero action follows the phase-indexed Base14/Strike reference.  A future
    Base actor supplies only bounded 14-DOF residuals.  This environment exists
    to validate arm-reaction stability and build physically realized RSI
    prefixes before PPO is allowed.
    """

    actions: A3StrikeConditionedBaseActionsCfg = A3StrikeConditionedBaseActionsCfg()
    observations: A3StrikeConditionedBaseObservationsCfg = A3StrikeConditionedBaseObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.fix_base = False
        self.commands.motion.debug_vis = False
        self.scene.contact_forces.debug_vis = False
        self.scene.terrain.visual_material = None

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


@configclass
class A3FloatingF0EnvCfg(A3StrikeConditionedBaseEnvCfg):
    """Unified fixed/floating migration plant used by the F0 paired audit."""

    actions: A3F0ActionsCfg = A3F0ActionsCfg()
    observations: A3F0ObservationsCfg = A3F0ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Shared strike-ready stance: slight hip/knee flexion keeps the
        # floating-base comparison physically meaningful and matches the
        # established Stage-A support posture.
        # The current six-motion manifest is authored in this world-frame
        # support location.  Keep x/y aligned with motion frame zero while
        # replacing only the nominal upright height with the validated flexed
        # height; otherwise default-pose reset creates a metre-scale fake
        # strike error before the swing starts.
        self.scene.robot.init_state.pos = (3.1500, -0.3500, 1.0400)
        # The strike-only motion library was recorded with this root frame
        # orientation. The A3 asset default is the opposite yaw frame
        # ([1, 0, 0, 0]); using it here rotates the otherwise-correct upper
        # pose by 180 degrees and creates a metre-scale racket error.
        self.scene.robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)
        self.scene.robot.init_state.joint_pos = {
            **self.scene.robot.init_state.joint_pos,
            ".*_hip_pitch_joint": -0.1600,
            ".*_knee_joint": 0.3200,
            ".*_ankle_pitch_joint": -0.1550,
            "left_hip_roll_joint": 0.0800,
            "right_hip_roll_joint": -0.0800,
        }
        # F0 compares fixed and floating variants of this same config. The
        # evaluator changes only this flag after construction for the fixed
        # branch; all observation, target and actuator settings stay shared.
        self.scene.robot.spawn.fix_base = False
        self.commands.motion.sample_random_start_phase = False
        # Use the validated floating-base contract: reset into the slightly
        # flexed support stance, then blend into strike frame zero.
        self.commands.motion.prelude_steps = 50
        self.commands.motion.hold_last_frame_steps = 0
        self.commands.motion.return_to_default_steps = 0
        self.commands.motion.reset_to_default_pose = True
        self.actions.joint_pos.action_mask = (1.0,) * 12 + (0.0, 0.0)
        self.actions.joint_pos.smooth_raw_bound = True
        self.actions.joint_pos.base_reference_mode = "default"
        self.actions.joint_pos.phase_gate_joint_names = (
            "left_hip_yaw_joint",
            "right_hip_yaw_joint",
        )
        self.actions.joint_pos.phase_gate_min_scale = 0.15
        self.actions.joint_pos.phase_gate_start = 0.12
        self.actions.joint_pos.phase_gate_end = 0.45
        self.actions.joint_pos.phase_gate_tail_release_steps = 0
        self.actions.joint_pos.ready_hold_residual_release_steps = 0
        # Preserve the validated flexed ready-pose blend.  The frozen upper
        # policy's 12-step shoulder lead is restored over the first 12 motion
        # frames, so it is fully active before the hit frame.
        self.actions.joint_pos.upper_prelude_release_steps = 12


@configclass
class A3StrikeStabilizerAEnvCfg(A3StrikeConditionedBaseEnvCfg):
    """Stage-A plant: replay upper body/waist; learn only 12 leg residuals.

    The public action tensor remains Base14 so later stages can reopen the two
    waist residual channels without changing policy plumbing.  For this stage
    their mask is structurally zero: waist motion is a known strike reference,
    not an actuator the policy may use to chase racket errors.
    """

    rewards: A3StrikeStabilizerARewardsCfg = A3StrikeStabilizerARewardsCfg()
    terminations: A3StrikeStabilizerATerminationsCfg = A3StrikeStabilizerATerminationsCfg()
    events: A3StrikeStabilizerAEventsCfg = A3StrikeStabilizerAEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Stage A is forehand-only.  Do not infer swing family from the
        # target's lateral position: a valid forehand can be represented on
        # either side of base-Y after a stance/reference transformation.  The
        # current forehand checkpoint was trained with this legacy channel
        # constant at -1, so preserve that one-dimensional input exactly.
        # A future unified forehand/backhand policy must use a real semantic
        # manifest/planner label instead of this compatibility constant.
        self.observations.policy.swing_type.func = mdp.fixed_swing_type
        self.observations.policy.swing_type.params = {"value": -1.0}
        # The working point is a receiving/striking stance, not the old
        # straight nominal stand.  This is a reference pose only; it does not
        # prescribe the learned leg residual sequence.
        self.scene.robot.init_state.pos = (0.0, 0.0, 1.0400)
        # Keep the stabilizer's reset frame identical to the strike motion
        # root. Older Stage-A checkpoints trained with the asset default yaw
        # are not compatible with this corrected observation/action contract.
        self.scene.robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)
        self.scene.robot.init_state.joint_pos = {
            **self.scene.robot.init_state.joint_pos,
            ".*_hip_pitch_joint": -0.1600,
            ".*_knee_joint": 0.3200,
            ".*_ankle_pitch_joint": -0.1550,
            "left_hip_roll_joint": 0.0800,
            "right_hip_roll_joint": -0.0800,
        }
        self.actions.joint_pos.action_mask = (1.0,) * 12 + (0.0, 0.0)
        # The physical envelope stays at +/-0.25, but tanh removes the hard
        # execution discontinuity.  Latent saturation is separately visible
        # to the raw_action_excess reward above.
        self.actions.joint_pos.smooth_raw_bound = True
        self.actions.joint_pos.base_reference_mode = "default"
        # Hip yaw is quiet in the stable ready stance, smoothly becomes
        # available before peak swing loading, then smoothly retracts through
        # the settling tail.  It remains part of the 12-DOF leg policy.
        self.actions.joint_pos.phase_gate_joint_names = (
            "left_hip_yaw_joint",
            "right_hip_yaw_joint",
        )
        self.actions.joint_pos.phase_gate_min_scale = 0.15
        self.actions.joint_pos.phase_gate_start = 0.12
        self.actions.joint_pos.phase_gate_end = 0.45
        self.actions.joint_pos.phase_gate_tail_release_steps = 50
        # Stage A is a finite task: simulate the entire swing from frame zero,
        # hold the final arm/waist pose while the legs absorb the swing
        # momentum, smoothly return the arm/waist to the ready pose, then
        # require a second stable hold.  The leg policy must demonstrate a
        # complete strike cycle; it must never learn from a direct reset into
        # a mid-swing state.
        self.commands.motion.sample_random_start_phase = False
        self.commands.motion.prelude_steps = 50
        self.commands.motion.reset_to_default_pose = True
        self.commands.motion.hold_last_frame_steps = 75
        self.commands.motion.return_to_default_steps = 50
        # The learned stabilizer is task-active during the full swing, final
        # hold and smooth return.  Once ready is reached, hand the legs back to
        # the verified PD strike-ready plant instead of retaining a permanent
        # residual bias.  At 50 Hz this is a one-second minimum-jerk release.
        self.actions.joint_pos.ready_hold_residual_release_steps = 50


@configclass
class A3StrikeStabilizerAUnifiedEnvCfg(A3StrikeStabilizerAEnvCfg):
    """Stage-A with an explicit manifest semantic label for mixed strokes.

    This is intentionally separate from :class:`A3StrikeStabilizerAEnvCfg` so
    the forehand-only Robust-B checkpoint remains reproducible.  The action
    contract is unchanged (12 leg residual DOFs, waist masked); only the
    one-dimensional swing-family observation becomes semantic.
    """

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.swing_type.func = mdp.manifest_swing_type
        self.observations.policy.swing_type.params = {"command_name": "racket_target"}


@configclass
class A3RetrainStrikeStabilizerEnvCfg(A3StrikeStabilizerAUnifiedEnvCfg):
    """Current-contract Stage-A plant for the reproducible retraining chain.

    The historical Stage-A class is intentionally left unchanged.  This class
    keeps its leg-only Base14 contract and rewards, but uses the corrected
    strike work point shared by the current F0/F1 audits.
    """

    def __post_init__(self):
        super().__post_init__()
        # Keep the corrected world-frame work point and 180-degree root frame
        # used by the current backhand F0/F1 contract.  The inherited joint
        # initialization is the validated flexed ready stance.
        self.scene.robot.init_state.pos = (3.1500, -0.3500, 1.0400)
        self.scene.robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)
        self.scene.robot.spawn.fix_base = False


@configclass
class A3FloatingF1EnvCfg(A3FloatingF0EnvCfg):
    """F1 in-place migration: frozen model_900 upper body, trainable legs only."""

    actions: A3F1ActionsCfg = A3F1ActionsCfg()
    observations: A3F1ObservationsCfg = A3F1ObservationsCfg()
    rewards: A3F1StrikeAwareRewardsCfg = A3F1StrikeAwareRewardsCfg()
    terminations: A3StrikeStabilizerATerminationsCfg = A3StrikeStabilizerATerminationsCfg()
    events: A3StrikeStabilizerAEventsCfg = A3StrikeStabilizerAEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        # F1 remains an in-place migration experiment.  The leg policy may
        # brace and shift the COM, but it has no target-driven base command.
        self.scene.robot.spawn.fix_base = False
        self.commands.motion.sample_random_start_phase = False
        # Match the validated Stage-A start contract without inheriting its
        # post-strike hold/return tail.  Stage-A is only a warm start; the
        # upper strike still begins from the same flexed support stance.
        self.commands.motion.prelude_steps = 50
        self.commands.motion.hold_last_frame_steps = 0
        self.commands.motion.return_to_default_steps = 0
        self.commands.motion.reset_to_default_pose = True
        # Keep the warm-start actor's explicit semantic stroke label without
        # inheriting Stage-A's separate ready-pose initialization.
        self.observations.policy.swing_type.func = mdp.manifest_swing_type
        self.observations.policy.swing_type.params = {"command_name": "racket_target"}
        self.actions.joint_pos.action_mask = (1.0,) * 12 + (0.0, 0.0)
        self.actions.joint_pos.smooth_raw_bound = True
        self.actions.joint_pos.base_reference_mode = "default"
        self.actions.joint_pos.phase_gate_joint_names = (
            "left_hip_yaw_joint",
            "right_hip_yaw_joint",
        )
        self.actions.joint_pos.phase_gate_min_scale = 0.15
        self.actions.joint_pos.phase_gate_start = 0.12
        self.actions.joint_pos.phase_gate_end = 0.45
        self.actions.joint_pos.phase_gate_tail_release_steps = 0
        self.actions.joint_pos.ready_hold_residual_release_steps = 0


@configclass
class A3FloatingUpperCorrectionEnvCfg(A3FloatingF0EnvCfg):
    """Frozen model_3396 support with PPO correction only on upper joints."""

    actions: A3UpperCorrectionActionsCfg = A3UpperCorrectionActionsCfg()
    observations: A3UpperCorrectionObservationsCfg = A3UpperCorrectionObservationsCfg()
    rewards: A3F1StrikeAwareRewardsCfg = A3F1StrikeAwareRewardsCfg()
    terminations: A3StrikeStabilizerATerminationsCfg = A3StrikeStabilizerATerminationsCfg()
    events: A3StrikeStabilizerAEventsCfg = A3StrikeStabilizerAEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.upper_prelude_release_steps = 12
        # F1 inherits a Base14 execution-gap penalty.  This task's public
        # action is instead the 10-D upper correction, so its indices must
        # match that public contract rather than the frozen leg actor.
        self.rewards.raw_action_excess.params["action_indices"] = tuple(range(10))
        self.rewards.action_execution_gap.params["action_indices"] = tuple(range(10))


@configclass
class A3FloatingJointCoordinatorEnvCfg(A3FloatingF0EnvCfg):
    """Final fixed-stance floating-base strike controller with frozen priors."""

    actions: A3JointCoordinatorActionsCfg = A3JointCoordinatorActionsCfg()
    observations: A3JointCoordinatorObservationsCfg = A3JointCoordinatorObservationsCfg()
    rewards: A3JointCoordinatorRewardsCfg = A3JointCoordinatorRewardsCfg()
    terminations: A3StrikeStabilizerATerminationsCfg = A3StrikeStabilizerATerminationsCfg()
    events: A3StrikeStabilizerAEventsCfg = A3StrikeStabilizerAEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Keep the exact F0/J0 plant: same flexed ready state, root frame,
        # 50-step prelude and 12-step shoulder lead release.  The only new
        # trainable authority is the coordinator correction around frozen
        # model_3396/model_900 outputs.
        self.actions.joint_pos.upper_prelude_release_steps = 12
        self.actions.joint_pos.action_mask = (1.0,) * 12 + (0.0, 0.0)
        self.rewards.raw_action_excess.params["action_indices"] = tuple(range(22))
        self.rewards.raw_action_excess.params["raw_limit"] = 0.80
        self.rewards.action_execution_gap.params["action_indices"] = tuple(range(22))
        self.rewards.action_execution_gap.params["deadband"] = 0.05


@configclass
class A3FloatingJointCoordinatorV2EnvCfg(A3FloatingJointCoordinatorEnvCfg):
    """Single-strike coordinator with authority sized for observed shoulder lag.

    This deliberately remains a residual policy around the frozen Stage-A and
    fixed-base strike priors.  It changes only the trainable correction trust
    region and strike-reward shaping; the plant, ready stance, prelude and
    lead-compensated upper reference remain unchanged.
    """

    def __post_init__(self):
        super().__post_init__()
        # Keep the verified leg authority unchanged.  The 1200-iteration audit
        # found the dominant controllable error at right shoulder pitch/yaw,
        # so widen only waist/arm channels needed to compensate it.
        self.actions.joint_pos.waist_correction_scale_rad = (0.020, 0.020, 0.025)
        self.actions.joint_pos.arm_correction_scale_rad = (
            0.050,  # right shoulder pitch
            0.040,  # right shoulder roll
            0.050,  # right shoulder yaw
            0.045,  # right elbow
            0.025,  # wrist roll
            0.025,  # wrist pitch
            0.025,  # wrist yaw
        )
        # At the observed 13 cm error the old 8 cm exponential kernel was
        # almost flat and active only near impact.  A broad term drives the
        # approach; a fine term preserves the incentive below 10 cm.
        self.rewards.racket_position.weight = 6.0
        self.rewards.racket_position.params["std"] = 0.14
        self.rewards.racket_position_y.weight = 3.0
        self.rewards.racket_position_y.params["std"] = 0.14
        self.rewards.racket_position_fine = RewTerm(
            func=mdp.racket_position_tracking_exp,
            weight=2.0,
            params={"command_name": "racket_target", "std": 0.06},
        )
        self.rewards.racket_hit_coupled.weight = 0.75
        self.rewards.racket_hit_coupled.params["pos_std"] = 0.14
        # Preserve a trust region but do not teach the old zero-correction
        # solution merely because waist/arm corrections are more useful.
        self.rewards.action_residual_l2.weight = -0.01
        self.rewards.coordinator_leg_l2.weight = -0.015
        self.rewards.coordinator_waist_l2.weight = -0.04
        self.rewards.coordinator_arm_l2.weight = -0.02


@configclass
class A3FloatingJointCoordinatorV3EnvCfg(A3FloatingJointCoordinatorV2EnvCfg):
    """V2 continuation that restores a usable strike-velocity gradient.

    V2 reaches the target position reliably, but its 0.75 m/s velocity kernel
    is effectively zero for the observed 1.5--2.6 m/s impact-speed errors.
    This keeps the V2 plant, priors, action authority and position/stability
    objectives intact while making speed correction learnable from model_900.
    """

    def __post_init__(self):
        super().__post_init__()
        # exp(-e^2 / 0.75^2) is nearly flat at the current error range.  A
        # 2.0 m/s kernel preserves gradient without weakening exact position.
        self.rewards.racket_velocity.weight = 3.0
        self.rewards.racket_velocity.params["std"] = 2.0
        # Keep the coupled impact objective aligned with the direct speed
        # reward, but do not let it replace the established position term.
        self.rewards.racket_hit_coupled.weight = 1.0
        self.rewards.racket_hit_coupled.params["vel_std"] = 2.0


@configclass
class A3FloatingJointCoordinatorV4EnvCfg(A3FloatingJointCoordinatorV2EnvCfg):
    """V2 continuation with speed improvement gated by hit placement.

    The V3 unconditional velocity reward improved mean speed slightly by
    sacrificing exact placement.  V4 restores the V2 objective and rewards
    speed only once the racket is inside the 10 cm placement corridor.
    """

    def __post_init__(self):
        super().__post_init__()
        # The inherited narrow velocity term was intentionally ineffective at
        # the observed error range.  Replace it with a broad but position-safe
        # velocity term; all V2 position, normal and stability terms remain.
        self.rewards.racket_velocity.weight = 0.0
        self.rewards.racket_velocity_position_gated.weight = 2.0
        self.rewards.racket_velocity_position_gated.params["velocity_std"] = 2.0
        self.rewards.racket_velocity_position_gated.params["position_threshold"] = 0.10
        self.rewards.racket_velocity_position_gated.params["position_excess_std"] = 0.025


@configclass
class A3FixedBaseBackhandRewardsCfg(A3NativeStrikeRewardsCfg):
    """Backhand residual rewards with a soft latent-action trust band."""

    action_residual_l2 = RewTerm(
        func=mdp.action_raw_l2,
        weight=-0.04,
        params={"action_name": "joint_pos"},
    )
    raw_action_excess = RewTerm(
        func=mdp.action_unbounded_excess_l2,
        weight=-1.0,
        params={
            "action_name": "joint_pos",
            "raw_limit": 0.20,
            "action_indices": tuple(range(10)),
        },
    )
    action_execution_gap = RewTerm(
        func=mdp.action_execution_gap_l2,
        weight=-0.10,
        params={
            "action_name": "joint_pos",
            "action_indices": tuple(range(10)),
            "deadband": 0.02,
        },
    )


@configclass
class A3FixedBaseStrikeObservationsCfg(A3StrikeConditionedBaseObservationsCfg):
    """Fixed-base strike observations with explicit target-error channels."""

    @configclass
    class PolicyCfg(A3StrikeConditionedBaseObservationsCfg.PolicyCfg):
        # Give the residual actor the two task errors explicitly.  The target
        # and actual states remain available as well, but these channels make
        # the intended interpolation relation direct and easy to audit.
        racket_target_error_pos_b = ObsTerm(
            func=mdp.racket_target_error_pos_b,
            params={"command_name": "racket_target"},
        )
        racket_target_error_vel_b = ObsTerm(
            func=mdp.racket_target_error_vel_b,
            params={"command_name": "racket_target"},
        )
        racket_target_error_normal_b = ObsTerm(
            func=mdp.racket_target_error_normal_b,
            params={"command_name": "racket_target"},
        )

    @configclass
    class CriticCfg(A3StrikeConditionedBaseObservationsCfg.CriticCfg):
        racket_target_error_pos_b = ObsTerm(
            func=mdp.racket_target_error_pos_b,
            params={"command_name": "racket_target"},
        )
        racket_target_error_vel_b = ObsTerm(
            func=mdp.racket_target_error_vel_b,
            params={"command_name": "racket_target"},
        )
        racket_target_error_normal_b = ObsTerm(
            func=mdp.racket_target_error_normal_b,
            params={"command_name": "racket_target"},
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class A3FixedBaseReferenceStrikeEnvCfg(A3NativeStrikeEnvCfg):
    """Fixed-base strike executor with explicit motion-reference preview.

    The reference-residual action already follows the processed motion, but a
    residual policy should also observe the current phase, reference pose,
    reference velocity, and short-horizon velocity preview.  Reusing the
    strike-conditioned observation contract makes the forehand/backhand
    conditioning explicit instead of asking PPO to infer phase from raw joint
    positions alone.
    """

    observations: A3FixedBaseStrikeObservationsCfg = A3FixedBaseStrikeObservationsCfg()


@configclass
class A3FixedBaseBackhandReferenceStrikeEnvCfg(A3FixedBaseReferenceStrikeEnvCfg):
    """Backhand-only fixed-base residual PPO contract from the reviewed plan.

    The default YAML is Stage 1B (manifest target, no target perturbation).
    Stage 1C reuses the same environment with
    ``racket_target.target_mode=manifest_perturbed``.
    """

    rewards: A3FixedBaseBackhandRewardsCfg = A3FixedBaseBackhandRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # A separate family policy must not infer the stroke from target-side
        # geometry.  Backhand is an explicit semantic contract (+1).
        self.observations.policy.swing_type.func = mdp.fixed_swing_type
        self.observations.policy.swing_type.params = {"value": 1.0}
