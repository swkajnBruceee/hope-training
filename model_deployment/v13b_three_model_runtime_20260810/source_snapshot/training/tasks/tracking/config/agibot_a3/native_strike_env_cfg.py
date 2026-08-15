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
from training.tasks.tracking.mdp import precision_rescue_commands as rescue_commands
from training.tasks.tracking.mdp import precision_rescue_rewards as rescue_rewards
from training.tasks.base_locomotion.mdp import (
    A3_PD_STAND_BASE_ACTION_SCALE_RAD,
    A3F1FrozenUpperBaseCompositePositionActionCfg,
    A3F0UpperBaseCompositePositionActionCfg,
    A3FrozenStageAJointCoordinatorActionCfg,
    A3UnifiedUpperReferenceTrackerActionCfg,
    A3ReferenceFreeTargetConditionedPositionActionCfg,
    A3TargetConditionedJointCoordinatorActionCfg,
    A3FrozenStageAUpperCorrectionActionCfg,
    A3FrozenAnchorArmAdapterPositionActionCfg,
    A3StrikeConditionedBaseCompositePositionActionCfg,
    RootHeightBelowMinimum,
)
from training.robots.agibot_a3 import (
    A3_BACKEND_JOINTS,
    A3_BASE_ACTION_JOINTS,
    A3_FEET_BODIES,
    A3_LEFT_LEG_JOINTS,
    A3_NATIVE_STRIKE_JOINTS,
    A3_RIGHT_ARM_JOINTS,
    A3_RIGHT_LEG_JOINTS,
    A3_REFERENCE_TRACKER_JOINTS,
    A3_STRIKE_V2_REFERENCE_JOINTS,
    A3_WAIST_JOINTS,
    AGIBOT_A3_NATIVE_STRIKE_ACTION_SCALE,
)
from training.tasks.tracking.config.agibot_a3.hope_env_cfg import (
    HOPECommandsCfg,
    HOPEEventCfg,
    HOPEPingPongAgibotA3EnvCfg,
)
from training.tasks.tracking.tracking_env_cfg import CommandsCfg, ActionsCfg, ObservationsCfg, RewardsCfg, TerminationsCfg
from training.utils.v13b_ready_stance import V13B_RIGHT_FRONT_READY


def _scale_gain_map(value, scale: float):
    if isinstance(value, dict):
        return {k: float(v) * scale for k, v in value.items()}
    return float(value) * scale


V13B_READY_JOINT_POSITIONS = dict(V13B_RIGHT_FRONT_READY.joint_positions)


def _apply_v13b_right_front_ready(env_cfg) -> None:
    """Apply the one V1.3B READY contract to the PhysX reset state."""
    joint_pos = env_cfg.scene.robot.init_state.joint_pos
    for pattern in (
        ".*_hip_pitch_joint",
        ".*_hip_roll_joint",
        ".*_knee_joint",
        ".*_ankle_pitch_joint",
        ".*_ankle_roll_joint",
    ):
        joint_pos.pop(pattern, None)
    joint_pos.update(V13B_READY_JOINT_POSITIONS)
    root_x, root_y, root_z = env_cfg.scene.robot.init_state.pos
    env_cfg.scene.robot.init_state.pos = (
        root_x,
        root_y,
        root_z + V13B_RIGHT_FRONT_READY.root_height_delta_m,
    )


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
class A3ReferenceTrackerActionsCfg(ActionsCfg):
    """P5D pure safe-reference plus 22-D residual action contract.

    No frozen Stage-A policy, upper policy, target adapter, control anchor, or
    motion-specific target feed-forward is present here.  Zero PPO output is
    exactly the safe reference command.
    """

    joint_pos = mdp.ReferenceResidualJointPositionActionCfg(
        asset_name="robot",
        joint_names=tuple(A3_REFERENCE_TRACKER_JOINTS),
        preserve_order=True,
        scale={
            **dict(zip(A3_LEFT_LEG_JOINTS + A3_RIGHT_LEG_JOINTS, A3_PD_STAND_BASE_ACTION_SCALE_RAD[:12])),
            "waist_yaw_joint": 0.10,
            "waist_roll_joint": 0.06,
            "waist_pitch_joint": 0.08,
            "right_shoulder_pitch_joint": 0.10,
            "right_shoulder_roll_joint": 0.10,
            "right_shoulder_yaw_joint": 0.10,
            "right_elbow_joint": 0.10,
            "right_wrist_roll_joint": 0.06,
            "right_wrist_pitch_joint": 0.05,
            "right_wrist_yaw_joint": 0.05,
        },
        raw_clip=1.0,
        reference_command_name="motion",
        reference_lookahead_steps=0,
        interpolate_reference=False,
        soft_limit_margin_rad_by_joint={
            "waist_yaw_joint": 0.08,
            "waist_roll_joint": 0.12,
            "waist_pitch_joint": 0.12,
            "right_shoulder_roll_joint": 0.08,
            "right_shoulder_pitch_joint": 0.04,
            "right_shoulder_yaw_joint": 0.04,
            "right_elbow_joint": 0.04,
            "right_wrist_roll_joint": 0.03,
            "right_wrist_pitch_joint": 0.03,
            "right_wrist_yaw_joint": 0.03,
        },
    )


@configclass
class A3PriorGuidedReferenceTrackerActionsCfg(ActionsCfg):
    """P5D bootstrap with frozen 3396/900 execution priors and a 10-D residual.

    ``model_3396`` retains sole ownership of the 12 leg-support channels and
    ``model_900`` retains the reviewed waist/right-arm swing prior.  The
    learnable public action is *only* a bounded correction on the three waist
    and seven right-arm joints.  This preserves the evaluated support-state
    machine while keeping P5D a generic reference tracker rather than a new
    full-body planner or a motion-specific target adapter.
    """

    joint_pos = A3FrozenStageAUpperCorrectionActionCfg(
        asset_name="robot",
        base_joint_names=tuple(A3_BASE_ACTION_JOINTS),
        backend_joint_names=tuple(A3_BACKEND_JOINTS),
        strike_joint_names=tuple(A3_STRIKE_V2_REFERENCE_JOINTS),
        upper_joint_names=tuple(A3_NATIVE_STRIKE_JOINTS),
        action_scale_rad=A3_PD_STAND_BASE_ACTION_SCALE_RAD,
        # Preserve the reviewed split: 3396 supplies only leg support;
        # model_900 supplies the strike prior; P5D may correct the upper ten
        # joints only.  The two historical waist outputs from 3396 are masked.
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
        upper_checkpoint="checkpoints/frozen_priors/model_900.pt",
        legacy_stage_a_checkpoint="checkpoints/frozen_priors/model_3396.pt",
        legacy_stage_a_observation_group="stage_a",
        # model_900 was trained under the V7 frozen-prior contract.  Keep
        # that contract explicit here: removing these fields makes the same
        # checkpoint run with a different plant (no shoulder lead and no
        # velocity target), which systematically shifts the shoulder/elbow/
        # wrist timing and invalidates P5D attribution.
        joint_reference_lookahead_steps={
            "right_shoulder_pitch_joint": 12.0,
            "right_shoulder_yaw_joint": 12.0,
        },
        joint_velocity_feedforward_mode="task_phase",
        joint_velocity_feedforward_beta=0.75,
        joint_velocity_feedforward_joint_names=(
            "right_shoulder_pitch_joint",
            "right_shoulder_yaw_joint",
        ),
        joint_velocity_feedforward_post_hit_decay_steps=6,
        upper_correction_scale_rad=(0.035,) * 10,
    )


@configclass
class A3UnifiedUpperReferenceTrackerActionsCfg(ActionsCfg):
    """P5U-1 action contract: model_3396 prior plus optional PPO balance residual."""

    joint_pos = A3UnifiedUpperReferenceTrackerActionCfg(
        asset_name="robot",
        base_joint_names=tuple(A3_BASE_ACTION_JOINTS),
        backend_joint_names=tuple(A3_BACKEND_JOINTS),
        strike_joint_names=tuple(A3_STRIKE_V2_REFERENCE_JOINTS),
        upper_joint_names=tuple(A3_NATIVE_STRIKE_JOINTS),
        action_scale_rad=A3_PD_STAND_BASE_ACTION_SCALE_RAD,
        action_mask=(1.0,) * 12 + (0.0, 0.0),
        raw_clip=0.50,
        smooth_raw_bound=True,
        scale=dict(AGIBOT_A3_NATIVE_STRIKE_ACTION_SCALE),
        clip_to_soft_joint_limits=True,
        reference_command_name="motion",
        base_reference_mode="default",
        joint_names=tuple(A3_NATIVE_STRIKE_JOINTS),
        preserve_order=True,
        upper_waist_joint_names=tuple(A3_WAIST_JOINTS),
        # Contract A: no hard command lead and no velocity feedforward.  The
        # actor receives +1/+3/+6/+12 preview through its observation group.
        reference_lookahead_steps=0,
        joint_reference_lookahead_steps={},
        joint_velocity_feedforward_mode="none",
        joint_velocity_feedforward_beta=0.0,
        joint_velocity_feedforward_joint_names=(),
        upper_correction_scale_rad=(0.035,) * 10,
        legacy_stage_a_checkpoint="checkpoints/frozen_priors/model_3396.pt",
        legacy_stage_a_observation_group="stage_a",
        legacy_stage_a_yaw_adapter=True,
    )


@configclass
class A3UnifiedUpperReferenceTrackerGlobalPhaseActionsCfg(ActionsCfg):
    """Contract B: ten position residuals plus one shared continuous phase offset."""

    joint_pos = A3UnifiedUpperReferenceTrackerActionCfg(
        asset_name="robot",
        base_joint_names=tuple(A3_BASE_ACTION_JOINTS),
        backend_joint_names=tuple(A3_BACKEND_JOINTS),
        strike_joint_names=tuple(A3_STRIKE_V2_REFERENCE_JOINTS),
        upper_joint_names=tuple(A3_NATIVE_STRIKE_JOINTS),
        action_scale_rad=A3_PD_STAND_BASE_ACTION_SCALE_RAD,
        action_mask=(1.0,) * 12 + (0.0, 0.0),
        raw_clip=0.50,
        smooth_raw_bound=True,
        scale=dict(AGIBOT_A3_NATIVE_STRIKE_ACTION_SCALE),
        clip_to_soft_joint_limits=True,
        reference_command_name="motion",
        base_reference_mode="default",
        joint_names=tuple(A3_NATIVE_STRIKE_JOINTS),
        preserve_order=True,
        upper_waist_joint_names=tuple(A3_WAIST_JOINTS),
        reference_lookahead_steps=0,
        joint_reference_lookahead_steps={},
        joint_velocity_feedforward_mode="none",
        joint_velocity_feedforward_beta=0.0,
        joint_velocity_feedforward_joint_names=(),
        upper_correction_scale_rad=(0.035,) * 10,
        legacy_stage_a_checkpoint="checkpoints/frozen_priors/model_3396.pt",
        legacy_stage_a_observation_group="stage_a",
        legacy_stage_a_yaw_adapter=True,
        phase_contract="B",
        global_phase_limit_steps=4.0,
    )


@configclass
class A3UnifiedUpperReferenceTrackerGroupedPhaseActionsCfg(ActionsCfg):
    """Contract C: position residual plus global and shoulder/elbow/wrist phases."""

    joint_pos = A3UnifiedUpperReferenceTrackerActionCfg(
        asset_name="robot",
        base_joint_names=tuple(A3_BASE_ACTION_JOINTS),
        backend_joint_names=tuple(A3_BACKEND_JOINTS),
        strike_joint_names=tuple(A3_STRIKE_V2_REFERENCE_JOINTS),
        upper_joint_names=tuple(A3_NATIVE_STRIKE_JOINTS),
        action_scale_rad=A3_PD_STAND_BASE_ACTION_SCALE_RAD,
        action_mask=(1.0,) * 12 + (0.0, 0.0),
        raw_clip=0.50,
        smooth_raw_bound=True,
        scale=dict(AGIBOT_A3_NATIVE_STRIKE_ACTION_SCALE),
        clip_to_soft_joint_limits=True,
        reference_command_name="motion",
        base_reference_mode="default",
        joint_names=tuple(A3_NATIVE_STRIKE_JOINTS),
        preserve_order=True,
        upper_waist_joint_names=tuple(A3_WAIST_JOINTS),
        reference_lookahead_steps=0,
        joint_reference_lookahead_steps={},
        joint_velocity_feedforward_mode="none",
        joint_velocity_feedforward_beta=0.0,
        joint_velocity_feedforward_joint_names=(),
        upper_correction_scale_rad=(0.035,) * 10,
        legacy_stage_a_checkpoint="checkpoints/frozen_priors/model_3396.pt",
        legacy_stage_a_observation_group="stage_a",
        legacy_stage_a_yaw_adapter=True,
        phase_contract="C",
        global_phase_limit_steps=4.0,
        shoulder_phase_limit_steps=2.0,
        elbow_phase_limit_steps=2.0,
        wrist_phase_limit_steps=2.0,
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
class A3TargetConditionedCoordinatorObservationsCfg(
    A3JointCoordinatorObservationsCfg
):
    """Split frozen-anchor and trainable external-target observation paths."""

    @configclass
    class CoordinatorUpperCfg(A3F0ObservationsCfg.PolicyCfg):
        racket_target_pos_b = ObsTerm(
            func=mdp.coordinator_racket_target_pos_b,
            params={"command_name": "racket_target"},
        )

    @configclass
    class CoordinatorCfg(A3JointCoordinatorObservationsCfg.CoordinatorCfg):
        coordinator = ObsTerm(
            func=mdp.joint_coordinator_target_conditioned_observation
        )

    policy: CoordinatorCfg = CoordinatorCfg()
    critic: CoordinatorCfg = CoordinatorCfg()
    coordinator_upper: CoordinatorUpperCfg = CoordinatorUpperCfg()


@configclass
class A3TargetConditionedRecoveryObservationsCfg(
    A3TargetConditionedCoordinatorObservationsCfg
):
    """P3 target observation plus a predictive lower-body support suffix."""

    @configclass
    class CoordinatorCfg(A3TargetConditionedCoordinatorObservationsCfg.CoordinatorCfg):
        coordinator = ObsTerm(
            func=mdp.TargetConditionedRecoveryObservation,
            params={
                "command_name": "motion",
                # Motion 3's deterministic failure has already crossed its
                # forward capture boundary before its nominal hit frame.  The
                # recovery branch must therefore establish a lower-body brace
                # in the final part of READY and have full authority at the
                # swing release; a post-hit-only residual is physically late.
                # P3's arm actor and target feedforward remain frozen.
                "gate_delay_steps": 0,
                "gate_ramp_steps": 16,
                "gate_lead_steps": 44,
                # P3's controlled strike runs through a 75-control READY
                # phase.  A brace that only begins in its last 16 controls
                # cannot shift the support state enough before swing release,
                # so make the lower-body gate a smooth READY-long ramp.
                "prelude_prepare_steps": 75,
            },
        )

    policy: CoordinatorCfg = CoordinatorCfg()
    critic: CoordinatorCfg = CoordinatorCfg()


@configclass
class A3JointCoordinatorPreviewObservationsCfg(A3JointCoordinatorObservationsCfg):
    """Coordinator contract with a zero-migratable 18-D upper dynamics preview."""

    @configclass
    class CoordinatorCfg(A3JointCoordinatorObservationsCfg.CoordinatorCfg):
        coordinator = ObsTerm(func=mdp.joint_coordinator_observation_with_upper_preview)

    policy: CoordinatorCfg = CoordinatorCfg()
    critic: CoordinatorCfg = CoordinatorCfg()


@configclass
class A3JointCoordinatorMomentumPreviewObservationsCfg(A3JointCoordinatorObservationsCfg):
    """Coordinator contract with a canonical 18-D upper momentum preview."""

    @configclass
    class CoordinatorCfg(A3JointCoordinatorObservationsCfg.CoordinatorCfg):
        coordinator = ObsTerm(func=mdp.joint_coordinator_observation_with_momentum_preview)

    policy: CoordinatorCfg = CoordinatorCfg()
    critic: CoordinatorCfg = CoordinatorCfg()


@configclass
class A3JointCoordinatorStaggerSupportObservationsCfg(A3JointCoordinatorObservationsCfg):
    """Legacy coordinator state plus an explicit 19-D stagger-support contract."""

    @configclass
    class CoordinatorCfg(A3JointCoordinatorObservationsCfg.CoordinatorCfg):
        coordinator = ObsTerm(func=mdp.joint_coordinator_observation_with_stagger_support)

    policy: CoordinatorCfg = CoordinatorCfg()
    critic: CoordinatorCfg = CoordinatorCfg()


@configclass
class A3JointCoordinatorWideStaggerSupportObservationsCfg(
    A3JointCoordinatorObservationsCfg
):
    """Legacy coordinator state plus a complete 23-D 2-D support contract."""

    @configclass
    class CoordinatorCfg(A3JointCoordinatorObservationsCfg.CoordinatorCfg):
        coordinator = ObsTerm(
            func=mdp.joint_coordinator_observation_with_wide_stagger_support
        )

    policy: CoordinatorCfg = CoordinatorCfg()
    critic: CoordinatorCfg = CoordinatorCfg()


@configclass
class A3JointCoordinatorWideStaggerRecoveryObservationsCfg(
    A3JointCoordinatorObservationsCfg
):
    """V22 state plus capture-point rate and a post-hit recovery gate."""

    @configclass
    class CoordinatorCfg(A3JointCoordinatorObservationsCfg.CoordinatorCfg):
        coordinator = ObsTerm(
            func=mdp.JointCoordinatorWideStaggerRecoveryObservation,
            params={
                "command_name": "motion",
                "gate_delay_steps": 2,
                "gate_ramp_steps": 8,
                "capture_rate_scale_mps": 1.0,
            },
        )

    policy: CoordinatorCfg = CoordinatorCfg()
    critic: CoordinatorCfg = CoordinatorCfg()


@configclass
class A3JointCoordinatorBentReadyRecoveryObservationsCfg(
    A3JointCoordinatorObservationsCfg
):
    """V28 state: V22 support plus explicit bent-READY settling signals."""

    @configclass
    class CoordinatorCfg(A3JointCoordinatorObservationsCfg.CoordinatorCfg):
        coordinator = ObsTerm(
            func=mdp.JointCoordinatorBentReadyRecoveryObservation,
            params={
                "command_name": "motion",
                "gate_delay_steps": 2,
                "gate_ramp_steps": 8,
                "capture_rate_scale_mps": 1.0,
            },
        )

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

    # Disabled by default so historical V2/V5 runs remain reproducible.  The
    # full-cycle recovery curriculum explicitly enables it to intervene before
    # the root-height fall termination becomes the only remaining signal.
    post_strike_root_tilt_l2 = RewTerm(
        func=mdp.post_strike_root_tilt_l2,
        weight=0.0,
        params={"command_name": "motion"},
    )
    post_strike_recovery_progress = RewTerm(
        func=mdp.PostStrikeRootRecoveryProgress,
        weight=0.0,
        params={"command_name": "motion"},
    )
    pre_hit_root_tilt_l2 = RewTerm(
        func=mdp.pre_hit_root_tilt_l2,
        weight=0.0,
        params={"command_name": "motion"},
    )
    pre_hit_root_angular_velocity_l2 = RewTerm(
        func=mdp.pre_hit_root_angular_velocity_l2,
        weight=0.0,
        params={"command_name": "motion"},
    )
    pre_hit_root_forward_velocity_l2 = RewTerm(
        func=mdp.pre_hit_root_forward_velocity_l2,
        weight=0.0,
        params={"command_name": "motion"},
    )
    strike_approach_pitch_rate_deadband_l2 = RewTerm(
        func=mdp.strike_approach_pitch_rate_deadband_l2,
        weight=0.0,
        params={"command_name": "motion", "window_steps": 30, "deadband": 0.06},
    )
    strike_approach_forward_velocity_deadband_l2 = RewTerm(
        func=mdp.strike_approach_forward_velocity_deadband_l2,
        weight=0.0,
        params={"command_name": "motion", "window_steps": 30, "deadband": 0.05},
    )
    exact_strike_pitch_rate_deadband_l2 = RewTerm(
        func=mdp.exact_strike_pitch_rate_deadband_l2,
        weight=0.0,
        params={"command_name": "motion", "deadband": 0.06},
    )
    exact_strike_forward_velocity_deadband_l2 = RewTerm(
        func=mdp.exact_strike_forward_velocity_deadband_l2,
        weight=0.0,
        params={"command_name": "motion", "deadband": 0.05},
    )
    post_hit_forward_velocity_deadband_l2 = RewTerm(
        func=mdp.post_hit_forward_velocity_deadband_l2,
        weight=0.0,
        params={
            "command_name": "motion",
            "deadband": 0.06,
            "delay_steps": 2,
            "ramp_steps": 8,
        },
    )
    post_hit_pitch_rate_deadband_l2 = RewTerm(
        func=mdp.post_hit_pitch_rate_deadband_l2,
        weight=0.0,
        params={
            "command_name": "motion",
            "deadband": 0.08,
            "delay_steps": 2,
            "ramp_steps": 8,
        },
    )
    post_hit_capture_point_center_l2 = RewTerm(
        func=mdp.post_hit_capture_point_center_l2,
        weight=0.0,
        params={
            "command_name": "motion",
            "deadband": 0.04,
            "delay_steps": 2,
            "ramp_steps": 8,
        },
    )
    post_hit_capture_point_barrier_l2 = RewTerm(
        func=mdp.post_hit_capture_point_barrier_l2,
        weight=0.0,
        params={
            "command_name": "motion",
            "target_margin": 0.06,
            "delay_steps": 2,
            "ramp_steps": 8,
        },
    )
    post_hit_capture_point_center_progress = RewTerm(
        func=mdp.PostHitCapturePointCenterProgress,
        weight=0.0,
        params={"command_name": "motion", "delay_steps": 2, "ramp_steps": 8},
    )
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
    stagger_capture_point_margin_l2 = RewTerm(
        func=mdp.stagger_capture_point_margin_l2,
        weight=0.0,
        params={"target_margin": 0.04},
    )
    stagger_lateral_capture_point_margin_l2 = RewTerm(
        func=mdp.stagger_lateral_capture_point_margin_l2,
        weight=0.0,
        params={"target_margin": 0.035},
    )


    stagger_minimum_foot_load = RewTerm(
        func=mdp.stagger_minimum_foot_load,
        weight=0.0,
        params={"minimum_body_weight_fraction": 0.08},
    )
    stagger_sagittal_span_l2 = RewTerm(
        func=mdp.stagger_sagittal_span_l2,
        weight=0.0,
        params={"target_span": 0.08, "deadband": 0.015},
    )
    stagger_lateral_span_l2 = RewTerm(
        func=mdp.stagger_lateral_span_l2,
        weight=0.0,
        params={"target_span": 0.42, "deadband": 0.03},
    )


@configclass
class A3BentReadyRecoveryRewardsCfg(A3JointCoordinatorRewardsCfg):
    """V28 return-phase objective; strike terms remain inherited unchanged."""

    bent_ready_arm_score = RewTerm(
        func=mdp.post_strike_bent_ready_arm_score,
        weight=3.0,
        params={"command_name": "motion", "position_std": 0.15, "velocity_std": 0.50},
    )
    bent_ready_progress = RewTerm(
        func=mdp.PostStrikeBentReadyProgress,
        weight=2.0,
        params={"command_name": "motion"},
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
    # Kept effectively disabled by default for historical task compatibility.
    # Full-cycle coordinator tasks opt into a conservative 30 degree envelope
    # through their YAML contract so PPO cannot learn from an already-falling
    # 30--60 degree trajectory until root height finally crosses 0.65 m.
    recovery_tilt = DoneTerm(
        func=mdp.SustainedRootTiltExceeded,
        params={
            "max_tilt_rad": 1.55,
            "required_steps": 3,
        },
    )
    # Strict visual-fall gate.  The legacy 1.55 rad (~89 deg) envelope is
    # retained above for compatibility, but is not sufficient to catch a
    # robot that is visibly down while its root is still above 0.65 m.
    strict_fall = DoneTerm(
        func=mdp.StrictRootFallExceeded,
        params={
            "max_tilt_rad": 0.785398,  # 45 degrees
            "minimum_height": 0.82,
            "max_torso_tilt_rad": 0.785398,  # 45 degrees; catches torso collapse while root stays upright
            "minimum_torso_height": 0.70,
            "required_steps": 2,
        },
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
            # Strict visual-fall auditing should not wait for a 10 N impact.
            "threshold": 1.0,
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
class A3ProgressiveFallAssistEventsCfg(A3StrikeStabilizerAEventsCfg):
    """P5U bootstrap events with a decaying anti-fall wrench."""

    fall_assist = EventTerm(
        func=mdp.apply_progressive_fall_assist,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        is_global_time=True,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["pelvis_link"]),
            # Bootstrap assistance is intentionally strong enough to keep the
            # fixed upper-body reference physically recoverable for the full
            # 10 s learning window.  This is still a bounded external wrench;
            # it does not teleport pose or disable fall termination.
            "max_torque_nm": 100.0,
            "kp_nm_per_rad": 30.0,
            "kd_nms_per_rad": 12.0,
            "tilt_deadband_rad": 0.05,
            # 24 control steps per PPO iteration: 1500 iterations = 36000
            # control steps.  Adaptive feedback may restore assistance when
            # strict-fall rate rises, but never lets it decay faster than this
            # schedule.
            "anneal_steps": 36000,
            "minimum_scale": 0.10,
            "adaptive_enabled": True,
            "adapt_interval_steps": 24,
            "failure_rate_high": 0.0025,
            "failure_rate_low": 0.0008,
            "adaptive_increase": 0.03,
            "adaptive_decrease": 0.01,
            "torso_max_torque_nm": 100.0,
            "torso_kp_nm_per_rad": 40.0,
            "torso_kd_nms_per_rad": 12.0,
            "emergency_tilt_rad": 0.35,
            "emergency_gain": 4.0,
            "torso_emergency_tilt_rad": 0.35,
            "torso_emergency_gain": 7.0,
            "max_force_n": 0.0,
            "force_kp_n_per_rad": 0.0,
            "force_kd_ns_per_mps": 0.0,
            "torso_max_force_n": 0.0,
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
    # F0 is an evaluation plant, but it must still terminate on the same
    # unified physical fall contract as P5U/P5D.  Inheriting the native
    # strike terminations silently left only ``time_out`` active, allowing a
    # visibly collapsed torso to continue as a nominal episode.
    terminations: A3StrikeStabilizerATerminationsCfg = A3StrikeStabilizerATerminationsCfg()

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
        self.commands.motion.post_hit_guard_steps = 75
        self.commands.motion.recovery_timeout_steps = 250
        self.commands.motion.recovery_ready_hold_steps = 15
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
class A3ReferenceTrackerObservationsCfg(ObservationsCfg):
    """Deployable state plus explicit current/future safe-reference preview."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        command = None
        motion_anchor_pos_b = None
        motion_anchor_ori_b = None
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.10, n_max=0.10))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.03, n_max=0.03))
        feet_contact = ObsTerm(
            func=mdp.feet_contact_state,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=A3_FEET_BODIES),
                "threshold": 10.0,
            },
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=A3_REFERENCE_TRACKER_JOINTS, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=A3_REFERENCE_TRACKER_JOINTS, preserve_order=True)},
            noise=Unoise(n_min=-0.35, n_max=0.35),
        )
        reference_joint_pos = ObsTerm(
            func=mdp.motion_joint_pos,
            params={"command_name": "motion", "asset_cfg": SceneEntityCfg("robot", joint_names=A3_REFERENCE_TRACKER_JOINTS, preserve_order=True)},
        )
        reference_joint_vel = ObsTerm(
            func=mdp.motion_joint_vel,
            params={"command_name": "motion", "asset_cfg": SceneEntityCfg("robot", joint_names=A3_REFERENCE_TRACKER_JOINTS, preserve_order=True)},
        )
        reference_joint_pos_error = ObsTerm(
            func=mdp.motion_joint_position_error,
            params={"command_name": "motion", "asset_cfg": SceneEntityCfg("robot", joint_names=A3_REFERENCE_TRACKER_JOINTS, preserve_order=True)},
        )
        reference_joint_vel_error = ObsTerm(
            func=mdp.motion_joint_velocity_error,
            params={"command_name": "motion", "asset_cfg": SceneEntityCfg("robot", joint_names=A3_REFERENCE_TRACKER_JOINTS, preserve_order=True)},
        )
        reference_joint_pos_8 = ObsTerm(
            func=mdp.motion_joint_pos,
            params={"command_name": "motion", "asset_cfg": SceneEntityCfg("robot", joint_names=A3_REFERENCE_TRACKER_JOINTS, preserve_order=True), "lookahead_steps": 8},
        )
        reference_joint_vel_8 = ObsTerm(
            func=mdp.motion_joint_vel,
            params={"command_name": "motion", "asset_cfg": SceneEntityCfg("robot", joint_names=A3_REFERENCE_TRACKER_JOINTS, preserve_order=True), "lookahead_steps": 8},
        )
        reference_joint_pos_16 = ObsTerm(
            func=mdp.motion_joint_pos,
            params={"command_name": "motion", "asset_cfg": SceneEntityCfg("robot", joint_names=A3_REFERENCE_TRACKER_JOINTS, preserve_order=True), "lookahead_steps": 16},
        )
        reference_joint_vel_16 = ObsTerm(
            func=mdp.motion_joint_vel,
            params={"command_name": "motion", "asset_cfg": SceneEntityCfg("robot", joint_names=A3_REFERENCE_TRACKER_JOINTS, preserve_order=True), "lookahead_steps": 16},
        )
        # Multi-timescale preview is required for phase-lead learning. These
        # terms are inputs to P5D only; the frozen 900 prior is unchanged.
        reference_joint_pos_1 = ObsTerm(
            func=mdp.motion_joint_pos,
            params={"command_name": "motion", "asset_cfg": SceneEntityCfg("robot", joint_names=A3_REFERENCE_TRACKER_JOINTS, preserve_order=True), "lookahead_steps": 1},
        )
        reference_joint_vel_1 = ObsTerm(
            func=mdp.motion_joint_vel,
            params={"command_name": "motion", "asset_cfg": SceneEntityCfg("robot", joint_names=A3_REFERENCE_TRACKER_JOINTS, preserve_order=True), "lookahead_steps": 1},
        )
        reference_joint_pos_3 = ObsTerm(
            func=mdp.motion_joint_pos,
            params={"command_name": "motion", "asset_cfg": SceneEntityCfg("robot", joint_names=A3_REFERENCE_TRACKER_JOINTS, preserve_order=True), "lookahead_steps": 3},
        )
        reference_joint_vel_3 = ObsTerm(
            func=mdp.motion_joint_vel,
            params={"command_name": "motion", "asset_cfg": SceneEntityCfg("robot", joint_names=A3_REFERENCE_TRACKER_JOINTS, preserve_order=True), "lookahead_steps": 3},
        )
        reference_joint_pos_6 = ObsTerm(
            func=mdp.motion_joint_pos,
            params={"command_name": "motion", "asset_cfg": SceneEntityCfg("robot", joint_names=A3_REFERENCE_TRACKER_JOINTS, preserve_order=True), "lookahead_steps": 6},
        )
        reference_joint_vel_6 = ObsTerm(
            func=mdp.motion_joint_vel,
            params={"command_name": "motion", "asset_cfg": SceneEntityCfg("robot", joint_names=A3_REFERENCE_TRACKER_JOINTS, preserve_order=True), "lookahead_steps": 6},
        )
        reference_joint_pos_12 = ObsTerm(
            func=mdp.motion_joint_pos,
            params={"command_name": "motion", "asset_cfg": SceneEntityCfg("robot", joint_names=A3_REFERENCE_TRACKER_JOINTS, preserve_order=True), "lookahead_steps": 12},
        )
        reference_joint_vel_12 = ObsTerm(
            func=mdp.motion_joint_vel,
            params={"command_name": "motion", "asset_cfg": SceneEntityCfg("robot", joint_names=A3_REFERENCE_TRACKER_JOINTS, preserve_order=True), "lookahead_steps": 12},
        )
        strike_phase = ObsTerm(func=mdp.motion_phase, params={"command_name": "motion"})
        strike_phase_sin = ObsTerm(func=mdp.motion_phase_sin, params={"command_name": "motion"})
        strike_phase_cos = ObsTerm(func=mdp.motion_phase_cos, params={"command_name": "motion"})
        time_to_strike = ObsTerm(func=mdp.time_to_strike, params={"command_name": "racket_target"})
        marked_hit_step = ObsTerm(func=mdp.motion_hit_step_normalized, params={"command_name": "motion"})
        racket_target_pos_b = ObsTerm(func=mdp.racket_target_pos_b, params={"command_name": "racket_target"})
        racket_target_vel_b = ObsTerm(func=mdp.racket_target_vel_b, params={"command_name": "racket_target"})
        racket_target_normal_b = ObsTerm(func=mdp.racket_target_normal_b, params={"command_name": "racket_target"})
        racket_pos_b = ObsTerm(func=mdp.racket_pos_b, params={"command_name": "racket_target"})
        racket_lin_vel_b = ObsTerm(func=mdp.racket_lin_vel_b, params={"command_name": "racket_target"})
        racket_normal_b = ObsTerm(func=mdp.racket_normal_b, params={"command_name": "racket_target"})
        racket_target_error_pos_b = ObsTerm(func=mdp.racket_target_error_pos_b, params={"command_name": "racket_target"})
        racket_target_error_vel_b = ObsTerm(func=mdp.racket_target_error_vel_b, params={"command_name": "racket_target"})
        racket_target_error_normal_b = ObsTerm(func=mdp.racket_target_error_normal_b, params={"command_name": "racket_target"})
        reference_racket_pos_b = ObsTerm(func=mdp.motion_racket_pos_b, params={"command_name": "motion"})
        reference_racket_vel_b = ObsTerm(func=mdp.motion_racket_vel_b, params={"command_name": "motion"})
        reference_racket_normal_b = ObsTerm(func=mdp.motion_racket_normal_b, params={"command_name": "motion"})
        reference_racket_pos_b_1 = ObsTerm(func=mdp.motion_racket_pos_b, params={"command_name": "motion", "lookahead_steps": 1})
        reference_racket_vel_b_1 = ObsTerm(func=mdp.motion_racket_vel_b, params={"command_name": "motion", "lookahead_steps": 1})
        reference_racket_normal_b_1 = ObsTerm(func=mdp.motion_racket_normal_b, params={"command_name": "motion", "lookahead_steps": 1})
        reference_racket_pos_b_3 = ObsTerm(func=mdp.motion_racket_pos_b, params={"command_name": "motion", "lookahead_steps": 3})
        reference_racket_vel_b_3 = ObsTerm(func=mdp.motion_racket_vel_b, params={"command_name": "motion", "lookahead_steps": 3})
        reference_racket_normal_b_3 = ObsTerm(func=mdp.motion_racket_normal_b, params={"command_name": "motion", "lookahead_steps": 3})
        reference_racket_pos_b_6 = ObsTerm(func=mdp.motion_racket_pos_b, params={"command_name": "motion", "lookahead_steps": 6})
        reference_racket_vel_b_6 = ObsTerm(func=mdp.motion_racket_vel_b, params={"command_name": "motion", "lookahead_steps": 6})
        reference_racket_normal_b_6 = ObsTerm(func=mdp.motion_racket_normal_b, params={"command_name": "motion", "lookahead_steps": 6})
        reference_racket_pos_b_12 = ObsTerm(func=mdp.motion_racket_pos_b, params={"command_name": "motion", "lookahead_steps": 12})
        reference_racket_vel_b_12 = ObsTerm(func=mdp.motion_racket_vel_b, params={"command_name": "motion", "lookahead_steps": 12})
        reference_racket_normal_b_12 = ObsTerm(func=mdp.motion_racket_normal_b, params={"command_name": "motion", "lookahead_steps": 12})
        reference_racket_pos_error_b = ObsTerm(func=mdp.motion_racket_pos_error_b, params={"command_name": "motion"})
        reference_racket_vel_error_b = ObsTerm(func=mdp.motion_racket_vel_error_b, params={"command_name": "motion"})
        reference_racket_normal_error_b = ObsTerm(func=mdp.motion_racket_normal_error_b, params={"command_name": "motion"})
        actions = ObsTerm(func=mdp.last_action, params={"action_name": "joint_pos"})

    @configclass
    class CriticCfg(PolicyCfg):
        """Asymmetric critic: privileged world root state during training only."""

        root_pos_w = ObsTerm(
            func=mdp.root_pos_w,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        root_quat_w = ObsTerm(
            func=mdp.root_quat_w,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        root_lin_vel_w = ObsTerm(
            func=mdp.root_lin_vel_w,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        root_ang_vel_w = ObsTerm(
            func=mdp.root_ang_vel_w,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class A3PriorGuidedReferenceTrackerObservationsCfg(
    A3ReferenceTrackerObservationsCfg
):
    """P5D tracker observation plus private frozen-prior input contracts."""

    # These groups are consumed only inside the frozen 900/3396 action term.
    # The learnable P5D actor still receives the reference-preview policy
    # group above and never receives a motion ID or control anchor.
    upper: A3F0ObservationsCfg.PolicyCfg = A3F0ObservationsCfg.PolicyCfg()
    stage_a: A3UpperCorrectionObservationsCfg.StageACfg = (
        A3UpperCorrectionObservationsCfg.StageACfg()
    )


@configclass
class A3UnifiedUpperReferenceTrackerObservationsCfg(A3ReferenceTrackerObservationsCfg):
    """Public reference-preview observations plus private model_3396 input."""

    # Only the frozen lower-support action term consumes this group.  It is
    # intentionally not part of the public policy group, so the new actor has
    # no model/reference/seed ID and no access to the legacy upper prior.
    stage_a: A3UpperCorrectionObservationsCfg.StageACfg = (
        A3UpperCorrectionObservationsCfg.StageACfg()
    )


@configclass
class A3ReferenceTrackerRewardsCfg(A3F1StrikeAwareRewardsCfg):
    """Dense reference tracking with hit-window task and recovery checks."""

    reference_joint_position = RewTerm(
        func=mdp.motion_joint_position_error_exp,
        weight=3.0,
        params={
            "command_name": "motion",
            "std": 0.20,
            "asset_cfg": SceneEntityCfg("robot", joint_names=A3_REFERENCE_TRACKER_JOINTS, preserve_order=True),
        },
    )
    reference_joint_velocity = RewTerm(
        func=mdp.motion_joint_velocity_error_exp,
        weight=1.5,
        params={
            "command_name": "motion",
            "std": 2.5,
            "asset_cfg": SceneEntityCfg("robot", joint_names=A3_REFERENCE_TRACKER_JOINTS, preserve_order=True),
        },
    )
    racket_position = RewTerm(func=mdp.racket_position_tracking_exp, weight=8.0, params={"command_name": "racket_target", "std": 0.12})
    racket_position_y = RewTerm(func=mdp.racket_position_axis_tracking_exp, weight=2.0, params={"command_name": "racket_target", "axis": 1, "std": 0.12})
    racket_position_fine = RewTerm(func=mdp.racket_position_tracking_exp, weight=2.0, params={"command_name": "racket_target", "std": 0.05})
    racket_velocity = RewTerm(func=mdp.racket_velocity_tracking_exp, weight=3.0, params={"command_name": "racket_target", "std": 1.5})
    racket_velocity_magnitude = RewTerm(
        func=mdp.racket_velocity_magnitude_tracking_exp,
        weight=2.0,
        params={"command_name": "racket_target", "std": 1.0, "half_window_steps": 3},
    )
    racket_velocity_direction = RewTerm(
        func=mdp.racket_velocity_direction_tracking,
        weight=2.0,
        params={"command_name": "racket_target", "half_window_steps": 3},
    )
    racket_signed_velocity = RewTerm(
        func=mdp.racket_signed_velocity_tracking,
        weight=2.0,
        params={"command_name": "racket_target", "half_window_steps": 3},
    )
    racket_pass_through = RewTerm(
        func=mdp.racket_pass_through_reward,
        weight=3.0,
        params={"command_name": "racket_target", "position_gate": 0.10, "minimum_speed": 0.5, "half_window_steps": 3},
    )
    racket_stop_at_target = RewTerm(
        func=mdp.racket_stop_at_target_penalty,
        weight=2.0,
        params={"command_name": "racket_target", "position_gate": 0.10, "minimum_speed": 0.5, "half_window_steps": 3},
    )
    racket_reverse_motion = RewTerm(
        func=mdp.racket_reverse_motion_penalty,
        weight=2.0,
        params={"command_name": "racket_target", "half_window_steps": 3},
    )
    racket_hit_timing = RewTerm(
        func=mdp.racket_hit_timing_kernel,
        weight=1.0,
        params={"command_name": "racket_target", "half_window_steps": 3},
    )
    racket_normal = RewTerm(func=mdp.racket_normal_tracking_exp, weight=2.0, params={"command_name": "racket_target", "std": 0.50})
    racket_hit_coupled = RewTerm(
        func=mdp.racket_hit_coupled_tracking_exp,
        weight=1.0,
        params={"command_name": "racket_target", "pos_std": 0.12, "vel_std": 1.5, "normal_std": 0.50},
    )
    racket_hit_precision = RewTerm(
        func=mdp.racket_exact_hit_precision_tracking_exp,
        weight=8.0,
        params={
            "command_name": "racket_target",
            "pos_std": 0.08,
            "vel_std": 1.0,
            "normal_std": 0.35,
            "time_std": 0.04,
        },
    )
    # The safe reference is the nominal plan.  A weak residual trust region
    # prevents PPO from replacing it with a second unconstrained planner.
    action_residual_l2 = RewTerm(func=mdp.action_raw_l2, weight=-0.01, params={"action_name": "joint_pos"})
    phase_magnitude = RewTerm(func=mdp.phase_magnitude_penalty, weight=0.02, params={"action_name": "joint_pos"})
    phase_rate = RewTerm(func=mdp.phase_rate_penalty, weight=0.05, params={"action_name": "joint_pos"})
    phase_group_consistency = RewTerm(func=mdp.phase_group_consistency_penalty, weight=0.02, params={"action_name": "joint_pos"})
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-20.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=A3_REFERENCE_TRACKER_JOINTS, preserve_order=True)},
    )
    feet_slip = RewTerm(
        func=mdp.feet_slip_l2,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=A3_FEET_BODIES), "threshold": 10.0},
    )
    # Dense early warning complements the terminal fall penalty.  It is
    # inactive in the normal upright range and rises before strict_fall fires.
    strict_fall_risk = RewTerm(
        func=mdp.strict_fall_risk_l2,
        weight=-25.0,
        params={
            "minimum_upright": 0.80,
            "minimum_height": 0.90,
            "minimum_torso_upright": 0.85,
            "minimum_torso_height": 0.80,
        },
    )
    # The finite reference is held after the swing.  These two tail-only
    # terms damp visible torso roll/pitch sway without suppressing the hit or
    # follow-through and without penalizing the intended yaw heading.
    post_strike_torso_angular_velocity_l2 = RewTerm(
        func=mdp.post_strike_torso_angular_velocity_l2,
        weight=-0.20,
        params={
            "command_name": "motion",
            "torso_body_name": "torso_Link",
            "deadband": 0.06,
        },
    )
    post_strike_torso_tilt_l2 = RewTerm(
        func=mdp.post_strike_torso_tilt_l2,
        weight=-0.15,
        params={
            "command_name": "motion",
            "torso_body_name": "torso_Link",
        },
    )
    recovery_completion_bonus = RewTerm(
        func=mdp.recovery_completion_bonus,
        weight=0.0,
    )
    terminal_remaining_horizon = RewTerm(
        func=mdp.terminal_remaining_horizon_penalty,
        weight=0.0,
        params={"horizon_steps": 250},
    )
    fall = RewTerm(func=mdp.is_terminated, weight=-150.0)
    alive = RewTerm(func=mdp.is_alive, weight=1.0)


@configclass
class A3FloatingReferenceTrackerEnvCfg(A3StrikeConditionedBaseEnvCfg):
    """P5D generic floating-base safe-reference execution task.

    This is intentionally a tracker, not a target adapter: it receives a
    reference trajectory from the manifest and learns only bounded residuals
    that improve actual tracking under the same joint safety projection.
    """

    actions: A3ReferenceTrackerActionsCfg = A3ReferenceTrackerActionsCfg()
    observations: A3ReferenceTrackerObservationsCfg = A3ReferenceTrackerObservationsCfg()
    rewards: A3ReferenceTrackerRewardsCfg = A3ReferenceTrackerRewardsCfg()
    terminations: A3StrikeStabilizerATerminationsCfg = A3StrikeStabilizerATerminationsCfg()
    events: A3StrikeStabilizerAEventsCfg = A3StrikeStabilizerAEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        # A3NativeStrikeEnvCfg rewrites actions into its historical 10-D
        # waist/right-arm executor.  Restore P5D's explicit 22-D tracker
        # contract after that inherited compatibility mutation.
        tracker_template = A3ReferenceTrackerActionsCfg().joint_pos
        self.actions.joint_pos.joint_names = tuple(tracker_template.joint_names)
        self.actions.joint_pos.scale = dict(tracker_template.scale)
        self.actions.joint_pos.reference_lookahead_steps = 0
        self.actions.joint_pos.joint_reference_lookahead_steps = {}
        self.actions.joint_pos.soft_limit_margin_rad_by_joint = dict(
            tracker_template.soft_limit_margin_rad_by_joint
        )
        self.scene.robot.spawn.fix_base = False
        # Formal P1 materialized references are expressed in this scene root
        # frame.  Do not inherit F0's historical (3.15, -0.35) work point:
        # that creates metre-scale fake tracking error before PPO acts.
        self.scene.robot.init_state.pos = (-0.5000, -0.7625, 1.0400)
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.commands.motion.expected_root_quaternion_wxyz = (1.0, 0.0, 0.0, 0.0)
        self.scene.robot.init_state.joint_pos = {
            **self.scene.robot.init_state.joint_pos,
            ".*_hip_pitch_joint": -0.1600,
            ".*_knee_joint": 0.3200,
            ".*_ankle_pitch_joint": -0.1550,
            "left_hip_roll_joint": 0.0800,
            "right_hip_roll_joint": -0.0800,
        }
        # Match the formal P1 timing but leave target interpretation canonical.
        self.commands.motion.sample_random_start_phase = False
        self.commands.motion.prelude_steps = 50
        self.commands.motion.hold_last_frame_steps = 0
        self.commands.motion.return_to_default_steps = 0
        self.commands.motion.reset_to_default_pose = True
        self.commands.racket_target.target_mode = "manifest"
        self.commands.racket_target.manifest_nominal_probability = 1.0
        self.commands.racket_target.strike_time_std_s = 0.0
        self.commands.racket_target.adapter_external_offset_half_range = (0.0, 0.0, 0.0)
        self.commands.racket_target.adapter_external_zero_probability = 1.0
        self.commands.racket_target.adapter_external_paired = False


@configclass
class A3FloatingPriorGuidedReferenceTrackerEnvCfg(
    A3FloatingF0EnvCfg
):
    """P5D bootstrap plant retaining the verified 3396/900 support chain.

    This class is intentionally separate from ``A3FloatingReferenceTracker``:
    the latter remains the zero-prior ablation.  Neither class grants the
    frozen policies authority to relabel a canonical goal or to approve a
    teacher; they only supply a stable execution initialisation for the shared
    residual tracker.
    """

    actions: A3PriorGuidedReferenceTrackerActionsCfg = (
        A3PriorGuidedReferenceTrackerActionsCfg()
    )
    observations: A3PriorGuidedReferenceTrackerObservationsCfg = (
        A3PriorGuidedReferenceTrackerObservationsCfg()
    )
    rewards: A3ReferenceTrackerRewardsCfg = A3ReferenceTrackerRewardsCfg()
    terminations: A3StrikeStabilizerATerminationsCfg = A3StrikeStabilizerATerminationsCfg()
    events: A3StrikeStabilizerAEventsCfg = A3StrikeStabilizerAEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        # A3F1StrikeAwareRewardsCfg was authored for a public Base14 action
        # and inherits indices 0..11 for the old leg actor.  P5D's public
        # tensor is instead exactly the 10-D upper correction; leaving those
        # indices in place causes an out-of-range CUDA gather before any
        # physics or balance conclusion can be drawn.
        self.rewards.raw_action_excess.params["action_indices"] = tuple(range(10))
        self.rewards.action_execution_gap.params["action_indices"] = tuple(range(10))
        # The P1 materialized reference and its canonical goal use this
        # physical root frame, rather than F0's historical work point.
        self.scene.robot.init_state.pos = (-0.5000, -0.7625, 1.0400)
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.commands.motion.expected_root_quaternion_wxyz = (1.0, 0.0, 0.0, 0.0)
        self.commands.motion.sample_random_start_phase = False
        self.commands.motion.prelude_steps = 50
        self.commands.motion.hold_last_frame_steps = 0
        self.commands.motion.return_to_default_steps = 0
        self.commands.motion.reset_to_default_pose = True
        self.commands.racket_target.target_mode = "manifest"
        self.commands.racket_target.manifest_nominal_probability = 1.0
        self.commands.racket_target.strike_time_std_s = 0.0
        self.commands.racket_target.adapter_external_offset_half_range = (0.0, 0.0, 0.0)
        self.commands.racket_target.adapter_external_zero_probability = 1.0
        self.commands.racket_target.adapter_external_paired = False


@configclass
class A3FloatingUnifiedUpperReferenceTrackerEnvCfg(A3FloatingF0EnvCfg):
    """P5U-1 floating-base unified upper tracker.

    ``model_3396`` remains the nominal lower support and ``model_900`` is not
    constructed or called anywhere in this environment.  Legacy variants may
    expose only the ten-dimensional upper residual; the production NoAssist
    variant additionally enables a bounded 12-D lower balance residual around
    model_3396.
    """

    actions: A3UnifiedUpperReferenceTrackerActionsCfg = A3UnifiedUpperReferenceTrackerActionsCfg()
    observations: A3UnifiedUpperReferenceTrackerObservationsCfg = (
        A3UnifiedUpperReferenceTrackerObservationsCfg()
    )
    rewards: A3ReferenceTrackerRewardsCfg = A3ReferenceTrackerRewardsCfg()
    terminations: A3StrikeStabilizerATerminationsCfg = A3StrikeStabilizerATerminationsCfg()
    events: A3ProgressiveFallAssistEventsCfg = A3ProgressiveFallAssistEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        # The inherited strike-aware reward bundle contains legacy Base14
        # action-index defaults.  Legacy P5U exposes ten upper residuals, so
        # every action-indexed diagnostic must be explicitly narrowed to 0:10
        # before the first CUDA step.
        self.rewards.raw_action_excess.params["action_indices"] = tuple(range(10))
        self.rewards.action_execution_gap.params["action_indices"] = tuple(range(10))
        # P5 canonical manifests use the same materialized world frame as the
        # existing reference-tracker audit, not F0's historical work point.
        self.scene.robot.init_state.pos = (-0.5000, -0.7625, 1.0400)
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.commands.motion.expected_root_quaternion_wxyz = (1.0, 0.0, 0.0, 0.0)
        self.scene.robot.init_state.joint_pos = {
            **self.scene.robot.init_state.joint_pos,
            ".*_hip_pitch_joint": -0.1600,
            ".*_knee_joint": 0.3200,
            ".*_ankle_pitch_joint": -0.1550,
            "left_hip_roll_joint": 0.0800,
            "right_hip_roll_joint": -0.0800,
        }
        self.scene.robot.spawn.fix_base = False
        self.commands.motion.sample_random_start_phase = False
        self.commands.motion.prelude_steps = 50
        self.commands.motion.hold_last_frame_steps = 0
        self.commands.motion.return_to_default_steps = 0
        self.commands.motion.reset_to_default_pose = True
        self.commands.racket_target.target_mode = "manifest"
        self.commands.racket_target.manifest_nominal_probability = 1.0
        self.commands.racket_target.strike_time_std_s = 0.0
        self.commands.racket_target.adapter_external_offset_half_range = (0.0, 0.0, 0.0)
        self.commands.racket_target.adapter_external_zero_probability = 1.0
        self.commands.racket_target.adapter_external_paired = False


@configclass
class A3FloatingUnifiedUpperReferenceTrackerNoAssistEnvCfg(
    A3FloatingUnifiedUpperReferenceTrackerEnvCfg
):
    """P5U unified tracker with no external fall-assist wrench.

    The regular P5U class retains the historical progressive fall-assist
    bootstrap.  This separate environment keeps the same READY stance,
    action contract, rewards, and terminations while removing that external
    torque event for unbiased training on the safe augmented motion bank.
    """

    events: A3StrikeStabilizerAEventsCfg = A3StrikeStabilizerAEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        # model_3396 is a nominal prior, not the final lower-body controller.
        # Give PPO explicit authority over all twelve leg channels while
        # preserving the frozen prior as the additive baseline.  The action
        # term ramps this residual in during the READY prelude.
        self.actions.joint_pos.lower_balance_residual_enabled = True
        # Contract A: 12 lower residuals + 10 upper residuals.  The action
        # term computes the same dimension at runtime; keep the reward index
        # contract explicit at config time because the cfg object has no
        # bound ActionTerm instance yet.
        action_dim = 12 + len(self.actions.joint_pos.upper_joint_names)
        self.rewards.raw_action_excess.params["action_indices"] = tuple(range(action_dim))
        self.rewards.action_execution_gap.params["action_indices"] = tuple(range(action_dim))


@configclass
class A3ReferenceFreeCommandsCfg(HOPECommandsCfg):
    """Command set with no motion/manifest term at runtime."""

    # Temporary construction placeholder: the inherited A3 cfg post-init
    # touches these fields.  V1.3B sets it back to ``None`` before the
    # ManagerBasedRLEnv is built, so no motion command is active at runtime.
    motion = HOPECommandsCfg().motion
    racket_target = mdp.ReferenceFreeRacketTargetCommandCfg(
        asset_name="robot",
        debug_vis=False,
        racket_body_name="__force_wrist_offset_racket_fk__",
        wrist_body_name="right_wrist_yaw_Link",
        racket_fk_mode="wrist_offset",
        target_mode="reference_free_global",
        strike_window_s=0.06,
        strike_time_std_s=0.05,
    )


@configclass
class A3WorkspaceExpansionCommandsCfg(A3ReferenceFreeCommandsCfg):
    """Pure V1.3B commands; the manifest is anchor metadata only."""

    racket_target = mdp.ReferenceFreeRacketTargetCommandCfg(
        asset_name="robot",
        debug_vis=False,
        racket_body_name="__force_wrist_offset_racket_fk__",
        wrist_body_name="right_wrist_yaw_Link",
        racket_fk_mode="wrist_offset",
        target_mode="reference_free_global",
        strike_window_s=0.06,
        strike_time_std_s=0.05,
        workspace_expansion_enabled=True,
        workspace_sampling_mode="audited_anchor_bank",
        workspace_anchor_bank_enabled=True,
        workspace_keep_motion_anchor_final=False,
    )


@configclass
class A3AnnealedPriorCommandsCfg(A3ReferenceFreeCommandsCfg):
    """Training-only command set retaining a private motion source.

    The public policy group remains reference-free.  The motion term exists
    only so the frozen model_3396 prior can receive its private 126-D stage-A
    observation during the annealing run; the final V1.3B task removes it.
    """

    motion = HOPECommandsCfg().motion
    # This command is private to the frozen 3396/900 historical observation
    # contracts.  Its manifest target remains coherent with ``motion`` while
    # the public ``racket_target`` samples the independent V1.3B 10-D goal.
    # It is deliberately never named by the deployable 98-D observation or a
    # reward term.
    teacher_racket_target = HOPECommandsCfg().racket_target


@configclass
class A3PrecisionRescueCommandsCfg(A3AnnealedPriorCommandsCfg):
    """CompletePriors private-prior setup with the same public local sampler.

    Only the command *class* differs: it records Rescue episode accounting.
    It does not change sampling, StrikeEvent, timing, or public observations.
    """

    racket_target = rescue_commands.PrecisionRescueRacketTargetCommandCfg(
        asset_name="robot",
        debug_vis=False,
        racket_body_name="__force_wrist_offset_racket_fk__",
        wrist_body_name="right_wrist_yaw_Link",
        racket_fk_mode="wrist_offset",
        target_mode="reference_free_global",
        strike_window_s=0.06,
        strike_time_std_s=0.05,
    )


@configclass
class A3ReferenceFreeTargetActionsCfg(ActionsCfg):
    """Action manager wrapper exposing the 26-D V1.3B term."""

    joint_pos = A3ReferenceFreeTargetConditionedPositionActionCfg(
        asset_name="robot",
        base_joint_names=tuple(A3_BASE_ACTION_JOINTS),
        backend_joint_names=tuple(A3_BACKEND_JOINTS),
        strike_joint_names=tuple(A3_STRIKE_V2_REFERENCE_JOINTS),
        upper_joint_names=tuple(A3_NATIVE_STRIKE_JOINTS),
        joint_names=tuple(A3_NATIVE_STRIKE_JOINTS),
        action_scale_rad=A3_PD_STAND_BASE_ACTION_SCALE_RAD,
        action_mask=(1.0,) * 14,
        scale=dict(AGIBOT_A3_NATIVE_STRIKE_ACTION_SCALE),
        direct_lower_scale_rad=(0.24,) * 12,
        direct_upper_scale_rad=(0.55,) * 10,
        direct_scale_config_path="cfg/target_conditioned/direct_action_scale_v13b.yaml",
        clip_to_soft_joint_limits=True,
        raw_clip=1.0,
        smooth_raw_bound=True,
        microstep_enabled=True,
    )


@configclass
class A3ReferenceFreeTargetObservationsCfg(ObservationsCfg):
    """Deployable robot state + exactly one canonical 10-D target."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        command = None
        motion_anchor_pos_b = None
        motion_anchor_ori_b = None
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.10, n_max=0.10))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.20, n_max=0.20))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=A3_REFERENCE_TRACKER_JOINTS, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=A3_REFERENCE_TRACKER_JOINTS, preserve_order=True)},
            noise=Unoise(n_min=-0.50, n_max=0.50),
        )
        racket_pos_b = ObsTerm(func=mdp.racket_pos_b, params={"command_name": "racket_target"})
        racket_lin_vel_b = ObsTerm(func=mdp.racket_lin_vel_b, params={"command_name": "racket_target"})
        racket_normal_b = ObsTerm(func=mdp.racket_normal_b, params={"command_name": "racket_target"})
        # One and only one canonical 10-D goal: [position, velocity, normal, signed time].
        strike_goal_10d = ObsTerm(
            func=mdp.racket_target_goal_10d_b,
            params={
                "command_name": "racket_target",
                "position_mean": (0.44237322, -0.34721070, 0.09162542),
                "position_std": (0.04256963, 0.29942963, 0.06187854),
                "time_std": 1.0,
                "time_clip_s": 4.0,
            },
        )
        swing_type = None
        actions = ObsTerm(func=mdp.last_action, params={"action_name": "joint_pos"})

    @configclass
    class CriticCfg(ObservationsCfg.PrivilegedCfg):
        command = None
        motion_anchor_pos_b = None
        motion_anchor_ori_b = None
        body_pos = None
        body_ori = None
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=A3_REFERENCE_TRACKER_JOINTS, preserve_order=True)})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=A3_REFERENCE_TRACKER_JOINTS, preserve_order=True)})
        racket_pos_b = ObsTerm(func=mdp.racket_pos_b, params={"command_name": "racket_target"})
        racket_lin_vel_w = ObsTerm(func=mdp.racket_lin_vel_w, params={"command_name": "racket_target"})
        racket_normal_w = ObsTerm(func=mdp.racket_normal_w, params={"command_name": "racket_target"})
        strike_goal_10d = ObsTerm(
            func=mdp.racket_target_goal_10d_b,
            params={
                "command_name": "racket_target",
                "position_mean": (0.44237322, -0.34721070, 0.09162542),
                "position_std": (0.04256963, 0.29942963, 0.06187854),
                "time_std": 1.0,
                "time_clip_s": 4.0,
            },
        )
        actions = ObsTerm(func=mdp.last_action, params={"action_name": "joint_pos"})
        episode_time_left = ObsTerm(func=mdp.episode_time_left)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class A3AnnealedPriorTargetObservationsCfg(A3ReferenceFreeTargetObservationsCfg):
    """Reference-free actor plus private 3396 and model_900 observations."""

    # Exact 56-D historical model_900 observation contract.  This group is
    # only queried while the upper prior alpha is nonzero.
    upper: A3F0ObservationsCfg.PolicyCfg = A3F0ObservationsCfg.PolicyCfg()
    stage_a: A3UpperCorrectionObservationsCfg.StageACfg = (
        A3UpperCorrectionObservationsCfg.StageACfg()
    )


@configclass
class A3ReferenceFreeTargetRewardsCfg(RewardsCfg):
    """Goal/event rewards with no motion imitation or teacher dependency."""

    motion_global_anchor_pos = None
    motion_global_anchor_ori = None
    motion_body_pos = None
    motion_body_ori = None
    motion_body_lin_vel = None
    motion_body_ang_vel = None

    racket_position = RewTerm(func=mdp.racket_position_tracking_exp, weight=4.0, params={"command_name": "racket_target", "std": 0.04})
    # Impact speed and face orientation are first-class objectives.  Position
    # remains primary, but these weights are intentionally strong enough that
    # the policy cannot obtain a good strike score while merely arriving at
    # the right point with the wrong racket velocity/normal.
    racket_velocity = RewTerm(func=mdp.racket_velocity_tracking_exp, weight=2.5, params={"command_name": "racket_target", "std": 0.50})
    racket_normal = RewTerm(func=mdp.racket_normal_tracking_exp, weight=3.0, params={"command_name": "racket_target", "std": 0.1745329})
    racket_hit_precision = RewTerm(
        func=mdp.racket_exact_hit_precision_tracking_exp,
        weight=4.0,
        params={
            "command_name": "racket_target",
            "pos_std": 0.04,
            "vel_std": 0.50,
            "normal_std": 0.1745329,
            "time_std": 0.05,
            "pos_coeff": 0.40,
            "velocity_coeff": 0.30,
            "normal_coeff": 0.30,
        },
    )
    pre_hit_progress = RewTerm(func=mdp.racket_target_progress, weight=0.25, params={"command_name": "racket_target", "scale_m": 0.10})
    # One 10-second V1.3B episode contains one strike opportunity.  Preserve
    # the first 150 ms of follow-through, then make a continued torso sway or
    # forward surge costly.  The gate uses only public signed time-to-hit;
    # private motion/model priors never enter these rewards.
    post_hit_torso_angular_velocity = RewTerm(
        func=mdp.post_hit_goal_torso_angular_velocity_l2,
        weight=-0.20,
        params={
            "command_name": "racket_target",
            "torso_body_name": "torso_Link",
            "deadband": 0.06,
            "follow_through_s": 0.15,
            "ramp_s": 0.35,
        },
    )
    post_hit_torso_tilt = RewTerm(
        func=mdp.post_hit_goal_torso_tilt_l2,
        weight=-0.15,
        params={
            "command_name": "racket_target",
            "torso_body_name": "torso_Link",
            "follow_through_s": 0.15,
            "ramp_s": 0.35,
        },
    )
    post_hit_forward_velocity = RewTerm(
        func=mdp.post_hit_goal_forward_velocity_deadband_l2,
        weight=-0.05,
        params={
            "command_name": "racket_target",
            "deadband": 0.08,
            "follow_through_s": 0.15,
            "ramp_s": 0.35,
        },
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    action_residual_l2 = RewTerm(func=mdp.action_raw_l2, weight=-0.01, params={"action_name": "joint_pos"})
    joint_limit = RewTerm(func=mdp.joint_pos_limits, weight=-10.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=A3_REFERENCE_TRACKER_JOINTS, preserve_order=True)})
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-6, params={"asset_cfg": SceneEntityCfg("robot", joint_names=A3_REFERENCE_TRACKER_JOINTS, preserve_order=True)})
    feet_slip = RewTerm(func=mdp.feet_slip_l2, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=A3_FEET_BODIES), "threshold": 10.0})
    strict_fall_risk = RewTerm(func=mdp.strict_fall_risk_l2, weight=-25.0, params={"minimum_upright": 0.80, "minimum_height": 0.90, "minimum_torso_upright": 0.85, "minimum_torso_height": 0.80})
    fall = RewTerm(func=mdp.is_terminated, weight=-150.0)
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # Compatibility placeholders used by the inherited A3 post-init; kept at
    # zero so the reference-free reward bundle has no hidden legacy term.
    raw_action_excess = RewTerm(func=mdp.action_unbounded_excess_l2, weight=0.0, params={"action_name": "joint_pos", "action_indices": tuple(range(26))})
    action_execution_gap = RewTerm(func=mdp.action_execution_gap_l2, weight=0.0, params={"action_name": "joint_pos", "action_indices": tuple(range(26))})


@configclass
class A3PrecisionRescueRewardsCfg(A3ReferenceFreeTargetRewardsCfg):
    """Current exact rewards plus opt-in wide p/v recovery shaping."""

    # The wrappers preserve the exact functions/parameters byte-for-byte in
    # value semantics and solely add per-episode accounting to the Rescue
    # command.  Exact position and pre-hit progress remain inherited; the
    # small wide position term supplies recovery shaping for the current
    # large-position-error failure mode.
    racket_position_wide_recovery = RewTerm(
        func=rescue_rewards.racket_position_tracking_wide_recovery_exp,
        weight=0.75,
        params={"command_name": "racket_target", "std": 0.12, "audit_weight": 0.75, "time_std_s": 0.05},
    )
    racket_velocity = RewTerm(
        func=rescue_rewards.racket_velocity_tracking_exact_audited,
        weight=2.5,
        params={"command_name": "racket_target", "std": 0.50, "audit_weight": 2.5},
    )
    racket_normal = RewTerm(
        func=rescue_rewards.racket_normal_tracking_exact_audited,
        weight=3.0,
        params={"command_name": "racket_target", "std": 0.1745329, "audit_weight": 3.0},
    )
    racket_normal_wide = RewTerm(
        func=rescue_rewards.racket_normal_tracking_wide_exp,
        weight=1.5,
        params={"command_name": "racket_target", "std": 0.60, "audit_weight": 1.5},
    )
    racket_velocity_wide = RewTerm(
        func=rescue_rewards.racket_velocity_tracking_position_gated_wide_exp,
        weight=1.25,
        params={
            "command_name": "racket_target",
            "velocity_std": 2.0,
            "position_threshold": 0.02,
            "position_excess_std": 0.05,
            "audit_weight": 1.25,
        },
    )


@configclass
class A3FloatingTargetConditionedReferenceFreeV13BEnvCfg(A3FloatingUnifiedUpperReferenceTrackerEnvCfg):
    """V1.3B: reference-free student, direct 26-D action, global 10-D goal."""

    commands: A3ReferenceFreeCommandsCfg = A3ReferenceFreeCommandsCfg()
    actions: A3ReferenceFreeTargetActionsCfg = A3ReferenceFreeTargetActionsCfg()
    observations: A3ReferenceFreeTargetObservationsCfg = A3ReferenceFreeTargetObservationsCfg()
    rewards: A3ReferenceFreeTargetRewardsCfg = A3ReferenceFreeTargetRewardsCfg()
    terminations: A3StrikeStabilizerATerminationsCfg = A3StrikeStabilizerATerminationsCfg()
    events: HOPEEventCfg = HOPEEventCfg()
    reference_free_mode: bool = True
    teacher_alpha_initial: float = 0.0
    target_global_probability_final: float = 1.0
    policy_goal_contract_version: str = "policy_strike_goal_10d/racket_contact_v1"

    def __post_init__(self):
        # Run the established A3 floating plant setup, then replace the
        # reference-bearing managers before ManagerBasedRLEnv construction.
        super().__post_init__()
        self.commands.motion = None
        self.commands.racket_target.target_mode = "reference_free_global"
        self.commands.racket_target.strike_time_std_s = 0.05
        self.commands.racket_target.racket_body_name = "__force_wrist_offset_racket_fk__"
        self.actions.joint_pos.base_joint_names = tuple(A3_BASE_ACTION_JOINTS)
        self.actions.joint_pos.backend_joint_names = tuple(A3_BACKEND_JOINTS)
        self.actions.joint_pos.strike_joint_names = tuple(A3_STRIKE_V2_REFERENCE_JOINTS)
        self.actions.joint_pos.upper_joint_names = tuple(A3_NATIVE_STRIKE_JOINTS)
        self.actions.joint_pos.joint_names = tuple(A3_NATIVE_STRIKE_JOINTS)
        self.actions.joint_pos.scale = dict(AGIBOT_A3_NATIVE_STRIKE_ACTION_SCALE)
        self.actions.joint_pos.action_scale_rad = A3_PD_STAND_BASE_ACTION_SCALE_RAD
        self.actions.joint_pos.action_mask = (1.0,) * 14
        self.actions.joint_pos.ready_joint_positions = dict(V13B_READY_JOINT_POSITIONS)
        self.scene.robot.spawn.fix_base = False
        self.scene.robot.init_state.pos = (-0.5000, -0.7625, 1.0400)
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        _apply_v13b_right_front_ready(self)
        # train.py may receive the same stance parameters through task YAML.
        # Mark the unshifted root reference so that override application
        # replaces this contract rather than adding the pelvis correction a
        # second time.
        self.v13b_right_front_ready_contract = True
        self.v13b_ready_root_reference_z = 1.0400
        self.events.sample_leg_policy_handoff = None
        self.v13b_policy_progress = 0.0
        self.v13b_private_motion_disabled = False
        self.rewards.undesired_contacts = None
        self.terminations.anchor_pos = None
        self.terminations.anchor_ori = None
        self.terminations.ee_body_pos = None
        if self.commands.motion is not None:
            raise RuntimeError("V1.3B reference-free contract failed: motion command still active")
        if self.observations.policy.swing_type is not None:
            raise RuntimeError("V1.3B actor contract failed: swing_type is still exposed")
        if len(self.actions.joint_pos.direct_lower_scale_rad) != 12:
            raise RuntimeError("V1.3B direct lower action scale must be 12-D")


@configclass
class A3FloatingTargetConditionedReferenceFreeV13BAnnealedPriorEnvCfg(
    A3FloatingTargetConditionedReferenceFreeV13BEnvCfg
):
    """Training-only complete priors, with a reference-free public actor."""

    commands: A3AnnealedPriorCommandsCfg = A3AnnealedPriorCommandsCfg()
    observations: A3AnnealedPriorTargetObservationsCfg = A3AnnealedPriorTargetObservationsCfg()
    training_only_annealed_prior: bool = True

    def __post_init__(self):
        super().__post_init__()
        # The parent intentionally disables motion for final deployment.  This
        # private branch restores it only for the frozen stage-A prior.
        self.commands.motion = HOPECommandsCfg().motion
        # The replacement above must not re-introduce the generic tracking
        # reset curriculum.  CompletePriors starts from the exact deployed
        # right-front READY state; robustness perturbations are a separate
        # later experiment and are not part of the V1.3B start contract.
        self.commands.motion.pose_range = {}
        self.commands.motion.velocity_range = {}
        self.commands.motion.joint_position_range = (0.0, 0.0)
        self.commands.motion.reset_perturbation_probability = 0.0
        self.commands.motion.hard_case_probability = 0.0
        self.commands.motion.hard_case_motion_ids = ()
        self.commands.motion.hard_case_velocity_range = {}
        if (
            self.commands.motion.pose_range
            or self.commands.motion.velocity_range
            or tuple(self.commands.motion.joint_position_range) != (0.0, 0.0)
            or float(self.commands.motion.reset_perturbation_probability) != 0.0
            or float(self.commands.motion.hard_case_probability) != 0.0
        ):
            raise RuntimeError(
                "CompletePriors reset contract failed: private motion reset perturbation is nonzero"
            )
        print(
            "[V1.3B] CompletePriors reset contract: "
            f"pose_range={self.commands.motion.pose_range} "
            f"velocity_range={self.commands.motion.velocity_range} "
            f"joint_position_range={self.commands.motion.joint_position_range} "
            f"reset_perturbation_probability={self.commands.motion.reset_perturbation_probability} "
            f"hard_case_probability={self.commands.motion.hard_case_probability}",
            flush=True,
        )
        # Replacing the dataclass instance above also replaces the assignments
        # made by the inherited A3 flat/tracker setup.  The private stage-A
        # observation group still needs the same canonical anchor and tracked
        # body list as model_3396's original 126-D contract.
        self.commands.motion.anchor_body_name = "torso_Link"
        self.commands.motion.body_names = [
            "torso_Link",
            "right_shoulder_roll_Link",
            "right_elbow_Link",
            "right_wrist_yaw_Link",
        ]
        self.commands.motion.expected_root_quaternion_wxyz = (1.0, 0.0, 0.0, 0.0)
        self.commands.motion.sample_random_start_phase = False
        self.commands.motion.prelude_steps = 50
        self.commands.motion.hold_last_frame_steps = 500
        self.commands.motion.return_to_default_steps = 0
        self.commands.motion.reset_to_default_pose = True
        private_target = self.commands.teacher_racket_target
        private_target.motion_command_name = "motion"
        private_target.target_mode = "manifest"
        private_target.racket_body_name = "__force_wrist_offset_racket_fk__"
        private_target.wrist_body_name = "right_wrist_yaw_Link"
        private_target.racket_fk_mode = "wrist_offset"
        # Random 10-D targets are independent of the private teacher motion
        # from iteration zero.  Keep the motion *name* available solely for
        # the private historical 56-D/126-D teacher observations (their
        # explicit stroke label and READY prelude timing); it is never used to
        # generate the racket target and never enters the public 98-D actor.
        self.commands.racket_target.motion_command_name = "motion"
        self.commands.racket_target.target_mode = "reference_free_global"
        # Early public-goal alignment prevents the frozen private priors from
        # striking a different point/time than the 10-D reward.  It hands off
        # by the lower-prior zero point; motion is never exposed to the actor.
        self.commands.racket_target.motion_alignment_enabled = True
        self.commands.racket_target.motion_alignment_start_progress = 0.0
        self.commands.racket_target.motion_alignment_end_progress = 0.60
        self.commands.racket_target.motion_alignment_include_prelude_s = False
        self.commands.racket_target.motion_alignment_time_range_s = (0.20, 0.60)
        self.commands.racket_target.private_motion_disable_progress = 0.70
        # Preserve the exact historical teacher inputs.  In particular, the
        # frozen priors must never see the public random target: that target
        # is intentionally not tied to their private reference motion.
        for group_name in ("upper", "stage_a"):
            group = getattr(self.observations, group_name)
            for term_name in (
                "racket_target_pos_b",
                "racket_target_vel_b",
                "racket_target_normal_b",
                "time_to_strike",
                "swing_type",
            ):
                term = getattr(group, term_name, None)
                if term is not None:
                    term.params = {**dict(term.params), "command_name": "teacher_racket_target"}
        self.actions.joint_pos.annealed_3396_prior_enabled = True
        self.actions.joint_pos.annealed_3396_prior_checkpoint = (
            "checkpoints/frozen_priors/model_3396.pt"
        )
        self.actions.joint_pos.direct_scale_config_path = (
            "cfg/target_conditioned/direct_action_scale_v13b_annealed_prior.yaml"
        )
        self.actions.joint_pos.annealed_3396_prior_observation_group = "stage_a"
        self.actions.joint_pos.annealed_3396_prior_alpha_start = 1.00
        self.actions.joint_pos.annealed_3396_prior_alpha_zero_progress = 0.70
        self.actions.joint_pos.annealed_900_upper_prior_enabled = True
        self.actions.joint_pos.annealed_900_upper_prior_checkpoint = (
            "checkpoints/frozen_priors/model_900.pt"
        )
        self.actions.joint_pos.annealed_900_upper_prior_observation_group = "upper"
        self.actions.joint_pos.annealed_900_upper_prior_reference_command = "motion"
        self.actions.joint_pos.annealed_900_upper_prior_raw_clip = 0.50
        # Reproduce the reviewed model_900 reference composition while it is
        # active.  These channels are never exposed to the public actor.
        self.actions.joint_pos.joint_reference_lookahead_steps = {
            "right_shoulder_pitch_joint": 12.0,
            "right_shoulder_yaw_joint": 12.0,
        }
        self.actions.joint_pos.joint_velocity_feedforward_mode = "task_phase"
        self.actions.joint_pos.joint_velocity_feedforward_beta = 0.75
        self.actions.joint_pos.joint_velocity_feedforward_joint_names = (
            "right_shoulder_pitch_joint",
            "right_shoulder_yaw_joint",
        )


@configclass
class A3FloatingTargetConditionedReferenceFreeV13BCompletePriorsPrecisionRescueEnvCfg(
    A3FloatingTargetConditionedReferenceFreeV13BAnnealedPriorEnvCfg
):
    """Opt-in Precision Rescue continuation; CompletePriors is not mutated.

    This retains the exact CompletePriors plant, READY/reset contract, public
    98-D actor and 26-D action path.  The only task-local deltas are the two
    broad p/v reward kernels and a continuation scheduler populated after
    checkpoint selection.  Workspace expansion is explicitly forbidden.
    """

    commands: A3PrecisionRescueCommandsCfg = A3PrecisionRescueCommandsCfg()
    rewards: A3PrecisionRescueRewardsCfg = A3PrecisionRescueRewardsCfg()
    precision_rescue_enabled: bool = True
    precision_rescue_source_checkpoint: str = ""
    precision_rescue_source_iteration: int = -1
    precision_rescue_source_progress: float = -1.0
    precision_rescue_source_lower_alpha: float = -1.0
    precision_rescue_source_upper_alpha: float = -1.0
    precision_rescue_hold_updates: int = 300
    precision_rescue_upper_step: float = 0.05
    precision_rescue_schedule_total_updates: int = 50000
    precision_rescue_upper_probe_interval_updates: int = 200
    precision_rescue_upper_probe_max_steps: int = 600
    precision_rescue_upper_probe_consecutive_passes: int = 2
    precision_rescue_upper_probe_min_survival: float = 0.95
    precision_rescue_upper_probe_min_hit_rate: float = 0.95
    precision_rescue_upper_probe_max_position_error_m: float = 0.03
    precision_rescue_upper_probe_max_normal_error_deg: float = 35.0
    precision_rescue_upper_probe_max_velocity_error_mps: float = 1.2
    precision_rescue_upper_probe_seed: int = 20260810
    precision_rescue_upper_gate_file: str = ""
    precision_rescue_upper_gate_run_id: str = ""
    # Explicit rescue-only knob for controlled reward ablations.  The
    # default preserves the original PrecisionRescue contract; exposing it on
    # the structured env config prevents ad-hoc Python edits between tests.
    precision_rescue_wide_position_weight: float = 0.75

    def __post_init__(self):
        super().__post_init__()
        if bool(getattr(self.commands.racket_target, "workspace_expansion_enabled", False)):
            raise RuntimeError("PrecisionRescue must use current CompletePriors local sampler")
        self.rewards.racket_position_wide_recovery.weight = float(
            self.precision_rescue_wide_position_weight
        )
        term = self.actions.joint_pos
        term.precision_rescue_enabled = True
        term.precision_rescue_source_checkpoint = str(self.precision_rescue_source_checkpoint)
        term.precision_rescue_source_progress = float(self.precision_rescue_source_progress)
        term.precision_rescue_source_lower_alpha = float(self.precision_rescue_source_lower_alpha)
        term.precision_rescue_source_upper_alpha = float(self.precision_rescue_source_upper_alpha)
        term.precision_rescue_hold_updates = int(self.precision_rescue_hold_updates)
        term.precision_rescue_upper_step = float(self.precision_rescue_upper_step)
        term.precision_rescue_schedule_total_updates = int(self.precision_rescue_schedule_total_updates)
        if float(self.precision_rescue_source_progress) < 0.0:
            # Static auditing may instantiate this config before checkpoint
            # selection.  A real ManagerBasedRLEnv may not: the ActionTerm
            # will fail closed rather than reset continuation state to zero.
            term.precision_rescue_enabled = False
        # Scheduler construction is deferred until checkpoint selection fills
        # historical fields.  This permits static config/audit work without
        # inventing a source checkpoint or starting a run accidentally.
        self.v13b_precision_rescue_schedule = None
        if float(self.precision_rescue_source_progress) >= 0.0:
            from training.utils.v13b_precision_rescue import PrecisionRescuePriorSchedule

            self.v13b_precision_rescue_schedule = PrecisionRescuePriorSchedule(
                source_progress=float(self.precision_rescue_source_progress),
                source_lower_alpha=float(self.precision_rescue_source_lower_alpha),
                source_upper_alpha=float(self.precision_rescue_source_upper_alpha),
                total_chain_updates=int(self.precision_rescue_schedule_total_updates),
                hold_updates=int(self.precision_rescue_hold_updates),
                upper_step=float(self.precision_rescue_upper_step),
            )
        print(
            "[V1.3B PrecisionRescue] opt-in task configured: "
            f"source={self.precision_rescue_source_checkpoint or '<selection-pending>'} "
            f"workspace_expansion_enabled=false schedule_ready="
            f"{self.v13b_precision_rescue_schedule is not None}",
            flush=True,
        )


@configclass
class A3FloatingTargetConditionedReferenceFreeV13BWorkspaceExpansionEnvCfg(
    A3FloatingTargetConditionedReferenceFreeV13BEnvCfg
):
    """Pure-V1.3B continuation over an audited anchor metadata bank.

    This class intentionally inherits the deployable reference-free plant,
    not the annealed-prior class.  No motion command, model_900 or model_3396
    action path is constructed.
    """

    commands: A3WorkspaceExpansionCommandsCfg = A3WorkspaceExpansionCommandsCfg()
    training_only_annealed_prior: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.commands.motion is not None:
            raise RuntimeError("WorkspaceExpansion must not instantiate a motion command")
        term = self.actions.joint_pos
        if bool(getattr(term, "annealed_3396_prior_enabled", False)):
            raise RuntimeError("WorkspaceExpansion lower prior must be disabled")
        if bool(getattr(term, "annealed_900_upper_prior_enabled", False)):
            raise RuntimeError("WorkspaceExpansion upper prior must be disabled")
        print(
            "[V1.3B WorkspaceExpansion] pure actor contract: "
            "p5u_migration_loaded=false model18900_loaded=false "
            "model900_loaded=false model3396_loaded=false "
            "upper_prior_alpha=0 lower_prior_alpha=0 reference_action_enabled=false",
            flush=True,
        )


@configclass
class A3FloatingUnifiedUpperReferenceTrackerGlobalPhaseEnvCfg(
    A3FloatingUnifiedUpperReferenceTrackerEnvCfg
):
    """P5U Contract B: shared continuous global phase action."""

    actions: A3UnifiedUpperReferenceTrackerGlobalPhaseActionsCfg = (
        A3UnifiedUpperReferenceTrackerGlobalPhaseActionsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.rewards.raw_action_excess.params["action_indices"] = tuple(range(11))
        self.rewards.action_execution_gap.params["action_indices"] = tuple(range(11))


@configclass
class A3FloatingUnifiedUpperReferenceTrackerGroupedPhaseEnvCfg(
    A3FloatingUnifiedUpperReferenceTrackerEnvCfg
):
    """P5U Contract C: global plus shoulder/elbow/wrist phase actions."""

    actions: A3UnifiedUpperReferenceTrackerGroupedPhaseActionsCfg = (
        A3UnifiedUpperReferenceTrackerGroupedPhaseActionsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.rewards.raw_action_excess.params["action_indices"] = tuple(range(14))
        self.rewards.action_execution_gap.params["action_indices"] = tuple(range(14))


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
class A3FloatingJointCoordinatorV5PreviewEnvCfg(A3FloatingJointCoordinatorV2EnvCfg):
    """V2 impact controller with an appended anticipatory dynamics preview."""

    def __post_init__(self):
        super().__post_init__()
        self.observations = A3JointCoordinatorPreviewObservationsCfg()


@configclass
class A3FloatingJointCoordinatorV6MomentumPreviewEnvCfg(A3FloatingJointCoordinatorV2EnvCfg):
    """V19 P0 plant with canonical upper momentum preview."""

    def __post_init__(self):
        super().__post_init__()
        self.observations = A3JointCoordinatorMomentumPreviewObservationsCfg()
        self.commands.motion.require_upper_momentum = True


@configclass
class A3FloatingJointCoordinatorV7StaggeredRecoveryEnvCfg(A3FloatingJointCoordinatorV2EnvCfg):
    """204-D coordinator plant for support-only staggered-stance adaptation.

    The actual stagger geometry is an explicit task-YAML override so scans can
    share this plant without creating multiple nearly identical Gym classes.
    """

    pass


@configclass
class A3FloatingJointCoordinatorV8StaggerSupportEnvCfg(A3FloatingJointCoordinatorV2EnvCfg):
    """Staggered support plant with explicit geometry, load and capture state."""

    def __post_init__(self):
        super().__post_init__()
        self.observations = A3JointCoordinatorStaggerSupportObservationsCfg()


@configclass
class A3FloatingJointCoordinatorV9WideStaggerSupportEnvCfg(
    A3FloatingJointCoordinatorV2EnvCfg
):
    """Wide stagger plant with explicit sagittal and lateral support state."""

    def __post_init__(self):
        super().__post_init__()
        self.observations = A3JointCoordinatorWideStaggerSupportObservationsCfg()


@configclass
class A3FloatingJointCoordinatorV10WideStaggerRecoveryEnvCfg(
    A3FloatingJointCoordinatorV9WideStaggerSupportEnvCfg
):
    """V22 plant with a strictly post-hit recovery-adapter contract."""

    def __post_init__(self):
        super().__post_init__()
        self.observations = A3JointCoordinatorWideStaggerRecoveryObservationsCfg()


@configclass
class A3FloatingJointCoordinatorV11BentReadyRecoveryEnvCfg(
    A3FloatingJointCoordinatorV9WideStaggerSupportEnvCfg
):
    """V28 plant: frozen V25/V27 stack plus a bounded settling adapter."""

    def __post_init__(self):
        super().__post_init__()
        self.observations = A3JointCoordinatorBentReadyRecoveryObservationsCfg()
        # Preserve all V2/V25 reward mutations performed by the inherited
        # __post_init__ chain (notably the enabled fine strike-position term).
        # V28 adds only return/READY objectives; replacing the complete reward
        # config here would silently discard that qualified impact contract.
        bent_ready_terms = A3BentReadyRecoveryRewardsCfg()
        self.rewards.bent_ready_arm_score = bent_ready_terms.bent_ready_arm_score
        self.rewards.bent_ready_progress = bent_ready_terms.bent_ready_progress


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
class A3FixedBaseTargetAdapterObservationsCfg(A3FixedBaseStrikeObservationsCfg):
    """P0 adapter observations plus an anchor-only model_900 group."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        # This is deliberately a standalone 25-D contract, not the generic
        # tracker observation plus an adapter suffix.
        command = None
        motion_anchor_pos_b = None
        motion_anchor_ori_b = None
        base_lin_vel = None
        base_ang_vel = None
        joint_pos = None
        joint_vel = None
        actions = None
        adapter = ObsTerm(func=mdp.target_adapter_observation, params={"command_name": "racket_target"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class AnchorCfg(A3NativeStrikeObservationsCfg.PolicyCfg):
        # The checkpoint was trained with the native 10-D upper action
        # history, while the public P0 adapter action is 7-D.
        actions = ObsTerm(func=mdp.f0_upper_last_action)
        racket_target_pos_b = ObsTerm(
            func=mdp.racket_anchor_target_pos_b, params={"command_name": "racket_target"}
        )
        racket_target_vel_b = ObsTerm(
            func=mdp.racket_anchor_target_vel_b, params={"command_name": "racket_target"}
        )
        racket_target_normal_b = ObsTerm(
            func=mdp.racket_anchor_target_normal_b, params={"command_name": "racket_target"}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    anchor: AnchorCfg = AnchorCfg()


@configclass
class A3FixedBaseTargetAdapterActionsCfg(ActionsCfg):
    """Seven-dimensional right-arm residual around the frozen anchor swing."""

    joint_pos = A3FrozenAnchorArmAdapterPositionActionCfg(
        asset_name="robot",
        base_joint_names=tuple(A3_BASE_ACTION_JOINTS),
        backend_joint_names=tuple(A3_BACKEND_JOINTS),
        strike_joint_names=tuple(A3_STRIKE_V2_REFERENCE_JOINTS),
        upper_joint_names=tuple(A3_NATIVE_STRIKE_JOINTS),
        adapter_joint_names=tuple(A3_RIGHT_ARM_JOINTS),
        action_scale_rad=A3_PD_STAND_BASE_ACTION_SCALE_RAD,
        action_mask=(1.0,) * len(A3_BASE_ACTION_JOINTS),
        raw_clip=1.0,
        upper_raw_clip=0.50,
        scale=dict(AGIBOT_A3_NATIVE_STRIKE_ACTION_SCALE),
        clip_to_soft_joint_limits=True,
        reference_command_name="motion",
        base_reference_mode="default",
        joint_names=tuple(A3_NATIVE_STRIKE_JOINTS),
        preserve_order=True,
        upper_observation_group="anchor",
        # P0 needs centimetre-scale wrist displacement.  Keep the physical
        # envelope at +/- 0.05 rad independently from model_900's clip.
        adapter_scale_rad=(0.10,) * 7,
        adapter_raw_clip=0.5,
        # P2 doubles the P1 centimetre envelope.  The corresponding physical
        # feedforward authority is 0.10 rad, or 0.12 rad for motion 1's weak
        # local-y direction.  The PPO contribution is still scaled by 0.1.
        adapter_raw_clip_by_motion=(1.0, 1.2, 1.0, 1.0, 1.0, 1.0),
        adapter_ramp_in_steps=10,
        adapter_ramp_out_steps=8,
        adapter_policy_residual_gain=0.1,
        # Clip-aware inverse calibrated on the full {-1, 0, +1}^3 motion-0
        # target grid.  This distributes authority across the arm so combined
        # x/y/z requests remain useful after the per-joint +/-0.5 raw bound.
        adapter_feedforward_pinv=(
            (-28.301, -34.989, 17.405),
            (-3.559, 49.620, -1.078),
            (-22.918, -5.274, -32.356),
            (25.083, -7.513, -33.957),
            (36.379, -39.350, 6.404),
            (-2.251, 48.212, -24.392),
            (2.716, -49.938, 1.417),
        ),
        # P1 keeps the same public 7-D residual policy but conditions its
        # analytic local controller on the selected anchor motion.
        adapter_feedforward_pinv_by_motion=(
            (
                (-28.301, -34.989, 17.405),
                (-3.559, 49.620, -1.078),
                (-22.918, -5.274, -32.356),
                (25.083, -7.513, -33.957),
                (36.379, -39.350, 6.404),
                (-2.251, 48.212, -24.392),
                (2.716, -49.938, 1.417),
            ),
            (
                (-11.735, -60.000, -20.667),
                (5.863, 60.000, -16.778),
                (-16.117, 19.600, -24.251),
                (33.519, -32.230, -22.487),
                (4.866, -60.000, 13.138),
                (11.017, 60.000, -21.383),
                (-24.825, -60.000, 8.000),
            ),
            (
                (-8.506, -50.000, -8.837),
                (-14.670, 50.000, 21.155),
                (-33.767, 50.000, -9.707),
                (11.381, 3.591, -32.116),
                (14.411, 45.170, 8.235),
                (-4.815, 50.000, -16.394),
                (-14.469, -50.000, 1.731),
            ),
            (
                (-13.126, -50.000, -11.936),
                (4.522, 50.000, -12.598),
                (-19.650, 50.000, -17.841),
                (29.566, -24.297, -28.357),
                (9.609, 22.421, 9.940),
                (11.703, 50.000, -30.902),
                (-19.752, -50.000, -3.189),
            ),
            (
                (-23.914, 50.000, -7.509),
                (-36.659, 50.000, -6.209),
                (-25.521, -32.171, -15.293),
                (16.411, -37.653, -23.893),
                (13.539, 50.000, 3.348),
                (-10.183, 50.000, -15.141),
                (-8.728, -50.000, 11.612),
            ),
            (
                (-23.346, 50.000, -14.901),
                (-40.328, 50.000, -19.191),
                (-17.277, -30.850, -17.621),
                (25.203, -28.137, -23.681),
                (8.467, 50.000, 9.681),
                (-2.914, 50.000, -20.696),
                (-8.778, -50.000, 10.589),
            ),
        ),
        adapter_feedforward_target_transform=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
    )


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
class A3FixedBaseTargetAdapterEnvCfg(A3FixedBaseReferenceStrikeEnvCfg):
    """P0 fixed-base motion-0 target residual adapter."""

    actions: A3FixedBaseTargetAdapterActionsCfg = A3FixedBaseTargetAdapterActionsCfg()
    observations: A3FixedBaseTargetAdapterObservationsCfg = A3FixedBaseTargetAdapterObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.fix_base = True
        self.commands.motion.manifest_subset_size = 1
        self.commands.racket_target.target_mode = "manifest"
        self.commands.racket_target.manifest_nominal_probability = 1.0
        # Every paired sibling must evaluate the same point on the reference
        # swing.  A per-env strike-time jitter turns a 40-ms phase difference
        # into several centimetres of false "incremental" position error.
        self.commands.racket_target.strike_time_std_s = 0.0
        self.commands.racket_target.adapter_external_offset_half_range = (0.01, 0.01, 0.01)
        self.commands.racket_target.adapter_external_zero_probability = 0.20
        self.commands.racket_target.adapter_external_paired = True
        # P0 is deliberately an *incremental* identification task.  The
        # frozen anchor still carries a roughly 10-cm absolute placement
        # error, so the inherited absolute target reward would teach a fixed
        # bias before it could learn ``external delta -> racket delta``.
        # Preserve velocity/normal softly and keep the motion route as a weak
        # safety prior; the paired displacement objective is the clear winner.
        self.rewards.racket_position.weight = 0.0
        self.rewards.racket_position_y.weight = 0.0
        self.rewards.racket_position_fine.weight = 0.0
        self.rewards.racket_position_y_fine.weight = 0.0
        self.rewards.racket_hit_coupled.weight = 0.0
        self.rewards.racket_velocity.weight = 0.5
        self.rewards.racket_normal.weight = 0.5
        for term_name in (
            "motion_body_pos",
            "motion_body_ori",
            "motion_torso_ori",
            "motion_native_joint_pos",
            "motion_body_lin_vel",
            "motion_body_ang_vel",
        ):
            term = getattr(self.rewards, term_name, None)
            if term is not None:
                term.weight *= 0.20
        self.rewards.racket_incremental_position = RewTerm(
            func=mdp.racket_paired_incremental_position_tracking,
            weight=0.0,
            # At a 1-cm target a 4-cm kernel rates zero response at 0.94,
            # which is indistinguishable from useful gain.  P0 needs the
            # centimetre-scale kernel to make an unchanged racket costly.
            params={"command_name": "racket_target", "std": 0.012},
        )
        self.rewards.racket_incremental_direction = RewTerm(
            func=mdp.racket_paired_incremental_direction_gain,
            weight=0.0,
            params={"command_name": "racket_target"},
        )
        self.rewards.racket_incremental_dense_huber = RewTerm(
            func=mdp.racket_paired_incremental_dense_huber,
            weight=0.5,
            params={"command_name": "racket_target", "scale_m": 0.01},
        )
        self.rewards.racket_incremental_gain = RewTerm(
            func=mdp.racket_paired_incremental_gain_loss,
            weight=0.25,
            params={"command_name": "racket_target"},
        )
        self.rewards.racket_incremental_cross_axis = RewTerm(
            func=mdp.racket_paired_incremental_cross_axis_loss,
            weight=2.0,
            params={"command_name": "racket_target"},
        )
        self.rewards.target_adapter_zero_hold = RewTerm(
            func=mdp.target_adapter_zero_action_hold,
            weight=0.25,
            params={"command_name": "racket_target"},
        )


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


@configclass
class A3FloatingTargetConditionedActionsCfg(ActionsCfg):
    """V30's 22-D coordinator plus an internal target feedforward path."""

    joint_pos = A3TargetConditionedJointCoordinatorActionCfg(
        asset_name="robot",
        base_joint_names=tuple(A3_BASE_ACTION_JOINTS),
        backend_joint_names=tuple(A3_BACKEND_JOINTS),
        strike_joint_names=tuple(A3_STRIKE_V2_REFERENCE_JOINTS),
        upper_joint_names=tuple(A3_NATIVE_STRIKE_JOINTS),
        action_scale_rad=A3_PD_STAND_BASE_ACTION_SCALE_RAD,
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
        adapter_joint_names=tuple(A3_RIGHT_ARM_JOINTS),
    )


@configclass
class A3FloatingTargetConditionedCoordinatorEnvCfg(A3FloatingJointCoordinatorV2EnvCfg):
    """P3 floating-base replay with the fixed-base adapter held feedforward-only."""

    actions: A3FloatingTargetConditionedActionsCfg = A3FloatingTargetConditionedActionsCfg()
    observations: A3TargetConditionedCoordinatorObservationsCfg = (
        A3TargetConditionedCoordinatorObservationsCfg()
    )
    rewards: A3JointCoordinatorRewardsCfg = A3JointCoordinatorRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # P3 is a paired system-identification task.  Independent observation
        # corruption across siblings would appear as false Cartesian
        # cross-axis response, so keep the frozen actors and coordinator's
        # private upper copy deterministic here.
        for group_name in ("upper", "stage_a", "coordinator_upper"):
            getattr(self.observations, group_name).enable_corruption = False
        self.commands.motion.manifest_subset_size = 6
        self.commands.racket_target.target_mode = "manifest"
        self.commands.racket_target.manifest_nominal_probability = 1.0
        self.commands.racket_target.strike_time_std_s = 0.0
        self.commands.racket_target.adapter_external_offset_half_range = (0.02, 0.02, 0.02)
        self.commands.racket_target.adapter_external_zero_probability = 0.0
        self.commands.racket_target.adapter_external_paired = False
        self.commands.racket_target.external_delta_receipt_frame = True

        # Keep the incremental terms available for the focused floating-base
        # continuation below.  They are inactive when paired sampling is off
        # (the replay default), so this does not alter the replay contract.
        self.rewards.racket_incremental_dense_huber = RewTerm(
            func=mdp.racket_paired_incremental_dense_huber,
            weight=0.5,
            params={"command_name": "racket_target", "scale_m": 0.01},
        )
        self.rewards.racket_incremental_gain = RewTerm(
            func=mdp.racket_paired_incremental_gain_loss,
            weight=0.25,
            params={"command_name": "racket_target"},
        )
        self.rewards.racket_incremental_cross_axis = RewTerm(
            func=mdp.racket_paired_incremental_cross_axis_loss,
            weight=2.0,
            params={"command_name": "racket_target"},
        )
        self.rewards.target_adapter_zero_hold = RewTerm(
            func=mdp.target_adapter_zero_action_hold,
            weight=0.25,
            params={"command_name": "racket_target"},
        )

        # Reuse the audited fixed-base matrices verbatim.  This is a replay
        # integration test; no floating-base target adapter is trained here.
        source = A3FixedBaseTargetAdapterActionsCfg().joint_pos
        target = self.actions.joint_pos
        target.adapter_scale_rad = source.adapter_scale_rad
        target.adapter_raw_clip = source.adapter_raw_clip
        target.adapter_raw_clip_by_motion = source.adapter_raw_clip_by_motion
        target.adapter_ramp_in_steps = source.adapter_ramp_in_steps
        target.adapter_ramp_out_steps = source.adapter_ramp_out_steps
        target.adapter_feedforward_pinv = source.adapter_feedforward_pinv
        target.adapter_feedforward_pinv_by_motion = source.adapter_feedforward_pinv_by_motion
        target.adapter_feedforward_target_transform = source.adapter_feedforward_target_transform


@configclass
class A3FloatingTargetConditionedRecoveryEnvCfg(
    A3FloatingTargetConditionedCoordinatorEnvCfg
):
    """P3 plant with an observation suffix reserved for post-hit braking."""

    observations: A3TargetConditionedRecoveryObservationsCfg = (
        A3TargetConditionedRecoveryObservationsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        # P4-A measured motion 3's stable native-time (1.56 s) hit centre
        # relative to its manifest anchor.  This is deliberately an input
        # coordinate calibration, rather than a large residual action: the
        # local adapter remains confined to its validated centimetre range.
        # Other motions retain the uncalibrated P3 contract until separately
        # audited.  Delayed hit schedules also remain uncalibrated because
        # P4-A showed their centre and tail stability are time-dependent.
        self.actions.joint_pos.adapter_control_anchor_offset_b_by_motion = (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (-0.0955253, 0.0660839, -0.0309181),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        )
        self.actions.joint_pos.adapter_control_anchor_calibration_hit_steps_by_motion = (
            -1,
            -1,
            -1,
            # Manifest frame, not the externally reported control step.
            # P4-A native schedule: prelude 50 + hit frame 30 + two commit
            # updates = control-step 78 (1.56 s).
            30,
            -1,
            -1,
        )
        self.actions.joint_pos.adapter_control_anchor_calibration_prelude_steps_by_motion = (
            -1,
            -1,
            -1,
            50,
            -1,
            -1,
        )


@configclass
class A3FloatingTargetConditionedRecoveryYCompEnvCfg(
    A3FloatingTargetConditionedRecoveryEnvCfg
):
    """Isolated P4-D candidate: conservative local y/cross-axis pre-compensation."""

    def __post_init__(self):
        super().__post_init__()
        # T = 0.75 I + 0.25 J^{-1}, using P4-C's calibrated native-time
        # motion-3 response.  This is deliberately partial: it improves the
        # weak y gain without turning a 1-cm cube corner into a clipped arm
        # command.  P4-C remains unchanged as the frozen fallback.
        identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        self.actions.joint_pos.adapter_feedforward_target_transform_by_motion = (
            identity,
            identity,
            identity,
            (
                (1.007454, -0.047725, 0.014104),
                (0.172997, 1.176075, -0.004985),
                (-0.027995, 0.010253, 1.006042),
            ),
            identity,
            identity,
        )


@configclass
class A3FloatingTargetConditionedRecoveryMotion0CalibratedEnvCfg(
    A3FloatingTargetConditionedRecoveryYCompEnvCfg
):
    """P5: retain P4-D motion 3 and add motion 0's measured native-time centre."""

    def __post_init__(self):
        super().__post_init__()
        # P5-A, motion 0, frame 30 / READY prelude 50.  This is the measured
        # physical hit centre relative to its manifest anchor; it belongs in
        # the external-coordinate bridge, never in the centimetre adapter.
        self.actions.joint_pos.adapter_control_anchor_offset_b_by_motion = (
            (-0.0343194, 0.0407395, -0.0581275),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (-0.0955253, 0.0660839, -0.0309181),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        )
        self.actions.joint_pos.adapter_control_anchor_calibration_hit_steps_by_motion = (
            30,
            -1,
            -1,
            30,
            -1,
            -1,
        )
        self.actions.joint_pos.adapter_control_anchor_calibration_prelude_steps_by_motion = (
            50,
            -1,
            -1,
            50,
            -1,
            -1,
        )


@configclass
class A3FloatingTargetConditionedRecoveryMotion2CalibratedEnvCfg(
    A3FloatingTargetConditionedRecoveryMotion0CalibratedEnvCfg
):
    """P7: retain P5 contracts and add motion 2's native-time hit centre."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.adapter_control_anchor_offset_b_by_motion = (
            (-0.0343194, 0.0407395, -0.0581275),
            (0.0, 0.0, 0.0),
            (-0.0168705, 0.0431306, -0.0826364),
            (-0.0955253, 0.0660839, -0.0309181),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        )
        self.actions.joint_pos.adapter_control_anchor_calibration_hit_steps_by_motion = (
            30,
            -1,
            30,
            30,
            -1,
            -1,
        )
        self.actions.joint_pos.adapter_control_anchor_calibration_prelude_steps_by_motion = (
            50,
            -1,
            50,
            50,
            -1,
            -1,
        )


@configclass
class A3FloatingTargetConditionedRecoveryMotion4CalibratedEnvCfg(
    A3FloatingTargetConditionedRecoveryMotion2CalibratedEnvCfg
):
    """P8: retain P7 contracts and add motion 4's native-time hit centre."""

    def __post_init__(self):
        super().__post_init__()
        # P8-A, motion 4, frame 30 / READY prelude 50.  The native-time
        # response is well conditioned enough to first preserve its measured
        # coordinates verbatim; any local cross-axis compensation remains a
        # separate, reversible follow-up from this control-anchor contract.
        self.actions.joint_pos.adapter_control_anchor_offset_b_by_motion = (
            (-0.0343194, 0.0407395, -0.0581275),
            (0.0, 0.0, 0.0),
            (-0.0168705, 0.0431306, -0.0826364),
            (-0.0955253, 0.0660839, -0.0309181),
            (-0.0312324, 0.0621604, -0.0230464),
            (0.0, 0.0, 0.0),
        )
        self.actions.joint_pos.adapter_control_anchor_calibration_hit_steps_by_motion = (
            30,
            -1,
            30,
            30,
            30,
            -1,
        )
        self.actions.joint_pos.adapter_control_anchor_calibration_prelude_steps_by_motion = (
            50,
            -1,
            50,
            50,
            50,
            -1,
        )


@configclass
class A3FloatingTargetConditionedRecoveryMotion5CalibratedEnvCfg(
    A3FloatingTargetConditionedRecoveryMotion4CalibratedEnvCfg
):
    """P9: retain P8 contracts and add motion 5's native-time hit centre."""

    def __post_init__(self):
        super().__post_init__()
        # P9-A, motion 5, frame 30 / READY prelude 50.  This is a measured
        # centre offset only.  The y-to-z coupling seen in its first local
        # grid is intentionally not folded into this layer, so an optional
        # compensation can later be evaluated independently of calibration.
        self.actions.joint_pos.adapter_control_anchor_offset_b_by_motion = (
            (-0.0343194, 0.0407395, -0.0581275),
            (0.0, 0.0, 0.0),
            (-0.0168705, 0.0431306, -0.0826364),
            (-0.0955253, 0.0660839, -0.0309181),
            (-0.0312324, 0.0621604, -0.0230464),
            (-0.0744438, 0.0363935, -0.0106046),
        )
        self.actions.joint_pos.adapter_control_anchor_calibration_hit_steps_by_motion = (
            30,
            -1,
            30,
            30,
            30,
            30,
        )
        self.actions.joint_pos.adapter_control_anchor_calibration_prelude_steps_by_motion = (
            50,
            -1,
            50,
            50,
            50,
            50,
        )
        # P10 selection contract: select in the same measured control-centre
        # coordinates consumed by the adapter.  Motion 1 is deliberately
        # unavailable: its native full-tail audit still terminates after hit.
        self.commands.motion.external_control_anchor_offset_b_by_motion = (
            (-0.0343194, 0.0407395, -0.0581275),
            (0.0, 0.0, 0.0),
            (-0.0168705, 0.0431306, -0.0826364),
            (-0.0955253, 0.0660839, -0.0309181),
            (-0.0312324, 0.0621604, -0.0230464),
            (-0.0744438, 0.0363935, -0.0106046),
        )
        self.commands.motion.external_control_anchor_enabled_by_motion = (
            True,
            False,
            True,
            True,
            True,
            True,
        )
        # Every admitted P10 control centre has been exercised only in its
        # local +/-1 cm cube.  This task-level ceiling is intersected with
        # caller policy rather than trusting play.yaml's older P3 +/-2 cm
        # defaults.  The disabled motion retains a zero range for transparent
        # diagnostics, but can never be selected in the first place.
        self.commands.motion.external_control_local_half_range_by_motion = (
            (0.01, 0.01, 0.01),
            (0.0, 0.0, 0.0),
            (0.01, 0.01, 0.01),
            (0.01, 0.01, 0.01),
            (0.01, 0.01, 0.01),
            (0.01, 0.01, 0.01),
        )


@configclass
class A3FloatingTargetConditionedRecoveryMotion1TrainEnvCfg(
    A3FloatingTargetConditionedRecoveryMotion5CalibratedEnvCfg
):
    """P11: isolate motion 1's post-hit recovery without admitting it to P10.

    This is a training-only contract. It fixes every reset to manifest motion
    1 and removes paired external perturbations so PPO can attribute return
    stability to the lower-body residual rather than target-side variation.
    The inherited P10 selector still leaves motion 1 disabled for user-facing
    external target execution.
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.fixed_motion_id = 1
        self.commands.motion.manifest_balance_strokes = False
        self.commands.racket_target.adapter_external_paired = False
        self.commands.racket_target.adapter_external_offset_half_range = (0.0, 0.0, 0.0)
        self.commands.racket_target.adapter_external_zero_probability = 1.0
