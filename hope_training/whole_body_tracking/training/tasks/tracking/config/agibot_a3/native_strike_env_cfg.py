"""A3 native-MOTION strike policy task.

This task keeps the existing Isaac A3 body for training, but changes the RL
contract to match the intended real A3 deployment path:

* native MOTION/MC owns standing, balance, legs, head, and the non-paddle arm;
* the learned policy commands only waist + right arm joint targets;
* manifest/reference motion is used as a strike teacher and motion prior, not
  as a full-body actor command.
"""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import training.tasks.tracking.mdp as mdp
from training.robots.agibot_a3 import (
    A3_NATIVE_STRIKE_JOINTS,
    A3_RIGHT_ARM_JOINTS,
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
class A3NativeStrikeTerminationsCfg(TerminationsCfg):
    pass


@configclass
class A3NativeStrikeEnvCfg(HOPEPingPongAgibotA3EnvCfg):
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
        waist_actuator = self.scene.robot.actuators.get("waist")
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
        arm_actuator = self.scene.robot.actuators.get("arms")
        if arm_actuator is not None:
            arm_actuator.stiffness = _scale_gain_map(arm_actuator.stiffness, 2.0)
            arm_actuator.damping = _scale_gain_map(arm_actuator.damping, 2.0)
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
