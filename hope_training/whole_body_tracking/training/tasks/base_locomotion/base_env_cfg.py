"""Minimal deterministic environment definition for A3 Base Stand smoke."""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import training.tasks.base_locomotion.mdp as mdp
from training.robots.agibot_a3 import (
    A3_ANCHOR_BODY,
    A3_BACKEND_JOINTS,
    A3_BASE_ACTION_JOINTS,
    A3_FEET_BODIES,
    A3_POLICY_JOINTS,
)


A3_BASE_ACTION_SCALE_RAD = (
    0.6875,
    0.4583333333333333,
    0.6875,
    0.32,
    0.591,
    0.27375,
    0.6875,
    0.4583333333333333,
    0.6875,
    0.32,
    0.591,
    0.27375,
    0.12,
    0.14,
)
A3_NOMINAL_BODY_HEIGHT_M = 1.0684

# In Stand v0, only the two feet may contact the ground.  Case is exact and
# follows the prepared A3 URDF; do not normalize link names.
A3_NON_FOOT_CONTACT_REGEX = (
    r"^(?!left_ankle_roll_Link$)(?!right_ankle_roll_Link$).+$"
)


@configclass
class A3BaseStandSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=None,
    )
    robot: ArticulationCfg = MISSING
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=False,
        force_threshold=1.0,
        debug_vis=False,
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


@configclass
class CommandsCfg:
    """Stand v0 intentionally has no sampled command term."""

    pass


@configclass
class ActionsCfg:
    base = mdp.A3BaseCompositePositionActionCfg(
        asset_name="robot",
        base_joint_names=tuple(A3_BASE_ACTION_JOINTS),
        backend_joint_names=tuple(A3_BACKEND_JOINTS),
        action_scale_rad=A3_BASE_ACTION_SCALE_RAD,
        action_mask=(1.0,) * len(A3_BASE_ACTION_JOINTS),
        raw_clip=0.25,
        clip_to_soft_joint_limits=True,
    )


_OBS_PARAMS = {
    "policy_joint_names": tuple(A3_POLICY_JOINTS),
    "torso_body_name": A3_ANCHOR_BODY,
    "nominal_body_height_m": A3_NOMINAL_BODY_HEIGHT_M,
}


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actor = ObsTerm(func=mdp.A3BaseActorObservation, params=_OBS_PARAMS)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        critic = ObsTerm(
            func=mdp.A3BaseCriticObservation,
            params={
                **_OBS_PARAMS,
                "contact_sensor_name": "contact_forces",
                "foot_body_names": tuple(A3_FEET_BODIES),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class RewardsCfg:
    # Small first-pass shaping set.  No gait, strike, command, contact-timing,
    # randomization-robustness, or task-success rewards are present.
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    pelvis_upright = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    torso_upright = RewTerm(
        func=mdp.torso_upright_l2,
        weight=-1.0,
        params={"torso_body_name": A3_ANCHOR_BODY},
    )
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-4.0,
        params={"target_height": A3_NOMINAL_BODY_HEIGHT_M},
    )
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_posture = RewTerm(
        func=mdp.joint_posture_l2,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(A3_BASE_ACTION_JOINTS))},
    )
    joint_velocity = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(A3_BASE_ACTION_JOINTS))},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[A3_NON_FOOT_CONTACT_REGEX]
            ),
            "threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    torso_tilt = DoneTerm(
        func=mdp.torso_tilt_exceeded,
        params={"torso_body_name": A3_ANCHOR_BODY, "max_tilt_rad": 0.8},
    )
    base_height = DoneTerm(
        func=mdp.RootHeightBelowMinimum,
        params={"minimum_height": 0.75},
    )
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[A3_NON_FOOT_CONTACT_REGEX]
            ),
            "threshold": 1.0,
        },
    )
    joint_limit = DoneTerm(
        func=mdp.HardJointPositionLimitExceeded,
        params={"tolerance_rad": 1.0e-4},
    )
    nonfinite_state = DoneTerm(func=mdp.nonfinite_robot_state)


@configclass
class CurriculumCfg:
    """No curriculum is allowed during deterministic Stand smoke."""

    pass


@configclass
class A3BaseStandEnvCfg(ManagerBasedRLEnvCfg):
    scene: A3BaseStandSceneCfg = A3BaseStandSceneCfg(num_envs=64, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.viewer.eye = (1.8, 1.8, 1.4)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
