"""Isolated H1-flat-style zero-command stand task for the A3 embodiment."""

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

import training.tasks.replica_locomotion.mdp as mdp
from training.robots.agibot_a3 import A3_ANCHOR_BODY, A3_FEET_BODIES, A3_LEFT_LEG_JOINTS, A3_RIGHT_LEG_JOINTS


A3_REPLICA_LEG_JOINTS = tuple(A3_LEFT_LEG_JOINTS + A3_RIGHT_LEG_JOINTS)
# Same physical residual scales as the current Base task, limited to its leg subset.
A3_REPLICA_ACTION_SCALE_RAD = (
    0.6875, 0.4583333333333333, 0.6875, 0.32, 0.591, 0.27375,
    0.6875, 0.4583333333333333, 0.6875, 0.32, 0.591, 0.27375,
)


@configclass
class A3ReplicaSceneCfg(InteractiveSceneCfg):
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
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=1.0
    )
    light = AssetBaseCfg(
        prim_path="/World/light", spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0)
    )


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=1.0,
        rel_heading_envs=0.0,
        heading_command=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.A3ReplicaLegPositionActionCfg(
        asset_name="robot", joint_names=A3_REPLICA_LEG_JOINTS, action_scale_rad=A3_REPLICA_ACTION_SCALE_RAD
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(A3_REPLICA_LEG_JOINTS))}
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(A3_REPLICA_LEG_JOINTS))}
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class RewardsCfg:
    # These weights follow Isaac Lab's H1 flat velocity task.  Zero commands
    # turn the tracking terms into a standing-velocity objective.
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(A3_REPLICA_LEG_JOINTS))},
    )
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.25e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(A3_REPLICA_LEG_JOINTS))},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(A3_FEET_BODIES)),
            "threshold": 0.6,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(A3_FEET_BODIES)),
            "asset_cfg": SceneEntityCfg("robot", body_names=list(A3_FEET_BODIES)),
        },
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    ankle_joint_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    torso_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[A3_ANCHOR_BODY]), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    pass


@configclass
class A3BaseStandReplicaEnvCfg(ManagerBasedRLEnvCfg):
    scene: A3ReplicaSceneCfg = A3ReplicaSceneCfg(num_envs=256, env_spacing=2.5)
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
        self.scene.contact_forces.update_period = self.sim.dt
        self.viewer.eye = (1.8, 1.8, 1.4)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
