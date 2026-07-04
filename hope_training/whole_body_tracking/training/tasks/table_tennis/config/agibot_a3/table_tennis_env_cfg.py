"""Agibot A3 specialization of the table-tennis environment.

Drops the official Agibot A3 ping-pong articulation into the HOPE table-tennis scene, standing on the
P1 side and facing P2, and wires up the A3 per-joint action scale. Everything else (scene, ball
aerodynamics, observations, rewards, events, terminations) is inherited from
:class:`~training.tasks.table_tennis.table_tennis_env_cfg.TableTennisEnvCfg`.
"""

from __future__ import annotations

import copy

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from training.robots.agibot_a3 import AGIBOT_A3_ACTION_SCALE, AGIBOT_A3_CFG
from training.tasks.table_tennis import geometry
from training.tasks.table_tennis import mdp
from training.tasks.table_tennis.table_tennis_env_cfg import TableTennisEnvCfg
from training.tasks.table_tennis.geometry import ServeConfig

# Pelvis height above the floor in the A3 standing keyframe (= AGIBOT_A3_CFG init z).
A3_STAND_PELVIS_HEIGHT: float = float(AGIBOT_A3_CFG.init_state.pos[2])


@configclass
class AgibotA3TableTennisEnvCfg(TableTennisEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Deep-copy so we never mutate the shared global AGIBOT_A3_CFG (its init_state is reused
        # by the WBC tracking configs).
        robot = copy.deepcopy(AGIBOT_A3_CFG)
        robot.prim_path = "{ENV_REGEX_NS}/Robot"
        # Stand at the P1 side, on the floor (HOPE z = -0.76), facing +X toward P2.
        robot.init_state.pos = (
            geometry.P1_STAND_X,
            geometry.P1_STAND_Y,
            geometry.FLOOR_Z + A3_STAND_PELVIS_HEIGHT,
        )
        # Identity orientation = facing +X (toward P2). If the A3 URDF forward axis turns out to be
        # -X, set this to (0.0, 0.0, 0.0, 1.0) (180 deg about Z).
        robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.scene.robot = robot

        # Per-joint action scale (0.25 * effort / stiffness), matching the A3 deploy decoder.
        self.actions.joint_pos.scale = AGIBOT_A3_ACTION_SCALE


@configclass
class HitFixedBaseObservationsCfg:
    """Compact observations for the fixed-base racket-to-ball touch task."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)
        ball_pos_b = ObsTerm(func=mdp.ball_position_b, noise=Unoise(n_min=-0.01, n_max=0.01))
        ball_vel_b = ObsTerm(func=mdp.ball_velocity_b, noise=Unoise(n_min=-0.05, n_max=0.05))
        racket_pos_b = ObsTerm(func=mdp.racket_position_b, noise=Unoise(n_min=-0.01, n_max=0.01))
        racket_vel_b = ObsTerm(func=mdp.racket_velocity_b, noise=Unoise(n_min=-0.05, n_max=0.05))
        racket_normal_b = ObsTerm(func=mdp.racket_normal_b, noise=Unoise(n_min=-0.02, n_max=0.02))
        racket_to_ball_b = ObsTerm(func=mdp.racket_to_ball_b, noise=Unoise(n_min=-0.01, n_max=0.01))
        predicted_hit_pos_b = ObsTerm(func=mdp.predicted_hit_position_b, noise=Unoise(n_min=-0.01, n_max=0.01))
        time_to_hit = ObsTerm(func=mdp.time_to_hit, noise=Unoise(n_min=-0.01, n_max=0.01))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        ball_pos_b = ObsTerm(func=mdp.ball_position_b)
        ball_vel_b = ObsTerm(func=mdp.ball_velocity_b)
        racket_pos_b = ObsTerm(func=mdp.racket_position_b)
        racket_vel_b = ObsTerm(func=mdp.racket_velocity_b)
        racket_normal_b = ObsTerm(func=mdp.racket_normal_b)
        racket_to_ball_b = ObsTerm(func=mdp.racket_to_ball_b)
        predicted_hit_pos_b = ObsTerm(func=mdp.predicted_hit_position_b)
        time_to_hit = ObsTerm(func=mdp.time_to_hit)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class HitFixedBaseRewardsCfg:
    """Rewards for the first learning milestone: touch the incoming ball."""

    alive = RewTerm(func=mdp.is_alive, weight=0.2)
    racket_ball_proximity = RewTerm(func=mdp.racket_ball_proximity_exp, weight=4.0, params={"std": 0.20})
    racket_closing_speed = RewTerm(func=mdp.racket_closing_speed, weight=1.0, params={"max_speed": 6.0})
    touch_ball_forward = RewTerm(
        func=mdp.touch_ball_forward,
        weight=15.0,
        params={"distance_threshold": 0.07, "min_forward_velocity": 0.2},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)


@configclass
class HitFixedBaseTerminationsCfg:
    """Terminate on success, timeout, or when the ball leaves play."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    touch_success = DoneTerm(
        func=mdp.ball_touched_by_racket,
        params={"distance_threshold": 0.07, "min_forward_velocity": 0.2},
    )
    ball_out_of_bounds = DoneTerm(
        func=mdp.ball_out_of_bounds,
        params={"bounds": geometry.OutOfBoundsBox().as_dict(), "asset_cfg": SceneEntityCfg("ball")},
    )


@configclass
class HitFixedBaseTouchRewardsCfg:
    """Curriculum stage 1: learn stable racket-ball contact with the right arm."""

    alive = RewTerm(func=mdp.is_alive, weight=0.03)
    predicted_hit_position_coarse = RewTerm(
        func=mdp.racket_predicted_hit_position_exp,
        weight=2.5,
        params={"std": 0.35, "hit_x": -0.10, "min_height": 0.12, "max_height": 0.40, "max_time": 0.45},
    )
    predicted_hit_position_fine = RewTerm(
        func=mdp.racket_predicted_hit_position_exp,
        weight=2.0,
        params={"std": 0.07, "hit_x": -0.10, "min_height": 0.12, "max_height": 0.40, "max_time": 0.45},
    )
    predicted_hit_lateral = RewTerm(
        func=mdp.racket_predicted_hit_lateral_exp,
        weight=2.5,
        params={
            "std": 0.07,
            "hit_x": -0.10,
            "min_height": 0.12,
            "max_height": 0.40,
            "max_time": 0.45,
            "normal_axis": 1,
            "normal_sign": 1.0,
        },
    )
    predicted_hit_face = RewTerm(
        func=mdp.racket_predicted_hit_face_exp,
        weight=8.0,
        params={
            "lateral_std": 0.055,
            "normal_std": 0.045,
            "target_normal_dist": 0.025,
            "hit_x": -0.10,
            "min_height": 0.12,
            "max_height": 0.40,
            "max_time": 0.45,
            "normal_axis": 1,
            "normal_sign": 1.0,
        },
    )
    racket_ball_plane_alignment = RewTerm(
        func=mdp.racket_ball_plane_alignment_exp,
        weight=0.8,
        params={"std": 0.08, "normal_axis": 1, "normal_sign": 1.0},
    )
    racket_ball_face_contact = RewTerm(
        func=mdp.racket_ball_face_contact_exp,
        weight=3.0,
        params={"lateral_std": 0.06, "normal_std": 0.05, "normal_axis": 1, "normal_sign": 1.0},
    )
    racket_face_alignment = RewTerm(
        func=mdp.racket_face_alignment,
        weight=1.0,
        params={"normal_axis": 1, "normal_sign": 1.0},
    )
    racket_closing_speed = RewTerm(func=mdp.racket_closing_speed, weight=0.8, params={"max_speed": 6.0})
    racket_forward_swing = RewTerm(
        func=mdp.racket_forward_swing,
        weight=1.0,
        params={"max_speed": 4.0, "hit_x": -0.10, "max_time": 0.45},
    )
    racket_timed_forward_swing = RewTerm(
        func=mdp.racket_timed_forward_swing,
        weight=2.0,
        params={"max_speed": 4.0, "hit_x": -0.10, "min_time": 0.02, "max_time": 0.18},
    )
    first_contact = RewTerm(func=mdp.racket_ball_first_contact, weight=60.0)
    contact_forward_bonus = RewTerm(
        func=mdp.contact_ball_forward, weight=20.0, params={"min_forward_velocity": 0.2}
    )
    face_forward_touch = RewTerm(
        func=mdp.racket_face_ball_forward,
        weight=35.0,
        params={
            "min_forward_velocity": 0.2,
            "lateral_threshold": 0.10,
            "normal_threshold": 0.10,
            "normal_axis": 1,
            "normal_sign": 1.0,
        },
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.003)


@configclass
class HitFixedBaseTouchTerminationsCfg:
    """Terminate once the racket physically contacts the ball."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    touch_success = DoneTerm(
        func=mdp.ball_returned_from_racket_face,
        params={
            "min_forward_velocity": 0.2,
            "lateral_threshold": 0.10,
            "normal_threshold": 0.10,
            "normal_axis": 1,
            "normal_sign": 1.0,
        },
    )
    ball_out_of_bounds = DoneTerm(
        func=mdp.ball_out_of_bounds,
        params={"bounds": geometry.OutOfBoundsBox().as_dict(), "asset_cfg": SceneEntityCfg("ball")},
    )


@configclass
class AgibotA3HitFixedBaseEnvCfg(AgibotA3TableTennisEnvCfg):
    """Fixed-base table-tennis task for learning the first ball-touch behavior."""

    observations: HitFixedBaseObservationsCfg = HitFixedBaseObservationsCfg()
    rewards: HitFixedBaseRewardsCfg = HitFixedBaseRewardsCfg()
    terminations: HitFixedBaseTerminationsCfg = HitFixedBaseTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 64
        self.episode_length_s = 3.0
        self.scene.robot.spawn.fix_base = True

        # Fixed-base racket training should isolate the striking arm. Keep the torso/waist, left arm,
        # head, and legs at their reset/default pose; PPO only controls the right arm joints.
        self.actions.joint_pos.joint_names = [
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]
        self.actions.joint_pos.scale = {
            "right_shoulder_pitch_joint": AGIBOT_A3_ACTION_SCALE[".*_shoulder_pitch_joint"],
            "right_shoulder_roll_joint": AGIBOT_A3_ACTION_SCALE[".*_shoulder_roll_joint"],
            "right_shoulder_yaw_joint": AGIBOT_A3_ACTION_SCALE[".*_shoulder_yaw_joint"],
            "right_elbow_joint": AGIBOT_A3_ACTION_SCALE[".*_elbow_joint"],
            "right_wrist_roll_joint": AGIBOT_A3_ACTION_SCALE[".*_wrist_roll_joint"],
            "right_wrist_pitch_joint": AGIBOT_A3_ACTION_SCALE[".*_wrist_pitch_joint"],
            "right_wrist_yaw_joint": AGIBOT_A3_ACTION_SCALE[".*_wrist_yaw_joint"],
        }

        # A narrow, reachable serve distribution for the first learning milestone. The ball still
        # travels from the P2 side toward the P1 robot, but lateral/height variance is intentionally
        # small so early PPO gradients are dominated by reaching and touching rather than search.
        self.events.serve_ball.params["serve_cfg"] = ServeConfig(
            pos_x_range=(1.85, 2.05),
            pos_y_range=(-0.90, -0.62),
            pos_z_range=(0.45, 0.60),
            vel_x_range=(-3.2, -2.4),
            vel_y_range=(-0.15, 0.15),
            vel_z_range=(-0.05, 0.25),
        )


@configclass
class AgibotA3HitFixedBaseTouchEnvCfg(AgibotA3HitFixedBaseEnvCfg):
    """Easier fixed-base curriculum stage: reach the ball before learning to return it."""

    rewards: HitFixedBaseTouchRewardsCfg = HitFixedBaseTouchRewardsCfg()
    terminations: HitFixedBaseTouchTerminationsCfg = HitFixedBaseTouchTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Keep the first curriculum stage focused, but do not make the ball a dead short ball: it must
        # bounce once on the P1 half and continue past the near table edge so "racket near ball" means
        # reaching a realistic returnable ball, not chasing a ball that dies on the table.
        self.events.serve_ball.params["serve_cfg"] = ServeConfig(
            pos_x_range=(2.00, 2.20),
            pos_y_range=(-0.86, -0.66),
            pos_z_range=(0.55, 0.72),
            vel_x_range=(-7.0, -6.2),
            vel_y_range=(-0.06, 0.06),
            vel_z_range=(-0.05, 0.25),
        )
