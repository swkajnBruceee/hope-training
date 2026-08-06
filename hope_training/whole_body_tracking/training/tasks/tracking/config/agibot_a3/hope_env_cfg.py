"""Agibot A3 — HOPE ping-pong WBC (BeyondMimic + HITTER racket-target tracking).

This environment extends the A3 motion-tracking baseline
(:class:`AgibotA3FlatEnvCfg`) with the HITTER racket objective:

* a :class:`RacketTargetCommand` that samples the desired racket state (position/velocity/normal)
  and desired base XY each swing, and computes the actual racket state by FK through ``T_mount``;
* HOPE actor observations (desired racket pos/vel/normal in base frame,
  time-to-strike, desired base XY rel-base) plus projected gravity, with privileged actual racket
  state on the critic;
* HITTER goal rewards (base-position before strike; racket pos/vel/normal in a window around strike),
  on top of the BeyondMimic imitation reward and the regularization reward;
* extended domain randomization for sim-to-real.

Default usage trains one swing style per policy (forehand or backhand), selected by the local
``motion_file`` or optional registry clip. The swing-type observation is therefore omitted from the
actor by default; enable ``swing_type`` on the policy group only if you train a single unified policy.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import training.tasks.tracking.mdp as mdp
from training.tasks.tracking.config.agibot_a3.flat_env_cfg import AgibotA3FlatEnvCfg
from training.tasks.tracking.tracking_env_cfg import (
    CommandsCfg,
    EventCfg,
    ObservationsCfg,
    RewardsCfg,
)

##
# Commands: motion (imitation) + racket target.
##


@configclass
class HOPECommandsCfg(CommandsCfg):
    racket_target = mdp.RacketTargetCommandCfg(
        asset_name="robot",
        motion_command_name="motion",
        debug_vis=False,
        # Paddle face normal = racket-local +Y (blade is thin along Y; +Y is the red/hitting face).
        # The std-pingpong URDF blade is thin along local Y. sign=+1 -> red (forehand)
        # face; use -1 for the black face if you train a backhand-only policy.
        # NOTE: cfg/task/HOPEPingPong.yaml also sets mount_normal_axis and (via train.py) overrides
        # this for the Hydra path — keep the two in sync.
        mount_normal_axis=1,
        mount_normal_sign=1.0,
    )


##
# Observations: HITTER actor (desired targets only) + privileged critic (actual racket state).
##


@configclass
class HOPEObservationsCfg(ObservationsCfg):
    @configclass
    class HOPEPolicyCfg(ObservationsCfg.PolicyCfg):
        # Deployable WBC actor: proprioception + strike command only.
        # Remove inherited reference-motion actor observations from ObservationsCfg.PolicyCfg.
        command = None
        motion_anchor_pos_b = None
        motion_anchor_ori_b = None
        base_lin_vel = None

        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_target_pos_b = ObsTerm(func=mdp.base_target_pos_b, params={"command_name": "racket_target"})
        racket_target_pos_b = ObsTerm(
            func=mdp.racket_target_pos_b,
            params={"command_name": "racket_target"},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        racket_target_vel_b = ObsTerm(func=mdp.racket_target_vel_b, params={"command_name": "racket_target"})
        racket_target_normal_b = ObsTerm(func=mdp.racket_target_normal_b, params={"command_name": "racket_target"})
        time_to_strike = ObsTerm(func=mdp.time_to_strike, params={"command_name": "racket_target"})
        swing_type = ObsTerm(func=mdp.swing_type, params={"command_name": "racket_target"})

    @configclass
    class HOPECriticCfg(ObservationsCfg.PrivilegedCfg):
        base_target_pos_b = ObsTerm(func=mdp.base_target_pos_b, params={"command_name": "racket_target"})
        racket_target_pos_b = ObsTerm(func=mdp.racket_target_pos_b, params={"command_name": "racket_target"})
        racket_target_vel_b = ObsTerm(func=mdp.racket_target_vel_b, params={"command_name": "racket_target"})
        racket_target_normal_b = ObsTerm(func=mdp.racket_target_normal_b, params={"command_name": "racket_target"})
        racket_target_vel_w = ObsTerm(func=mdp.racket_target_vel_w, params={"command_name": "racket_target"})
        racket_target_normal_w = ObsTerm(func=mdp.racket_target_normal_w, params={"command_name": "racket_target"})
        time_to_strike = ObsTerm(func=mdp.time_to_strike, params={"command_name": "racket_target"})
        # actual racket state (FK) — privileged, never available on hardware
        racket_pos_b = ObsTerm(func=mdp.racket_pos_b, params={"command_name": "racket_target"})
        racket_lin_vel_w = ObsTerm(func=mdp.racket_lin_vel_w, params={"command_name": "racket_target"})
        racket_normal_w = ObsTerm(func=mdp.racket_normal_w, params={"command_name": "racket_target"})
        episode_time_left = ObsTerm(func=mdp.episode_time_left)

    policy: HOPEPolicyCfg = HOPEPolicyCfg()
    critic: HOPECriticCfg = HOPECriticCfg()


##
# Rewards: imitation (inherited) + goal (racket/base) + regularization.
# Weights are HOPE tuning choices (HITTER does not publish reward weights/kernels).
##


@configclass
class HOPERewardsCfg(RewardsCfg):
    # r_goal: racket state tracking, active only in the strike window. Starter std values
    # are set near acceptance tolerances; tune for your motions.
    racket_position = RewTerm(
        func=mdp.racket_position_tracking_exp,
        weight=4.0,
        params={"command_name": "racket_target", "std": 0.075},  # target < 7.5 cm
    )
    racket_velocity = RewTerm(
        func=mdp.racket_velocity_tracking_exp,
        weight=2.0,
        params={"command_name": "racket_target", "std": 0.5},  # target < 0.5 m/s
    )
    racket_normal = RewTerm(
        func=mdp.racket_normal_tracking_exp,
        weight=2.0,
        params={"command_name": "racket_target", "std": 0.262},  # radians, target < 15 deg
    )
    # r_goal — base repositioning, active only before the strike.
    base_position = RewTerm(
        func=mdp.base_position_tracking_exp,
        weight=1.0,
        params={"command_name": "racket_target", "std": 0.3},
    )
    # r_regularization — energy / torque smoothness (action_rate_l2 already inherited).
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)


##
# Domain randomization (HITTER + standard sim-to-real reconstruction).
#
# HITTER publishes no DR table; it states PD gains are FIXED. The mass/friction/push/observation-noise
# terms below are standard BeyondMimic practice; PD-gain and motor-strength randomization are added for
# sim-to-real robustness and can be disabled to match HITTER exactly.
#
# Already provided by the base EventCfg: friction (physics_material, startup), CoM (startup),
# joint default pos (startup), external pushes (push_robot, interval). Observation noise comes from the
# per-term Unoise + enable_corruption on the policy observation group.
##


@configclass
class HOPEEventCfg(EventCfg):
    # link mass randomization (±15%)
    randomize_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.85, 1.15),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    # PD gain / motor strength randomization (±20%). NOTE: HITTER keeps PD fixed; this is a
    # sim-to-real robustness choice. Set to None to disable.
    randomize_pd_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )


##
# Environment configuration.
##


@configclass
class HOPEPingPongAgibotA3EnvCfg(AgibotA3FlatEnvCfg):
    commands: HOPECommandsCfg = HOPECommandsCfg()
    observations: HOPEObservationsCfg = HOPEObservationsCfg()
    rewards: HOPERewardsCfg = HOPERewardsCfg()
    events: HOPEEventCfg = HOPEEventCfg()

    def __post_init__(self):
        # AgibotA3FlatEnvCfg sets the robot, action scale, motion anchor/body names, and the A3
        # contact/termination/CoM body names (all valid for the inherited HOPE* cfg subclasses).
        super().__post_init__()
