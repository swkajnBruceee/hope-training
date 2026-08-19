from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

import numpy as np


# =====================================================================
# Paths
# =====================================================================

THIS_FILE = Path(__file__).resolve()

WORKTREE = THIS_FILE.parents[3]

WBT_ROOT = (
    WORKTREE
    / "hope_training"
    / "whole_body_tracking"
)

DEPLOY_ROOT = (
    WORKTREE.parent
    / "hope-deploy-baseline"
    / "a3_deploy"
    / "a3_deploy_example"
)

DEPLOY_REFERENCE = (
    DEPLOY_ROOT
    / "reference"
)

MODEL_PATH = (
    DEPLOY_ROOT
    / "models"
    / "model_21800"
    / "policy"
    / "exported"
    / "policy.onnx"
)

DEPLOY_YAML = (
    DEPLOY_ROOT
    / "models"
    / "model_21800"
    / "policy"
    / "params"
    / "deploy.yaml"
)

RUNTIME_YAML = (
    DEPLOY_ROOT
    / "config"
    / "hope_pingpong_runtime.yaml"
)


sys.path.insert(
    0,
    str(WBT_ROOT),
)

sys.path.insert(
    0,
    str(DEPLOY_REFERENCE),
)


# Stage 5B2B local ONNXRuntime dependency.
#
# Isaac Sim's bundled Python does not contain
# onnxruntime.  Earlier Stage-5 benchmarks use
# this worktree-local isolated dependency tree.
LOCAL_DEPS = (
    WORKTREE
    / ".local_deps"
)

if LOCAL_DEPS.exists():
    sys.path.insert(
        0,
        str(LOCAL_DEPS),
    )


# =====================================================================
# Isaac application
# =====================================================================

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description=(
        "Stage 5B2B: persistent multi-env "
        "model_21800 executor smoke."
    )
)

parser.add_argument(
    "--num_envs",
    type=int,
    default=16,
)

parser.add_argument(
    "--duration",
    type=float,
    default=3.2,
)

AppLauncher.add_app_launcher_args(
    parser
)

args_cli = parser.parse_args()

app_launcher = AppLauncher(
    args_cli
)

simulation_app = app_launcher.app


# =====================================================================
# Imports requiring Isaac application
# =====================================================================

import torch
import yaml

import isaaclab.sim as sim_utils

from isaaclab.scene import InteractiveScene
from isaaclab.utils.math import quat_apply, quat_rotate_inverse


from training.tasks.table_tennis.config.agibot_a3.table_tennis_env_cfg import (
    AgibotA3TableTennisEnvCfg,
)

from training.tasks.table_tennis import geometry
from training.tasks.table_tennis.ball import compute_aero_wrench

from training.robots.agibot_a3 import (
    A3_MOUNT_OFFSET,
    A3_WRIST_BODY,
    AGIBOT_A3_JOINT_NAMES,
)


from a3_deploy_onnx_ref_pingpong.action_adapter import (
    ActionAdapter,
)

from a3_deploy_onnx_ref_pingpong.lifecycle import (
    LifecycleConfig,
    SwingLifecycle,
)

from a3_deploy_onnx_ref_pingpong.observation import (
    RobotState,
    build_observation,
)

from a3_deploy_onnx_ref_pingpong.onnx_policy import (
    OnnxPolicy,
)

from a3_deploy_onnx_ref_pingpong.racket_command import (
    RacketCommand,
)


# =====================================================================
# Frozen integration contract
# =====================================================================

PHYSICS_DT = 1.0 / 400.0

DECIMATION = 8

CONTROL_DT = (
    PHYSICS_DT
    * DECIMATION
)

LEAD_TIME = 1.20


# Stage-4C winning action.
RACKET_TARGET_DY = 0.05

RACKET_TARGET_VEL_W = np.asarray(
    [
        3.44068113587798,
        -0.6659088846041841,
        1.4249536866108532,
    ],
    dtype=np.float64,
)


# Stage-4 calibrated physical intercept
# for:
#
# ball_y  = -0.5375
# ball_z  =  0.43
# ball_vx = -6.0
#
# Stage 5B2B does NOT launch that ball.
# This is only the known command target used to
# exercise model_21800.
CALIBRATED_HIT_POS_L = np.asarray(
    [
        -0.10,
        -0.5375000238418579,
        0.2721405533256116,
    ],
    dtype=np.float64,
)


COMMAND_TARGET_POS_L = (
    CALIBRATED_HIT_POS_L.copy()
)

COMMAND_TARGET_POS_L[1] += (
    RACKET_TARGET_DY
)


BALL_PARK_POS_L = np.asarray(
    [
        2.22,
        -0.5375,
        0.43,
    ],
    dtype=np.float64,
)


# =====================================================================
# Stage 5B2C physical incoming-ball contract.
#
# Same controlled Stage-4B/4C failure case.
# =====================================================================

BALL_INIT_POS_L = np.asarray(
    [
        2.22,
        -0.5375,
        0.43,
    ],
    dtype=np.float64,
)

BALL_INIT_VEL_W = np.asarray(
    [
        -6.0,
        0.0,
        0.08,
    ],
    dtype=np.float64,
)

BALL_INIT_SPIN_W = np.zeros(
    3,
    dtype=np.float64,
)

# Stage-4 physical trajectory calibration for this x/z/vx class.
# Lateral y shift does not change the x-flight dynamics because vy=0
# and the currently configured Magnus term is zero.
CALIBRATED_CROSS_TIME_S = 0.4591565

LAUNCH_DELAY_S = max(
    0.0,
    LEAD_TIME
    - CALIBRATED_CROSS_TIME_S,
)

CONTACT_FORCE_THRESHOLD = 0.05
FACE_LATERAL_THRESHOLD = 0.10
FACE_NORMAL_THRESHOLD = 0.10

RACKET_NORMAL_AXIS = 1
RACKET_NORMAL_SIGN = 1.0

HIGH_MARGIN_X_LO = 1.90
HIGH_MARGIN_X_HI = 2.20
HIGH_MARGIN_Y_LO = -0.90
HIGH_MARGIN_Y_HI = -0.60


JOINT_NAMES = tuple(
    AGIBOT_A3_JOINT_NAMES
)

if len(JOINT_NAMES) != 31:
    raise RuntimeError(
        "Expected 31 canonical A3 joints"
    )


# =====================================================================
# Helpers
# =====================================================================

def percentile(values, q):

    values = sorted(values)

    index = int(
        round(
            (len(values) - 1)
            * q
        )
    )

    return values[index]


def robot_state_batch_to_numpy(
    robot,
    joint_ids,
):

    # One logical GPU -> CPU copy for all
    # executor state needed by all environments.
    packed = torch.cat(
        [
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            robot.data.root_ang_vel_b,
            robot.data.joint_pos[
                :,
                joint_ids,
            ],
            robot.data.joint_vel[
                :,
                joint_ids,
            ],
        ],
        dim=-1,
    )

    return (
        packed
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float64,
            copy=True,
        )
    )


def unpack_robot_state(
    packed,
    env_id,
):

    x = packed[env_id]

    return RobotState(
        base_pos_w=x[0:3],
        base_quat_w=x[3:7],
        base_ang_vel_b=x[7:10],
        q=x[10:41],
        qd=x[41:72],
    )


def park_all_balls(
    ball,
    env_origins,
):

    n = int(
        env_origins.shape[0]
    )

    device = ball.device

    pose = torch.zeros(
        (n, 7),
        device=device,
        dtype=torch.float32,
    )

    local = torch.as_tensor(
        BALL_PARK_POS_L,
        device=device,
        dtype=torch.float32,
    )

    pose[:, :3] = (
        env_origins
        + local.unsqueeze(0)
    )

    # quaternion (w,x,y,z)
    pose[:, 3] = 1.0

    velocity = torch.zeros(
        (n, 6),
        device=device,
        dtype=torch.float32,
    )

    ball.write_root_pose_to_sim(
        pose
    )

    ball.write_root_velocity_to_sim(
        velocity
    )



def set_all_ball_state(
    ball,
    env_origins,
    pos_l,
    lin_vel_w,
    ang_vel_w,
):

    n = int(
        env_origins.shape[0]
    )

    device = ball.device

    pose = torch.zeros(
        (n, 7),
        device=device,
        dtype=torch.float32,
    )

    pose[:, :3] = (
        env_origins
        + torch.as_tensor(
            pos_l,
            device=device,
            dtype=torch.float32,
        ).unsqueeze(0)
    )

    pose[:, 3] = 1.0

    velocity = torch.zeros(
        (n, 6),
        device=device,
        dtype=torch.float32,
    )

    velocity[:, :3] = torch.as_tensor(
        lin_vel_w,
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)

    velocity[:, 3:] = torch.as_tensor(
        ang_vel_w,
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)

    ball.write_root_pose_to_sim(
        pose
    )

    ball.write_root_velocity_to_sim(
        velocity
    )


def racket_state_batch(
    robot,
    wrist_body_id,
    mount_offset,
):

    wpos = robot.data.body_pos_w[
        :,
        wrist_body_id,
    ]

    wquat = robot.data.body_quat_w[
        :,
        wrist_body_id,
    ]

    wlin = robot.data.body_lin_vel_w[
        :,
        wrist_body_id,
    ]

    wang = robot.data.body_ang_vel_w[
        :,
        wrist_body_id,
    ]

    offset_b = (
        mount_offset
        .unsqueeze(0)
        .expand(
            wpos.shape[0],
            -1,
        )
    )

    offset_w = quat_apply(
        wquat,
        offset_b,
    )

    racket_pos_w = (
        wpos
        + offset_w
    )

    racket_vel_w = (
        wlin
        + torch.cross(
            wang,
            offset_w,
            dim=-1,
        )
    )

    normal_b = torch.zeros_like(
        racket_pos_w
    )

    normal_b[
        :,
        RACKET_NORMAL_AXIS,
    ] = RACKET_NORMAL_SIGN

    racket_normal_w = quat_apply(
        wquat,
        normal_b,
    )

    return (
        racket_pos_w,
        racket_vel_w,
        racket_normal_w,
    )


def phase_name(
    lifecycle,
):

    phase = lifecycle.phase

    return str(
        getattr(
            phase,
            "value",
            phase,
        )
    )


# =====================================================================
# Main
# =====================================================================

def main():

    n_envs = int(
        args_cli.num_envs
    )

    if n_envs <= 0:
        raise RuntimeError(
            "num_envs must be positive"
        )


    print(
        "=================================================="
    )
    print(
        "MODEL_21800 SINGLE-ENV STAGE-4 EXECUTION PARITY"
    )
    print(
        "=================================================="
    )

    print(
        "num_envs =",
        n_envs,
    )

    print(
        "physics_hz =",
        1.0 / PHYSICS_DT,
    )

    print(
        "executor_hz =",
        1.0 / CONTROL_DT,
    )

    print(
        "decimation =",
        DECIMATION,
    )


    # -------------------------------------------------------------
    # Verify resources.
    # -------------------------------------------------------------

    for path in (
        MODEL_PATH,
        DEPLOY_YAML,
        RUNTIME_YAML,
    ):

        if not path.exists():

            raise FileNotFoundError(
                path
            )


    with open(
        RUNTIME_YAML,
        "r",
        encoding="utf-8",
    ) as f:

        runtime = yaml.safe_load(
            f
        )


    runtime_hz = float(
        runtime["control_hz"]
    )

    if not math.isclose(
        runtime_hz,
        1.0 / CONTROL_DT,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):

        raise RuntimeError(
            "model_21800 control-rate mismatch"
        )


    lifecycle_cfg = LifecycleConfig(
        dt=CONTROL_DT,
        follow_through_s=float(
            runtime[
                "lifecycle"
            ][
                "follow_through_s"
            ]
        ),
        recovery_s=float(
            runtime[
                "lifecycle"
            ][
                "recovery_s"
            ]
        ),
        ready_time_to_strike=float(
            runtime[
                "lifecycle"
            ][
                "ready_time_to_strike"
            ]
        ),
        ready_reach_x=float(
            runtime[
                "lifecycle"
            ][
                "ready_reach_x"
            ]
        ),
        ready_reach_y=float(
            runtime[
                "lifecycle"
            ][
                "ready_reach_y"
            ]
        ),
        ready_reach_z=float(
            runtime[
                "lifecycle"
            ][
                "ready_reach_z"
            ]
        ),
    )


    # IMPORTANT:
    # one mutable lifecycle object per environment.
    lifecycles = [
        SwingLifecycle(
            lifecycle_cfg
        )
        for _ in range(
            n_envs
        )
    ]


    if (
        len(
            {
                id(x)
                for x in lifecycles
            }
        )
        != n_envs
    ):

        raise RuntimeError(
            "Lifecycle instances are not independent"
        )


    print(
        "INDEPENDENT_LIFECYCLES=PASS"
    )


    # One shared ORT session.
    policy = OnnxPolicy(
        MODEL_PATH,
        providers=[
            "CPUExecutionProvider",
        ],
    )


    adapter = (
        ActionAdapter.from_yaml(
            DEPLOY_YAML
        )
    )


    # -------------------------------------------------------------
    # Scene.
    # -------------------------------------------------------------

    env_cfg = (
        AgibotA3TableTennisEnvCfg()
    )

    env_cfg.scene.num_envs = (
        n_envs
    )

    env_cfg.scene.env_spacing = (
        4.0
    )

    # Whole-body floating robot.
    env_cfg.scene.robot.spawn.fix_base = (
        False
    )

    env_cfg.sim.dt = (
        PHYSICS_DT
    )

    env_cfg.sim.render_interval = 1

    env_cfg.sim.device = (
        args_cli.device
    )

    env_cfg.sim.physx.enable_ccd = (
        True
    )


    sim = sim_utils.SimulationContext(
        env_cfg.sim
    )


    scene = InteractiveScene(
        env_cfg.scene
    )


    sim.reset()

    scene.reset()


    robot = scene["robot"]

    ball = scene["ball"]

    sensor = scene.sensors[
        "racket_ball_contact"
    ]


    # -------------------------------------------------------------
    # Exact official HOPE ball aerodynamics at every 400-Hz
    # physics substep.
    # -------------------------------------------------------------

    aero_cfg = env_cfg.ball_aerodynamics

    if not aero_cfg.enabled:
        raise RuntimeError(
            "Stage5B2C requires HOPE aerodynamics enabled"
        )


    aero_force_b = torch.zeros(
        (
            n_envs,
            1,
            3,
        ),
        device=ball.device,
        dtype=torch.float32,
    )

    aero_torque_b = torch.zeros_like(
        aero_force_b
    )


    def apply_ball_aerodynamics(
        dt: float,
    ):

        lin_vel_w = (
            ball.data.root_lin_vel_w
        )

        ang_vel_w = (
            ball.data.root_ang_vel_w
        )

        force_w, torque_w = (
            compute_aero_wrench(
                lin_vel_w,
                ang_vel_w,
                float(
                    geometry.BALL_MASS
                ),
                aero_cfg,
            )
        )

        quat_w = (
            ball.data.root_quat_w
        )

        aero_force_b[
            :,
            0,
            :,
        ] = quat_rotate_inverse(
            quat_w,
            force_w,
        )

        aero_torque_b[
            :,
            0,
            :,
        ] = quat_rotate_inverse(
            quat_w,
            torque_w,
        )

        ball.set_external_force_and_torque(
            aero_force_b,
            aero_torque_b,
        )

        ball.write_data_to_sim()


    sim.add_physics_callback(
        "hope_stage5b2c_ball_aerodynamics",
        apply_ball_aerodynamics,
    )

    print(
        "OFFICIAL_HOPE_AERODYNAMICS=PASS"
    )


    wrist_ids, wrist_names = (
        robot.find_bodies(
            A3_WRIST_BODY,
            preserve_order=True,
        )
    )

    if len(wrist_ids) != 1:
        raise RuntimeError(
            f"Could not uniquely resolve "
            f"{A3_WRIST_BODY}: "
            f"{wrist_names}"
        )

    wrist_body_id = int(
        wrist_ids[0]
    )

    mount_offset = torch.as_tensor(
        A3_MOUNT_OFFSET,
        device=robot.device,
        dtype=torch.float32,
    )


    if int(scene.num_envs) != n_envs:

        raise RuntimeError(
            "Scene num_envs mismatch"
        )


    print(
        "VECTOR_SCENE_CREATION=PASS"
    )


    # -------------------------------------------------------------
    # Canonical joint order.
    # -------------------------------------------------------------

    joint_ids, resolved_names = (
        robot.find_joints(
            list(
                JOINT_NAMES
            ),
            preserve_order=True,
        )
    )


    if (
        len(joint_ids) != 31
        or tuple(
            resolved_names
        )
        != JOINT_NAMES
    ):

        raise RuntimeError(
            "Canonical 31-joint resolution failed"
        )


    print(
        "CANONICAL_JOINT_RESOLUTION=PASS"
    )


    device = robot.device

    dtype = (
        robot.data.joint_pos.dtype
    )


    default_q_t = torch.as_tensor(
        adapter.default_q,
        device=device,
        dtype=dtype,
    )


    q0 = (
        default_q_t
        .unsqueeze(0)
        .repeat(
            n_envs,
            1,
        )
    )


    qd0 = torch.zeros_like(
        q0
    )


    root_state0 = (
        robot.data.default_root_state
        .clone()
    )


    root_state0[
        :,
        :3,
    ] += scene.env_origins


    def reset_all_robots():

        robot.write_root_pose_to_sim(
            root_state0[
                :,
                :7,
            ]
        )

        robot.write_root_velocity_to_sim(
            root_state0[
                :,
                7:,
            ]
        )

        robot.write_joint_state_to_sim(
            q0,
            qd0,
            joint_ids=joint_ids,
        )

        robot.set_joint_position_target(
            q0,
            joint_ids=joint_ids,
        )



    # =========================================================
    # STAGE5C1-B2 PERSISTENT EPISODES
    # =========================================================

    B2_EPISODES = int(
        os.environ.get(
            "STAGE5C4_FINAL_TEST_EPISODES",
            "8",
        )
    )


    if B2_EPISODES < 1:

        raise RuntimeError(
            "STAGE5C4_FINAL_TEST_EPISODES must be >= 1"
        )

    b2_episode_gates = []
    b2_action_snapshots = []
    b2_reward_snapshots = []
    b2_initial_state_snapshots = []
    b2_contact_counts = []
    b2_legal_counts = []


    print()
    print(
        "=================================================="
    )
    print(
        "STAGE5C1-B2 PERSISTENT ISAAC PROCESS"
    )
    print(
        "=================================================="
    )

    print(
        "B2_NUM_ENVS=",
        n_envs,
    )

    print(
        "B2_EPISODES=",
        B2_EPISODES,
    )

    print(
        "ISAAC_PROCESS_REUSED_ACROSS_EPISODES=TRUE"
    )

    print(
        "ONNX_SESSION_REUSED_ACROSS_EPISODES=TRUE"
    )

    print(
        "ACTION_ADAPTER_REUSED_ACROSS_EPISODES=TRUE"
    )



    # =========================================================
    # STAGE5C2-B0
    # VECTORIZED PHYSICAL ORACLE CALIBRATION SMOKE
    #
    # This deliberately exits before the Stage5C1 strike loop.
    #
    # Goal:
    #   randomized incoming state
    #       ->
    #   same Isaac physics / aero / table collision
    #       ->
    #   true x = -0.10 m crossing
    #       ->
    #   [y,z,vx,vy,vz,time]
    #
    # No high-level policy is trained here.
    # =========================================================

    print()
    print(
        "=================================================="
    )
    print(
        "STAGE5C2-B0 VECTORIZED PHYSICAL ORACLE"
    )
    print(
        "=================================================="
    )


    # ---------------------------------------------------------
    # Initial curriculum.
    #
    # env_0 remains the known nominal positive-control incoming.
    # env_1..N-1 are randomized.
    # ---------------------------------------------------------



    # =========================================================
    # STAGE5C3-B
    # TRAIN-READY HIGH-LEVEL ENVIRONMENT SUPPORT
    # =========================================================

    STAGE5C3_OBS_CENTER = np.asarray(
        [
            -0.70,
             0.27,
            -4.00,
             0.00,
             0.70,
             0.47,
        ],
        dtype=np.float64,
    )

    STAGE5C3_OBS_SCALE = np.asarray(
        [
            0.20,
            0.15,
            1.00,
            1.00,
            1.25,
            0.12,
        ],
        dtype=np.float64,
    )


    # Smoke/validation mode only:
    #
    # env0 retains the frozen positive-control incoming/action.
    # This MUST be disabled when a real RL trainer is connected.
    STAGE5C3_VALIDATION_ANCHOR = False


    stage5c3_incoming_snapshots = []
    stage5c3_oracle_snapshots = []
    stage5c3_policy_obs_snapshots = []
    stage5c3_terminated_snapshots = []
    stage5c3_truncated_snapshots = []
    stage5c3_unstable_snapshots = []

    stage5c3_external_action_provider_calls = 0



    # =========================================================

    # =========================================================
    # STAGE5C3-C3-C DETERMINISTIC CHECKPOINT EVALUATOR
    #
    # Evaluation-only policy:
    #
    #       action = tanh(actor mean)
    #
    # No stochastic noise.
    # No replay.
    # No backward.
    # No optimizer.
    # =========================================================

    class Stage5C3SACActor(
        torch.nn.Module
    ):

        def __init__(self):

            super().__init__()

            self.backbone = torch.nn.Sequential(
                torch.nn.Linear(
                    6,
                    128,
                ),
                torch.nn.ReLU(),
                torch.nn.Linear(
                    128,
                    128,
                ),
                torch.nn.ReLU(),
            )

            self.mean_head = torch.nn.Linear(
                128,
                4,
            )

            self.log_std_head = torch.nn.Linear(
                128,
                4,
            )


        def forward(
            self,
            obs,
        ):

            h = self.backbone(
                obs
            )

            mean = self.mean_head(
                h
            )

            log_std = torch.clamp(
                self.log_std_head(
                    h
                ),
                min=-5.0,
                max=2.0,
            )

            return (
                mean,
                log_std,
            )


    stage5c3_eval_checkpoint_path = (
        os.environ.get(
            "STAGE5C3_EVAL_CHECKPOINT"
        )
    )


    if not stage5c3_eval_checkpoint_path:

        raise RuntimeError(
            "STAGE5C3_EVAL_CHECKPOINT "
            "must be provided"
        )


    stage5c3_eval_checkpoint = torch.load(
        stage5c3_eval_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )


    if (
        stage5c3_eval_checkpoint.get(
            "schema"
        )
        != "STAGE5C3_C3_SAC_PILOT_V1"
    ):

        raise RuntimeError(
            "Unexpected evaluator checkpoint schema"
        )


    if (
        "actor"
        not in stage5c3_eval_checkpoint
    ):

        raise RuntimeError(
            "Checkpoint actor state missing"
        )


    stage5c3_c1_policy = (
        Stage5C3SACActor()
        .to(
            device=device,
            dtype=torch.float32,
        )
    )


    stage5c3_c1_policy.load_state_dict(
        stage5c3_eval_checkpoint[
            "actor"
        ],
        strict=True,
    )


    stage5c3_c1_policy.eval()


    stage5c3_eval_checkpoint_loaded_exact = bool(
        all(
            torch.equal(
                stage5c3_c1_policy
                .state_dict()[
                    key
                ]
                .detach()
                .cpu(),
                value.detach().cpu(),
            )
            for key, value
            in stage5c3_eval_checkpoint[
                "actor"
            ].items()
        )
    )


    if not stage5c3_eval_checkpoint_loaded_exact:

        raise RuntimeError(
            "Checkpoint actor load mismatch"
        )


    stage5c3_c1_initial_parameters = [
        parameter
        .detach()
        .cpu()
        .clone()
        for parameter
        in stage5c3_c1_policy.parameters()
    ]


    stage5c3_c1_policy_parameter_count = int(
        sum(
            parameter.numel()
            for parameter
            in stage5c3_c1_policy.parameters()
        )
    )


    # These compatibility counters MUST remain zero.
    stage5c3_c1_optimizer_steps = 0
    stage5c3_c1_backward_calls = 0


    stage5c3_c1_policy_action_snapshots = []

    stage5c3_eval_repeat_deltas = []

    stage5c3_eval_stochastic_samples = 0


    # =========================================================
    # STAGE5C3-C3-D HELD-OUT INCOMING SEED SPLIT
    # =========================================================

    STAGE5C3_C3_D_EVAL_INCOMING_SEED_BASE = int(
        os.environ.get(
            "STAGE5C3_HELDOUT_EVAL_SEED_BASE",
            "1000000",
        )
    )


    stage5c3_c3_d_training_seed_base = int(
        stage5c3_eval_checkpoint[
            "config"
        ][
            "incoming_seed_base"
        ]
    )


    stage5c3_c3_d_training_episode_count = int(
        stage5c3_eval_checkpoint[
            "completed_episodes"
        ]
    )


    stage5c3_c3_d_training_seeds = tuple(
        stage5c3_c3_d_training_seed_base
        + i
        for i in range(
            stage5c3_c3_d_training_episode_count
        )
    )


    stage5c3_c3_d_eval_seeds = tuple(
        STAGE5C3_C3_D_EVAL_INCOMING_SEED_BASE
        + i
        for i in range(
            B2_EPISODES
        )
    )


    stage5c3_c3_d_seed_overlap = sorted(
        set(
            stage5c3_c3_d_training_seeds
        )
        .intersection(
            stage5c3_c3_d_eval_seeds
        )
    )


    stage5c3_c3_d_seed_disjoint = bool(
        len(
            stage5c3_c3_d_seed_overlap
        )
        == 0
    )


    if not stage5c3_c3_d_seed_disjoint:

        raise RuntimeError(
            "Training/evaluation incoming "
            "seed overlap detected"
        )


    print(
        "STAGE5C3_C3_D_TRAIN_INCOMING_SEED_BASE=",
        stage5c3_c3_d_training_seed_base,
    )

    print(
        "STAGE5C3_C3_D_TRAIN_COMPLETED_EPISODES=",
        stage5c3_c3_d_training_episode_count,
    )

    print(
        "STAGE5C3_C3_D_TRAIN_SEEDS=",
        list(
            stage5c3_c3_d_training_seeds
        ),
    )

    print(
        "STAGE5C3_C3_D_EVAL_INCOMING_SEED_BASE=",
        STAGE5C3_C3_D_EVAL_INCOMING_SEED_BASE,
    )

    print(
        "STAGE5C3_C3_D_EVAL_SEEDS=",
        list(
            stage5c3_c3_d_eval_seeds
        ),
    )

    print(
        "STAGE5C3_C3_D_SEED_OVERLAP=",
        stage5c3_c3_d_seed_overlap,
    )

    print(
        "STAGE5C3_C3_D_SEED_DISJOINT=",
        "PASS"
        if stage5c3_c3_d_seed_disjoint
        else "FAIL",
    )



    print(
        "STAGE5C3_C3_C_CHECKPOINT_SCHEMA=",
        stage5c3_eval_checkpoint[
            "schema"
        ],
    )

    print(
        "STAGE5C3_C3_C_CHECKPOINT_COMPLETED_EPISODES=",
        int(
            stage5c3_eval_checkpoint[
                "completed_episodes"
            ]
        ),
    )

    print(
        "STAGE5C3_C3_C_CHECKPOINT_SAC_UPDATES=",
        int(
            stage5c3_eval_checkpoint[
                "sac_update_steps"
            ]
        ),
    )

    print(
        "STAGE5C3_C3_C_ACTOR_PARAMETER_COUNT=",
        stage5c3_c1_policy_parameter_count,
    )

    print(
        "STAGE5C3_C3_C_CHECKPOINT_ACTOR_LOAD="
        "PASS"
    )

    print(
        "STAGE5C3_C3_C_ACTION_MODE="
        "DETERMINISTIC_TANH_MEAN"
    )

    print(
        "STAGE5C3_C3_C_STOCHASTIC_SAMPLING=DISABLED"
    )


    def stage5c3_external_action_provider(
        policy_obs,
        episode_index,
    ):
        """Deterministic SAC checkpoint evaluator."""

        nonlocal stage5c3_external_action_provider_calls


        policy_obs = np.asarray(
            policy_obs,
            dtype=np.float64,
        )


        if (
            policy_obs.shape
            != (
                n_envs,
                6,
            )
            or not np.isfinite(
                policy_obs
            ).all()
        ):

            raise RuntimeError(
                "Invalid deterministic evaluator observation"
            )


        obs_tensor = torch.as_tensor(
            policy_obs,
            device=device,
            dtype=torch.float32,
        )


        with torch.no_grad():

            mean_a, _ = (
                stage5c3_c1_policy(
                    obs_tensor
                )
            )

            action_a = torch.tanh(
                mean_a
            )


            # Repeat on identical observations to prove the
            # evaluator has no sampling path.
            mean_b, _ = (
                stage5c3_c1_policy(
                    obs_tensor
                )
            )

            action_b = torch.tanh(
                mean_b
            )


        repeat_delta = float(
            torch.max(
                torch.abs(
                    action_a
                    - action_b
                )
            )
            .detach()
            .cpu()
            .item()
        )


        stage5c3_eval_repeat_deltas.append(
            repeat_delta
        )


        normalized_action = (
            action_a
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float64,
                copy=True,
            )
        )


        if (
            normalized_action.shape
            != (
                n_envs,
                4,
            )
            or not np.isfinite(
                normalized_action
            ).all()
            or np.any(
                normalized_action
                < -1.0
                - 1.0e-6
            )
            or np.any(
                normalized_action
                > 1.0
                + 1.0e-6
            )
        ):

            raise RuntimeError(
                "Invalid deterministic evaluator action"
            )


        stage5c3_c1_policy_action_snapshots.append(
            normalized_action.copy()
        )


        stage5c3_external_action_provider_calls += 1


        return normalized_action


    print()
    print(
        "=================================================="
    )
    print(
        "STAGE5C3-B TRAIN-READY ENVIRONMENT"
    )
    print(
        "=================================================="
    )

    print(
        "STAGE5C3_HIGH_LEVEL_DECISIONS_PER_BALL=1"
    )

    print(
        "STAGE5C3_CONTEXTUAL_BANDIT_ONE_STEP_MDP=TRUE"
    )

    print(
        "STAGE5C3_EXTERNAL_ACTION_PROVIDER="
        "SAC_CHECKPOINT_DETERMINISTIC_TANH_MEAN"
    )

    print(
        "STAGE5C3_VALIDATION_ANCHOR="
        "FALSE_C1_POLICY_INTEGRATION"
    )

    print(
        "NEW_HIGH_LEVEL_RL_TRAINING_STARTED=FALSE"
    )

    for episode_idx in range(
        B2_EPISODES
    ):
        stage5c2_rng = np.random.default_rng(
            STAGE5C3_C3_D_EVAL_INCOMING_SEED_BASE + episode_idx
        )

        stage5c2_ball_pos_l = np.zeros(
            (
                n_envs,
                3,
            ),
            dtype=np.float64,
        )

        stage5c2_ball_vel_w = np.zeros(
            (
                n_envs,
                3,
            ),
            dtype=np.float64,
        )

        stage5c2_ball_spin_w = np.zeros(
            (
                n_envs,
                3,
            ),
            dtype=np.float64,
        )


        stage5c2_ball_pos_l[
            :,
            0,
        ] = 2.22

        stage5c2_ball_pos_l[
            :,
            1,
        ] = stage5c2_rng.uniform(
            -0.90,
            -0.60,
            size=n_envs,
        )

        stage5c2_ball_pos_l[
            :,
            2,
        ] = stage5c2_rng.uniform(
            0.32,
            0.54,
            size=n_envs,
        )


        stage5c2_ball_vel_w[
            :,
            0,
        ] = stage5c2_rng.uniform(
            -7.00,
            -5.00,
            size=n_envs,
        )

        stage5c2_ball_vel_w[
            :,
            1,
        ] = 0.0

        stage5c2_ball_vel_w[
            :,
            2,
        ] = 0.08



        # Stage5C3-C1:
        # env0 is NOT a positive-control exception.
        # All environments use the sampled curriculum incoming.
        print(
            "STAGE5C3_C1_ENV0_INCOMING_OVERRIDE="
            "DISABLED"
        )


        print(
            "STAGE5C2_RANDOM_INCOMING_VARIABLES="
            "[ball_y,ball_z,ball_vx]"
        )

        print(
            "STAGE5C2_RANDOM_Y_RANGE=",
            [
                float(
                    np.min(
                        stage5c2_ball_pos_l[
                            1:,
                            1,
                        ]
                    )
                )
                if n_envs > 1
                else float(
                    stage5c2_ball_pos_l[
                        0,
                        1,
                    ]
                ),
                float(
                    np.max(
                        stage5c2_ball_pos_l[
                            1:,
                            1,
                        ]
                    )
                )
                if n_envs > 1
                else float(
                    stage5c2_ball_pos_l[
                        0,
                        1,
                    ]
                ),
            ],
        )

        print(
            "STAGE5C2_RANDOM_Z_RANGE=",
            [
                float(
                    np.min(
                        stage5c2_ball_pos_l[
                            1:,
                            2,
                        ]
                    )
                )
                if n_envs > 1
                else float(
                    stage5c2_ball_pos_l[
                        0,
                        2,
                    ]
                ),
                float(
                    np.max(
                        stage5c2_ball_pos_l[
                            1:,
                            2,
                        ]
                    )
                )
                if n_envs > 1
                else float(
                    stage5c2_ball_pos_l[
                        0,
                        2,
                    ]
                ),
            ],
        )

        print(
            "STAGE5C2_RANDOM_VX_RANGE=",
            [
                float(
                    np.min(
                        stage5c2_ball_vel_w[
                            1:,
                            0,
                        ]
                    )
                )
                if n_envs > 1
                else float(
                    stage5c2_ball_vel_w[
                        0,
                        0,
                    ]
                ),
                float(
                    np.max(
                        stage5c2_ball_vel_w[
                            1:,
                            0,
                        ]
                    )
                )
                if n_envs > 1
                else float(
                    stage5c2_ball_vel_w[
                        0,
                        0,
                    ]
                ),
            ],
        )


        # ---------------------------------------------------------
        # Full-batch ball-state writer.
        # ---------------------------------------------------------

        def stage5c2_write_ball_batch(
            pos_l_np,
            lin_vel_np,
            ang_vel_np,
        ):

            pose = torch.zeros(
                (
                    n_envs,
                    7,
                ),
                device=ball.device,
                dtype=torch.float32,
            )

            pose[
                :,
                :3,
            ] = (
                scene.env_origins
                + torch.as_tensor(
                    pos_l_np,
                    device=ball.device,
                    dtype=torch.float32,
                )
            )

            # quaternion wxyz
            pose[
                :,
                3,
            ] = 1.0


            velocity = torch.cat(
                [
                    torch.as_tensor(
                        lin_vel_np,
                        device=ball.device,
                        dtype=torch.float32,
                    ),
                    torch.as_tensor(
                        ang_vel_np,
                        device=ball.device,
                        dtype=torch.float32,
                    ),
                ],
                dim=1,
            )


            ball.write_root_pose_to_sim(
                pose
            )

            ball.write_root_velocity_to_sim(
                velocity
            )


        # ---------------------------------------------------------
        # Move all robots out of the incoming-ball corridor.
        #
        # Same principle already validated in Stage4A calibration.
        # ---------------------------------------------------------

        calibration_root = (
            root_state0.clone()
        )

        calibration_root[
            :,
            0,
        ] = (
            scene.env_origins[
                :,
                0,
            ]
            - 4.0
        )

        calibration_root[
            :,
            1,
        ] = (
            scene.env_origins[
                :,
                1,
            ]
            + 2.0
        )


        robot.write_root_pose_to_sim(
            calibration_root[
                :,
                :7,
            ]
        )

        robot.write_root_velocity_to_sim(
            calibration_root[
                :,
                7:,
            ]
        )

        robot.write_joint_state_to_sim(
            q0,
            qd0,
            joint_ids=joint_ids,
        )

        robot.set_joint_position_target(
            q0,
            joint_ids=joint_ids,
        )


        # Keep balls fixed while relocated robot states synchronize.
        zero_vel = np.zeros(
            (
                n_envs,
                3,
            ),
            dtype=np.float64,
        )


        for _ in range(
            4
        ):

            stage5c2_write_ball_batch(
                stage5c2_ball_pos_l,
                zero_vel,
                zero_vel,
            )

            scene.write_data_to_sim()

            sim.step(
                render=False
            )

            scene.update(
                PHYSICS_DT
            )


        print(
            "CALIBRATION_ROBOTS_RELOCATED=PASS"
        )


        # ---------------------------------------------------------
        # Launch all calibration balls simultaneously.
        # ---------------------------------------------------------

        stage5c2_write_ball_batch(
            stage5c2_ball_pos_l,
            stage5c2_ball_vel_w,
            stage5c2_ball_spin_w,
        )

        scene.write_data_to_sim()


        hit_x = -0.10

        crossed = np.zeros(
            n_envs,
            dtype=bool,
        )

        bounce_seen = np.zeros(
            n_envs,
            dtype=bool,
        )

        oracle_pos_l = np.full(
            (
                n_envs,
                3,
            ),
            np.nan,
            dtype=np.float64,
        )

        oracle_vel_w = np.full(
            (
                n_envs,
                3,
            ),
            np.nan,
            dtype=np.float64,
        )

        oracle_cross_time = np.full(
            n_envs,
            np.nan,
            dtype=np.float64,
        )


        previous_pos_l = (
            stage5c2_ball_pos_l.copy()
        )

        previous_vel_w = (
            stage5c2_ball_vel_w.copy()
        )


        calibration_time = 0.0

        max_calibration_s = 1.20

        # Raw sensor-force diagnostic.  This may contain forces
        # unrelated to a true ball-racket contact and is therefore
        # NOT itself a blocking contact criterion.
        max_calibration_contact_force = 0.0

        # Stage4A-compatible physical contact diagnostics:
        #
        # true calibration contact =
        #   force > threshold
        #   AND racket face-close geometry
        #
        # face-close =
        #   lateral distance < FACE_LATERAL_THRESHOLD
        #   AND abs(normal distance) < FACE_NORMAL_THRESHOLD
        calibration_contact_seen = np.zeros(
            n_envs,
            dtype=bool,
        )

        max_face_gated_contact_force = 0.0

        min_calibration_racket_ball_distance = np.full(
            n_envs,
            np.inf,
            dtype=np.float64,
        )

        min_calibration_face_lateral_distance = np.full(
            n_envs,
            np.inf,
            dtype=np.float64,
        )

        min_calibration_abs_face_normal_distance = np.full(
            n_envs,
            np.inf,
            dtype=np.float64,
        )


        while (
            calibration_time
            < max_calibration_s
            and not np.all(
                crossed
            )
        ):

            robot.set_joint_position_target(
                q0,
                joint_ids=joint_ids,
            )

            scene.write_data_to_sim()

            sim.step(
                render=False
            )

            scene.update(
                PHYSICS_DT
            )


            calibration_time += PHYSICS_DT


            current_pos_l = (
                ball.data.root_pos_w
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float64,
                    copy=True,
                )
                - scene.env_origins
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float64,
                    copy=False,
                )
            )


            current_vel_w = (
                ball.data.root_lin_vel_w
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float64,
                    copy=True,
                )
            )


            # -----------------------------------------------------
            # First table-like bounce.
            # -----------------------------------------------------

            bounce_now = (
                ~bounce_seen
                & (
                    previous_vel_w[
                        :,
                        2,
                    ]
                    < 0.0
                )
                & (
                    current_vel_w[
                        :,
                        2,
                    ]
                    > 0.0
                )
                & (
                    current_pos_l[
                        :,
                        0,
                    ]
                    >= 0.0
                )
                & (
                    current_pos_l[
                        :,
                        0,
                    ]
                    <= geometry.TABLE_LENGTH
                )
                & (
                    current_pos_l[
                        :,
                        1,
                    ]
                    >= -geometry.TABLE_WIDTH
                )
                & (
                    current_pos_l[
                        :,
                        1,
                    ]
                    <= 0.0
                )
                & (
                    current_pos_l[
                        :,
                        2,
                    ]
                    < 0.10
                )
            )

            bounce_seen[
                bounce_now
            ] = True


            # -----------------------------------------------------
            # First physical crossing of x = -0.10 m.
            # -----------------------------------------------------

            crossing_now = (
                ~crossed
                & (
                    previous_pos_l[
                        :,
                        0,
                    ]
                    > hit_x
                )
                & (
                    current_pos_l[
                        :,
                        0,
                    ]
                    <= hit_x
                )
            )


            ids = np.flatnonzero(
                crossing_now
            )


            for env_id in ids:

                denominator = (
                    previous_pos_l[
                        env_id,
                        0,
                    ]
                    - current_pos_l[
                        env_id,
                        0,
                    ]
                )

                alpha = (
                    previous_pos_l[
                        env_id,
                        0,
                    ]
                    - hit_x
                ) / max(
                    denominator,
                    1.0e-12,
                )


                oracle_pos_l[
                    env_id
                ] = (
                    previous_pos_l[
                        env_id
                    ]
                    + alpha
                    * (
                        current_pos_l[
                            env_id
                        ]
                        - previous_pos_l[
                            env_id
                        ]
                    )
                )


                # Same temporal interpolation for velocity.
                oracle_vel_w[
                    env_id
                ] = (
                    previous_vel_w[
                        env_id
                    ]
                    + alpha
                    * (
                        current_vel_w[
                            env_id
                        ]
                        - previous_vel_w[
                            env_id
                        ]
                    )
                )


                oracle_cross_time[
                    env_id
                ] = (
                    calibration_time
                    - PHYSICS_DT
                    + alpha
                    * PHYSICS_DT
                )


                crossed[
                    env_id
                ] = True


            # -----------------------------------------------------
            # Stage4A-compatible calibration contact semantics.
            #
            # IMPORTANT:
            # Raw contact-sensor force alone is not sufficient.
            # A true ball-racket contact additionally requires that
            # the ball be geometrically close to the racket face.
            # -----------------------------------------------------

            (
                calibration_racket_pos_w_t,
                _,
                calibration_racket_normal_w_t,
            ) = racket_state_batch(
                robot,
                wrist_body_id,
                mount_offset,
            )


            calibration_ball_pos_w_t = (
                ball.data.root_pos_w
            )


            calibration_delta_w_t = (
                calibration_ball_pos_w_t
                - calibration_racket_pos_w_t
            )


            calibration_signed_normal_t = torch.sum(
                calibration_delta_w_t
                * calibration_racket_normal_w_t,
                dim=-1,
            )


            calibration_lateral_vec_t = (
                calibration_delta_w_t
                - calibration_signed_normal_t.unsqueeze(-1)
                * calibration_racket_normal_w_t
            )


            calibration_lateral_dist_t = (
                torch.linalg.vector_norm(
                    calibration_lateral_vec_t,
                    dim=-1,
                )
            )


            calibration_center_dist_t = (
                torch.linalg.vector_norm(
                    calibration_delta_w_t,
                    dim=-1,
                )
            )


            calibration_face_close_t = (
                (
                    calibration_lateral_dist_t
                    < FACE_LATERAL_THRESHOLD
                )
                & (
                    torch.abs(
                        calibration_signed_normal_t
                    )
                    < FACE_NORMAL_THRESHOLD
                )
            )


            calibration_force_mag_t = (
                torch.linalg.vector_norm(
                    sensor.data.net_forces_w,
                    dim=-1,
                )
            )


            # Convert arbitrary per-env sensor shape into one maximum
            # force value per environment.
            if calibration_force_mag_t.ndim == 1:

                calibration_force_per_env_t = (
                    calibration_force_mag_t
                )

            else:

                calibration_force_per_env_t = (
                    calibration_force_mag_t
                    .reshape(
                        n_envs,
                        -1,
                    )
                    .amax(
                        dim=1
                    )
                )


            calibration_true_contact_t = (
                (
                    calibration_force_per_env_t
                    > CONTACT_FORCE_THRESHOLD
                )
                & calibration_face_close_t
            )


            calibration_force_per_env = (
                calibration_force_per_env_t
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float64,
                    copy=True,
                )
            )


            calibration_face_close = (
                calibration_face_close_t
                .detach()
                .cpu()
                .numpy()
                .astype(
                    bool,
                    copy=True,
                )
            )


            calibration_true_contact = (
                calibration_true_contact_t
                .detach()
                .cpu()
                .numpy()
                .astype(
                    bool,
                    copy=True,
                )
            )


            calibration_center_dist = (
                calibration_center_dist_t
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float64,
                    copy=True,
                )
            )


            calibration_lateral_dist = (
                calibration_lateral_dist_t
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float64,
                    copy=True,
                )
            )


            calibration_abs_normal_dist = (
                torch.abs(
                    calibration_signed_normal_t
                )
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float64,
                    copy=True,
                )
            )


            max_calibration_contact_force = max(
                max_calibration_contact_force,
                float(
                    np.max(
                        calibration_force_per_env
                    )
                ),
            )


            if np.any(
                calibration_face_close
            ):

                max_face_gated_contact_force = max(
                    max_face_gated_contact_force,
                    float(
                        np.max(
                            calibration_force_per_env[
                                calibration_face_close
                            ]
                        )
                    ),
                )


            calibration_contact_seen |= (
                calibration_true_contact
            )


            min_calibration_racket_ball_distance = np.minimum(
                min_calibration_racket_ball_distance,
                calibration_center_dist,
            )

            min_calibration_face_lateral_distance = np.minimum(
                min_calibration_face_lateral_distance,
                calibration_lateral_dist,
            )

            min_calibration_abs_face_normal_distance = np.minimum(
                min_calibration_abs_face_normal_distance,
                calibration_abs_normal_dist,
            )


            previous_pos_l = (
                current_pos_l
            )

            previous_vel_w = (
                current_vel_w
            )


        # ---------------------------------------------------------
        # Construct final 6D physical oracle.
        # ---------------------------------------------------------

        oracle_obs = np.concatenate(
            [
                oracle_pos_l[
                    :,
                    1:3,
                ],
                oracle_vel_w,
                oracle_cross_time[
                    :,
                    None,
                ],
            ],
            axis=1,
        )


        # ---------------------------------------------------------
        # Compare against the intentionally-wrong CV launch model.
        # Diagnostic only.
        # ---------------------------------------------------------

        cv_time = (
            (
                stage5c2_ball_pos_l[
                    :,
                    0,
                ]
                - hit_x
            )
            / np.maximum(
                -stage5c2_ball_vel_w[
                    :,
                    0,
                ],
                1.0e-6,
            )
        )


        cv_hit_z = (
            stage5c2_ball_pos_l[
                :,
                2,
            ]
            + stage5c2_ball_vel_w[
                :,
                2,
            ]
            * cv_time
        )


        cv_vs_physical_z_error = np.abs(
            cv_hit_z
            - oracle_pos_l[
                :,
                2,
            ]
        )


        # ---------------------------------------------------------
        # Acceptance gates.
        # ---------------------------------------------------------

        all_crossed = bool(
            np.all(
                crossed
            )
        )

        all_bounced = bool(
            np.all(
                bounce_seen
            )
        )

        finite_oracle = bool(
            np.isfinite(
                oracle_obs
            ).all()
        )


        hit_x_error = float(
            np.max(
                np.abs(
                    oracle_pos_l[
                        :,
                        0,
                    ]
                    - hit_x
                )
            )
        )


        time_sane = bool(
            np.all(
                oracle_cross_time
                > 0.25
            )
            and np.all(
                oracle_cross_time
                < 0.80
            )
        )


        lateral_sane = bool(
            np.all(
                np.abs(
                    oracle_pos_l[
                        :,
                        1,
                    ]
                    - stage5c2_ball_pos_l[
                        :,
                        1,
                    ]
                )
                < 0.05
            )
        )


        height_sane = bool(
            np.all(
                oracle_pos_l[
                    :,
                    2,
                ]
                > 0.03
            )
            and np.all(
                oracle_pos_l[
                    :,
                    2,
                ]
                < 0.45
            )
        )


        calibration_contact_count = int(
            np.sum(
                calibration_contact_seen
            )
        )

        calibration_no_racket_contact = bool(
            calibration_contact_count == 0
        )


        nominal_cross_time_error = float(
            abs(
                oracle_cross_time[
                    0
                ]
                - CALIBRATED_CROSS_TIME_S
            )
        )


        nominal_control_ok = bool(
            nominal_cross_time_error
            < 0.01
            and abs(
                oracle_pos_l[
                    0,
                    1,
                ]
                - (
                    COMMAND_TARGET_POS_L[
                        1
                    ]
                    - RACKET_TARGET_DY
                )
            )
            < 0.01
            and abs(
                oracle_pos_l[
                    0,
                    2,
                ]
                - COMMAND_TARGET_POS_L[
                    2
                ]
            )
            < 0.02
        )


        oracle_diversity = (
            np.ptp(
                oracle_obs,
                axis=0,
            )
        )


        diversity_ok = bool(
            n_envs == 1
            or (
                oracle_diversity[
                    0
                ]
                > 0.05
                and oracle_diversity[
                    1
                ]
                > 0.03
                and oracle_diversity[
                    2
                ]
                > 0.30
            )
        )


        oracle_gate = bool(
            all_crossed
            and all_bounced
            and finite_oracle
            and hit_x_error
            < 1.0e-6
            and time_sane
            and lateral_sane
            and height_sane
            and calibration_no_racket_contact
            and diversity_ok
            and oracle_obs.shape
            == (
                n_envs,
                6,
            )
        )


        print()
        print(
            "=================================================="
        )
        print(
            "STAGE5C2-B0 ORACLE RESULTS"
        )
        print(
            "=================================================="
        )

        print(
            "ORACLE_OBS_SHAPE=",
            oracle_obs.shape,
        )

        print(
            "ALL_PHYSICAL_HIT_PLANE_CROSSED=",
            all_crossed,
        )

        print(
            "ALL_TABLE_BOUNCE_SEEN=",
            all_bounced,
        )

        print(
            "ORACLE_FINITE=",
            finite_oracle,
        )

        print(
            "ORACLE_HIT_X_MAX_ERROR_M=",
            hit_x_error,
        )

        print(
            "ORACLE_CROSS_TIME_RANGE_S=",
            [
                float(
                    np.min(
                        oracle_cross_time
                    )
                ),
                float(
                    np.max(
                        oracle_cross_time
                    )
                ),
            ],
        )

        print(
            "ORACLE_HIT_Y_RANGE_M=",
            [
                float(
                    np.min(
                        oracle_pos_l[
                            :,
                            1,
                        ]
                    )
                ),
                float(
                    np.max(
                        oracle_pos_l[
                            :,
                            1,
                        ]
                    )
                ),
            ],
        )

        print(
            "ORACLE_HIT_Z_RANGE_M=",
            [
                float(
                    np.min(
                        oracle_pos_l[
                            :,
                            2,
                        ]
                    )
                ),
                float(
                    np.max(
                        oracle_pos_l[
                            :,
                            2,
                        ]
                    )
                ),
            ],
        )

        print(
            "ORACLE_HIT_VX_RANGE_MPS=",
            [
                float(
                    np.min(
                        oracle_vel_w[
                            :,
                            0,
                        ]
                    )
                ),
                float(
                    np.max(
                        oracle_vel_w[
                            :,
                            0,
                        ]
                    )
                ),
            ],
        )

        print(
            "ORACLE_HIT_VY_RANGE_MPS=",
            [
                float(
                    np.min(
                        oracle_vel_w[
                            :,
                            1,
                        ]
                    )
                ),
                float(
                    np.max(
                        oracle_vel_w[
                            :,
                            1,
                        ]
                    )
                ),
            ],
        )

        print(
            "ORACLE_HIT_VZ_RANGE_MPS=",
            [
                float(
                    np.min(
                        oracle_vel_w[
                            :,
                            2,
                        ]
                    )
                ),
                float(
                    np.max(
                        oracle_vel_w[
                            :,
                            2,
                        ]
                    )
                ),
            ],
        )

        print(
            "ORACLE_DIVERSITY=",
            oracle_diversity.tolist(),
        )

        print(
            "ORACLE_DIVERSITY_OK=",
            diversity_ok,
        )

        print(
            "MAX_CALIBRATION_RAW_SENSOR_FORCE_N=",
            max_calibration_contact_force,
        )

        print(
            "MAX_CALIBRATION_FACE_GATED_FORCE_N=",
            max_face_gated_contact_force,
        )

        print(
            "CALIBRATION_TRUE_RACKET_CONTACT_COUNT=",
            f"{calibration_contact_count}/{n_envs}",
        )

        print(
            "CALIBRATION_TRUE_RACKET_CONTACT_MASK=",
            calibration_contact_seen.tolist(),
        )

        print(
            "CALIBRATION_MIN_RACKET_BALL_DISTANCE_RANGE_M=",
            [
                float(
                    np.min(
                        min_calibration_racket_ball_distance
                    )
                ),
                float(
                    np.max(
                        min_calibration_racket_ball_distance
                    )
                ),
            ],
        )

        print(
            "CALIBRATION_MIN_FACE_LATERAL_RANGE_M=",
            [
                float(
                    np.min(
                        min_calibration_face_lateral_distance
                    )
                ),
                float(
                    np.max(
                        min_calibration_face_lateral_distance
                    )
                ),
            ],
        )

        print(
            "CALIBRATION_MIN_ABS_FACE_NORMAL_RANGE_M=",
            [
                float(
                    np.min(
                        min_calibration_abs_face_normal_distance
                    )
                ),
                float(
                    np.max(
                        min_calibration_abs_face_normal_distance
                    )
                ),
            ],
        )

        print(
            "CALIBRATION_CONTACT_SEMANTICS="
            "FORCE_AND_FACE_CLOSE_STAGE4A_COMPATIBLE"
        )

        print(
            "CALIBRATION_NO_RACKET_CONTACT=",
            calibration_no_racket_contact,
        )

        print(
            "NOMINAL_ENV0_CROSS_TIME_S=",
            float(
                oracle_cross_time[
                    0
                ]
            ),
        )

        print(
            "NOMINAL_ENV0_CROSS_TIME_ERROR_S=",
            nominal_cross_time_error,
        )

        print(
            "NOMINAL_ENV0_ORACLE=",
            oracle_obs[
                0
            ].tolist(),
        )

        print(
            "NOMINAL_CONTROL_REPRODUCTION=",
            nominal_control_ok,
        )

        print(
            "CV_PREDICTOR_Z_ERROR_MEAN_M=",
            float(
                np.mean(
                    cv_vs_physical_z_error
                )
            ),
        )

        print(
            "CV_PREDICTOR_Z_ERROR_MAX_M=",
            float(
                np.max(
                    cv_vs_physical_z_error
                )
            ),
        )

        print(
            "ORACLE_SEMANTICS="
            "PHYSICAL_AERO_TABLE_BOUNCE_HIT_PLANE_STATE"
        )

        print(
            "STAGE5C2B0_PHYSICAL_ORACLE_SMOKE=",
            "PASS"
            if oracle_gate
            else "FAIL",
        )


        if not oracle_gate:

            raise RuntimeError(
                "Stage5C2-B0 physical oracle gate failed"
            )



        # =========================================================
        # STAGE5C2-B1
        # Continue from the validated physical oracle into the
        # actual frozen-model_21800 strike.
        # =========================================================

        print()
        print(
            "=================================================="
        )
        print(
            "STAGE5C2-B1 PHYSICAL STRIKE CONTINUATION"
        )
        print(
            "=================================================="
        )

        stage5c2_oracle_flight_time_s = (
            oracle_cross_time.copy()
        )

        stage5c2_launch_delay_s = np.maximum(
            0.0,
            LEAD_TIME
            - stage5c2_oracle_flight_time_s,
        )

        stage5c2_oracle_contract_ok = bool(
            oracle_gate
            and oracle_obs.shape == (n_envs, 6)
            and np.isfinite(oracle_obs).all()
            and np.isfinite(
                stage5c2_launch_delay_s
            ).all()
            and np.all(
                stage5c2_launch_delay_s >= 0.0
            )
        )

        if not stage5c2_oracle_contract_ok:
            raise RuntimeError(
                "Stage5C2-B1 received invalid physical oracle"
            )


        stage5c2_park_pos_l_batch = np.repeat(
            BALL_PARK_POS_L[
                None,
                :,
            ],
            n_envs,
            axis=0,
        ).astype(
            np.float64,
            copy=True,
        )

        stage5c2_zero_vel_batch = np.zeros(
            (
                n_envs,
                3,
            ),
            dtype=np.float64,
        )


        def stage5c2_write_selected_ball_states(
            pos_l_np,
            lin_vel_np,
            ang_vel_np,
            env_ids_np,
        ):
            """Write root state only for selected environments."""

            env_ids_np = np.asarray(
                env_ids_np,
                dtype=np.int64,
            ).reshape(-1)

            if env_ids_np.size == 0:
                return

            if np.any(env_ids_np < 0) or np.any(
                env_ids_np >= n_envs
            ):
                raise RuntimeError(
                    "Invalid Stage5C2 selected env id"
                )

            def select_rows(x):
                x = np.asarray(
                    x,
                    dtype=np.float64,
                )

                if x.shape == (3,):
                    return np.repeat(
                        x[None, :],
                        env_ids_np.size,
                        axis=0,
                    )

                if x.shape != (n_envs, 3):
                    raise RuntimeError(
                        f"Unexpected selected-state shape {x.shape}"
                    )

                return x[
                    env_ids_np
                ]


            pos_selected = select_rows(
                pos_l_np
            )

            lin_selected = select_rows(
                lin_vel_np
            )

            ang_selected = select_rows(
                ang_vel_np
            )


            env_ids_t = torch.as_tensor(
                env_ids_np,
                device=ball.device,
                dtype=torch.long,
            )


            pose = torch.zeros(
                (
                    env_ids_np.size,
                    7,
                ),
                device=ball.device,
                dtype=torch.float32,
            )

            pose[
                :,
                :3,
            ] = (
                scene.env_origins[
                    env_ids_t
                ]
                + torch.as_tensor(
                    pos_selected,
                    device=ball.device,
                    dtype=torch.float32,
                )
            )

            pose[
                :,
                3,
            ] = 1.0


            velocity = torch.zeros(
                (
                    env_ids_np.size,
                    6,
                ),
                device=ball.device,
                dtype=torch.float32,
            )

            velocity[
                :,
                :3,
            ] = torch.as_tensor(
                lin_selected,
                device=ball.device,
                dtype=torch.float32,
            )

            velocity[
                :,
                3:,
            ] = torch.as_tensor(
                ang_selected,
                device=ball.device,
                dtype=torch.float32,
            )


            ball.write_root_pose_to_sim(
                pose,
                env_ids=env_ids_t,
            )

            ball.write_root_velocity_to_sim(
                velocity,
                env_ids=env_ids_t,
            )


        print(
            "B1_ORACLE_OBS_SHAPE=",
            oracle_obs.shape,
        )

        print(
            "B1_ORACLE_FLIGHT_TIME_RANGE_S=",
            [
                float(
                    np.min(
                        stage5c2_oracle_flight_time_s
                    )
                ),
                float(
                    np.max(
                        stage5c2_oracle_flight_time_s
                    )
                ),
            ],
        )

        print(
            "B1_LAUNCH_DELAY_RANGE_S=",
            [
                float(
                    np.min(
                        stage5c2_launch_delay_s
                    )
                ),
                float(
                    np.max(
                        stage5c2_launch_delay_s
                    )
                ),
            ],
        )

        print(
            "B1_ORACLE_TIME_SEMANTICS="
            "PHYSICAL_FLIGHT_TIME_FROM_LAUNCH_TO_HIT_PLANE"
        )

        print(
            "B1_POLICY_STRIKE_CLOCK_S=",
            LEAD_TIME,
        )

        print(
            "STAGE5C3_C1_ACTION_DEPENDS_ON_POLICY_OBS=TRUE"
        )

        print(
            "B1_MODEL21800_FROZEN=TRUE"
        )




        # =====================================================
        # Stage5C3 reset contract:
        # this incoming state and this oracle belong to the
        # CURRENT episode.
        # =====================================================

        stage5c3_incoming_snapshot = np.concatenate(
            [
                stage5c2_ball_pos_l,
                stage5c2_ball_vel_w,
            ],
            axis=1,
        )

        stage5c3_incoming_snapshots.append(
            stage5c3_incoming_snapshot.copy()
        )

        stage5c3_oracle_snapshots.append(
            oracle_obs.copy()
        )

        print(
            "STAGE5C3_EPISODE_INCOMING_ORACLE_READY "
            f"episode={episode_idx}"
        )


        print()
        print(
            "##################################################"
        )

        print(
            f"B2 EPISODE "
            f"{episode_idx + 1}/{B2_EPISODES} START"
        )

        print(
            "##################################################"
        )


        # Fresh mutable lifecycle for every ball/environment.
        lifecycles = [
            SwingLifecycle(
                lifecycle_cfg
            )
            for _ in range(
                n_envs
            )
        ]


        b2_lifecycle_reset_ok = bool(
            len(lifecycles) == n_envs
            and len(
                {
                    id(x)
                    for x in lifecycles
                }
            ) == n_envs
            and all(
                phase_name(x) == "ready"
                for x in lifecycles
            )
        )


        print(
            "B2_LIFECYCLE_RESET=",
            b2_lifecycle_reset_ok,
        )

        reset_all_robots()


        # -------------------------------------------------------------
        # Initial settle.
        # -------------------------------------------------------------

        # Stage-4 execution-parity contract:
        # after robot reset, settle exactly 0.20 s before
        # capturing strike-start state/base target.
        PRESTRIKE_SETTLE_S = 0.20

        settle_steps = int(
            round(
                PRESTRIKE_SETTLE_S
                / PHYSICS_DT
            )
        )

        print(
            "prestrike_settle_s =",
            PRESTRIKE_SETTLE_S,
        )


        for _ in range(
            settle_steps
        ):

            park_all_balls(
                ball,
                scene.env_origins,
            )

            scene.write_data_to_sim()

            sim.step(
                render=False
            )

            scene.update(
                PHYSICS_DT
            )


        torch.cuda.synchronize()



        # =====================================================
        # STAGE5C1-B2-R1 FINAL PARK AFTER ROBOT SETTLE
        #
        # The settle loop parks the ball BEFORE each 400-Hz
        # physics step.  Therefore the final substep leaves one
        # 2.5-ms gravity increment in the ball state.
        #
        # Re-park here, with NO subsequent physics step, so the
        # episode observation/action contract starts from the
        # intended parked-ball state.
        # =====================================================

        park_all_balls(
            ball,
            scene.env_origins,
        )

        scene.write_data_to_sim()

        print(
            "B2_FINAL_POST_SETTLE_BALL_REPARK=TRUE"
        )

        initial_packed = (
            robot_state_batch_to_numpy(
                robot,
                joint_ids,
            )
        )


        if (
            initial_packed.shape
            != (
                n_envs,
                72,
            )
        ):

            raise RuntimeError(
                "Batched robot state shape mismatch"
            )


        print(
            "BATCHED_ROBOT_STATE=PASS"
        )


        base_target_xy = (
            initial_packed[
                :,
                0:2,
            ]
            .copy()
        )


        initial_base_xy = (
            base_target_xy.copy()
        )


        env_origins_np = (
            scene.env_origins
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float64,
                copy=True,
            )
        )




        # =========================================================
        # STAGE5C1-B1 HIGH-LEVEL ACTION INTERFACE
        #
        # One action is sampled ONCE per environment and held
        # throughout this complete ball/strike episode.
        #
        # normalized action:
        #   [dy, vx, vy, vz] in [-1,+1]^4
        #
        # physical bounds are the Stage-4C tested bounds.
        # =========================================================

        stage5c1_action_lo = np.asarray(
            [
                -0.10,
                3.10,
                -1.30,
                1.25,
            ],
            dtype=np.float64,
        )

        stage5c1_action_hi = np.asarray(
            [
                0.10,
                4.50,
                -0.45,
                1.95,
            ],
            dtype=np.float64,
        )


        # =====================================================
        # STAGE5C3 POLICY-FACING OBSERVATION
        # =====================================================

        stage5c3_raw_obs = (
            oracle_obs.copy()
        )

        stage5c3_policy_obs = np.clip(
            (
                stage5c3_raw_obs
                - STAGE5C3_OBS_CENTER[
                    None,
                    :
                ]
            )
            / STAGE5C3_OBS_SCALE[
                None,
                :
            ],
            -5.0,
            5.0,
        )


        stage5c3_policy_obs_ok = bool(
            stage5c3_policy_obs.shape
            == (
                n_envs,
                6,
            )
            and np.isfinite(
                stage5c3_policy_obs
            ).all()
            and np.all(
                stage5c3_policy_obs
                >= -5.0
                - 1.0e-12
            )
            and np.all(
                stage5c3_policy_obs
                <= 5.0
                + 1.0e-12
            )
        )


        if not stage5c3_policy_obs_ok:

            raise RuntimeError(
                "Stage5C3 policy observation normalization failed"
            )


        stage5c3_policy_obs_snapshots.append(
            stage5c3_policy_obs.copy()
        )


        # =====================================================
        # EXTERNAL HIGH-LEVEL ACTION INTERFACE
        #
        # During this Stage5C3-B implementation smoke the
        # provider is random, but action acquisition occurs
        # through the same function boundary that PPO/SAC will
        # later replace.
        # =====================================================

        stage5c1_action_normalized = (
            stage5c3_external_action_provider(
                stage5c3_policy_obs,
                episode_idx,
            )
        )


        stage5c1_action_physical = (
            stage5c1_action_lo[
                None,
                :,
            ]
            + 0.5
            * (
                stage5c1_action_normalized
                + 1.0
            )
            * (
                stage5c1_action_hi
                - stage5c1_action_lo
            )[
                None,
                :,
            ]
        )


        # ---------------------------------------------------------

        print(
            "STAGE5C3_ANCHOR_INJECTION_THIS_EPISODE=",
            STAGE5C3_VALIDATION_ANCHOR,
        )


        # =====================================================
        # STAGE5C3-C1
        # NO ACTION ANCHOR INJECTION
        # =====================================================

        stage5c1_anchor_physical = np.asarray(
            [
                0.05,
                3.44068113587798,
                -0.6659088846041841,
                1.4249536866108532,
            ],
            dtype=np.float64,
        )

        print(
            "STAGE5C3_C1_ACTION_ANCHOR_OVERRIDE="
            "DISABLED"
        )


        # ---------------------------------------------------------
        # Stage5C2-B1 privileged physical-oracle observation.
        #
        # [0] physical hit-plane y
        # [1] physical hit-plane z
        # [2] physical hit-plane vx
        # [3] physical hit-plane vy
        # [4] physical hit-plane vz
        # [5] physical flight time from launch to hit plane
        #
        # This is NOT the team predictor.
        # ---------------------------------------------------------

        stage5c1_obs = (
            oracle_obs.copy()
        )

        stage5c2_obs_mapping_ok = bool(
            stage5c1_obs.shape
            == (
                n_envs,
                6,
            )
            and np.isfinite(
                stage5c1_obs
            ).all()
            and np.allclose(
                stage5c1_obs[
                    :,
                    0:2,
                ],
                oracle_pos_l[
                    :,
                    1:3,
                ],
                atol=1.0e-12,
                rtol=0.0,
            )
            and np.allclose(
                stage5c1_obs[
                    :,
                    2:5,
                ],
                oracle_vel_w,
                atol=1.0e-12,
                rtol=0.0,
            )
            and np.allclose(
                stage5c1_obs[
                    :,
                    5,
                ],
                stage5c2_oracle_flight_time_s,
                atol=1.0e-12,
                rtol=0.0,
            )
        )

        print()
        print(
            "=================================================="
        )
        print(
            "STAGE5C1-B1 HIGH-LEVEL INTERFACE"
        )
        print(
            "=================================================="
        )

        print(
            "OBS_SHAPE=",
            stage5c1_obs.shape,
        )

        print(
            "ACTION_NORMALIZED_SHAPE=",
            stage5c1_action_normalized.shape,
        )

        print(
            "ACTION_PHYSICAL_SHAPE=",
            stage5c1_action_physical.shape,
        )


        print(
            "OBS_SEMANTICS="
            "PHYSICAL_ORACLE_HIT_STATE"
        )

        print(
            "OBS_TIME_FIELD="
            "ORACLE_FLIGHT_TIME_S"
        )

        print(
            "OBS_PRIVILEGED_ORACLE=TRUE"
        )

        print(
            "OBS_TEAM_PREDICTOR=FALSE"
        )

        print(
            "ACTION_SAMPLED_ONCE_PER_BALL=TRUE"
        )

        print(
            "ACTION_HELD_FOR_COMPLETE_STRIKE=TRUE"
        )

        print(
            "MODEL21800_FROZEN=TRUE"
        )

        print(
            "ACTION_PHYSICAL_MIN=",
            np.min(
                stage5c1_action_physical,
                axis=0,
            ).tolist(),
        )

        print(
            "ACTION_PHYSICAL_MAX=",
            np.max(
                stage5c1_action_physical,
                axis=0,
            ).tolist(),
        )

        for stage5c1_env_id in range(
            min(
                n_envs,
                8,
            )
        ):

            print(
                f"ACTION_ENV_{stage5c1_env_id}="
                f"{stage5c1_action_physical[stage5c1_env_id].tolist()}"
            )


        # ---------------------------------------------------------
        # Stage5C2-B1 physical-oracle target command.
        #
        # target_x = oracle_hit_x
        # target_y = oracle_hit_y + dy
        # target_z = oracle_hit_z
        # ---------------------------------------------------------

        stage5c2_target_pos_l = (
            oracle_pos_l.copy()
        )

        stage5c2_target_pos_l[
            :,
            1,
        ] += (
            stage5c1_action_physical[
                :,
                0,
            ]
        )

        target_pos_w = (
            env_origins_np
            + stage5c2_target_pos_l
        )


        stage5c2_target_mapping_ok = bool(
            target_pos_w.shape
            == (
                n_envs,
                3,
            )
            and np.isfinite(
                target_pos_w
            ).all()
            and np.allclose(
                stage5c2_target_pos_l[
                    :,
                    0,
                ],
                oracle_pos_l[
                    :,
                    0,
                ],
                atol=1.0e-12,
                rtol=0.0,
            )
            and np.allclose(
                stage5c2_target_pos_l[
                    :,
                    1,
                ],
                oracle_pos_l[
                    :,
                    1,
                ]
                + stage5c1_action_physical[
                    :,
                    0,
                ],
                atol=1.0e-12,
                rtol=0.0,
            )
            and np.allclose(
                stage5c2_target_pos_l[
                    :,
                    2,
                ],
                oracle_pos_l[
                    :,
                    2,
                ],
                atol=1.0e-12,
                rtol=0.0,
            )
        )


        # env0 retains the nominal incoming + frozen Stage4C
        # winning action, so its resulting target should reproduce
        # the known fixed-ball target.
        stage5c2_nominal_target_reproduction = bool(
            np.allclose(
                stage5c2_target_pos_l[
                    0
                ],
                COMMAND_TARGET_POS_L,
                atol=1.0e-5,
                rtol=0.0,
            )
        )


        print(
            "B1_TARGET_MAPPING_OK=",
            stage5c2_target_mapping_ok,
        )

        print(
            "B1_NOMINAL_TARGET_REPRODUCTION=",
            stage5c2_nominal_target_reproduction,
        )

        print(
            "B1_ENV0_TARGET_POS_L=",
            stage5c2_target_pos_l[
                0
            ].tolist(),
        )


        # Per-environment persistent executor state.
        # -------------------------------------------------------------

        last_action = np.zeros(
            (
                n_envs,
                31,
            ),
            dtype=np.float32,
        )


        b2_last_action_reset_ok = bool(
            last_action.shape == (n_envs, 31)
            and np.count_nonzero(last_action) == 0
        )



        total_clamps = np.zeros(
            n_envs,
            dtype=np.int64,
        )


        max_tilt_deg = np.zeros(
            n_envs,
            dtype=np.float64,
        )


        max_base_drift_m = np.zeros(
            n_envs,
            dtype=np.float64,
        )


        min_root_z = np.full(
            n_envs,
            np.inf,
            dtype=np.float64,
        )


        finite_ok = True


        # -------------------------------------------------------------
        # Stage5B2C physical-ball state.
        # -------------------------------------------------------------

        launched = False

        actual_launch_time_s = None


        # Stage5C2-B1 true per-environment launch state.
        stage5c2_launched = np.zeros(
            n_envs,
            dtype=bool,
        )

        stage5c2_actual_launch_time_s = np.full(
            n_envs,
            np.nan,
            dtype=np.float64,
        )


        first_contact = np.zeros(
            n_envs,
            dtype=bool,
        )

        net_cross_seen = np.zeros(
            n_envs,
            dtype=bool,
        )

        outgoing_cross_net = np.zeros(
            n_envs,
            dtype=bool,
        )

        net_cross_z = np.full(
            n_envs,
            np.nan,
            dtype=np.float64,
        )

        opponent_bounce = np.zeros(
            n_envs,
            dtype=bool,
        )

        opponent_bounce_pos_l = np.full(
            (
                n_envs,
                3,
            ),
            np.nan,
            dtype=np.float64,
        )

        max_contact_force = np.zeros(
            n_envs,
            dtype=np.float64,
        )

        min_racket_ball_distance = np.full(
            n_envs,
            np.inf,
            dtype=np.float64,
        )

        previous_ball_pos_l = np.repeat(
            BALL_PARK_POS_L[
                None,
                :,
            ],
            n_envs,
            axis=0,
        ).astype(
            np.float64,
            copy=True,
        )

        previous_ball_vel_w = np.zeros(
            (
                n_envs,
                3,
            ),
            dtype=np.float64,
        )


        print(
            "ball_init_pos_l =",
            BALL_INIT_POS_L.tolist(),
        )

        print(
            "ball_init_vel_w =",
            BALL_INIT_VEL_W.tolist(),
        )

        print(
            "calibrated_cross_time_s =",
            CALIBRATED_CROSS_TIME_S,
        )

        print(
            "ball_launch_delay_s =",
            LAUNCH_DELAY_S,
        )

        print(
            "command_target_pos_l =",
            COMMAND_TARGET_POS_L.tolist(),
        )

        print(
            "command_target_vel_w =",
            RACKET_TARGET_VEL_W.tolist(),
        )


        # -------------------------------------------------------------
        # Persistent 50-Hz executor.
        # -------------------------------------------------------------



        # =========================================================
        # STAGE5C1-B2 PRE-SHOT RESET AUDIT
        # =========================================================

        b2_metrics_reset_ok = bool(
            launched is False
            and actual_launch_time_s is None
            and not np.any(first_contact)
            and not np.any(net_cross_seen)
            and not np.any(outgoing_cross_net)
            and not np.any(opponent_bounce)
            and np.all(max_contact_force == 0.0)
            and np.all(np.isinf(min_racket_ball_distance))
            and np.all(total_clamps == 0)
            and finite_ok
        )


        b2_ball_pos_l_pre = (
            ball.data.root_pos_w
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=True)
            - env_origins_np
        )

        b2_ball_vel_pre = (
            ball.data.root_lin_vel_w
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=True)
        )


        b2_ball_park_position_error_m = float(
            np.max(
                np.abs(
                    b2_ball_pos_l_pre
                    - BALL_PARK_POS_L[None, :]
                )
            )
        )

        b2_ball_park_velocity_max_mps = float(
            np.max(
                np.abs(
                    b2_ball_vel_pre
                )
            )
        )

        b2_ball_park_ok = bool(
            b2_ball_park_position_error_m < 1.0e-5
            and b2_ball_park_velocity_max_mps < 1.0e-5
        )


        print(
            "B2_PRE_SHOT_LAST_ACTION_RESET=",
            b2_last_action_reset_ok,
        )

        print(
            "B2_PRE_SHOT_METRICS_RESET=",
            b2_metrics_reset_ok,
        )

        print(
            "B2_PRE_SHOT_BALL_PARK_POSITION_ERROR_M=",
            b2_ball_park_position_error_m,
        )

        print(
            "B2_PRE_SHOT_BALL_PARK_VELOCITY_MAX_MPS=",
            b2_ball_park_velocity_max_mps,
        )

        print(
            "B2_PRE_SHOT_BALL_PARK=",
            b2_ball_park_ok,
        )

        control_steps = int(
            round(
                float(
                    args_cli.duration
                )
                / CONTROL_DT
            )
        )


        tick_times_ms = []


        for tick in range(
            control_steps
        ):

            torch.cuda.synchronize()

            tick_t0 = (
                time.perf_counter()
            )


            strike_elapsed = (
                tick
                * CONTROL_DT
            )


            # One whole-batch state transfer.
            packed = (
                robot_state_batch_to_numpy(
                    robot,
                    joint_ids,
                )
            )


            raw_action = np.zeros(
                (
                    n_envs,
                    31,
                ),
                dtype=np.float32,
            )


            q_des_all = np.zeros(
                (
                    n_envs,
                    31,
                ),
                dtype=np.float64,
            )


            # ---------------------------------------------------------
            # Same high-level command in every local environment.
            #
            # Each environment nevertheless has:
            #   * its own RobotState
            #   * its own SwingLifecycle
            #   * its own last_action
            #
            # All environments share:
            #   * one immutable ActionAdapter
            #   * one ORT OnnxPolicy session
            # ---------------------------------------------------------

            for env_id in range(
                n_envs
            ):

                state = (
                    unpack_robot_state(
                        packed,
                        env_id,
                    )
                )


                remaining_tts = max(
                    LEAD_TIME
                    - strike_elapsed,
                    0.0,
                )


                if not first_contact[
                    env_id
                ]:

                    cmd = RacketCommand(
                        task_id=1,
                        task_revision=tick,
                        swing_sign=1,
                        position=target_pos_w[
                            env_id
                        ],
                        velocity=stage5c1_action_physical[env_id, 1:4],
                        time_to_strike=remaining_tts,
                    )

                else:

                    cmd = None


                target = (
                    lifecycles[
                        env_id
                    ].update(
                        cmd,
                        state,
                    )
                )


                obs = build_observation(
                    state=state,
                    target=target,
                    last_action=last_action[
                        env_id
                    ],
                    default_q=adapter.default_q,
                    base_target_xy=base_target_xy[
                        env_id
                    ],
                )


                if (
                    obs.shape
                    != (
                        110,
                    )
                    or not np.isfinite(
                        obs
                    ).all()
                ):

                    raise RuntimeError(
                        f"Invalid executor observation "
                        f"in env {env_id}"
                    )


                # Stage-4 stability diagnostics,
                # evaluated from the exact deploy observation.
                gravity_b = (
                    obs[
                        96:99
                    ]
                )


                upright_cos = float(
                    np.clip(
                        -gravity_b[2],
                        -1.0,
                        1.0,
                    )
                )


                tilt_deg = (
                    math.degrees(
                        math.acos(
                            upright_cos
                        )
                    )
                )


                drift_m = float(
                    np.linalg.norm(
                        state.base_pos_w[
                            :2
                        ]
                        - initial_base_xy[
                            env_id
                        ]
                    )
                )


                max_tilt_deg[
                    env_id
                ] = max(
                    max_tilt_deg[
                        env_id
                    ],
                    tilt_deg,
                )


                max_base_drift_m[
                    env_id
                ] = max(
                    max_base_drift_m[
                        env_id
                    ],
                    drift_m,
                )


                min_root_z[
                    env_id
                ] = min(
                    min_root_z[
                        env_id
                    ],
                    float(
                        state.base_pos_w[
                            2
                        ]
                    ),
                )


                action_i = (
                    policy.infer_target(
                        obs,
                        target.time_to_strike,
                        lifecycles[
                            env_id
                        ].swing_sign,
                        CONTROL_DT,
                    )
                )


                action_i = np.asarray(
                    action_i,
                    dtype=np.float32,
                ).reshape(
                    31
                )


                if not np.isfinite(
                    action_i
                ).all():

                    raise RuntimeError(
                        f"Non-finite action "
                        f"in env {env_id}"
                    )


                # Exact passive-head contract.
                action_i[
                    3
                ] = 0.0

                action_i[
                    4
                ] = 0.0


                raw_action[
                    env_id
                ] = action_i


                unclamped = (
                    adapter.default_q
                    + action_i.astype(
                        np.float64
                    )
                    * adapter.action_scale
                )


                q_des = (
                    np.clip(
                        unclamped,
                        adapter.clamp_lower,
                        adapter.clamp_upper,
                    )
                )


                total_clamps[
                    env_id
                ] += int(
                    np.count_nonzero(
                        np.abs(
                            q_des
                            - unclamped
                        )
                        > 1.0e-10
                    )
                )


                # Passive head position = deploy default.
                q_des[3] = (
                    adapter.default_q[
                        3
                    ]
                )

                q_des[4] = (
                    adapter.default_q[
                        4
                    ]
                )


                q_des_all[
                    env_id
                ] = q_des


            finite_ok = bool(
                finite_ok
                and np.isfinite(
                    q_des_all
                ).all()
            )


            # One whole-batch CPU -> GPU action write.
            q_des_t = torch.as_tensor(
                q_des_all,
                device=device,
                dtype=dtype,
            )


            robot.set_joint_position_target(
                q_des_t,
                joint_ids=joint_ids,
            )


            # Exact NEXT-observation action history.
            last_action = (
                raw_action.copy()
            )


            # ---------------------------------------------------------
            # Eight 400-Hz physics substeps.
            #
            # Balls remain parked intentionally:
            # this stage isolates executor integration.
            # ---------------------------------------------------------

            for substep in range(
                DECIMATION
            ):

                physics_time = (
                    tick
                    * CONTROL_DT
                    + substep
                    * PHYSICS_DT
                )



                # -----------------------------------------------------
                # Stage5C2-B1 per-environment launch scheduling.
                #
                # Launch is checked at substep==0 to retain the
                # frozen Stage4/Stage5C1 50-Hz launch-boundary
                # semantics.  Different incoming flight times
                # therefore map to different control-boundary
                # launch times.
                # -----------------------------------------------------

                if substep == 0:

                    stage5c2_due_mask = (
                        ~stage5c2_launched
                        & (
                            strike_elapsed
                            >= stage5c2_launch_delay_s
                            - 1.0e-9
                        )
                    )

                    stage5c2_due_ids = np.flatnonzero(
                        stage5c2_due_mask
                    )


                    if stage5c2_due_ids.size > 0:

                        stage5c2_write_selected_ball_states(
                            stage5c2_ball_pos_l,
                            stage5c2_ball_vel_w,
                            stage5c2_ball_spin_w,
                            stage5c2_due_ids,
                        )


                        stage5c2_launched[
                            stage5c2_due_ids
                        ] = True

                        stage5c2_actual_launch_time_s[
                            stage5c2_due_ids
                        ] = float(
                            strike_elapsed
                        )


                        # Outcome interpolation must start from
                        # the actual launch state for newly
                        # launched environments.
                        previous_ball_pos_l[
                            stage5c2_due_ids
                        ] = stage5c2_ball_pos_l[
                            stage5c2_due_ids
                        ]

                        previous_ball_vel_w[
                            stage5c2_due_ids
                        ] = stage5c2_ball_vel_w[
                            stage5c2_due_ids
                        ]


                        if (
                            0 in stage5c2_due_ids
                            and actual_launch_time_s is None
                        ):
                            actual_launch_time_s = float(
                                strike_elapsed
                            )


                        print(
                            "B1_BALL_LAUNCH "
                            f"t={strike_elapsed:.6f}s "
                            f"env_ids={stage5c2_due_ids.tolist()}"
                        )


                # Hold only those balls that have NOT yet launched.
                # Already-launched environments are left untouched.
                stage5c2_waiting_ids = np.flatnonzero(
                    ~stage5c2_launched
                )


                if stage5c2_waiting_ids.size > 0:

                    stage5c2_write_selected_ball_states(
                        stage5c2_park_pos_l_batch,
                        stage5c2_zero_vel_batch,
                        stage5c2_zero_vel_batch,
                        stage5c2_waiting_ids,
                    )


                # Backward-compatible aggregate flag used by the
                # inherited outcome/smoke checks.
                launched = bool(
                    np.all(
                        stage5c2_launched
                    )
                )

                scene.write_data_to_sim()

                sim.step(
                    render=False
                )

                scene.update(
                    PHYSICS_DT
                )


                # -----------------------------------------------------
                # Vector physical diagnostics at 400 Hz.
                # -----------------------------------------------------

                ball_pos_w_t = (
                    ball.data.root_pos_w
                )

                ball_vel_w_t = (
                    ball.data.root_lin_vel_w
                )


                (
                    racket_pos_w_t,
                    racket_vel_w_t,
                    racket_normal_w_t,
                ) = racket_state_batch(
                    robot,
                    wrist_body_id,
                    mount_offset,
                )


                rel_t = (
                    ball_pos_w_t
                    - racket_pos_w_t
                )

                signed_normal_t = (
                    rel_t
                    * racket_normal_w_t
                ).sum(
                    dim=-1
                )

                lateral_t = (
                    rel_t
                    - signed_normal_t.unsqueeze(
                        -1
                    )
                    * racket_normal_w_t
                )


                center_distance_t = (
                    torch.linalg.vector_norm(
                        rel_t,
                        dim=-1,
                    )
                )

                lateral_distance_t = (
                    torch.linalg.vector_norm(
                        lateral_t,
                        dim=-1,
                    )
                )


                force_raw = (
                    sensor.data.net_forces_w
                )

                force_env_t = (
                    torch.linalg.vector_norm(
                        force_raw,
                        dim=-1,
                    )
                    .reshape(
                        n_envs,
                        -1,
                    )
                    .amax(
                        dim=1
                    )
                )


                face_close_t = (
                    (
                        lateral_distance_t
                        < FACE_LATERAL_THRESHOLD
                    )
                    & (
                        torch.abs(
                            signed_normal_t
                        )
                        < FACE_NORMAL_THRESHOLD
                    )
                )


                physical_contact_t = (
                    face_close_t
                    & (
                        force_env_t
                        > CONTACT_FORCE_THRESHOLD
                    )
                )


                ball_pos_l = (
                    ball_pos_w_t
                    - scene.env_origins
                )


                ball_pos_l_np = (
                    ball_pos_l
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(
                        np.float64,
                        copy=True,
                    )
                )

                ball_vel_w_np = (
                    ball_vel_w_t
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(
                        np.float64,
                        copy=True,
                    )
                )

                physical_contact_np = (
                    physical_contact_t
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(
                        bool,
                        copy=True,
                    )
                )

                force_env_np = (
                    force_env_t
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(
                        np.float64,
                        copy=True,
                    )
                )

                center_distance_np = (
                    center_distance_t
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(
                        np.float64,
                        copy=True,
                    )
                )


                max_contact_force = np.maximum(
                    max_contact_force,
                    force_env_np,
                )

                min_racket_ball_distance = np.minimum(
                    min_racket_ball_distance,
                    center_distance_np,
                )


                if launched:

                    new_contact = (
                        physical_contact_np
                        & ~first_contact
                    )

                    first_contact[
                        new_contact
                    ] = True


                    # -------------------------------------------------
                    # Outgoing net crossing after first racket contact.
                    # -------------------------------------------------

                    prev_x = (
                        previous_ball_pos_l[
                            :,
                            0,
                        ]
                    )

                    curr_x = (
                        ball_pos_l_np[
                            :,
                            0,
                        ]
                    )


                    crossing = (
                        first_contact
                        & ~net_cross_seen
                        & (
                            prev_x
                            < geometry.NET_X
                        )
                        & (
                            curr_x
                            >= geometry.NET_X
                        )
                        & (
                            ball_vel_w_np[
                                :,
                                0,
                            ]
                            > 0.0
                        )
                    )


                    ids = np.flatnonzero(
                        crossing
                    )


                    for env_id in ids:

                        dx = (
                            curr_x[
                                env_id
                            ]
                            - prev_x[
                                env_id
                            ]
                        )

                        alpha = (
                            geometry.NET_X
                            - prev_x[
                                env_id
                            ]
                        ) / max(
                            dx,
                            1.0e-9,
                        )

                        z = (
                            previous_ball_pos_l[
                                env_id,
                                2,
                            ]
                            + alpha
                            * (
                                ball_pos_l_np[
                                    env_id,
                                    2,
                                ]
                                - previous_ball_pos_l[
                                    env_id,
                                    2,
                                ]
                            )
                        )

                        net_cross_z[
                            env_id
                        ] = z

                        net_cross_seen[
                            env_id
                        ] = True

                        outgoing_cross_net[
                            env_id
                        ] = bool(
                            z
                            > (
                                geometry.NET_HEIGHT
                                + geometry.BALL_RADIUS
                            )
                        )


                    # -------------------------------------------------
                    # Strict opponent-table physical bounce.
                    # Same state criterion used by Stage 4.
                    # -------------------------------------------------

                    inside_opponent = (
                        (
                            ball_pos_l_np[
                                :,
                                0,
                            ]
                            >= 1.40
                        )
                        & (
                            ball_pos_l_np[
                                :,
                                0,
                            ]
                            <= 2.71
                        )
                        & (
                            ball_pos_l_np[
                                :,
                                1,
                            ]
                            >= -1.495
                        )
                        & (
                            ball_pos_l_np[
                                :,
                                1,
                            ]
                            <= -0.03
                        )
                    )


                    near_surface = (
                        (
                            ball_pos_l_np[
                                :,
                                2,
                            ]
                            >= (
                                0.02
                                - 0.025
                            )
                        )
                        & (
                            ball_pos_l_np[
                                :,
                                2,
                            ]
                            <= (
                                0.02
                                + 0.040
                            )
                        )
                    )


                    returned_forward = (
                        ball_vel_w_np[
                            :,
                            0,
                        ]
                        > 0.2
                    )

                    upward_after_bounce = (
                        ball_vel_w_np[
                            :,
                            2,
                        ]
                        > 0.05
                    )


                    bounce_now = (
                        first_contact
                        & ~opponent_bounce
                        & inside_opponent
                        & near_surface
                        & returned_forward
                        & upward_after_bounce
                    )


                    opponent_bounce[
                        bounce_now
                    ] = True

                    opponent_bounce_pos_l[
                        bounce_now
                    ] = ball_pos_l_np[
                        bounce_now
                    ]


                previous_ball_pos_l = (
                    ball_pos_l_np
                )

                previous_ball_vel_w = (
                    ball_vel_w_np
                )


            torch.cuda.synchronize()


            tick_times_ms.append(
                (
                    time.perf_counter()
                    - tick_t0
                )
                * 1000.0
            )


        # -------------------------------------------------------------
        # Final state / parity diagnostics.
        # -------------------------------------------------------------

        final_packed = (
            robot_state_batch_to_numpy(
                robot,
                joint_ids,
            )
        )


        final_root_w = (
            final_packed[
                :,
                0:3,
            ]
        )


        final_root_l = (
            final_root_w
            - env_origins_np
        )


        # Spread across otherwise-identical environments.
        base_local_spread = float(
            np.max(
                np.ptp(
                    final_root_l,
                    axis=0,
                )
            )
        )


        action_spread = float(
            np.max(
                np.ptp(
                    last_action,
                    axis=0,
                )
            )
        )


        phases = [
            phase_name(x)
            for x in lifecycles
        ]


        phase_counts = {
            name: phases.count(
                name
            )
            for name in sorted(
                set(
                    phases
                )
            )
        }


        tick_mean = statistics.mean(
            tick_times_ms
        )

        tick_p95 = percentile(
            tick_times_ms,
            0.95,
        )


        print()
        print(
            "=================================================="
        )
        print(
            "STAGE 5B2B RESULTS"
        )
        print(
            "=================================================="
        )


        print(
            "control_ticks =",
            control_steps,
        )


        print(
            "full_tick_mean_ms =",
            tick_mean,
        )


        print(
            "full_tick_p95_ms =",
            tick_p95,
        )


        print(
            "full_tick_20ms_budget_pct =",
            tick_mean
            / 20.0
            * 100.0,
        )


        print(
            "finite =",
            finite_ok,
        )


        print(
            "phase_counts =",
            phase_counts,
        )


        print(
            "min_root_z_all =",
            float(
                np.min(
                    min_root_z
                )
            ),
        )


        print(
            "max_tilt_deg_all =",
            float(
                np.max(
                    max_tilt_deg
                )
            ),
        )


        print(
            "max_base_drift_m_all =",
            float(
                np.max(
                    max_base_drift_m
                )
            ),
        )


        print(
            "max_final_local_base_spread_m =",
            base_local_spread,
        )


        print(
            "max_final_action_spread =",
            action_spread,
        )


        print(
            "clamp_events_per_env =",
            total_clamps.tolist(),
        )


        print(
            "unique_clamp_counts =",
            sorted(
                set(
                    total_clamps.tolist()
                )
            ),
        )



        # -------------------------------------------------------------
        # Stage5B2C physical return metrics.
        # -------------------------------------------------------------

        legal_return = (
            first_contact
            & outgoing_cross_net
            & opponent_bounce
        )


        high_margin = np.zeros(
            n_envs,
            dtype=bool,
        )


        valid_bounce = np.flatnonzero(
            opponent_bounce
        )


        for env_id in valid_bounce:

            bx = opponent_bounce_pos_l[
                env_id,
                0,
            ]

            by = opponent_bounce_pos_l[
                env_id,
                1,
            ]

            high_margin[
                env_id
            ] = bool(
                legal_return[
                    env_id
                ]
                and HIGH_MARGIN_X_LO
                <= bx
                <= HIGH_MARGIN_X_HI
                and HIGH_MARGIN_Y_LO
                <= by
                <= HIGH_MARGIN_Y_HI
            )


        contact_count = int(
            np.count_nonzero(
                first_contact
            )
        )

        cross_count = int(
            np.count_nonzero(
                outgoing_cross_net
            )
        )

        landing_count = int(
            np.count_nonzero(
                opponent_bounce
            )
        )

        legal_count = int(
            np.count_nonzero(
                legal_return
            )
        )

        high_margin_count = int(
            np.count_nonzero(
                high_margin
            )
        )


        # =========================================================
        # STAGE5C1-B1 REWARD SMOKE
        #
        # This is deliberately simple and hierarchical:
        #
        # miss       = 0
        # contact    adds 1
        # cross-net  adds 2
        # legal      adds 6
        # quality    [0,1], ONLY after legal
        #
        # Dynamic-stability penalty is deliberately deferred to
        # Stage5C1-B2 where reset/stability state is tested across
        # repeated episodes.
        # =========================================================

        stage5c1_landing_center = np.asarray(
            [
                2.05,
                -0.75,
            ],
            dtype=np.float64,
        )

        stage5c1_landing_error = np.full(
            n_envs,
            np.nan,
            dtype=np.float64,
        )

        stage5c1_quality_bonus = np.zeros(
            n_envs,
            dtype=np.float64,
        )


        for stage5c1_env_id in np.flatnonzero(
            legal_return
        ):

            stage5c1_bounce_xy = (
                opponent_bounce_pos_l[
                    stage5c1_env_id,
                    :2,
                ]
            )

            stage5c1_err = float(
                np.linalg.norm(
                    stage5c1_bounce_xy
                    - stage5c1_landing_center
                )
            )

            stage5c1_landing_error[
                stage5c1_env_id
            ] = stage5c1_err

            stage5c1_quality_bonus[
                stage5c1_env_id
            ] = float(
                np.exp(
                    -(
                        stage5c1_err
                        / 0.35
                    ) ** 2
                )
            )



        # =====================================================
        # STAGE5C3 TRAIN REWARD V1
        # =====================================================

        stage5c3_min_root_clearance_per_env = (
            min_root_z
            - float(
                geometry.FLOOR_Z
            )
        )


        stage5c3_unstable = (
            ~np.isfinite(
                stage5c3_min_root_clearance_per_env
            )
            | ~np.isfinite(
                max_tilt_deg
            )
            | ~np.isfinite(
                max_base_drift_m
            )
            | (
                stage5c3_min_root_clearance_per_env
                <= 0.75
            )
            | (
                max_tilt_deg
                >= 35.0
            )
            | (
                max_base_drift_m
                >= 0.30
            )
        )


        if not finite_ok:

            stage5c3_unstable[:] = True


        stage5c3_unstable_snapshots.append(
            stage5c3_unstable.copy()
        )


        stage5c1_reward = (
            first_contact.astype(
                np.float64
            )
            + 2.0
            * outgoing_cross_net.astype(
                np.float64
            )
            + 6.0
            * legal_return.astype(
                np.float64
            )
            + stage5c1_quality_bonus
            - 4.0
            * stage5c3_unstable.astype(
                np.float64
            )
        )


        # One high-level step is one complete strike.
        stage5c3_terminated = np.ones(
            n_envs,
            dtype=bool,
        )

        stage5c3_truncated = np.zeros(
            n_envs,
            dtype=bool,
        )


        stage5c3_terminated_snapshots.append(
            stage5c3_terminated.copy()
        )

        stage5c3_truncated_snapshots.append(
            stage5c3_truncated.copy()
        )


        print(
            "STAGE5C3_TERMINATED_COUNT=",
            f"{int(np.sum(stage5c3_terminated))}/{n_envs}",
        )

        print(
            "STAGE5C3_TRUNCATED_COUNT=",
            f"{int(np.sum(stage5c3_truncated))}/{n_envs}",
        )

        print(
            "STAGE5C3_UNSTABLE_COUNT=",
            f"{int(np.sum(stage5c3_unstable))}/{n_envs}",
        )



        # ---------------------------------------------------------
        # Contract checks.
        # ---------------------------------------------------------

        stage5c1_obs_shape_ok = (
            stage5c1_obs.shape
            == (
                n_envs,
                6,
            )
        )

        stage5c1_action_shape_ok = (
            stage5c1_action_physical.shape
            == (
                n_envs,
                4,
            )
            and stage5c1_action_normalized.shape
            == (
                n_envs,
                4,
            )
        )

        stage5c1_finite_ok = bool(
            np.isfinite(
                stage5c1_obs
            ).all()
            and np.isfinite(
                stage5c1_action_normalized
            ).all()
            and np.isfinite(
                stage5c1_action_physical
            ).all()
            and np.isfinite(
                stage5c1_reward
            ).all()
        )

        stage5c1_bounds_ok = bool(
            np.all(
                stage5c1_action_normalized
                >= -1.0
                - 1.0e-12
            )
            and np.all(
                stage5c1_action_normalized
                <= 1.0
                + 1.0e-12
            )
            and np.all(
                stage5c1_action_physical
                >= stage5c1_action_lo[
                    None,
                    :
                ]
                - 1.0e-12
            )
            and np.all(
                stage5c1_action_physical
                <= stage5c1_action_hi[
                    None,
                    :
                ]
                + 1.0e-12
            )
        )


        stage5c1_roundtrip = (
            stage5c1_action_lo[
                None,
                :
            ]
            + 0.5
            * (
                stage5c1_action_normalized
                + 1.0
            )
            * (
                stage5c1_action_hi
                - stage5c1_action_lo
            )[
                None,
                :
            ]
        )

        stage5c1_mapping_ok = bool(
            np.allclose(
                stage5c1_roundtrip,
                stage5c1_action_physical,
                atol=1.0e-10,
                rtol=0.0,
            )
        )


        if n_envs > 1:

            stage5c1_action_diversity = float(
                np.max(
                    np.ptp(
                        stage5c1_action_physical,
                        axis=0,
                    )
                )
            )

            stage5c1_diversity_ok = bool(
                stage5c1_action_diversity
                > 1.0e-6
            )

        else:

            stage5c1_action_diversity = 0.0
            stage5c1_diversity_ok = True


        stage5c1_anchor_ok = bool(
            np.allclose(
                stage5c1_action_physical[
                    0
                ],
                stage5c1_anchor_physical,
                atol=1.0e-12,
                rtol=0.0,
            )
        )


        stage5c1_outcome_logic_ok = bool(
            np.all(
                ~legal_return
                | (
                    first_contact
                    & outgoing_cross_net
                    & opponent_bounce
                )
            )
            and np.all(
                ~high_margin
                | legal_return
            )
        )


        # Positive-control env_0 uses the frozen Stage4C winning
        # command, so it should still achieve a legal return.
        stage5c1_anchor_legal = bool(
            legal_return[
                0
            ]
        )


        stage5c1_smoke_gate = bool(
            launched
            and stage5c1_obs_shape_ok
            and stage5c1_action_shape_ok
            and stage5c1_finite_ok
            and stage5c1_bounds_ok
            and stage5c1_mapping_ok
            and stage5c1_diversity_ok
            and stage5c1_outcome_logic_ok
        )


        print()
        print(
            "=================================================="
        )
        print(
            "STAGE5C1-B1 RANDOM ACTION / REWARD SMOKE"
        )
        print(
            "=================================================="
        )

        print(
            "obs_shape_ok =",
            stage5c1_obs_shape_ok,
        )

        print(
            "action_shape_ok =",
            stage5c1_action_shape_ok,
        )

        print(
            "finite_ok =",
            stage5c1_finite_ok,
        )

        print(
            "action_bounds_ok =",
            stage5c1_bounds_ok,
        )

        print(
            "normalized_physical_mapping_ok =",
            stage5c1_mapping_ok,
        )

        print(
            "action_diversity =",
            stage5c1_action_diversity,
        )

        print(
            "action_diversity_ok =",
            stage5c1_diversity_ok,
        )

        print(
            "anchor_action_ok =",
            stage5c1_anchor_ok,
        )

        print(
            "anchor_env0_legal =",
            stage5c1_anchor_legal,
        )

        print(
            "outcome_logic_ok =",
            stage5c1_outcome_logic_ok,
        )

        print(
            "random_contact_count =",
            f"{contact_count}/{n_envs}",
        )

        print(
            "random_cross_net_count =",
            f"{cross_count}/{n_envs}",
        )

        print(
            "random_landing_count =",
            f"{landing_count}/{n_envs}",
        )

        print(
            "random_legal_count =",
            f"{legal_count}/{n_envs}",
        )

        print(
            "random_high_margin_count =",
            f"{high_margin_count}/{n_envs}",
        )

        print(
            "reward_min =",
            float(
                np.min(
                    stage5c1_reward
                )
            ),
        )

        print(
            "reward_mean =",
            float(
                np.mean(
                    stage5c1_reward
                )
            ),
        )

        print(
            "reward_max =",
            float(
                np.max(
                    stage5c1_reward
                )
            ),
        )

        print(
            "reward_finite =",
            bool(
                np.isfinite(
                    stage5c1_reward
                ).all()
            ),
        )

        print(
            "LANDING_QUALITY_ONLY_AFTER_LEGAL=TRUE"
        )

        print(
            "STABILITY_PENALTY_INCLUDED="
            "FALSE_B1_SMOKE_ONLY"
        )


        print(
            "FINAL_ORACLE_HIT_VELOCITY_INCLUDED="
            "TRUE_STAGE5C2_B1"
        )

        print(
            "STAGE5C1B1_RANDOM_ACTION_SMOKE=",
            "PASS"
            if stage5c1_smoke_gate
            else "FAIL",
        )



        print()
        print(
            "=================================================="
        )

        print(
            "STAGE 5B2C PHYSICAL OUTCOMES"
        )

        print(
            "=================================================="
        )


        print(
            "launched =",
            launched,
        )

        print(
            "actual_launch_time_s =",
            actual_launch_time_s,
        )

        print(
            "expected_stage4_launch_boundary_s =",
            0.760000,
        )

        launch_timing_parity = bool(
            actual_launch_time_s is not None
            and abs(
                actual_launch_time_s
                - 0.760000
            )
            < 1.0e-9
        )

        print(
            "STAGE4_FIXED_BALL_LAUNCH_TIMING_PARITY_DIAGNOSTIC=",
            "PASS"
            if launch_timing_parity
            else "FAIL",
        )


        stage5c2_all_launched = bool(
            np.all(
                stage5c2_launched
            )
        )

        stage5c2_launch_times_finite = bool(
            np.isfinite(
                stage5c2_actual_launch_time_s
            ).all()
        )


        if stage5c2_launch_times_finite:

            stage5c2_physical_arrival_time_s = (
                stage5c2_actual_launch_time_s
                + stage5c2_oracle_flight_time_s
            )

            stage5c2_arrival_alignment_error_s = (
                stage5c2_physical_arrival_time_s
                - LEAD_TIME
            )

            stage5c2_max_arrival_alignment_error_s = float(
                np.max(
                    np.abs(
                        stage5c2_arrival_alignment_error_s
                    )
                )
            )

        else:

            stage5c2_physical_arrival_time_s = np.full(
                n_envs,
                np.nan,
                dtype=np.float64,
            )

            stage5c2_arrival_alignment_error_s = np.full(
                n_envs,
                np.nan,
                dtype=np.float64,
            )

            stage5c2_max_arrival_alignment_error_s = float(
                "inf"
            )


        # Because launch is intentionally quantized to the same
        # 50-Hz boundary semantics as Stage4/Stage5C1, alignment
        # error may be up to one CONTROL_DT.
        stage5c2_launch_alignment_ok = bool(
            stage5c2_all_launched
            and stage5c2_launch_times_finite
            and stage5c2_max_arrival_alignment_error_s
            <= CONTROL_DT
            + 1.0e-9
        )


        print(
            "B1_ALL_ENVS_LAUNCHED=",
            stage5c2_all_launched,
        )

        print(
            "B1_ACTUAL_LAUNCH_TIME_RANGE_S=",
            [
                float(
                    np.min(
                        stage5c2_actual_launch_time_s
                    )
                ),
                float(
                    np.max(
                        stage5c2_actual_launch_time_s
                    )
                ),
            ]
            if stage5c2_launch_times_finite
            else "NONFINITE",
        )

        print(
            "B1_PHYSICAL_ARRIVAL_TIME_RANGE_S=",
            [
                float(
                    np.min(
                        stage5c2_physical_arrival_time_s
                    )
                ),
                float(
                    np.max(
                        stage5c2_physical_arrival_time_s
                    )
                ),
            ]
            if stage5c2_launch_times_finite
            else "NONFINITE",
        )

        print(
            "B1_MAX_ARRIVAL_ALIGNMENT_ERROR_S=",
            stage5c2_max_arrival_alignment_error_s,
        )

        print(
            "B1_LAUNCH_ALIGNMENT_TOLERANCE_S=",
            CONTROL_DT,
        )

        print(
            "B1_PER_ENV_LAUNCH_ALIGNMENT=",
            "PASS"
            if stage5c2_launch_alignment_ok
            else "FAIL",
        )


        print(
            "contact_count =",
            f"{contact_count}/{n_envs}",
        )

        print(
            "cross_net_count =",
            f"{cross_count}/{n_envs}",
        )

        print(
            "opponent_landing_count =",
            f"{landing_count}/{n_envs}",
        )

        print(
            "legal_return_count =",
            f"{legal_count}/{n_envs}",
        )

        print(
            "high_margin_count =",
            f"{high_margin_count}/{n_envs}",
        )


        print(
            "max_contact_force_range_N =",
            [
                float(
                    np.min(
                        max_contact_force
                    )
                ),
                float(
                    np.max(
                        max_contact_force
                    )
                ),
            ],
        )


        print(
            "min_racket_ball_distance_range_m =",
            [
                float(
                    np.min(
                        min_racket_ball_distance
                    )
                ),
                float(
                    np.max(
                        min_racket_ball_distance
                    )
                ),
            ],
        )


        finite_bounce = (
            opponent_bounce_pos_l[
                opponent_bounce
            ]
        )


        if finite_bounce.shape[0] > 0:

            print(
                "landing_x_range_m =",
                [
                    float(
                        np.min(
                            finite_bounce[
                                :,
                                0,
                            ]
                        )
                    ),
                    float(
                        np.max(
                            finite_bounce[
                                :,
                                0,
                            ]
                        )
                    ),
                ],
            )

            print(
                "landing_y_range_m =",
                [
                    float(
                        np.min(
                            finite_bounce[
                                :,
                                1,
                            ]
                        )
                    ),
                    float(
                        np.max(
                            finite_bounce[
                                :,
                                1,
                            ]
                        )
                    ),
                ],
            )


        finite_net = (
            net_cross_z[
                np.isfinite(
                    net_cross_z
                )
            ]
        )


        if finite_net.size > 0:

            print(
                "net_cross_z_range_m =",
                [
                    float(
                        np.min(
                            finite_net
                        )
                    ),
                    float(
                        np.max(
                            finite_net
                        )
                    ),
                ],
            )


        # -------------------------------------------------------------
        # Engineering smoke gates.
        #
        # These are integration sanity gates,
        # NOT new scientific robustness claims.
        # -------------------------------------------------------------

        lifecycle_gate = bool(
            all(
                phase == "ready"
                for phase in phases
            )
        )


        # World Z is NOT ground clearance in this HOPE scene.
        # The table-tennis floor is geometry.FLOOR_Z (currently -0.76 m),
        # so convert pelvis world-Z into pelvis height above the floor.
        min_root_clearance_m = (
            float(
                np.min(
                    min_root_z
                )
            )
            - float(
                geometry.FLOOR_Z
            )
        )


        stability_sanity = bool(
            finite_ok
            and min_root_clearance_m
            > 0.75
            and float(
                np.max(
                    max_tilt_deg
                )
            )
            < 35.0
            and float(
                np.max(
                    max_base_drift_m
                )
            )
            < 0.30
        )


        # Exact cross-environment numerical identity is intentionally
        # NOT a blocking requirement for floating humanoids evolved by
        # parallel GPU PhysX.  D1 showed sub-mm divergence initially
        # and gradual nonlinear growth during the swing.
        #
        # Keep spread values as diagnostics, while blocking only on
        # actual vector execution consistency.
        parallel_execution_sanity = bool(
            final_packed.shape
            == (
                n_envs,
                72,
            )
            and last_action.shape
            == (
                n_envs,
                31,
            )
            and len(lifecycles)
            == n_envs
            and len(phases)
            == n_envs
            and len(
                set(
                    phases
                )
            )
            == 1
            and np.isfinite(
                final_packed
            ).all()
            and np.isfinite(
                last_action
            ).all()
        )


        physical_pipeline_gate = bool(
            launched
            and contact_count == n_envs
            and cross_count == n_envs
            and landing_count == n_envs
            and legal_count == n_envs
        )


        print()
        print(
            "LIFECYCLE_RETURN_TO_READY=",
            "PASS"
            if lifecycle_gate
            else "FAIL",
        )


        print(
            "floor_z =",
            float(
                geometry.FLOOR_Z
            ),
        )


        print(
            "min_root_clearance_m =",
            min_root_clearance_m,
        )


        print(
            "MULTI_ENV_STABILITY_SANITY=",
            "PASS"
            if stability_sanity
            else "FAIL",
        )


        print(
            "IDENTICAL_ENV_PARITY_DIAGNOSTIC=",
            (
                "PASS"
                if (
                    base_local_spread
                    < 1.0e-3
                    and action_spread
                    < 1.0e-4
                )
                else "DIVERGENCE_OBSERVED_NONBLOCKING"
            ),
        )


        print(
            "PARALLEL_EXECUTION_SANITY=",
            "PASS"
            if parallel_execution_sanity
            else "FAIL",
        )


        print(
            "PHYSICAL_FIXED_BALL_PIPELINE=",
            "PASS"
            if physical_pipeline_gate
            else "FAIL",
        )


        if high_margin_count == n_envs:

            hm_status = "FULL_16_OF_16"

        elif high_margin_count > 0:

            hm_status = (
                f"PARTIAL_{high_margin_count}_OF_{n_envs}"
            )

        else:

            hm_status = "NONE"


        print(
            "HIGH_MARGIN_REPRODUCTION=",
            hm_status,
        )


        stage4_outcome_parity = bool(
            launch_timing_parity
            and legal_count == n_envs
            and high_margin_count == n_envs
        )


        print(
            "STAGE4_OUTCOME_PARITY=",
            "PASS"
            if stage4_outcome_parity
            else "FAIL",
        )


        if not (
            lifecycle_gate
            and stability_sanity
            and parallel_execution_sanity
            and physical_pipeline_gate
            and stage4_outcome_parity
        ):

            print(
                "INHERITED_STAGE4_PARITY_GATE="
                "NONBLOCKING_FOR_RANDOM_ACTION"
            )


        print()
        print(
            "INHERITED_STAGE4_FIXED_ACTION_MARKER_NOT_USED"
        )


        # =========================================================
        # STAGE5C1-B2 SINGLE EPISODE ACCEPTANCE
        # =========================================================


        # Stage5C3-C1 V2:
        #
        # env0 is randomized, therefore the old Stage4 fixed-ball
        # 0.76-s launch-boundary parity is no longer a valid
        # blocking requirement.
        #
        # Timing correctness is instead enforced by the already
        # validated per-environment physical-arrival criterion:
        #
        #   B1_MAX_ARRIVAL_ALIGNMENT_ERROR_S <= CONTROL_DT
        #
        # represented by:
        #
        #   stage5c2_launch_alignment_ok
        #
        print(
            "STAGE5C3_C1_V2_FIXED_STAGE4_LAUNCH_PARITY="
            "NONBLOCKING_RANDOM_INCOMING"
        )

        print(
            "STAGE5C3_C1_V2_PER_ENV_LAUNCH_ALIGNMENT_BLOCKING=",
            stage5c2_launch_alignment_ok,
        )


        b2_episode_gate = bool(
            b2_lifecycle_reset_ok
            and b2_last_action_reset_ok
            and b2_metrics_reset_ok
            and b2_ball_park_ok
            and stage5c1_smoke_gate
            and stage5c2_launch_alignment_ok
            and lifecycle_gate
            and stability_sanity
            and parallel_execution_sanity
            and np.isfinite(stage5c1_reward).all()
        )


        b2_episode_gates.append(
            b2_episode_gate
        )

        b2_action_snapshots.append(
            stage5c1_action_physical.copy()
        )

        b2_reward_snapshots.append(
            stage5c1_reward.copy()
        )

        b2_initial_state_snapshots.append(
            initial_packed.copy()
        )

        b2_contact_counts.append(
            int(contact_count)
        )

        b2_legal_counts.append(
            int(legal_count)
        )


        print()
        print(
            "=================================================="
        )

        print(
            f"STAGE5C1-B2 EPISODE "
            f"{episode_idx + 1}/{B2_EPISODES}"
        )

        print(
            "=================================================="
        )

        print(
            "B2_EPISODE_GATE=",
            "PASS" if b2_episode_gate else "FAIL",
        )

        print(
            "B2_EPISODE_CONTACT_COUNT=",
            f"{contact_count}/{n_envs}",
        )

        print(
            "B2_EPISODE_LEGAL_COUNT=",
            f"{legal_count}/{n_envs}",
        )

        print(
            "B2_EPISODE_HIGH_MARGIN_COUNT=",
            f"{high_margin_count}/{n_envs}",
        )

        print(
            "B2_EPISODE_REWARD_MEAN=",
            float(np.mean(stage5c1_reward)),
        )


    # =========================================================
    # STAGE5C1-B2 AGGREGATE RESULTS
    # =========================================================

    b2_episode_count_ok = bool(
        len(b2_episode_gates)
        == B2_EPISODES
    )


    b2_random_action_changes = []

    if (
        B2_EPISODES > 1
        and n_envs > 1
    ):

        for idx in range(
            1,
            len(b2_action_snapshots),
        ):

            change = float(
                np.max(
                    np.abs(
                        b2_action_snapshots[idx][1:]
                        - b2_action_snapshots[idx - 1][1:]
                    )
                )
            )

            b2_random_action_changes.append(
                change
            )


        b2_fresh_action_ok = bool(
            len(b2_random_action_changes)
            == B2_EPISODES - 1
            and all(
                x > 1.0e-6
                for x in b2_random_action_changes
            )
        )

    else:

        b2_fresh_action_ok = True


    b2_anchor_ok = bool(
        len(b2_action_snapshots)
        == B2_EPISODES
        and all(
            np.allclose(
                action[0],
                stage5c1_anchor_physical,
                atol=1.0e-12,
                rtol=0.0,
            )
            for action in b2_action_snapshots
        )
    )


    b2_rewards_finite_ok = bool(
        len(b2_reward_snapshots)
        == B2_EPISODES
        and all(
            np.isfinite(reward).all()
            for reward in b2_reward_snapshots
        )
    )


    if len(b2_initial_state_snapshots) > 1:

        ref = b2_initial_state_snapshots[0]

        b2_initial_state_max_delta = float(
            max(
                np.max(
                    np.abs(state - ref)
                )
                for state
                in b2_initial_state_snapshots[1:]
            )
        )

    else:

        b2_initial_state_max_delta = 0.0


    b2_persistent_gate = bool(
        b2_episode_count_ok
        and all(b2_episode_gates)
        and b2_fresh_action_ok
        and b2_rewards_finite_ok
    )


    print()
    print(
        "=================================================="
    )
    print(
        "STAGE5C1-B2 PERSISTENT RESET RESULTS"
    )
    print(
        "=================================================="
    )

    print(
        "episode_gates =",
        b2_episode_gates,
    )

    print(
        "contact_counts =",
        b2_contact_counts,
    )

    print(
        "legal_counts =",
        b2_legal_counts,
    )

    print(
        "reward_means =",
        [
            float(np.mean(x))
            for x in b2_reward_snapshots
        ],
    )

    print(
        "random_action_change_nonanchor =",
        b2_random_action_changes,
    )

    print(
        "FRESH_RANDOM_ACTION_AFTER_RESET=",
        "PASS"
        if b2_fresh_action_ok
        else "FAIL",
    )

    print(
        "ANCHOR_ACTION_PRESERVED_ACROSS_EPISODES=",
        "PASS"
        if b2_anchor_ok
        else "FAIL",
    )

    print(
        "ALL_REWARDS_FINITE=",
        "PASS"
        if b2_rewards_finite_ok
        else "FAIL",
    )

    print(
        "INITIAL_STATE_MAX_DELTA_DIAGNOSTIC=",
        b2_initial_state_max_delta,
    )

    print(
        "INITIAL_STATE_EXACT_PARITY_BLOCKING=FALSE"
    )


    print(
        "B2_BALL_RESET_SEMANTICS="
        "FINAL_REPARK_AFTER_0P20S_ROBOT_SETTLE"
    )

    print(
        "STAGE5C1B2_PERSISTENT_RESET_SMOKE=",
        "PASS"
        if b2_persistent_gate
        else "FAIL",
    )


    # =========================================================
    # STAGE5C2-B1 FINAL INTEGRATION GATE
    # =========================================================

    stage5c2b1_gate = bool(
        oracle_gate
        and stage5c2_oracle_contract_ok
        and stage5c2_obs_mapping_ok
        and stage5c2_target_mapping_ok
        and stage5c2_launch_alignment_ok
        and stage5c2_all_launched
        and stage5c1_smoke_gate
        and b2_persistent_gate
        and np.isfinite(
            stage5c1_reward
        ).all()
    )


    print()
    print(
        "=================================================="
    )
    print(
        "STAGE5C2-B1 RANDOM-INCOMING STRIKE RESULTS"
    )
    print(
        "=================================================="
    )

    print(
        "B1_ORACLE_GATE=",
        oracle_gate,
    )

    print(
        "B1_OBS_MAPPING=",
        stage5c2_obs_mapping_ok,
    )

    print(
        "B1_TARGET_MAPPING=",
        stage5c2_target_mapping_ok,
    )

    print(
        "B1_NOMINAL_TARGET_REPRODUCTION_FINAL=",
        stage5c2_nominal_target_reproduction,
    )

    print(
        "B1_PER_ENV_LAUNCH_ALIGNMENT_FINAL=",
        stage5c2_launch_alignment_ok,
    )

    print(
        "B1_ENV0_ANCHOR_LEGAL=",
        stage5c1_anchor_legal,
    )

    print(
        "B1_CONTACT_COUNT=",
        f"{contact_count}/{n_envs}",
    )

    print(
        "B1_CROSS_NET_COUNT=",
        f"{cross_count}/{n_envs}",
    )

    print(
        "B1_LANDING_COUNT=",
        f"{landing_count}/{n_envs}",
    )

    print(
        "B1_LEGAL_RETURN_COUNT=",
        f"{legal_count}/{n_envs}",
    )

    print(
        "B1_HIGH_MARGIN_COUNT=",
        f"{high_margin_count}/{n_envs}",
    )

    print(
        "B1_RANDOM_LEGAL_RATE_BLOCKING=FALSE"
    )

    print(
        "B1_ACTION_POLICY="
        "RANDOM_SMOKE_NOT_LEARNED"
    )

    print(
        "B1_OBSERVATION="
        "PRIVILEGED_PHYSICAL_ORACLE"
    )

    print(
        "STAGE5C2B1_RANDOM_INCOMING_STRIKE_SMOKE=",
        "PASS"
        if stage5c2b1_gate
        else "FAIL",
    )


    if not stage5c2b1_gate:

        raise RuntimeError(
            "Stage5C2-B1 random incoming physical strike gate failed"
        )


    # =========================================================
    # STAGE5C3-B FINAL TRAIN-READY IMPLEMENTATION GATE
    # =========================================================

    stage5c3_episode_count_ok = bool(
        len(
            stage5c3_incoming_snapshots
        )
        == B2_EPISODES
        and len(
            stage5c3_oracle_snapshots
        )
        == B2_EPISODES
        and len(
            stage5c3_policy_obs_snapshots
        )
        == B2_EPISODES
        and len(
            stage5c3_terminated_snapshots
        )
        == B2_EPISODES
        and len(
            stage5c3_truncated_snapshots
        )
        == B2_EPISODES
    )


    stage5c3_incoming_change = []

    stage5c3_oracle_change = []


    for idx in range(
        1,
        len(
            stage5c3_incoming_snapshots
        ),
    ):

        # env0 is deliberately the fixed validation anchor.
        # Freshness is therefore evaluated over non-anchor
        # environments.
        if n_envs > 1:

            incoming_delta = float(
                np.max(
                    np.abs(
                        stage5c3_incoming_snapshots[
                            idx
                        ][
                            1:
                        ]
                        - stage5c3_incoming_snapshots[
                            idx - 1
                        ][
                            1:
                        ]
                    )
                )
            )

            oracle_delta = float(
                np.max(
                    np.abs(
                        stage5c3_oracle_snapshots[
                            idx
                        ][
                            1:
                        ]
                        - stage5c3_oracle_snapshots[
                            idx - 1
                        ][
                            1:
                        ]
                    )
                )
            )

        else:

            incoming_delta = 0.0
            oracle_delta = 0.0


        stage5c3_incoming_change.append(
            incoming_delta
        )

        stage5c3_oracle_change.append(
            oracle_delta
        )


    if (
        B2_EPISODES > 1
        and n_envs > 1
    ):

        stage5c3_fresh_incoming_ok = bool(
            all(
                x > 1.0e-9
                for x in stage5c3_incoming_change
            )
        )

        stage5c3_fresh_oracle_ok = bool(
            all(
                x > 1.0e-9
                for x in stage5c3_oracle_change
            )
        )

    else:

        stage5c3_fresh_incoming_ok = True
        stage5c3_fresh_oracle_ok = True


    stage5c3_policy_obs_ok_all = bool(
        all(
            obs.shape
            == (
                n_envs,
                6,
            )
            and np.isfinite(
                obs
            ).all()
            and np.all(
                obs >= -5.0
                - 1.0e-12
            )
            and np.all(
                obs <= 5.0
                + 1.0e-12
            )
            for obs
            in stage5c3_policy_obs_snapshots
        )
    )


    stage5c3_done_semantics_ok = bool(
        all(
            np.all(
                terminated
            )
            for terminated
            in stage5c3_terminated_snapshots
        )
        and all(
            not np.any(
                truncated
            )
            for truncated
            in stage5c3_truncated_snapshots
        )
    )


    stage5c3_action_provider_ok = bool(
        stage5c3_external_action_provider_calls
        == B2_EPISODES
    )


    stage5c3_rewards_finite = bool(
        all(
            np.isfinite(
                reward
            ).all()
            for reward
            in b2_reward_snapshots
        )
    )


    stage5c3_train_ready_gate = bool(
        stage5c3_episode_count_ok
        and stage5c3_fresh_incoming_ok
        and stage5c3_fresh_oracle_ok
        and stage5c3_policy_obs_ok_all
        and stage5c3_done_semantics_ok
        and stage5c3_action_provider_ok
        and stage5c3_rewards_finite
        and b2_persistent_gate
        and stage5c2b1_gate
    )


    print()
    print(
        "=================================================="
    )
    print(
        "STAGE5C3-B TRAIN-READY ENV RESULTS"
    )
    print(
        "=================================================="
    )

    print(
        "STAGE5C3_EPISODE_COUNT=",
        len(
            stage5c3_incoming_snapshots
        ),
    )

    print(
        "STAGE5C3_EPISODE_COUNT_OK=",
        stage5c3_episode_count_ok,
    )

    print(
        "STAGE5C3_INCOMING_CHANGE_NONANCHOR=",
        stage5c3_incoming_change,
    )

    print(
        "STAGE5C3_ORACLE_CHANGE_NONANCHOR=",
        stage5c3_oracle_change,
    )

    print(
        "STAGE5C3_FRESH_INCOMING_EACH_EPISODE=",
        stage5c3_fresh_incoming_ok,
    )

    print(
        "STAGE5C3_FRESH_ORACLE_EACH_EPISODE=",
        stage5c3_fresh_oracle_ok,
    )

    print(
        "STAGE5C3_POLICY_OBS_NORMALIZATION=",
        "PASS"
        if stage5c3_policy_obs_ok_all
        else "FAIL",
    )

    print(
        "STAGE5C3_EXTERNAL_ACTION_PROVIDER_CALLS=",
        stage5c3_external_action_provider_calls,
    )

    print(
        "STAGE5C3_EXTERNAL_ACTION_INTERFACE=",
        "PASS"
        if stage5c3_action_provider_ok
        else "FAIL",
    )

    print(
        "STAGE5C3_TERMINATED_TRUE_EVERY_EPISODE=",
        stage5c3_done_semantics_ok,
    )

    print(
        "STAGE5C3_TRUNCATED_FALSE_EVERY_EPISODE=",
        stage5c3_done_semantics_ok,
    )

    print(
        "STAGE5C3_REWARD_FINITE_ALL_EPISODES=",
        stage5c3_rewards_finite,
    )

    print(
        "STAGE5C3_REWARD_SEMANTICS="
        "CONTACT1_CROSS2_LEGAL6_QUALITY_MINUS4_UNSTABLE"
    )

    print(
        "STAGE5C3_C1_VALIDATION_ANCHOR_ACTIVE="
        "FALSE"
    )

    print(
        "STAGE5C3_TRAINING_ANCHOR_INJECTION="
        "FALSE_EXECUTED_C1"
    )

    print(
        "STAGE5C3_ONE_HIGH_LEVEL_STEP_PER_BALL="
        "TRUE"
    )

    print(
        "STAGE5C3_LOW_LEVEL_MODEL21800_FROZEN="
        "TRUE"
    )

    print(
        "NEW_HIGH_LEVEL_RL_TRAINING_STARTED="
        "FALSE"
    )

    print(
        "STAGE5C3B_TRAIN_READY_ENV_SMOKE=",
        "PASS"
        if stage5c3_train_ready_gate
        else "FAIL",
    )



    # =========================================================
    # STAGE5C3-C1 POLICY-INTEGRATION FINAL GATE
    # =========================================================

    stage5c3_c1_parameters_unchanged = bool(
        all(
            torch.equal(
                parameter
                .detach()
                .cpu(),
                reference,
            )
            for parameter, reference
            in zip(
                stage5c3_c1_policy.parameters(),
                stage5c3_c1_initial_parameters,
            )
        )
    )


    stage5c3_c1_policy_call_count_ok = bool(
        len(
            stage5c3_c1_policy_action_snapshots
        )
        == B2_EPISODES
        and stage5c3_external_action_provider_calls
        == B2_EPISODES
    )


    stage5c3_c1_action_mapping_ok = True


    if (
        len(
            stage5c3_c1_policy_action_snapshots
        )
        != len(
            b2_action_snapshots
        )
    ):

        stage5c3_c1_action_mapping_ok = False

    else:

        for normalized_action, applied_physical in zip(
            stage5c3_c1_policy_action_snapshots,
            b2_action_snapshots,
        ):

            expected_physical = (
                stage5c1_action_lo[
                    None,
                    :
                ]
                + 0.5
                * (
                    normalized_action
                    + 1.0
                )
                * (
                    stage5c1_action_hi
                    - stage5c1_action_lo
                )[
                    None,
                    :
                ]
            )


            if not np.allclose(
                expected_physical,
                applied_physical,
                atol=1.0e-10,
                rtol=0.0,
            ):

                stage5c3_c1_action_mapping_ok = False
                break


    stage5c3_c1_env0_incoming_bounds_ok = bool(
        len(
            stage5c3_incoming_snapshots
        )
        == B2_EPISODES
        and all(
            (
                -0.90
                <= incoming[
                    0,
                    1,
                ]
                <= -0.60
            )
            and (
                0.32
                <= incoming[
                    0,
                    2,
                ]
                <= 0.54
            )
            and (
                -7.00
                <= incoming[
                    0,
                    3,
                ]
                <= -5.00
            )
            for incoming
            in stage5c3_incoming_snapshots
        )
    )


    stage5c3_c1_env0_incoming_changes = []


    for idx in range(
        1,
        len(
            stage5c3_incoming_snapshots
        ),
    ):

        delta = float(
            np.max(
                np.abs(
                    stage5c3_incoming_snapshots[
                        idx
                    ][
                        0
                    ]
                    - stage5c3_incoming_snapshots[
                        idx - 1
                    ][
                        0
                    ]
                )
            )
        )

        stage5c3_c1_env0_incoming_changes.append(
            delta
        )


    stage5c3_c1_env0_fresh_ok = bool(
        B2_EPISODES <= 1
        or all(
            delta > 1.0e-9
            for delta
            in stage5c3_c1_env0_incoming_changes
        )
    )


    stage5c3_c1_no_anchor_action_ok = bool(
        len(
            b2_action_snapshots
        )
        == B2_EPISODES
        and all(
            not np.allclose(
                action[
                    0
                ],
                stage5c1_anchor_physical,
                atol=1.0e-10,
                rtol=0.0,
            )
            for action
            in b2_action_snapshots
        )
    )


    stage5c3_c1_anchor_disabled_ok = bool(
        not STAGE5C3_VALIDATION_ANCHOR
        and stage5c3_c1_env0_incoming_bounds_ok
        and stage5c3_c1_env0_fresh_ok
        and stage5c3_c1_no_anchor_action_ok
    )


    stage5c3_c1_no_training_update_ok = bool(
        stage5c3_c1_optimizer_steps
        == 0
        and stage5c3_c1_backward_calls
        == 0
        and stage5c3_c1_parameters_unchanged
    )


    stage5c3_c1_gate = bool(
        stage5c3_train_ready_gate
        and stage5c3_c1_policy_call_count_ok
        and stage5c3_c1_action_mapping_ok
        and stage5c3_c1_anchor_disabled_ok
        and stage5c3_c1_no_training_update_ok
    )


    print()
    print(
        "=================================================="
    )
    print(
        "STAGE5C3-C1 TORCH MLP POLICY INTEGRATION RESULTS"
    )
    print(
        "=================================================="
    )


    print(
        "STAGE5C3_C1_POLICY_ARCH="
        "SAC_6_128_128_MEAN4_LOGSTD4_EVAL_TANH_MEAN"
    )


    print(
        "STAGE5C3_C1_POLICY_PARAMETER_COUNT=",
        stage5c3_c1_policy_parameter_count,
    )


    print(
        "STAGE5C3_C1_POLICY_PROVIDER_CALLS=",
        stage5c3_external_action_provider_calls,
    )


    print(
        "STAGE5C3_C1_POLICY_CALL_COUNT_OK=",
        stage5c3_c1_policy_call_count_ok,
    )


    print(
        "STAGE5C3_C1_POLICY_TO_PHYSICAL_ACTION_MAPPING=",
        "PASS"
        if stage5c3_c1_action_mapping_ok
        else "FAIL",
    )


    print(
        "STAGE5C3_C1_ENV0_INCOMING_BOUNDS=",
        "PASS"
        if stage5c3_c1_env0_incoming_bounds_ok
        else "FAIL",
    )


    print(
        "STAGE5C3_C1_ENV0_INCOMING_CHANGE=",
        stage5c3_c1_env0_incoming_changes,
    )


    print(
        "STAGE5C3_C1_ENV0_FRESH_RANDOM_INCOMING=",
        "PASS"
        if stage5c3_c1_env0_fresh_ok
        else "FAIL",
    )


    print(
        "STAGE5C3_C1_ENV0_ACTION_FROM_POLICY_NOT_ANCHOR=",
        "PASS"
        if stage5c3_c1_no_anchor_action_ok
        else "FAIL",
    )


    print(
        "STAGE5C3_C1_ALL_ENVS_SAME_POLICY_PATH=",
        "PASS"
        if stage5c3_c1_anchor_disabled_ok
        else "FAIL",
    )


    print(
        "STAGE5C3_C1_POLICY_PARAMETERS_UNCHANGED=",
        "PASS"
        if stage5c3_c1_parameters_unchanged
        else "FAIL",
    )


    print(
        "STAGE5C3_C1_OPTIMIZER_STEPS=",
        stage5c3_c1_optimizer_steps,
    )


    print(
        "STAGE5C3_C1_BACKWARD_CALLS=",
        stage5c3_c1_backward_calls,
    )


    print(
        "STAGE5C3_C1_NO_TRAINING_UPDATE=",
        "PASS"
        if stage5c3_c1_no_training_update_ok
        else "FAIL",
    )


    print(
        "NEW_HIGH_LEVEL_RL_TRAINING_STARTED="
        "FALSE"
    )


    print(
        "STAGE5C3C1_POLICY_INTEGRATION_SMOKE=",
        "PASS"
        if stage5c3_c1_gate
        else "FAIL",
    )


    if not stage5c3_c1_gate:

        raise RuntimeError(
            "Stage5C3-C1 policy integration gate failed"
        )



    # =========================================================
    # STAGE5C3-C3-C DETERMINISTIC EVALUATOR FINAL GATE
    # =========================================================

    stage5c3_eval_parameters_unchanged = bool(
        all(
            torch.equal(
                parameter
                .detach()
                .cpu(),
                reference,
            )
            for parameter, reference
            in zip(
                stage5c3_c1_policy.parameters(),
                stage5c3_c1_initial_parameters,
            )
        )
    )


    stage5c3_eval_repeat_max_delta = float(
        max(
            stage5c3_eval_repeat_deltas
        )
        if stage5c3_eval_repeat_deltas
        else float("inf")
    )


    stage5c3_eval_deterministic_ok = bool(
        len(
            stage5c3_eval_repeat_deltas
        )
        == B2_EPISODES
        and stage5c3_eval_repeat_max_delta
        <= 1.0e-7
    )


    stage5c3_eval_no_training_ok = bool(
        stage5c3_c1_optimizer_steps
        == 0
        and stage5c3_c1_backward_calls
        == 0
        and stage5c3_eval_stochastic_samples
        == 0
        and stage5c3_eval_parameters_unchanged
    )


    stage5c3_eval_gate = bool(
        stage5c3_train_ready_gate
        and stage5c3_c1_gate
        and stage5c3_eval_checkpoint_loaded_exact
        and stage5c3_eval_deterministic_ok
        and stage5c3_eval_no_training_ok
        and stage5c3_c3_d_seed_disjoint
    )


    print()
    print(
        "=================================================="
    )
    print(
        "STAGE5C3-C3-C DETERMINISTIC CHECKPOINT EVALUATION"
    )
    print(
        "=================================================="
    )


    print(
        "STAGE5C3_C3_C_EVAL_EPISODES=",
        B2_EPISODES,
    )

    print(
        "STAGE5C3_C3_C_ACTION_MODE="
        "DETERMINISTIC_TANH_MEAN"
    )

    print(
        "STAGE5C3_C3_C_CHECKPOINT_ACTOR_LOAD=",
        "PASS"
        if stage5c3_eval_checkpoint_loaded_exact
        else "FAIL",
    )

    print(
        "STAGE5C3_C3_C_REPEAT_ACTION_MAX_DELTA=",
        stage5c3_eval_repeat_max_delta,
    )

    print(
        "STAGE5C3_C3_C_DETERMINISTIC_REPEAT=",
        "PASS"
        if stage5c3_eval_deterministic_ok
        else "FAIL",
    )

    print(
        "STAGE5C3_C3_C_STOCHASTIC_SAMPLE_COUNT=",
        stage5c3_eval_stochastic_samples,
    )

    print(
        "STAGE5C3_C3_C_OPTIMIZER_STEPS=",
        stage5c3_c1_optimizer_steps,
    )

    print(
        "STAGE5C3_C3_C_BACKWARD_CALLS=",
        stage5c3_c1_backward_calls,
    )

    print(
        "STAGE5C3_C3_C_ACTOR_PARAMETERS_UNCHANGED=",
        "PASS"
        if stage5c3_eval_parameters_unchanged
        else "FAIL",
    )

    print(
        "STAGE5C3_C3_C_NO_TRAINING_DURING_EVAL=",
        "PASS"
        if stage5c3_eval_no_training_ok
        else "FAIL",
    )

    print(
        "STAGE5C3_C3_C_LEGAL_COUNTS_DIAGNOSTIC=",
        b2_legal_counts,
    )

    print(
        "STAGE5C3_C3_C_CONTACT_COUNTS_DIAGNOSTIC=",
        b2_contact_counts,
    )

    print(
        "STAGE5C3_C3_C_EVALUATION_ONLY=TRUE"
    )

    print(
        "NEW_HIGH_LEVEL_RL_TRAINING_STARTED="
        "FALSE_BY_THIS_EVALUATOR"
    )

    print(
        "STAGE5C3C3C_DETERMINISTIC_CHECKPOINT_EVAL=",
        "PASS"
        if stage5c3_eval_gate
        else "FAIL",
    )



    stage5c3_c3_d_total_balls = int(
        n_envs
        * B2_EPISODES
    )


    stage5c3_c3_d_total_contacts = int(
        sum(
            b2_contact_counts
        )
    )


    stage5c3_c3_d_total_legal = int(
        sum(
            b2_legal_counts
        )
    )


    stage5c3_c3_d_legal_rate = float(
        stage5c3_c3_d_total_legal
        / stage5c3_c3_d_total_balls
    )


    print()
    print(
        "=================================================="
    )
    print(
        "STAGE5C3-C3-D HELD-OUT EVALUATION RESULTS"
    )
    print(
        "=================================================="
    )


    if stage5c3_c3_d_training_seeds:

        stage5c3_c3_d_train_seed_range_report = [
            min(
                stage5c3_c3_d_training_seeds
            ),
            max(
                stage5c3_c3_d_training_seeds
            ),
        ]

    else:

        stage5c3_c3_d_train_seed_range_report = (
            "EMPTY_PRETRAIN_CHECKPOINT"
        )


    print(
        "STAGE5C3_C3_D_TRAIN_SEED_RANGE=",
        stage5c3_c3_d_train_seed_range_report,
    )

    print(
        "STAGE5C3_C3_D_EVAL_SEED_RANGE=",
        [
            min(
                stage5c3_c3_d_eval_seeds
            ),
            max(
                stage5c3_c3_d_eval_seeds
            ),
        ],
    )

    print(
        "STAGE5C3_C3_D_SEED_OVERLAP=",
        stage5c3_c3_d_seed_overlap,
    )

    print(
        "STAGE5C3_C3_D_SEED_DISJOINT=",
        "PASS"
        if stage5c3_c3_d_seed_disjoint
        else "FAIL",
    )

    print(
        "STAGE5C3_C3_D_TOTAL_BALLS=",
        stage5c3_c3_d_total_balls,
    )

    print(
        "STAGE5C3_C3_D_CONTACT_TOTAL=",
        stage5c3_c3_d_total_contacts,
    )

    print(
        "STAGE5C3_C3_D_LEGAL_TOTAL=",
        stage5c3_c3_d_total_legal,
    )

    print(
        "STAGE5C3_C3_D_LEGAL_RATE_DIAGNOSTIC=",
        stage5c3_c3_d_legal_rate,
    )

    print(
        "STAGE5C3_C3_D_HELDOUT_SCOPE="
        "DISJOINT_FROM_FROZEN_CHECKPOINT_TRAINING_SEEDS"
    )

    print(
        "STAGE5C3_C3_D_OPTIMIZER_STEPS=0"
    )

    print(
        "STAGE5C3_C3_D_BACKWARD_CALLS=0"
    )

    print(
        "STAGE5C3_C3_D_STOCHASTIC_SAMPLE_COUNT=0"
    )

    print(
        "STAGE5C3_C3_D_PERFORMANCE_STATUS="
        "HELD_OUT_SEED_DIAGNOSTIC_SMALL_SAMPLE"
    )

    print(
        "STAGE5C3C3D_HELDOUT_EVALUATION_SMOKE=",
        "PASS"
        if stage5c3_eval_gate
        else "FAIL",
    )


    if not stage5c3_eval_gate:

        raise RuntimeError(
            "Stage5C3-C3-C deterministic "
            "checkpoint evaluator gate failed"
        )


    if not stage5c3_train_ready_gate:

        raise RuntimeError(
            "Stage5C3-B train-ready environment gate failed"
        )




    if not b2_persistent_gate:

        raise RuntimeError(
            "Stage5C1-B2 persistent reset gate failed"
        )




# =====================================================================
# Fast exit
# =====================================================================

if __name__ == "__main__":

    exit_code = 0

    try:

        main()

    except Exception:

        traceback.print_exc()

        exit_code = 1

    finally:

        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(
                exit_code
            )
