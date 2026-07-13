# A3 Native Motion Strike Policy Plan

Date: 2026-07-10

This document supersedes the old whole-body-first training plan as the active project route for the A3 table-tennis controller.

## Core Decision

Do not continue treating stable standing as a PPO skill to learn from scratch.

The active route is:

```text
A3 native MOTION / MC controller
        -> standing, balance, lower body

RL strike policy
        -> waist + right arm strike execution

Planner / manifest
        -> racket target position, velocity, normal, time-to-hit, stroke type
```

The 31-DOF Isaac PPO run remains useful as a baseline, but it is no longer the main engineering path.

## Current Baseline To Preserve

Keep this run as the whole-body PPO baseline:

```text
hope_training/whole_body_tracking/logs/rsl_rl/agibot_a3_hope_manifest/
2026-07-10_17-04-46_exact_command_k4_env1024_frame_adapter/model_999.pt
```

What it proved:

- Isaac training can run at `1024` environments on the laptop.
- The manifest frame adapter is necessary and working for the tracking-only scene.
- The balanced motion library loads correctly.
- Whole-body PPO can learn basic stability.

What it did not prove:

- Stable strike execution.
- Suitability for real A3 deployment.
- Need to train balance from scratch.

## AimDK Interface Findings

Local official SDK/documentation used:

```text
agibot_a3_aimdk/
```

Relevant files:

- `agibot_a3_aimdk/examples/mc/arm.py`
- `agibot_a3_aimdk/examples/mc/move_waist.py`
- `agibot_a3_aimdk/protocol/protobuf/aimdk/protocol/motion_control/motion/mc_motion_channel.proto`
- `agibot_a3_aimdk/protocol/protobuf/aimdk/protocol/mc/action/mc_action_cmd.proto`
- `agibot_a3_aimdk/protocol/protobuf/aimdk/protocol/mc/common/planning.proto`

Confirmed interfaces:

- `/motion/control/arm_joint_command`
  - ROS `sensor_msgs/JointState`
  - 14 arm joints
  - example publishes at 50 Hz
- `/motion/control/move_waist`
  - protobuf `MotionControlMoveWaistChannel`
  - `waist_pitch`: `[-0.5, 0.5]` rad
  - `waist_roll`: `[-0.3, 0.3]` rad
  - `waist_yaw`: `[-1.57, 1.57]` rad
  - `waist_height`: `[-0.4, 0.0]` m
  - example publishes at 20 Hz

Important MC action modes from `mc_action_cmd.proto`:

- `McAction_STAND_ARM_EXT_JOINT_SERVO = 203`
  - force-control standing
  - external upper-arm joint servo
- `McAction_LOCOMOTION_ARM_EXT_JOINT_SERVO = 302`
  - locomotion with external arm control
- `McAction_RL_LOCOMOTION_ARM_EXT_JOINT_SERVO = 402`
  - RL locomotion with external arm control
- `McAction_RL_WHOLE_BODY_EXT_JOINT_SERVO = 405`
  - waist whole-body external joint servo
- `McAction_RL_WHOLE_BODY_EXT_ONLINE_PLANNING = 407`
  - lower-body force-control standing, waist+upper-body planning

Planning groups include:

- `McPlanningGroup_RIGHT_ARM = 4`
- `McPlanningGroup_RIGHT_ARM_WAIST = 5`
- `McPlanningGroup_DUAL_ARM_WAIST = 8`
- `McPlanningGroup_WAIST_LIFT = 11`
- `McPlanningGroup_WAIST_PITCH = 12`

Current conclusion:

```text
Gate A: waist / torso control is present in the SDK.
```

Therefore the first recommended trainable action space is not arm-only. It is:

```text
waist command
+
right arm 7-DOF command
```

Arm-only remains the fallback route if AimSim shows that waist command cannot be safely combined with native balance for table-tennis swing timing.

## Immediate Stop Rule

Do not continue:

- `K=4 -> K=8` with the current 31-DOF whole-body PPO task.
- Long 31-DOF PPO runs.
- Reward tuning aimed mainly at making the current policy stand better.
- Ball physics integration before strike executor validation.

## Next Gate: AimSim Native Balance Validation

Before new PPO training, validate the official control architecture in AimSim.

Validation helper:

```text
tools/a3_native_strike_validation.py
```

Current local environment check:

```text
rclpy: OK
sensor_msgs: OK
ros2_plugin_proto: OK
aimdk.protocol_pb2: OK
```

The local command/publisher environment is configured. The missing piece is a running AimSim / MC control endpoint. Current ROS graph only exposes `/rosout` and `/parameter_events`, so command publishing can start but no arm, waist, or IMU state is returned yet.

Configured local additions:

- installed `a3_aimdk-3.1.0` Python package
- installed user-level CMake for local message builds
- built `ros2_plugin_proto` for this x86_64 machine into:

```text
agibot_a3_aimdk/prebuilt/ros2_plugin_proto_x86_64/
```

Use this project setup script before validation commands:

```bash
source tools/setup_a3_native_validation_env.sh
```

Dependency check:

```bash
source tools/setup_a3_native_validation_env.sh
python3 tools/a3_native_strike_validation.py --check-only
```

Dry publish check without AimSim/MC currently succeeds and reports no state feedback:

```bash
source tools/setup_a3_native_validation_env.sh
python3 tools/a3_native_strike_validation.py --mode combined --duration-s 3 --rate-hz 20
```

Observed result on 2026-07-10:

```text
arm_state_count: 0
waist_state_count: 0
imu_count: 0
notes: no arm state received on /motion/control/arm_joint_state
```

This means the publisher side is ready, not that native balance has passed.

### Test A: Native Stand + Arm Servo

Sequence:

```text
start motion control
GET_UP
MOTION / force-control standing mode
locomotion velocity = 0
publish /motion/control/arm_joint_command
```

Use a safe right-arm pose and keep the left arm nominal.

Pass criteria:

- arm command takes effect
- robot remains standing
- lower body continues active balance
- no large base roll/pitch drift

Command:

```bash
source tools/setup_a3_native_validation_env.sh
python3 tools/a3_native_strike_validation.py \
  --mode arm \
  --duration-s 10 \
  --rate-hz 50 \
  --arm-elbow-amp-rad 0.12 \
  --arm-shoulder-roll-amp-rad 0.08
```

### Test B: Native Stand + Waist Command

Sequence:

```text
standing / MOTION mode
publish /motion/control/move_waist
small pitch/roll/yaw/height commands
```

Start with small values:

```text
waist_pitch <= 0.05 rad
waist_roll  <= 0.03 rad
waist_yaw   <= 0.05 rad
height      >= -0.03 m
```

Pass criteria:

- command affects waist as expected
- base remains stable
- lower body compensates rather than going passive

Command:

```bash
source tools/setup_a3_native_validation_env.sh
python3 tools/a3_native_strike_validation.py \
  --mode waist \
  --duration-s 10 \
  --rate-hz 50 \
  --waist-pitch-amp-rad 0.05 \
  --waist-roll-amp-rad 0.03 \
  --waist-yaw-amp-rad 0.05
```

### Test C: Combined Waist + Right Arm Slow Swing

Use one golden forehand motion:

```text
right arm 7 joints
waist pitch/roll/yaw/height if available
```

Replay at:

```text
0.25x -> 0.5x -> 1.0x
```

Pass criteria:

- no severe base tilt
- no self-collision
- command stream remains smooth
- lower body balance compensation remains active

Command:

```bash
python3 tools/a3_native_strike_validation.py \
  --mode combined \
  --duration-s 15 \
  --rate-hz 50 \
  --frequency-hz 0.20 \
  --arm-elbow-amp-rad 0.10 \
  --arm-shoulder-roll-amp-rad 0.06 \
  --waist-pitch-amp-rad 0.04 \
  --waist-roll-amp-rad 0.02 \
  --waist-yaw-amp-rad 0.04
```

If an IMU topic is available in the AimSim environment, pass it explicitly:

```bash
--imu-topic <sensor_msgs/Imu topic>
```

The local low-level MuJoCo package exposes `/body_drive/*`, but it is not a substitute for this gate because it does not prove the official native MOTION balance controller remains active while `/motion/control/arm_joint_command` and `/motion/control/move_waist` are used.

## Motion Dataset Analysis

Before retargeting or PPO, analyze the existing `40` balanced motions:

```text
hope_training/whole_body_tracking/sample_motions/p2_fixed_competition_global_funnel_balanced20/manifest.json
```

Compute per motion:

- `waist_pitch_range_rad`
- `waist_roll_range_rad`
- `waist_yaw_range_rad`
- `waist_height_range_m` if present
- waist velocity at hit
- right-arm joint velocity at hit
- A3 API velocity-limit violations

Then run FK counterfactuals:

```text
original: waist(t) + right_arm(t)
fallback: fixed/native waist + right_arm(t)
```

Compare at hit:

- racket position delta
- racket velocity delta
- racket normal delta

Output:

```text
waist_dependency_report.json
```

Motion tiers:

- Tier A: arm/waist feasible now
  - hit position delta `< 3 cm`
  - hit velocity delta `< 0.5 m/s`
  - normal delta `< 8 deg`
- Tier B: moderate dependency
  - position `3-7 cm` or velocity `0.5-1.0 m/s`
- Tier C: high dependency
  - position `> 7 cm` or velocity `> 1.0 m/s`

## Implemented RL Environment

Implemented as a new task instead of mutating the existing 31-DOF task:

```text
cfg/task/HOPEA3NativeStrikeManifest.yaml
gym_task: HOPE-NativeStrike-AgibotA3-v0
env cfg: training/tasks/tracking/config/agibot_a3/native_strike_env_cfg.py
```

Policy action:

```text
waist_yaw
waist_roll
waist_pitch
right_shoulder_pitch
right_shoulder_roll
right_shoulder_yaw
right_elbow
right_wrist_roll
right_wrist_pitch
right_wrist_yaw
```

```text
3 waist DOF + 7 right-arm DOF = 10 actions
```

Actor observation:

```text
base_ang_vel               3
waist+right_arm_joint_pos 10
waist+right_arm_joint_vel 10
previous_action           10
projected_gravity          3
racket_target_pos_b        3
racket_target_vel_b        3
racket_target_normal_b     3
time_to_hit                1
stroke_type                1
```

Do not include reference motion state in actor observations.

Current smoke dimensions:

```text
actor obs: 47
critic obs: 110
action: 10
termination: time_out only
Isaac base: fixed for Stage-1 strike-executor training
```

Critic receives reference motion, actual racket state, and strike target state. Full-body motion
tracking is not the actor interface.

Validation completed on 2026-07-10:

```bash
cd hope_training/whole_body_tracking
source setup_train_env.sh
hope_isaac_py scripts/train.py task=HOPEA3NativeStrikeManifest algo=ppo headless=true \
  logger=tensorboard num_envs=8 max_iterations=1 run_name=native_strike_smoke
```

Result:

```text
PASS: completed 1 PPO iteration
Action Manager shape: 10
Policy observation shape: 47
Reward terms active: strike objective + upper-chain motion prior + smoothness/safety
```

K=1 forehand selection was also verified:

```text
MotionCommand loaded manifest: 1 motions; forehand=1, backhand=0
```

Implementation notes:

- Stage-1 native-strike Isaac training fixes the robot base. This intentionally removes lower-body
  balance from the learning problem; real deployment still relies on A3 native MOTION/MC for balance.
- `add_joint_default_pos` is disabled for this task because the existing event assumes full-action
  indexing and is unsafe for a 10-action subset.
- `push_robot` is disabled for the first strike-executor phase because native MC, not PPO, is meant
  to own disturbance recovery.
- `randomize_link_mass` is disabled for the fixed-base strike-executor phase.
- `manifest_subset_size=1` now selects one forehand by default; `2` selects one forehand and one
  backhand.

## Command Adapter Contract

Create one adapter used by training, AimSim validation, and deployment:

```text
A3StrikeCommandAdapter
```

Responsibilities:

- Convert policy output to waist/right-arm target commands.
- Interpolate policy rate to command publish rate.
- Enforce joint velocity limits.
- Enforce command gap constraints.
- Log every command sent.

Initial conservative rates:

```text
policy: 50 Hz
arm command publish: 100 Hz if supported, otherwise 50 Hz
waist command publish: 20-50 Hz after AimSim validation
```

Do not add deployment-only low-pass filters. If filtering is needed, it must exist in training and AimSim too.

## Reward Redefinition

New reward is strike execution, not whole-body tracking:

```text
r_total =
    racket strike objective
  + right-arm / waist motion prior
  + smoothness and safety regularization
```

Keep:

- racket position at hit
- racket velocity at hit
- racket normal at hit
- temporal Gaussian hit kernel
- arm/waist joint prior
- action rate penalty
- joint limit penalty
- API velocity-limit penalty

Remove from the main objective:

- full-body global body position tracking
- full-body orientation tracking
- full-body body velocity tracking
- leg tracking
- learned standing reward

Base attitude should be a safety metric and termination condition, not the main task reward.

## Training Sequence

For Isaac interface bring-up, short smoke runs are allowed before AimSim. Do not treat them as
deployment validation.

Recommended first real training run:

```bash
cd /home/bruce/桌面/HOPETableTennis/hope_training/whole_body_tracking
source setup_train_env.sh

hope_isaac_py scripts/train.py task=HOPEA3NativeStrikeManifest algo=ppo headless=true \
  logger=tensorboard num_envs=512 max_iterations=1000 \
  manifest_subset_size=1 \
  run_name=native_k1_fh_exact
```

Then inspect whether strike rewards become non-zero and whether base safety metrics remain sane.

Expansion sequence:

1. `K=1` golden forehand
   - `128-512` env
   - `500-1000` iterations
2. `K=1` golden backhand
3. `K=2`
   - one FH, one BH
4. `K=4`
   - two FH, two BH
5. `K=8`
   - only Tier A feasible motions
6. `K=20-40`
   - coverage expansion after per-motion strike success is acceptable

Acceptance gate before increasing K:

```text
per-motion strike success > 50%
preferred > 70%
pos p90 < 3 cm
vel p90 < 0.5 m/s
normal p90 < 10 deg
```

## Deployment Route

Training command schema and deployment command schema must stay identical:

```text
racket_target_pos_b
racket_target_vel_b
racket_target_normal_b
time_to_hit
stroke_type
```

Training source:

```text
manifest hit_event / strike_target
```

Deployment source:

```text
ball estimator -> planner -> same strike command
```

Do not train a policy that requires reference motion at deployment time.

## Immediate TODO

1. Preserve the current whole-body baseline and stop expanding it.
2. Keep the implemented Isaac native-strike task as the active RL path:
   - `HOPEA3NativeStrikeManifest`
   - action = waist + right arm, 10 DOF
3. Run K=1 forehand exact-command training and inspect visual replay.
4. Build AimSim tests for:
   - native standing + arm command
   - native standing + waist command
   - combined waist + right-arm slow swing
5. Generate `waist_dependency_report.json` for the 40 balanced motions.
6. Decide final action space after AimSim:
   - preferred: waist + right arm
   - fallback: right arm only
7. Implement `A3StrikeCommandAdapter`.
8. Restart scalable RL from `K=1`, not old `K=4`.
