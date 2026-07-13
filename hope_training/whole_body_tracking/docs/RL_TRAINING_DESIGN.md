# RL Training Design For A3 Table-Tennis Motions

Date: 2026-07-10

Status update:

The active project route has moved away from 31-DOF whole-body PPO as the main line. The current recommended route is native A3 standing/balance plus a waist/right-arm strike policy. See:

```text
hope_training/whole_body_tracking/docs/A3_NATIVE_STRIKE_POLICY_PLAN.md
```

This document remains useful for the historical whole-body baseline and the HITTER-like command schema, but it should not be used to justify continuing `K=4 -> K=8` whole-body PPO expansion.

This document defines the recommended reinforcement-learning path for the current Agibot A3 table-tennis dataset.

## Current Training Dataset

Use only the balanced dataset:

```text
hope_training/whole_body_tracking/sample_motions/p2_fixed_competition_global_funnel_balanced20/manifest.json
```

Contents:

- `40` replay-ready motions.
- `20` forehand.
- `20` backhand.
- Every motion has:
  - `motion_npz`
  - `library_motion_npz`
  - `hit_event`
  - `strike_target`
  - `fps=50`
  - `joint_pos=(80,31)`
  - `body_pos_w=(80,32,3)`

This dataset is ready for training, but the current RL code path is still mostly single-motion oriented.

## Project Status

The current training package already has useful pieces:

- `training/tasks/tracking/mdp/commands.py`
  - `MotionLoader`
  - `MotionCommand`
  - single `.npz` motion tracking
  - adaptive reference-state initialization over one clip

- `training/tasks/tracking/mdp/hope_commands.py`
  - `RacketTargetCommand`
  - desired racket position / velocity / normal target
  - time-to-strike
  - strike-window metrics
  - reference-perturbed target curriculum

- `training/tasks/tracking/mdp/hope_rewards.py`
  - racket position reward
  - racket velocity reward
  - racket normal reward
  - pre-strike base target reward

- `cfg/task/HOPEPingPong.yaml`
  - current HOPE task configuration
  - PPO-compatible single-motion training entry

- `scripts/train.py`
  - Hydra training entry
  - currently accepts `motion_file=...`
  - does not yet accept a motion manifest/library as the primary source

Main gap:

```text
current code: one motion_file per run
needed code: balanced manifest -> sample many motions per env/episode
```

## External References And Lessons

### DeepMimic

DeepMimic established the reference-motion RL pattern: phase-synchronized motion imitation, reference-state initialization, and reward terms for pose, velocity, end-effectors, and root motion. It is the right conceptual base for this project's first phase: make the robot dynamically track the retargeted motions before adding harder ball-return logic.

Source: https://xbpeng.github.io/projects/DeepMimic/index.html

### BeyondMimic

This project is already based on BeyondMimic-style motion tracking. BeyondMimic's key lesson is that robust whole-body skills should first be learned as high-quality motion-tracking primitives, then composed or guided for downstream tasks.

Source: https://arxiv.org/abs/2508.08241

Open-source codebase relevant to this project:

https://github.com/HybridRobotics/whole_body_tracking

### HITTER

HITTER uses a hierarchical design: a planner provides strike commands such as target racket position, velocity, and timing; an RL whole-body controller executes them. This matches our data model well because every current motion has `strike_target` and `hit_event`.

Source: https://arxiv.org/html/2508.21043v1

Project implication:

```text
do not train end-to-end ball gameplay first
train command-conditioned racket/motion control first
```

### SMASH

SMASH emphasizes scalable, diverse strike-motion coverage and dynamics-compatible tracking. This directly supports our global candidate-index workflow and the balanced20 dataset. The practical lesson is that coverage matters, but only after the tracker can actually execute the motions.

Source: https://arxiv.org/html/2604.01158v1

Project implication:

```text
train small balanced coverage set first
expand only after metrics expose weak regions
```

### Isaac Lab / RSL-RL

The project already uses Isaac Lab with RSL-RL. Isaac Lab documents the `RslRlVecEnvWrapper` and RSL-RL integration; RSL-RL is a lightweight GPU-oriented robotics RL library using algorithms such as PPO.

Sources:

- https://isaac-sim.github.io/IsaacLab/main/source/api/lab_rl/isaaclab_rl.html
- https://github.com/leggedrobotics/rsl_rl
- PPO paper: https://arxiv.org/abs/1707.06347

## Architecture Contract

The project should use a HITTER-like planner-commanded whole-body controller.

The deployable policy receives proprioception plus a strike command. It does not receive the reference motion as an actor observation.

```text
ball / planner / manifest target
        ↓
strike command
        ↓
A3 whole-body controller
```

Reference motion is still important, but only as a training teacher:

- reward target for body, joint, and racket motion
- privileged critic information
- reset / phase / diagnostics source
- dynamic executability check

Do not build the first trainable policy around this interface:

```text
Actor = proprioception + reference motion state + strike command
```

That policy would not transfer cleanly to the later planner stage unless a motion selector or motion generator is added. This project should avoid that extra architecture for now.

### Actor Interface

Actor observations:

- joint positions
- joint velocities
- projected gravity
- base angular velocity
- previous action
- `racket_target_pos_b`
- `racket_target_vel_b`
- `racket_target_normal_b`
- normalized `time_to_hit`
- stroke type indicator for a unified forehand/backhand policy

All command vectors exposed to the actor must be in the robot base frame:

```python
p_target_b = R_wb.T @ (p_target_w - p_base_w)
v_target_b = R_wb.T @ v_target_w
normal_target_b = R_wb.T @ normal_target_w
```

World-frame command values may be kept for reward, critic, logging, and debugging, but not as the actor command interface.

### Critic Interface

Critic-only privileged observations may include:

- all actor observations
- actual racket position, velocity, and normal
- reference joint position and velocity
- reference body position and velocity
- motion id
- phase / cycle step
- exact strike error
- failure bin / motion quality metadata

## Time Model

The design must distinguish an environment episode from a strike cycle.

The current motions are:

```text
80 frames / 50 Hz = 1.6 s
```

An environment episode may be shorter or longer than one motion. Therefore the command layer should track:

```text
episode_step
strike_cycle_id
cycle_step
motion_id
motion_length
hit_frame
time_to_hit
```

First implementation should use one strike cycle per episode:

```text
episode_length_s = 1.6-2.0
reset -> prepare -> strike -> follow-through -> episode end
```

After single-strike training is stable, extend to multi-strike episodes:

```text
10 s environment episode
├── strike cycle 0
├── strike cycle 1
├── strike cycle 2
└── ...
```

In multi-strike mode, a strike cycle can end and sample a new motion without resetting the environment. Reset the environment only on falls, catastrophic tracking failure, invalid state, or timeout.

All strike metrics must be event-based, not episode-based:

- `strike/events_total`
- `strike/events_success`
- `strike/pos_error_at_event`
- `strike/vel_error_at_event`
- `strike/normal_error_at_event`
- `strike/timing_error`
- `forehand/events_success`
- `backhand/events_success`

## Recommended Training Strategy

Do not start with full ball physics gameplay.

Use this sequence:

```text
Phase 0: environment sanity
Phase 1: single-motion executability
Phase 2: small multi-motion diagnostic
Phase 3: manifest exact-command WBC
Phase 4: balanced40 exact-strike training with validation
Phase 5: strike-manifold perturbation curriculum
Phase 6: multi-strike recovery
Phase 7: domain randomization
Phase 8: planner-commanded ball hitting
```

## Phase 0: Environment Sanity

Purpose:

Make sure Isaac Lab, RSL-RL, assets, and single-motion loading still launch after the data cleanup.

Recommended run:

```bash
cd hope_training/whole_body_tracking
source setup_train_env.sh
hope_isaac_py scripts/train.py task=TrackingFlat algo=ppo headless=true \
  logger=tensorboard num_envs=16 max_iterations=3 \
  motion_file=sample_motions/p2_fixed_competition_global_funnel_balanced20/forehand/<episode>.npz \
  run_name=sanity_one_motion
```

## Phase 1: Single-Motion Executability

This phase is a diagnostic, not the final policy foundation.

Goal:

Check whether one easy forehand and one easy backhand can be dynamically tracked by the robot in simulation.

Run:

- one forehand overfit
- one backhand overfit

Expected outcome:

- body tracking error trends down
- joint position and velocity tracking errors trend down
- no early falls
- actions do not saturate
- reset terminations are not dominated by anchor/body errors

If a single motion cannot overfit, stop and debug physics, PD gains, action scale, retarget quality, asset indexing, or reward scale before adding more data.

## Phase 2: Small Multi-Motion Diagnostic

Before training on all 40 motions, run a motion-count ablation:

```text
K=1: one golden forehand
K=2: one forehand + one backhand
K=4: two forehand + two backhand
K=8: four forehand + four backhand
K=40: balanced20 forehand + balanced20 backhand
```

This isolates whether failures are caused by:

- one bad motion
- forehand/backhand asymmetry
- multi-motion ambiguity
- policy capacity
- coverage difficulty

## Phase 3: Manifest Exact-Command WBC

Goal:

Train the deployable policy interface using exact manifest strike targets.

Actor interface is fixed:

```text
proprioception
+ racket target position in base frame
+ racket target velocity in base frame
+ racket target normal in base frame
+ normalized time_to_hit
+ stroke_type
```

Actor does not observe reference motion state.

Training target comes from:

```json
motion["hit_event"]
motion["strike_target"]
```

Use reference motion for reward and critic only.

Reward:

```text
r_total =
  w_motion * r_motion_tracking
  + w_strike_pos * r_racket_pos_event
  + w_strike_vel * r_racket_vel_event
  + w_strike_normal * r_racket_normal_event
  + w_base * r_base_pre_strike
  + regularization
```

Initial reward scales should be tuned from logged contribution, not fixed by schedule. Start with a strong motion reward and exact strike reward, then reduce motion scale only if strike metrics plateau while weighted motion reward dominates the return.

Log weighted contributions:

- `reward/motion_weighted_mean`
- `reward/strike_pos_weighted_mean`
- `reward/strike_vel_weighted_mean`
- `reward/strike_normal_weighted_mean`
- `penalty/action_rate_mean`
- `penalty/torque_mean`
- `penalty/joint_limit_mean`

## Phase 4: Balanced40 Training With Validation

Use the 40 replay-ready motions, but define a validation split.

Recommended split:

```text
train: 16 forehand + 16 backhand = 32
val:    4 forehand +  4 backhand = 8
```

Validation motions must not be used for PPO updates. Periodically run deterministic evaluation with actor mean actions.

Validation metrics:

- `val/strike_pos_p50`
- `val/strike_pos_p90`
- `val/strike_vel_p50`
- `val/strike_vel_p90`
- `val/strike_normal_p50`
- `val/strike_normal_p90`
- `val/composite_success`

If source rally or source session ids are available, prefer splitting by rally/session to reduce leakage.

## Phase 5: Strike-Manifold Perturbation

Only start after exact manifest targets are stable.

Perturbations must preserve strike-state consistency. Do not sample position, velocity, and normal as independent boxes.

First version:

- sample position offsets in the local strike frame
- sample small velocity speed and angle offsets
- sample small normal rotations
- reject commands whose velocity-normal geometry deviates too far from the manifest reference

Later version:

Use the global candidate index from the approximately 700 original samples to build a strike-state distribution. KNN, PCA, or a small GMM can provide more realistic command perturbations without sending all 700 motions through expensive optimization.

## Phase 6: Multi-Strike Recovery

After single-strike WBC is stable, train longer episodes:

```text
10 s episode
multiple strike cycles
forehand/backhand transitions
balance recovery
```

The actor interface stays unchanged. Only the command scheduler changes from one cycle per episode to multiple cycles per episode.

## Phase 7: Domain Randomization

Add domain randomization after the exact-command controller is stable:

- mass and inertia
- joint friction / damping
- actuator strength
- latency
- observation noise
- racket mount perturbation
- ball state noise after planner integration

## Phase 8: Ball / Planner Conditioned Hitting

Only start after exact-command WBC and multi-strike recovery work.

Goal:

Use ball state or planner output to command a strike target.

Two options:

### C1 Planner-Commanded

This is closest to HITTER.

Planner input:

- ball state
- desired return
- table geometry

Planner output:

- target racket position
- target racket velocity
- target normal
- time-to-strike
- optional base target

Policy input:

- same actor command vector as Phase 3

This is the safest path because the policy interface stays the same.

### C2 End-To-End Ball Observation

This is closer to a pure table-tennis RL environment, but it is harder.

Policy input:

- ball position/velocity
- predicted impact point
- time-to-hit
- proprioception

Rewards:

- contact
- return over net
- landing target
- spin/velocity control

This should come later because it is harder to debug.

## Required Code Changes

### 1. Motion Manifest Loader

Add a new loader beside `MotionLoader`:

```python
class MotionLibraryLoader:
    def __init__(self, manifest_path, body_indexes, device):
        ...
```

Responsibilities:

- parse `manifest.json`
- load all motion NPZ files
- convert motion arrays to GPU tensors
- keep metadata arrays:
  - `episode_id`
  - `stroke_type`
  - `hit_frame`
  - `racket_position_m`
  - `racket_velocity_mps`
  - `racket_normal_w`
  - `racket_tangent_w`
- support variable sampling weights
- expose per-env selected motion

For the current dataset all motions are 80 frames at 50 Hz. Do not implement this as a Python list of per-motion loaders in the step loop. Store padded tensors and gather by `motion_ids` and `cycle_steps`:

```text
joint_pos[M, T_max, 31]
joint_vel[M, T_max, 31]
body_pos_w[M, T_max, 32, 3]
body_quat_w[M, T_max, 32, 4]
motion_lengths[M]
hit_frame[M]
strike_pos_w[M, 3]
strike_vel_w[M, 3]
strike_normal_w[M, 3]
```

Per-step lookup should be tensor gather:

```python
joint_target = joint_pos[motion_ids, cycle_steps]
```

Required asserts:

- `fps == expected_fps`
- `joint_pos.shape[-1] == 31`
- `body_pos_w.shape[-2] == 32`
- `0 <= hit_frame < motion_length`
- tensors are finite
- racket normals are unit length within tolerance

Also record and validate hashes when available:

- `joint_order_hash`
- `body_order_hash`
- `robot_asset_hash`
- `racket_mount_hash`

### 2. Multi-Motion Command

Extend or replace `MotionCommand` with:

```python
ManifestMotionCommand
```

State:

- `motion_ids[num_envs]`
- `strike_cycle_ids[num_envs]`
- `cycle_steps[num_envs]`
- `episode_steps[num_envs]`
- per-motion tensors
- per-motion hit metadata

Sampling:

- balanced sampling by stroke
- curriculum subset sizes: `1 -> 2 -> 4 -> 8 -> 40`
- train/validation split support
- hard-sample weights from motion and phase failure bins

Hard sampling should track:

```python
failure_ema[motion_id, phase_bin]
```

Use a capped mixture:

```text
50% stroke-balanced uniform
25% motion-hard sampling
25% motion-phase-hard sampling
```

Cap final sample probability:

```text
p_motion <= 4 * uniform_probability
```

This prevents one failing edge-case motion from dominating PPO updates.

Reset behavior:

- sample motion id
- sample start phase:
  - overfit/smoke: fixed start
  - diagnostic tracking: reference-state initialization with adaptive bins
  - exact-command WBC: bias start before hit frame
- one-strike mode: environment ends after the strike cycle
- multi-strike mode: sample a new cycle without environment reset

### 3. Manifest Strike Target Command

Add mode to `RacketTargetCommand`:

```text
target_mode: manifest
```

When `target_mode=manifest`, target fields are loaded from the current motion metadata:

- `racket_target_pos_w`
- `racket_target_vel_w`
- `racket_target_normal_w`
- `time_to_strike`
- `base_target_pos_w`

The current `reference_perturbed` mode should remain available for later generalization.

Manifest position semantics:

```text
racket_target_pos_w = impact position
```

Do not apply the old moving-target extrapolation in manifest mode:

```python
target_pos_now = target_pos - target_vel * time_to_strike
```

Impact reward should compare the actual racket state to the manifest impact state at the hit event. If old extrapolation remains useful for another target mode, guard it explicitly:

```python
if cfg.target_mode == "manifest":
    target_pos_for_reward = command.racket_target_pos_w
else:
    target_pos_for_reward = extrapolated_target_pos
```

Actor command conversion must happen after manifest targets are loaded:

```python
racket_target_pos_b = R_wb.T @ (racket_target_pos_w - base_pos_w)
racket_target_vel_b = R_wb.T @ racket_target_vel_w
racket_target_normal_b = R_wb.T @ racket_target_normal_w
```

### 4. Strike Reward Timing

Use an event-centered temporal kernel instead of a hard box reward.

Config names should make the window semantics explicit:

```yaml
racket:
  strike_time_std_s: 0.04
  strike_eval_half_window_s: 0.12
```

Reward temporal weight:

```python
tau = time_to_hit
temporal_weight = torch.exp(-0.5 * (tau / strike_time_std_s) ** 2)
```

Use `strike_eval_half_window_s` for event logging and success evaluation, not as a discontinuous training reward boundary.

### 5. Config Changes

Add top-level training config fields:

```yaml
motion_manifest: sample_motions/p2_fixed_competition_global_funnel_balanced20/manifest.json
motion_file: null
```

Add task config:

```yaml
racket:
  target_mode: manifest
  strike_time_std_s: 0.04
  strike_eval_half_window_s: 0.12
  strike_success_pos_thresh: 0.06
  strike_success_vel_thresh: 0.8
  strike_success_normal_thresh_deg: 20.0
```

### 6. Logging

Must log by stroke and by strike event:

- `forehand/exact_strike_pos_error`
- `backhand/exact_strike_pos_error`
- `forehand/strike_composite_success`
- `backhand/strike_composite_success`
- `strike/events_total`
- `strike/events_success`
- `strike/pos_error_at_event`
- `strike/vel_error_at_event`
- `strike/normal_error_at_event`
- `motion_id_histogram`
- `motion_failure_topk`
- `motion_phase_failure_topk`

The current live metric logger is useful, but it averages across all envs. That will hide backhand failure if forehand dominates. Even with balanced sampling, stroke-split logging is required.

### 7. Train / Validation Split

The manifest loader should support deterministic splits:

```text
train: 16 forehand + 16 backhand
val:    4 forehand +  4 backhand
```

Validation motions are used only for deterministic rollouts and metrics, never for PPO updates.

## Recommended First Implementation Scope

Do not implement ball physics integration yet.

Implement:

1. `motion_manifest` support in `scripts/train.py`.
2. `MotionLibraryLoader`.
3. `ManifestMotionCommand`.
4. `RacketTargetCommand.target_mode="manifest"`.
5. Actor command conversion to base frame.
6. Disable old manifest target extrapolation.
7. Event-based strike metrics.
8. Train/validation split.
9. Stroke-balanced and capped hard sampling.
10. Minimal config:
   - `HOPEPingPongManifest`

Then run:

```text
single motion overfit
2-motion smoke
4-motion exact-command WBC
8-motion exact-command WBC
32-train / 8-val exact-command WBC
```

## Suggested Experiments

### Experiment 0: Environment Sanity

Purpose:

Make sure Isaac/RSL-RL still launches after recent data work.

```bash
hope_isaac_py scripts/train.py task=TrackingFlat algo=ppo headless=true \
  logger=tensorboard num_envs=16 max_iterations=3 \
  motion_file=sample_motions/p2_fixed_competition_global_funnel_balanced20/forehand/<episode>.npz \
  run_name=sanity_one_motion
```

### Experiment 1: Forehand Overfit

```bash
hope_isaac_py scripts/train.py task=TrackingFlat algo=ppo headless=true \
  logger=tensorboard num_envs=128 max_iterations=500 \
  motion_file=sample_motions/p2_fixed_competition_global_funnel_balanced20/forehand/<episode>.npz \
  run_name=overfit_forehand
```

### Experiment 2: Backhand Overfit

Same as above with one backhand clip.

### Experiment 3: Balanced20 Manifest Tracking

This experiment is now a diagnostic only. Prefer the exact-command WBC experiments below for the deployable policy path.

```bash
hope_isaac_py scripts/train.py task=TrackingManifest algo=ppo headless=true \
  logger=tensorboard num_envs=1024 max_iterations=3000 \
  motion_manifest=sample_motions/p2_fixed_competition_global_funnel_balanced20/manifest.json \
  run_name=balanced20_tracking
```

### Experiment 4: Exact-Command K=4

After manifest strike target implementation:

```bash
hope_isaac_py scripts/train.py task=HOPEPingPongManifest algo=ppo headless=true \
  logger=tensorboard num_envs=512 max_iterations=1000 \
  motion_manifest=sample_motions/p2_fixed_competition_global_funnel_balanced20/manifest.json \
  manifest_subset_size=4 \
  run_name=exact_command_k4
```

### Experiment 5: Exact-Command K=8

```bash
hope_isaac_py scripts/train.py task=HOPEPingPongManifest algo=ppo headless=true \
  logger=tensorboard num_envs=1024 max_iterations=2000 \
  motion_manifest=sample_motions/p2_fixed_competition_global_funnel_balanced20/manifest.json \
  manifest_subset_size=8 \
  run_name=exact_command_k8
```

### Experiment 6: Exact-Command Train/Val

```bash
hope_isaac_py scripts/train.py task=HOPEPingPongManifest algo=ppo headless=true \
  logger=tensorboard num_envs=1024 max_iterations=5000 \
  motion_manifest=sample_motions/p2_fixed_competition_global_funnel_balanced20/manifest.json \
  manifest_train_split=balanced32 \
  manifest_val_split=balanced8 \
  run_name=exact_command_balanced32_val8
```

## Success Criteria

### Tracking Acceptance

For each stroke:

- low reset termination from anchor/body errors
- falling/undesired contact not dominant
- mean body position tracking error decreases
- joint position and joint velocity errors decrease
- action delta and max action remain bounded

### Strike Acceptance

For each stroke:

- composite strike success >= `70%` for bootstrap acceptance
- position error p50 < `4 cm`
- position error p90 < `7.5 cm`
- velocity vector error p50 < `0.5 m/s`
- velocity vector error p90 < `0.8 m/s`
- normal error p50 < `10 deg`
- normal error p90 < `20 deg`
- no severe joint velocity/acceleration spikes
- backhand metrics not hidden by forehand averages

Golden-quality target for the best motions:

- position error p90 < `3 cm`
- velocity vector error p90 < `0.3-0.5 m/s`
- normal error p90 < `8-10 deg`

Treat `7.5 cm` as a coarse bootstrap gate, not the final controller quality target.

## Stop Conditions

Stop training and debug if:

- single-motion overfit cannot reduce tracking error
- backhand overfit fails while forehand succeeds
- exact-strike sample count is near zero
- action magnitude saturates early
- dynamic failures resemble retargeting quality failures
- policy learns to satisfy racket reward by breaking body stability
- validation success collapses while training success rises
- one motion or one motion-phase bin dominates the hard sampler
- manifest mode still uses moving target extrapolation

## Recommended Next Step

Implement the manifest loader and manifest exact-command WBC path before long training.

Do one very short sanity run only to verify Isaac/RSL-RL launch. Do not start a multi-hour run until these contracts are implemented:

- Actor does not observe reference motion.
- Actor command is in base frame.
- Manifest target position means impact position.
- Manifest mode does not use old moving-target extrapolation.
- Strike metrics are event-based.
- Train/validation split exists.
