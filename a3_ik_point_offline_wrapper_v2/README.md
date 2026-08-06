# A3 IkPointArmSource offline wrapper v2

This project reuses the supplied `mc` source tree to generate explicit
**forehand and backhand kinematic candidate** right-arm strike trajectories.
It does not start ROS, AimRT, HAL, the official motion controller, or a real
robot process.

## What changed in v2

- explicit `forehand`, `backhand`, and diagnostic `auto` goal support;
- stroke-labelled READY states with mismatch fail-fast checks;
- separate forehand/backhand example goals and READY templates;
- dual-stroke batch generation with per-stroke output directories;
- requested/selected stroke and READY metadata in JSON and NPZ;
- a summary file that keeps forehand and backhand candidates from being mixed.

The underlying `IkPointArmSource` already implements both IK branches. The main
v2 change is the surrounding data contract, not a replacement IK solver.

## Output status

A successful run produces only a `KINEMATIC_CANDIDATE`. It is not automatically
an Isaac-FK-qualified, fixed-base-qualified, floating-base-qualified, or teacher
trajectory.

## Dependencies

```bash
sudo apt install build-essential cmake libeigen3-dev python3-numpy
```

The wrapper prefers the supplied `mc/deps/fetchcontent/yaml-cpp_src` tree. If
that is absent, install `libyaml-cpp-dev`.

## Build and generate both stroke examples

Extract the original archive so `/path/to/mc/arm/...` exists, then run:

```bash
./scripts/build_and_run.sh /path/to/mc ./dual_stroke_example_output
```

The output layout is:

```text
dual_stroke_example_output/
  forehand/<goal_id>/
    trajectory_100hz.csv
    trajectory_100hz.npz
    normalized_goal.json
    diagnostics.json
  backhand/<goal_id>/
    ...
  generation_summary.json
  generation_summary.csv
```

## Generate one explicit stroke

Forehand:

```bash
./build/a3_generate_strike_reference \
  --goal examples/goal_forehand.yaml \
  --ready examples/ready_forehand_template.yaml \
  --planner-config /path/to/mc/arm/hit_ik_point.yaml \
  --robot-xml /path/to/mc/models/hit/kinematics/a3_t2d5.xml \
  --output-dir output/forehand/example \
  --control-hz 100
```

Backhand:

```bash
./build/a3_generate_strike_reference \
  --goal examples/goal_backhand.yaml \
  --ready examples/ready_backhand_phase0.yaml \
  --planner-config /path/to/mc/arm/hit_ik_point.yaml \
  --robot-xml /path/to/mc/models/hit/kinematics/a3_t2d5.xml \
  --output-dir output/backhand/example \
  --control-hz 100
```

## READY-state contract

Each READY YAML now declares:

```yaml
ready_id: unique_name
swing_type: forehand  # or backhand/shared/auto
```

An explicit forehand goal cannot silently use a READY marked `backhand`, and
vice versa. `--allow-ready-mismatch` exists only for diagnostics.

`ready_forehand_template.yaml` is built from the accepted forehand seed embedded
in the supplied source. It is **not** a PhysX-qualified forehand READY state.
Replace both templates with exact local post-reset/post-settle 31-DOF states.

For dataset production, prefer two explicit runs—one per stroke—over `auto`.
Auto selection evaluates branches from one current READY state and can bias the
chosen branch, which is undesirable for clean labelled training data.

## Dual-stroke manifest

`examples/dual_stroke_manifest.json` lists goal YAMLs and their explicit labels.
Run a larger dataset with:

```bash
python3 scripts/generate_dual_stroke_dataset.py \
  --binary build/a3_generate_strike_reference \
  --manifest /path/to/manifest.json \
  --ready-forehand /path/to/settled_forehand_ready.yaml \
  --ready-backhand /path/to/settled_backhand_ready.yaml \
  --planner-config /path/to/mc/arm/hit_ik_point.yaml \
  --robot-xml /path/to/mc/models/hit/kinematics/a3_t2d5.xml \
  --output-root /path/to/output \
  --csv-to-npz scripts/csv_to_npz.py \
  --continue-on-error
```

The manifest label must agree with the `swing_type` inside each goal YAML.
`auto` goals are intentionally rejected by this dataset wrapper.

## Planner-native example goals

The supplied example goals are centred inside the source planner's configured
forehand/backhand workspaces and use its nominal velocity and normal-angle
conventions. They are smoke-test templates, not replacements for your own
canonical task goals.

## Mandatory local checks

1. Replace both READY templates with exact local post-settle states.
2. Verify the generator's racket mount and FK against Isaac FK.
3. Keep canonical targets in world or immutable initial-base-heading coordinates.
4. Replay each candidate under fixed-base PhysX before adding it to a candidate manifest.
5. Perform floating-base qualification separately after the lower-body P0 gate passes.
6. Keep forehand/backhand qualification statistics separate until both contracts pass.
