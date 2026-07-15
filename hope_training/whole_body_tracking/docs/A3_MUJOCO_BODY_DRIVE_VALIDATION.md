# A3 MuJoCo Body-Drive Validation

This document defines the local A3 actuator-contract validation path. It is
separate from IsaacLab training and from the official A3 MOTION controller.

## What Is Configured

The local simulator is built from:

```text
agibot/A3_MuJoCo_Sim/aimrt_mujoco_sim
```

The project entry points are:

```text
tools/build_a3_mujoco_sim.sh
tools/setup_a3_mujoco_sim_env.sh
tools/start_a3_mujoco_sim.sh
scripts/replay_body_drive_npz.py
```

The simulator exposes the same body-drive ROS2 message contract used by the
deployment example:

```text
/body_drive/{waist,neck,arm,leg}_joint_command
/body_drive/{waist,neck,arm,leg}_joint_state
/body_drive/{pelvis,torso}_imu/data
/sim/a3/reset
```

The command adapter converts the project's 31-DOF Isaac articulation order to
the four body-drive groups by joint name. It never assumes that CSV order is
runtime order.

## Actuator Semantics

The simulator's body-drive motor path computes:

```text
ctrl = effort + stiffness * (position - q) + damping * (velocity - dq)
```

The MuJoCo model then applies the actuator force range, joint limits, gravity,
contact, friction, armature and free-base dynamics. The replay adapter sends
commands at 500 Hz and linearly interpolates the 50 Hz NPZ trajectory. By
default it first runs a 3-second production-style `PD_STAND` pre-roll using the
gains from `a3_policy_parameters.hpp`, then switches to the normal motion gains
for the NPZ clip.

The default `Kp/Kd` values come from the project's A3 starter/deployment
configuration (`training/robots/agibot_a3.py` and
`a3_policy_parameters.hpp`). They are not claimed to be the complete factory
calibration of a physical A3 unit.

## Important Boundary

This simulator validates:

- ROS2 body-drive message names, fields and group order;
- 500 Hz state/command timing;
- NPZ joint mapping;
- position/velocity PD tracking;
- actuator force saturation and joint-limit behavior;
- free-base gravity/contact response;
- pelvis/torso IMU recording.

It does **not** run the official A3 `MOTION` controller or native balance policy.
The replay adapter's `PD_STAND` pre-roll is only a local command/gain
approximation, not the official controller implementation. A full-body NPZ
replay therefore cannot be treated as
an A3 native-balance test. If the floating base falls in this environment, the
first interpretation is that the commanded trajectory has no balance policy
or balance compensation in the loop, not that the reference motion is invalid
for the real robot.

The official deployment example confirms this separation: its startup path
uses a production `PD_STAND` phase before handing control to the policy, and
its normal `MOTION` path provides policy-generated commands rather than simply
replaying a fixed full-body joint trajectory.

## First Probe

Probe output:

```text
eval_outputs/a3_mujoco_body_drive/T002_015_forehand_pdstand_motion_probe/
```

The reset-synchronized probe with the PD_STAND pre-roll measured approximately:

```text
command rate:       499.61 Hz
state coverage:     100%
active q MAE:       0.070 rad
active q P95:       0.324 rad
torso tilt:         grows from near upright to about 82 deg
```

This is a valid diagnostic result: the transport and reset path work, while
fixed-trajectory full-body replay does not provide native balance. It is not a
promotion gate for training or deployment.

## Promotion Rules

Do not promote a motion or checkpoint from this probe alone. The required
sequence is:

```text
Isaac zero-residual physical tracking
        |
        v
body-drive actuator/timing probe
        |
        v
A3 native PD_STAND + MOTION validation
        |
        v
hardware-in-the-loop / real robot validation
```

The `summary.json` and `body_drive_states.npz` files are the source of truth
for command timing, joint tracking, effort and IMU diagnostics. Historical
replays and old policy checkpoints remain in their existing archive locations
and must not be mixed with the current training manifest.

## Official Native Deployment Prerequisite Check

Run:

```bash
tools/check_a3_native_deploy_prereqs.sh
```

This checks the checked-in official deployment example for the A3 policy
parameter header, the standalone simulator executable, x86_64 ONNX Runtime
headers and library, an actual deployment policy model, and the active ROS
environment.

On this workstation the standalone simulator binary, x86_64 ONNX Runtime C++
package, and deployment policy model are absent. The repository's
`mujoco_sim_standalone/run.sh` points to `bin/start_mujoco_sim.sh`, but that
binary is not included in the checked-in package. Source-only CMake configure
therefore stops at `find_package(onnxruntime REQUIRED)`. These are external
runtime artifacts; they cannot be safely replaced with a dummy model.

The local body-drive simulator and `replay_body_drive_npz.py` are consequently
the highest-fidelity runnable validation currently available in this checkout:
they use the official message contract, the checked-in A3 model's physical
limits, and the official stand gains, but they do not reproduce the proprietary
native A3 `MOTION` balance policy.
