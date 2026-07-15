# Official A3 AimSim Runtime

This is the authoritative local validation path for A3 native motion control.
It is separate from the historical custom `A3_MuJoCo_Sim` body-drive probe and
from IsaacLab PPO training.

## Installed Assets

```text
AimSim/aimsim-3.3-py3-none-any.whl
third_party/aimsim_official/motion_control_humble/
third_party/aimsim_official/user_config/
.venv/aimsim/
```

The Humble package is selected because the project ROS workspace reports
`ROS_DISTRO=humble`. The Jazzy tarball is intentionally not extracted or mixed
with this runtime.

## Start

From the repository root:

```bash
source hope_training/whole_body_tracking/tools/setup_official_aimsim_env.sh
hope_training/whole_body_tracking/tools/start_official_aimsim_a3.sh
```

The launcher starts the official `motion_control` first, then the official
AimSim MuJoCo SIL process for `raise_a3_t2d5`. It uses the official HTTP action
service on `127.0.0.1:56322` and AimSim's state service on `127.0.0.1:8001`.

## Official Action Sequence

The official action sequence is:

```text
GET_UP -> MOTION -> locomotion / upper-body interface
```

Useful state checks:

```bash
curl http://127.0.0.1:8001/liveness
curl http://127.0.0.1:8001/imu
curl http://127.0.0.1:8001/joint_states
```

## Current Verification

- Official A3 T2D5 `motion_control` binary starts successfully.
- Official A3 T2D5 resources load, including 31 active joints and native
  GET_UP/PD_STAND/MOTION configurations.
- `GET_UP` returned `CommonState_SUCCESS` and reached `GET_UP_FINISHED`.
- `MOTION` returned `CommonState_SUCCESS`.
- AimSim liveness and joint/IMU HTTP endpoints are online.
- At verification time, pelvis pitch was about 4.4 degrees, torso pitch about
  0.5 degrees, and both roll values were near zero.

## Known Non-Fatal Warnings

The official x86 package logs missing real-robot calibration files under
`/agibot/data/param/calibration/`. These are hardware calibration resources,
not copied into this project and must not be fabricated. The official SIL
simulation still starts and completes the action transition, but those
warnings must be resolved or explicitly accepted before claiming a real-robot
calibration-equivalent result.

The package may also print a ROS2 `rcutils` allocation warning while publishing
the simulated command messages. It did not stop the official processes during
this validation and should be tracked separately from controller behavior.

## Stop

Press `Ctrl+C` in the launcher terminal. The project launcher stops the
official Motion Control child and its `iox-roudi` instance.
