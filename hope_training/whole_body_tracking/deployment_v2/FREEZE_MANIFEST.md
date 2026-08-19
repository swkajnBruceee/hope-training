# Deployment-Aligned High-Level SAC V2

## HOPE Open-Source Software Contract v1

Freeze date: 2026-08-19

Status:

```text
SOFTWARE_CONTRACT_FROZEN
PROTOTYPE_NOT_TRAINED
```

Canonical low-level: `model_21800`

Model SHA256:

`6bf1a2418f8538e23577a0153f2fe6a1e78dee91f41650a232259432a84a4dc8`

Architecture:

```text
HOPE PredictedStrike
→ HOPE solver
→ /racket/command
→ Deployment-Aligned High-Level SAC V2
→ Schema-2 Adapter
→ /racket/command_flat
→ frozen HOPE native runner
→ model_21800
→ 31D
```

Observation:

```text
OBS_DIM=7
[
 predicted_intercept_y_world,
 predicted_intercept_z_world,
 incoming_vx_world,
 incoming_vy_world,
 incoming_vz_world,
 control_time_to_intercept,
 swing_sign
]
```

Action:

```text
ACTION_DIM=3
normalized [-1,1]^3
→ side-specific HOPE planner velocity box
```

Physical semantics:

```text
[
 racket_vx_world,
 racket_vy_world,
 racket_vz_world
]
```

Forehand planner box:

```text
vx [1.57,2.55]
vy [0.10,0.52]
vz [0.41,1.35]
```

Backhand planner box:

```text
vx [1.55,2.52]
vy [-0.18,0.29]
vz [0.40,1.32]
```

Position contract:

`HOPE_OPEN_SOURCE_SOLVER_POSITION`

Frame contract:

`HOPE_OPEN_SOURCE_WORLD_TABLE_FRAME_CODE_0`

Side contract:

`NEAREST_STATION_THEN_FLIGHT_LOCK`

Timing contract:

`AGED_TTS_PLUS_WALL_REANCHOR`

Estimator metadata:

```text
sample_count=0 permitted
span_s=0.0 permitted
```

Explicit boundaries:

```text
REAL_HARDWARE_TCP_CALIBRATION_VALIDATED=FALSE
PHYSICAL_MOCAP_TABLE_CALIBRATION_VALIDATED=FALSE
SIM2REAL_VALIDATED=FALSE
REAL_ROBOT_VALIDATED=FALSE
SAC_V2_TRAINED=FALSE
ISAAC_V2_ENV_VALIDATED=FALSE
```

Next stage:

`V2-B Deployment-Aligned Isaac Environment`
