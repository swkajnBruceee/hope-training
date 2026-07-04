# Roadmap

## v1 Public Starter

- Keep the existing HOPE reference documents and rules.
- Add A3 ping-pong URDF/meshes and the Isaac Lab starter task.
- Provide one-command asset preparation for Isaac Lab.
- Provide a local smoke motion generator.
- Run `TrackingFlat` PPO smoke training with TensorBoard and no WandB
  requirement.
- Include Agibot-provided A3 reference materials under `agibot/`, including URDF
  variants, the MuJoCo/AimRT simulation reference, and the A3 deployment
  example.

## Optional: ROS and Mocap

- Keep the ROS 2 Jazzy workspace skeleton under `hope_ws/`.
- Keep motion-capture reference docs under `data/mocap/`.
- Do not vendor the upstream `vrpn_mocap` package in v1. Teams that need live
  VRPN should install or clone it separately into their ROS 2 workspace.
- Treat real arena calibration, OptiTrack deployment, and live planner wiring as
  integration work for each team.

## Optional: MuJoCo

- Keep the Agibot-provided MuJoCo/AimRT reference project under
  `agibot/A3_MuJoCo_Sim/`.
- Treat it as an optional reference path; the validated v1 onboarding flow is
  Isaac Lab setup and smoke training.
- Do not claim a public MuJoCo RL backend in v1.

## Future Work

- Retargeted public forehand/backhand motion examples.
- Better reward defaults and validated training recipes.
- Full real-robot deployment gate docs with reproduced dry-run, joint-order,
  command-scale, low-gain, emergency-stop, and safe-halt verification.
- CI for non-Isaac checks and optional GPU smoke jobs.
