# ROS Topics

The ROS 2 workspace under `hope_ws/` is optional in the public starter. It is
kept as a skeleton for teams that want to connect motion capture, planner, and
robot-control components.

## Mocap Relay Outputs

Expected public-facing topics after the VRPN relay:

- `/poses`: pose array containing the configured tracked objects.
- `/ball/point`: ball position when available.
- `/table/pose`: table/world reference pose.
- `/P1/pose`: Player 1 robot base pose.
- `/P2/pose`: Player 2 robot base pose.

Raw VRPN topics live under `/vrpn_mocap/...` and depend on the mocap server
object names.

`vrpn_mocap` is an external optional ROS 2 dependency. It is not vendored in
this starter branch.

## Planner Command

The starter message package includes:

- `msgs/msg/RacketCommand.msg`

QoS convention:

- `/poses` input: best-effort, depth 1, for high-rate mocap sensor data.
- `/racket/command` output: reliable, keep-last depth 10, because it is a
  control setpoint consumed by the WBC side.

This is reference material for future planner/WBC integration. The v1 quickstart
does not require live ROS topics.

## A3 Body-Drive Deployment Topics

The optional Agibot A3 deploy example under `agibot/code_deployment/` uses the A3
body-drive interface. The backend consumes six state streams and publishes four
command streams:

- State: `/body_drive/waist_joint_state`
- State: `/body_drive/leg_joint_state`
- State: `/body_drive/arm_joint_state`
- State: `/body_drive/neck_joint_state`
- State: `/body_drive/pelvis_imu/data`
- State: `/body_drive/torso_imu/data`
- Command: `/body_drive/waist_joint_command`
- Command: `/body_drive/leg_joint_command`
- Command: `/body_drive/arm_joint_command`
- Command: `/body_drive/neck_joint_command`

Use probe or dry-run modes first so command publication remains disabled during
transport and sync bring-up.

## Official A3 TA Whole-Body Command

The official A3 teleoperation path is separate from the body-drive state/command
topics above. Its input channel is:

- `/ta/whole_body_command`

The protobuf payload contains the official TA command groups: leg, waist, head,
arm, and optional hand/pelvis fields. The local mapping mirror is kept in
`hope_training/whole_body_tracking/tools/official_ta_mapping.py`; its policy
view is the official 29-DOF layout (12 leg + 3 waist + 14 arm), while the A3
deployment backend can expand to the robot's 31-DOF layout.

The official AimRT backend is only considered available when the executable is
built with both `ENABLE_A3_AIMRT_BACKEND=ON` and `ENABLE_A3_ROS_MSGS=ON`. The
backend-off executable must not be used as evidence that the TA channel works.
