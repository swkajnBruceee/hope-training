# Official A3 Control Chain Status

Updated: 2026-07-16

## Official policy asset audit (2026-07-17)

The local official AimSim package was audited with
`agibot/code_deployment/a3_deploy_example/tools/audit_official_policy_models.py`.
The deployment package contract remains:

```text
input tensor  = obs_dict [1, 1570] float32
output tensor = action   [1, 29]   float32
```

The audit found no local model satisfying all four conditions.  The closest
official A3 T2D5 assets are different contracts, for example:

```text
ling_chuang_policy_dof29/jieweiduixing.onnx  obs [1,157]  -> actions [1,29]
get_up 29dof models                        obs [1,151]  -> actions [1,29]
ling_chuang_policy models                  obs [1,137]  -> actions [1,25]
human_like models                          input [1,3630] -> output [1,15] + est_vel
```

They are official models, but they are not interchangeable with the
deployment runtime policy.  In particular, the local runtime also requires
the tensor names `obs_dict` and `action`; the `jieweiduixing` model was
actually passed through the runtime probe and rejected because its input is
named `obs`.  No model from these groups is therefore copied, renamed, or
used as a substitute.

The user-provided `/home/bruce/下载/a3_t2d5+.zip` was also inspected.  It
contains only the A3 T2D5 URDF, STL meshes, and README; it contains no ONNX
policy.  The official policy-compatible model remains an external asset that
must be supplied by the A3 deployment package/vendor before the TA to policy
to body-drive loop can be closed.

## Verified

- The official AimSim/MOTION SIL is running locally.
- The official TA source and protobuf schema are present in
  `agibot/code_deployment/a3_deploy_example`.
- The official Unitree SDK2 x86 library and DDS libraries are available in the
  deployment example's `thirdparty/` tree.
- The x86 deployment target builds with the real AimRT backend enabled:
  `ENABLE_A3_AIMRT_BACKEND=ON`, `ENABLE_A3_ROS_MSGS=ON`.
- The build generates `TaWholeBodyCommand` protobuf bindings and includes
  `A3AimrtBackend::SendCommand` plus the official AimRT Core symbols.
- A backend-only dry-run successfully loaded the iceoryx and ROS2 plugins and
  registered `/ta/whole_body_command` through both configured backends.
- A C++ ROS2 probe published one valid `BODY_30` `TaWholeBodyCommandChannel`
  protobuf frame to the encoded ROS2 topic, and the backend logged
  `accepted TA frame` with the decoded waist and right-arm fields.

The dry-run was intentionally stopped by a timeout. It used
`--dry-run`, so it did not load a policy or publish robot commands.

The message probe also ran with the deployment's dry-run flag. It validated
protobuf serialization, ROS2 wrapper transport, AimRT subscription, and A3
29-DOF conversion only; it did not publish body-drive joint commands.

## Runtime command used for the successful dry-run

Before starting the binary, run the read-only asset preflight:

```bash
cd /home/bruce/桌面/HOPETableTennis/agibot/code_deployment/a3_deploy_example
python3 tools/check_a3_deploy_assets.py --json
```

The current result is `dry_run_ready=true` and `policy_ready=false`. The
packaged transport libraries are present, but the configured official
`model_step_098000_a3.onnx` is not present locally. Do not substitute one of
the unrelated `a3_t2d0` or `a3_ultra_t2d5` motion-policy files: matching the
filename or output DOF count is not enough to establish observation and
normalization compatibility.

The ROS2 library path must be internally consistent. On this machine the
complete `hope_ros` prefix must come before the auxiliary `hope_ros310` prefix;
mixing individual ROS libraries from both prefixes causes ABI errors such as
`rcl_clock_time_started`.

```bash
cd /home/bruce/桌面/HOPETableTennis/agibot/code_deployment/a3_deploy_example
export LD_LIBRARY_PATH=/workspace/anaconda3/envs/hope_ros/lib:/workspace/anaconda3/envs/hope_ros310/lib:/tmp/onnxruntime-linux-x64-1.19.2/lib:/tmp/zmq-dev-probe/root/usr/lib/x86_64-linux-gnu:dist/a3_deploy_x86_64
timeout 20 dist/a3_deploy_x86_64/a3_deploy_onnx_ref \
  --runtime-cfg=src/a3/a3_deploy_onnx_ref/config/a3_runtime_config.yaml \
  --dry-run \
  --aimrt-cfg=src/a3/a3_deploy_onnx_ref/config/a3_aimrt_config.yaml
```

Expected evidence includes:

```text
Load plugin 'iceoryx_plugin' succeeded.
Load plugin 'ros2_plugin' succeeded.
subscribe topic '/ta/whole_body_command' success
teleop subscriber enabled: /ta/whole_body_command
```

## Important boundary

This proves the local official TA/AimRT transport and command schema are
usable. It does not prove that a real robot is connected, that official policy
ONNX assets are present, or that a command is safe on hardware. The official
MOTION process currently owns its active control path; a direct waist RPC was
accepted but not observed while MOTION was active. Therefore the next hardware
step is a controlled TA-to-SIL command test, followed by a no-load hardware
test, not blind policy deployment.

## Waist control isolation result (2026-07-16)

The waist path was tested at two different layers. They must not be treated as the
same interface:

1. `POST /channel/%2Fmotion%2Fcontrol%2Fwaist_joint_command/...` was accepted by
   the official MOTION HTTP/ROS2 input channel. While the official MOTION/PD_STAND
   action was active, the requested waist offset was not observed in the final joint
   state. The official `MotionControlModule` continued publishing its own
   `/body_drive/waist_joint_command`, so this test proves input acceptance only.
2. The deployment `A3AimrtBackend::SendCommand` was tested against the official
   low-level iceoryx body-drive channel with the official MotionControl publisher
   isolated. A target offset of approximately `[+0.08, +0.04, -0.02]` rad produced
   an observed waist delta of `[+0.0557, +0.0302, -0.0112]` rad in the official
   MuJoCo SIL. Yaw and roll followed most of the request; pitch moved in the
   requested direction but retained a larger residual under the short probe. This
   verifies the transport, 29-DOF adapter, body-drive topic, and SIL actuator path.
   It does not claim perfect position tracking or real-hardware safety.

The practical conclusion is:

```text
TA protobuf / ROS2 input       = message-level integration verified
body-drive iceoryx output      = low-level waist command path verified in SIL
official MOTION waist override = not the deployment command path
```

The local probe is guarded by `A3_LOCAL_SIL=1` and never targets a real robot:

```bash
cd /home/bruce/桌面/HOPETableTennis/agibot/code_deployment/a3_deploy_example
export A3_LOCAL_SIL=1
export LD_LIBRARY_PATH=/workspace/anaconda3/envs/hope_ros/lib:/workspace/anaconda3/envs/hope_ros310/lib:/tmp/onnxruntime-linux-x64-1.19.2/lib:/tmp/zmq-dev-probe/root/usr/lib/x86_64-linux-gnu:dist/a3_deploy_x86_64
dist/a3_deploy_x86_64/a3_deploy_onnx_ref \
  --runtime-cfg=src/a3/a3_deploy_onnx_ref/config/a3_runtime_config.yaml \
  --body-drive-probe \
  --aimrt-cfg=src/a3/a3_deploy_onnx_ref/config/a3_aimrt_config.yaml
```

Run this only with the official MotionControl publisher isolated. If both publishers
write the same body-drive topic, the last writer wins and the result is not a valid
adapter test.

## Build/runtime packaging fix

The deployment CMake now copies `libirobot_events_executor.so` beside the
AimRT ROS2 plugin. Without this dependency, the binary compiled successfully
but failed during plugin loading on a clean runtime directory.

When `onnxruntime_ROOT` is explicitly provided, the CMake package also copies
the ONNX Runtime shared libraries into `dist/a3_deploy_x86_64`. The packaged
dry-run therefore no longer needs the temporary ONNX Runtime directory in
`LD_LIBRARY_PATH`.

## Current Local SIL Limitation (2026-07-16)

The official MuJoCo SIL and the low-level body-drive waist probe remain
available. The official MOTION executable itself was also started and its
configuration was validated, but it does not remain persistent on this host
because its default file logger resolves to the unwritable `/agibot/log/mc`
path. The isolated body-drive result above is therefore the authoritative
low-level command-path evidence; it must not be described as an active
official MOTION override test. A user-writable official MOTION log
configuration is still required before repeating the high-level runtime test.

## High-Level Waist Ownership Audit (2026-07-16)

The direct waist probe was repeated after correcting command-rate handling: the
source NPZ is 50 Hz, while the public command probe is sent at 100 Hz. The
probe now interpolates the trajectory to 100 Hz while preserving the original
1.58 s duration. This prevents a false two-times speed-up in validation.

Results:

| Official action | Requested waist span (rad) | Observed span (rad) | Result |
| --- | ---: | ---: | --- |
| `MOTION` | `[0.307, 0.084, 0.184]` | `[0.00003, 0.00037, 0.00035]` | accepted by HTTP, not consumed |
| `PD_STAND` | `[0.307, 0.084, 0.132]` | `[0.0031, 0.0029, 0.0227]` | not meaningful tracking |

The raw reports are under:

```text
hope_training/whole_body_tracking/eval_outputs/
  official_a3_waist_reference_probe_20260716/
```

This does **not** mean the A3 waist cannot be controlled. The isolated official
deployment body-drive probe already observed a bounded waist response in SIL.
It means the standalone `motion_control` HTTP waist topic is not an
independent override while its high-level action is active.

The official deployment binary does register the TA whole-body channel:

```text
/ta/whole_body_command
  -> pb:aimdk.protocol.TaWholeBodyCommandChannel
  -> policy view waist(3) + arm(14) + leg(12)
```

The dry-run was verified through both AimRT iceoryx and the ROS2 wrapper; the
ROS2 encoded topic appeared as:

```text
/ta/whole_body_command/pb_3Aaimdk_2Eprotocol_2ETaWholeBodyCommandChannel
```

This is the correct next ownership path. TA message-level subscription and
field mapping are verified, but a real TA frame has not yet been promoted to a
command-output test in the official deployment loop because the local package
does not contain the configured official policy model. Until that test is run,
the active training checkpoint remains `sim_only` and must not be described as
deployment-ready.
