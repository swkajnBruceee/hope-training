# LTL Remote Server Usage Notes

> Security note: this document contains plaintext server passwords because the current workflow requires it. Do not publish this file to a public repository or share it outside the project team.

## Server Account

Primary server account:

```bash
ssh dbcloudlabAccess123@10.148.100.21 -p 20052
```

Password:

```text
Testtesttest_1
```

Old server account, kept only for reference:

```bash
ssh admin@10.148.100.21 -p 20028
```

Password:

```text
1Q2w3e4r_RoboMaster
```

## Directory Rule

The shared server must not be polluted outside this project directory. All project files, environments, logs, and tools are under:

```bash
/attached/remote-home-21/dbcloudlabAccess123/ltl
```

Important paths:

```bash
# Project code
/attached/remote-home-21/dbcloudlabAccess123/ltl/workspace/hope-training

# Isaac/learning environment
/attached/remote-home-21/dbcloudlabAccess123/ltl/env/conda-envs/hope310

# ROS environment
/attached/remote-home-21/dbcloudlabAccess123/ltl/env/conda-envs/hope_ros

# Helper scripts
/attached/remote-home-21/dbcloudlabAccess123/ltl/tools/enter_hope310.sh
/attached/remote-home-21/dbcloudlabAccess123/ltl/tools/enter_hope_ros.sh
```

The account's normal home directory is not the working area. Always use the `ltl` path above.

## Login And Enter Environments

SSH into the server:

```bash
ssh dbcloudlabAccess123@10.148.100.21 -p 20052
```

Enter the Isaac Sim / Isaac Lab / training environment:

```bash
source /attached/remote-home-21/dbcloudlabAccess123/ltl/tools/enter_hope310.sh
```

Enter the ROS 2 environment:

```bash
source /attached/remote-home-21/dbcloudlabAccess123/ltl/tools/enter_hope_ros.sh
```

After sourcing either script, the shell should automatically move to the corresponding project workspace.

## Current Verified Status

The server was verified on 2026-07-07 / 2026-07-08.

Training stack:

- Isaac Sim 4.5.0 is available.
- Isaac Lab 2.1.0 is available under the project workspace.
- PyTorch is available in the `hope310` environment.
- CUDA sees the RTX A6000 GPUs.
- Vulkan and Xvfb are available for headless simulation.
- `pytest tests -q` passed with `6 passed`.
- Training smoke tests passed for:
  - `HitFixedBaseTouch`
  - `HitFixedBase`
  - `TrackingFlat`
  - `HOPEPingPong`
- `play.py` can load a checkpoint headlessly and export `policy.onnx`.

ROS stack:

- ROS 2 Humble user-space environment is installed in `hope_ros`.
- `hope_ws` was rebuilt from a clean `build/install/log`.
- 8 ROS packages compiled successfully:
  - `bringup`
  - `calibration`
  - `common`
  - `decision`
  - `msgs`
  - `solver`
  - `tools`
  - `trajectory`
- `colcon test-result --verbose` summary:

```text
65 tests, 0 errors, 0 failures, 5 skipped
```

Runtime smoke-tested ROS nodes:

```text
decision/decision_node
solver/solver_node
trajectory/strike_prediction_node
trajectory/trajectory_overlay_udp_node
```

## Common Training Commands

Enter the training environment first:

```bash
source /attached/remote-home-21/dbcloudlabAccess123/ltl/tools/enter_hope310.sh
```

Go to the training code:

```bash
cd /attached/remote-home-21/dbcloudlabAccess123/ltl/workspace/hope-training
```

Run the Python tests:

```bash
pytest tests -q
```

Use headless mode for server training. This server does not need a physical desktop session for training.

The previous smoke tests used the repository's Isaac Lab training scripts under:

```bash
hope_training/whole_body_tracking/scripts
```

For longer training, start from the same commands/tasks that passed smoke testing, then increase iterations and logging options as needed.

## Common ROS Commands

Enter the ROS environment:

```bash
source /attached/remote-home-21/dbcloudlabAccess123/ltl/tools/enter_hope_ros.sh
```

Go to the ROS workspace:

```bash
cd /attached/remote-home-21/dbcloudlabAccess123/ltl/workspace/hope-training/hope_ws
```

Clean and rebuild:

```bash
rm -rf build install log
colcon build --event-handlers console_direct+ --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

Run tests:

```bash
colcon test --event-handlers console_direct+
colcon test-result --verbose
```

Check project packages:

```bash
ros2 pkg list | grep -E '^(bringup|calibration|common|decision|msgs|solver|tools|trajectory)$' | sort
```

Launch individual nodes:

```bash
ros2 run decision decision_node
ros2 run solver solver_node
ros2 run trajectory strike_prediction_node
ros2 run trajectory trajectory_overlay_udp_node
```

If you only want to check whether a node starts, use a timeout:

```bash
timeout 5s ros2 run decision decision_node
```

Exit code `124` from `timeout` means the node kept running until the timeout killed it. For long-running ROS nodes, that is usually a successful smoke test.

## Headless Graphics Notes

The server does not need a visible GUI for training, but Isaac Sim still needs graphics/driver support. The current setup uses headless execution with Xvfb/Vulkan available.

Useful checks:

```bash
vulkaninfo --summary
ldconfig -p | grep -i nvidia
```

If Xvfb is needed manually:

```bash
nohup Xvfb :99 -screen 0 1024x768x24 -ac -nolisten tcp > xvfb.log 2>&1 &
export DISPLAY=:99
export XDG_RUNTIME_DIR=/tmp
```

## Known Local Code Fixes Synced To Server

The following local files had fixes that were synced to the server during setup:

```text
hope_training/whole_body_tracking/scripts/create_smoke_motion.py
hope_training/whole_body_tracking/scripts/play.py
hope_training/whole_body_tracking/scripts/train.py
hope_training/whole_body_tracking/training/tasks/tracking/mdp/hope_observations.py
```

These fixes were needed for reliable headless smoke testing, correct failure reporting, deterministic short `play.py` runs, and Isaac Lab initialization behavior.

## Quick Health Check

Use this after logging in to confirm the server is still usable:

```bash
source /attached/remote-home-21/dbcloudlabAccess123/ltl/tools/enter_hope310.sh
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY
```

Then check ROS:

```bash
source /attached/remote-home-21/dbcloudlabAccess123/ltl/tools/enter_hope_ros.sh
cd /attached/remote-home-21/dbcloudlabAccess123/ltl/workspace/hope-training/hope_ws
colcon test-result --verbose
```

If both pass, the training and ROS chains are still in the expected state.
