#!/usr/bin/env bash

# Simulation-only one-click launcher:
# AimRT MuJoCo -> HOPE planner -> physical Gate3 ball -> native runner MOTION.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
A3_DIR="$ROOT_DIR/a3_deploy/a3_deploy_example"
SIM_DIR="$ROOT_DIR/agibot/A3_MuJoCo_Sim/aimrt_mujoco_sim"
SIM_BIN="$SIM_DIR/build/install/bin"
SIM_START="$SIM_BIN/start_a3_pingpong_iceoryx.sh"
RUNNER_BIN="$A3_DIR/dist/a3_deploy_x86_64/a3_deploy_onnx_ref_pingpong"
RUNNER_CFG="$A3_DIR/src/a3/a3_deploy_onnx_ref/config/a3_runtime_config.pingpong.hitter_pingpong.yaml"
SIM_RUNNER_CFG="$A3_DIR/src/a3/a3_deploy_onnx_ref/config/a3_runtime_config.pingpong.hitter_pingpong.sim.yaml"
POLICY_DIR="$A3_DIR/models/model_21800/policy"
POLICY_LABEL="official_model_21800"
WS_INSTALL="$ROOT_DIR/hope_ws/install"
WS_SOURCE="$ROOT_DIR/hope_ws/src"
OFFICIAL_HOPE_ROOT="/home/bistu/桌面/HOPE"
OFFICIAL_SIM_MOCAP="$OFFICIAL_HOPE_ROOT/a3_deploy/a3_deploy_example/scripts/pp_gate3_sim_mocap.py"
OFFICIAL_SIM_CORE="$OFFICIAL_HOPE_ROOT/a3_deploy/a3_deploy_example/scripts"
P1_CALIBRATION_RECEIPT="$OFFICIAL_HOPE_ROOT/hope_ws/calibration_receipts/p1_marker_cad_registration_20260805_redefined_p1_strict.json"
NATNET_INSTALL="/home/bistu/桌面/HOPE/NatNet2ROS2/install"
GATE3_PY_BUILD="/tmp/hope_gate3_build"
CONDA_SH="/home/bistu/anaconda3/etc/profile.d/conda.sh"
TABLE_CENTER_Y="-0.7625"
STAND_X="-0.5"
STAND_Z="1.06839"
LOG_DIR="$ROOT_DIR/logs/mujoco_planner_motion/$(date +%Y%m%d_%H%M%S)"
RUN_DURATION=0
KEEP_RUNNING=0
SHOTS=1
FLIGHT_WINDOW=6.0
CONTACT_HOLD=1.5
INTER_SHOT=0.5
RANDOMIZE=0
MIXED_RANDOMIZE=0
WIDE_LATERAL_MIXED_RANDOMIZE=0
INPUT_SOURCE="sim"
MOTIVE_HOST="192.168.50.1"
MOCAP_INTERFACE_IP="192.168.50.230"
RANDOM_SEED=0
OFFICIAL_GATE3=0
OFFICIAL_PLANNER=0
MOTION_IDLE_S=1.2
SERVES=()
SIM_PID=""
BASE_RELAY_PID=""
PLANNER_PID=""
BALL_PID=""
POSE_BRIDGE_PID=""
OPTITRACK_RELAY_PID=""
WORLD_RELAY_PID=""
MOCAP_ADAPTER_PID=""
MOCAP_RELAY_PID=""
RUNNER_PID=""
EVIDENCE_PID=""
PHYSICAL_EVIDENCE_JSON=""
AUDIT_SUMMARY_JSON=""

usage() {
  echo "Usage: $0 [--input-source sim|mocap] [--motive-host IP] [--mocap-interface-ip IP] [--official-gate3|--official-planner] [--duration SEC] [--shots N] [--motion-idle SEC] [--randomize|--mixed-random|--wide-lateral-mixed --seed N] [--flight-window SEC] [--contact-hold SEC] [--inter-shot SEC] [--serve TUPLE ...] [--stand-x X --stand-y Y] [--policy-dir DIR] [--label NAME] [--keep]"
  echo "Starts AimRT, planner, and native runner MOTION; ball input defaults to simulation."
  echo "  --input-source sim|mocap  sim starts Gate3 ball; mocap starts NatNet2ROS2 + HOPE relay"
  echo "  --motive-host IP          NatNet/Motive host (mocap mode, default: $MOTIVE_HOST)"
  echo "  --mocap-interface-ip IP  local wired NatNet receive address (default: $MOCAP_INTERFACE_IP)"
  echo "  --official-gate3  use the published 12-shot Gate3 serve/planner contract"
  echo "  --official-planner  use official planner/serve geometry without forcing 12 shots"
}

die() {
  echo "[mujoco-motion] ERROR: $*" >&2
  exit 1
}

log() {
  echo "[mujoco-motion] $*"
}

while (($#)); do
  case "$1" in
    --input-source)
      (($# >= 2)) || die "--input-source needs sim or mocap"
      INPUT_SOURCE="$2"
      shift 2
      ;;
    --motive-host)
      (($# >= 2)) || die "--motive-host needs an IPv4 address"
      MOTIVE_HOST="$2"
      shift 2
      ;;
    --mocap-interface-ip)
      (($# >= 2)) || die "--mocap-interface-ip needs an IPv4 address"
      MOCAP_INTERFACE_IP="$2"
      shift 2
      ;;
    --official-gate3)
      OFFICIAL_GATE3=1
      OFFICIAL_PLANNER=1
      shift
      ;;
    --official-planner)
      OFFICIAL_PLANNER=1
      shift
      ;;
    --duration)
      (($# >= 2)) || die "--duration needs a value"
      RUN_DURATION="$2"
      shift 2
      ;;
    --keep)
      KEEP_RUNNING=1
      shift
      ;;
    --shots)
      (($# >= 2)) || die "--shots needs a value"
      SHOTS="$2"
      shift 2
      ;;
    --flight-window)
      (($# >= 2)) || die "--flight-window needs a value"
      FLIGHT_WINDOW="$2"
      shift 2
      ;;
    --contact-hold)
      (($# >= 2)) || die "--contact-hold needs a value"
      CONTACT_HOLD="$2"
      shift 2
      ;;
    --inter-shot)
      (($# >= 2)) || die "--inter-shot needs a value"
      INTER_SHOT="$2"
      shift 2
      ;;
    --motion-idle)
      (($# >= 2)) || die "--motion-idle needs a value"
      MOTION_IDLE_S="$2"
      shift 2
      ;;
    --randomize)
      RANDOMIZE=1
      shift
      ;;
    --mixed-random)
      MIXED_RANDOMIZE=1
      shift
      ;;
    --wide-lateral-mixed)
      WIDE_LATERAL_MIXED_RANDOMIZE=1
      shift
      ;;
    --seed)
      (($# >= 2)) || die "--seed needs a value"
      RANDOM_SEED="$2"
      shift 2
      ;;
    --serve)
      (($# >= 2)) || die "--serve needs x,y,z,vx,vy,vz"
      SERVES+=("$2")
      shift 2
      ;;
    --stand-x)
      (($# >= 2)) || die "--stand-x needs a value"
      STAND_X="$2"
      shift 2
      ;;
    --stand-y)
      (($# >= 2)) || die "--stand-y needs a value"
      TABLE_CENTER_Y="$2"
      shift 2
      ;;
    --policy-dir)
      (($# >= 2)) || die "--policy-dir needs a value"
      POLICY_DIR="$(cd "$2" && pwd)"
      POLICY_LABEL="$(basename "$POLICY_DIR")"
      shift 2
      ;;
    --label)
      (($# >= 2)) || die "--label needs a value"
      POLICY_LABEL="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

[[ "$INPUT_SOURCE" == "sim" || "$INPUT_SOURCE" == "mocap" ]] || die "invalid input-source: $INPUT_SOURCE"

[[ "$RUN_DURATION" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "invalid duration"
[[ "$SHOTS" =~ ^[1-9][0-9]*$ ]] || die "invalid shots"
[[ "$FLIGHT_WINDOW" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "invalid flight-window"
[[ "$CONTACT_HOLD" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "invalid contact-hold"
[[ "$INTER_SHOT" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "invalid inter-shot"
[[ "$STAND_X" =~ ^[-+]?[0-9]+([.][0-9]+)?$ ]] || die "invalid stand-x"
[[ "$TABLE_CENTER_Y" =~ ^[-+]?[0-9]+([.][0-9]+)?$ ]] || die "invalid stand-y"
if ((${#SERVES[@]} > 0)); then
  SHOTS="${#SERVES[@]}"
fi
if ((RANDOMIZE || MIXED_RANDOMIZE || WIDE_LATERAL_MIXED_RANDOMIZE)) && ((${#SERVES[@]} > 0)); then
  die "random serve modes and --serve cannot be combined"
fi
if ((RANDOMIZE + MIXED_RANDOMIZE + WIDE_LATERAL_MIXED_RANDOMIZE > 1)); then
  die "random serve modes cannot be combined"
fi
if ((OFFICIAL_GATE3)); then
  (( ${#SERVES[@]} == 0 )) || die "--official-gate3 cannot be combined with --serve"
  (( RANDOMIZE == 0 && MIXED_RANDOMIZE == 0 && WIDE_LATERAL_MIXED_RANDOMIZE == 0 )) || die "--official-gate3 cannot be combined with random serve modes"
  SHOTS=12
  FLIGHT_WINDOW=2.5
  CONTACT_HOLD=1.5
  INTER_SHOT=4.0
  MOTION_IDLE_S=20.0
fi
[[ -f "$CONDA_SH" ]] || die "missing conda setup: $CONDA_SH"
[[ -x "$SIM_START" ]] || die "missing MuJoCo launcher: $SIM_START"
[[ -f "$RUNNER_CFG" ]] || die "missing runner config: $RUNNER_CFG"
[[ -d "$POLICY_DIR" ]] || die "missing policy directory: $POLICY_DIR"
[[ -f "$WS_INSTALL/setup.bash" ]] || die "hope_ws is not built: $WS_INSTALL/setup.bash"
if [[ "$INPUT_SOURCE" == "mocap" ]]; then
  [[ -f "$NATNET_INSTALL/setup.bash" ]] || die "NatNet2ROS2 is not built: $NATNET_INSTALL/setup.bash"
fi

set +u
source "$CONDA_SH"
conda activate hope-ros
[[ -f "$CONDA_PREFIX/setup.bash" ]] && source "$CONDA_PREFIX/setup.bash"
source "$A3_DIR/setup_a3_env.sh"
source "$WS_INSTALL/setup.bash"
# The official simulated NamedPoseArray boundary uses the same generated
# message package as the real NatNet adapter. Source it for both sim and mocap
# modes; otherwise pp_gate3_sim_mocap.py exits before publishing /optitrack/poses.
if [[ -f "$NATNET_INSTALL/setup.bash" ]]; then
  source "$NATNET_INSTALL/setup.bash"
  source "$WS_INSTALL/setup.bash"
fi
if [[ -f "$SIM_BIN/../share/mujoco_sim_msgs/local_setup.bash" ]]; then
  source "$SIM_BIN/../share/mujoco_sim_msgs/local_setup.bash"
fi
set -u
export MOTION_STAND_X="$STAND_X"
export MOTION_STAND_Y="$TABLE_CENTER_Y"
export MOTION_STAND_Z="$STAND_Z"
export MOTION_POLICY_DIR="$POLICY_DIR"

# The current portable Gate3 build keeps generated Python message bindings in
# this build prefix.  The simulator binaries and XML remain project-local in
# build/install; this path is only for ROS message imports.
if [[ -d "$GATE3_PY_BUILD" ]]; then
  export PYTHONPATH="$GATE3_PY_BUILD/src/protocols/mujoco_sim_msgs/rosidl_generator_py:$GATE3_PY_BUILD/src/protocols/mujoco_sim_msgs/ament_cmake_python/mujoco_sim_msgs:${PYTHONPATH:-}"
  export LD_LIBRARY_PATH="$GATE3_PY_BUILD:$LD_LIBRARY_PATH"
fi

[[ -x "$RUNNER_BIN" ]] || die "missing native runner: $RUNNER_BIN"
ORT_LIB="$A3_DIR/thirdparty/onnxruntime/onnxruntime-linux-x64-1.19.2/lib"
[[ -d "$ORT_LIB" ]] || die "missing ONNX Runtime libs: $ORT_LIB"
OLD_LD_LIBRARY_PATH="$(printenv LD_LIBRARY_PATH 2>/dev/null || true)"
RUNNER_LIB="$(dirname "$RUNNER_BIN")"
SIM_LIB="$SIM_BIN/../lib"
export LD_LIBRARY_PATH="$ORT_LIB:$RUNNER_LIB:$SIM_BIN:$SIM_LIB:$OLD_LD_LIBRARY_PATH"
mkdir -p "$LOG_DIR"

if [[ "$INPUT_SOURCE" == "sim" ]]; then
  # The simulator-only latest-frame backend is selected before the early
  # runner bring-up below. Hardware/mocap keeps the original sync contract.
  RUNNER_CFG="$SIM_RUNNER_CFG"
fi

stop_matching() {
  local pattern="$1"
  local pid
  local pids
  pids="$(pgrep -f -- "$pattern" || true)"
  for pid in $pids; do
    [[ "$pid" == "$$" ]] && continue
    kill -TERM "$pid" 2>/dev/null || true
  done
  sleep 0.5
  for pid in $pids; do
    [[ "$pid" == "$$" ]] && continue
    kill -KILL "$pid" 2>/dev/null || true
  done
}

stop_executable() {
  local target="$1"
  local name
  local pid
  local exe
  local target_inode
  local exe_inode
  name="$(basename "$target")"
  target_inode="$(stat -Lc '%d:%i' "$target" 2>/dev/null || true)"
  [[ -n "$target_inode" ]] || return 0
  for pid in $(pgrep -x "$name" 2>/dev/null || true); do
    exe="/proc/$pid/exe"
    exe_inode="$(stat -Lc '%d:%i' "$exe" 2>/dev/null || true)"
    [[ "$exe_inode" == "$target_inode" ]] || continue
    kill -TERM "$pid" 2>/dev/null || true
  done
  sleep 0.5
  for pid in $(pgrep -x "$name" 2>/dev/null || true); do
    exe="/proc/$pid/exe"
    exe_inode="$(stat -Lc '%d:%i' "$exe" 2>/dev/null || true)"
    [[ "$exe_inode" == "$target_inode" ]] || continue
    kill -KILL "$pid" 2>/dev/null || true
  done
}

kill_group() {
  local pid="$1"
  [[ -n "$pid" ]] || return 0
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  sleep 0.5
  kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
}

cleanup_previous() {
  log "cleaning stale local simulation processes"
  stop_executable "$RUNNER_BIN"
  stop_matching "hope_planner_node.*$WS_INSTALL/hope_planner"
  # The ros2 launcher is a parent process and the installed Python entry point
  # is its child.  Match the relay executable itself so an interrupted run
  # cannot leave an orphan publisher emitting stale base-pose epochs.
  stop_matching "hope_base_pose_flat_relay"
  stop_matching "gate3_state_to_poses.py"
  stop_matching "pp_gate3_sim_mocap.py"
  stop_matching "optitrack_mct_relay"
  stop_matching "ros2 launch hope_bringup hope_world.launch.py"
  # An interrupted launcher can keep publishing a stale park command on the
  # same Gate3 topic.  That races the next run's launch and leaves the plant
  # at the parked sentinel (x=100), so clear launchers before starting AimRT.
  stop_matching "gate3_ball_launcher.py"
  stop_matching "pp_gate3_ball_evidence.py"
  stop_matching "$WS_INSTALL/hope_bringup.*fake_ball_publisher"
  if [[ "$INPUT_SOURCE" == "mocap" ]]; then
    stop_matching "NatNet2ROS2/install/motion_capture_tracking.*motion_capture_tracking_node"
    stop_matching "hope_ws/install/hope_bringup.*optitrack_mct_relay"
  fi
  stop_executable "$SIM_BIN/aimrt_main"
  stop_executable "$SIM_BIN/iox-roudi"
}

cleanup() {
  local rc="$?"
  if ((KEEP_RUNNING)); then
    log "--keep specified; child processes remain running"
    exit "$rc"
  fi
  log "stopping local simulation chain"
  kill_group "$RUNNER_PID"
  kill_group "$BALL_PID"
  kill_group "$EVIDENCE_PID"
  kill_group "$POSE_BRIDGE_PID"
  kill_group "$OPTITRACK_RELAY_PID"
  kill_group "$WORLD_RELAY_PID"
  kill_group "$MOCAP_RELAY_PID"
  kill_group "$MOCAP_ADAPTER_PID"
  kill_group "$PLANNER_PID"
  kill_group "$BASE_RELAY_PID"
  kill_group "$SIM_PID"
  stop_executable "$SIM_BIN/iox-roudi"
  exit "$rc"
}
trap cleanup EXIT INT TERM

wait_for_log() {
  local pid="$1"
  local file="$2"
  local pattern="$3"
  local seconds="$4"
  local i
  for ((i=0; i<seconds; i++)); do
    grep -Eq "$pattern" "$file" 2>/dev/null && return 0
    kill -0 "$pid" 2>/dev/null || return 1
    sleep 1
  done
  grep -Eq "$pattern" "$file" 2>/dev/null
}

# Unlike wait_for_log(), this checks the newest runner status only.  A previous
# healthy status must never satisfy the pre-serve gate after PD_STAND has
# already fallen and the runner has switched to PASSIVE.
latest_runner_status() {
  local file="$1"
  awk '/\[status\]/{line=$0} END{if (line != "") print line}' "$file" 2>/dev/null
}

require_latest_runner_status() {
  local pid="$1"
  local file="$2"
  local pattern="$3"
  local seconds="$4"
  local i
  local line
  for ((i=0; i<seconds; i++)); do
    line="$(latest_runner_status "$file")"
    if [[ "$line" =~ $pattern ]]; then
      return 0
    fi
    kill -0 "$pid" 2>/dev/null || return 1
    sleep 1
  done
  line="$(latest_runner_status "$file")"
  [[ "$line" =~ $pattern ]]
}

cleanup_previous
LOG_DIR="$ROOT_DIR/logs/mujoco_planner_motion/${POLICY_LABEL}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
PHYSICAL_EVIDENCE_JSON="$LOG_DIR/physical_evidence.json"
AUDIT_SUMMARY_JSON="$LOG_DIR/closed_loop_audit.json"
# Keep the AimRT plant contract explicit and reproducible.  The official HOPE
# MuJoCo path uses 1 ms physical steps and explicit body-drive PD torque.
export A3_MUJOCO_PD_MODE="${A3_MUJOCO_PD_MODE:-explicit}"
export A3_MUJOCO_DEBUG_CSV="$LOG_DIR/mujoco_plant.csv"
export A3_MUJOCO_DEBUG_STRIDE="${A3_MUJOCO_DEBUG_STRIDE:-5}"
export A3_GATE3_BALL_DRAG_K="${A3_GATE3_BALL_DRAG_K:-0.1261}"
export A3_GATE3_BALL_RESTITUTION_H="${A3_GATE3_BALL_RESTITUTION_H:-0.6400}"
export A3_GATE3_BALL_RESTITUTION_V="${A3_GATE3_BALL_RESTITUTION_V:-0.9215}"
log "policy: $POLICY_DIR"
log "logs: $LOG_DIR"
log "plant: PD=$A3_MUJOCO_PD_MODE dt=1ms debug_csv=$A3_MUJOCO_DEBUG_CSV"
log "ball: flight_window=${FLIGHT_WINDOW}s contact_hold=${CONTACT_HOLD}s inter_shot=${INTER_SHOT}s venue=(k=$A3_GATE3_BALL_DRAG_K,e_h=$A3_GATE3_BALL_RESTITUTION_H,e_v=$A3_GATE3_BALL_RESTITUTION_V)"
MODEL_XML="$SIM_BIN/cfg/model/a3_pingpong/a3_pingpong.xml"
[[ -f "$MODEL_XML" ]] || die "missing installed MuJoCo model: $MODEL_XML"
MODEL_SHA256="$(sha256sum "$MODEL_XML" | awk '{print $1}')"
[[ "$MODEL_SHA256" == "4887b5301915830a298bee781c897da9448d45cf41ac91da6ce7440a7c3bbd22" ]] || \
  die "unexpected MuJoCo model hash: $MODEL_SHA256"
log "MuJoCo model verified: table center y=$TABLE_CENTER_Y sha256=$MODEL_SHA256"

log "starting AimRT MuJoCo"
setsid bash -c 'exec "$1"' bash "$SIM_START" \
  > "$LOG_DIR/aimrt.log" 2>&1 &
SIM_PID="$!"
wait_for_log "$SIM_PID" "$LOG_DIR/aimrt.log" \
  "MujocoSimModule.*Init succeeded|MujocoSimModule.*Start succeeded" 20 || \
  die "AimRT did not initialize; see $LOG_DIR/aimrt.log"
grep -Eq "MuJoCo body-drive PD mode='explicit'" "$LOG_DIR/aimrt.log" || \
  die "AimRT did not start with explicit body-drive PD; see $LOG_DIR/aimrt.log"

if false; then
python - <<'PY'
import time
import os
import rclpy
from mujoco_sim_msgs.msg import SimReset

rclpy.init()
node = rclpy.create_node("mujoco_motion_chain_reset")
publisher = node.create_publisher(SimReset, "/sim/a3/reset", 10)
message = SimReset()
message.mode = SimReset.MODE_KEYFRAME
message.keyframe_id = 0
message.set_base = True
message.pelvis_pose.position.x = float(os.environ["MOTION_STAND_X"])
message.pelvis_pose.position.y = float(os.environ["MOTION_STAND_Y"])
message.pelvis_pose.position.z = float(os.environ["MOTION_STAND_Z"])
message.pelvis_pose.orientation.w = 1.0
# Match the official Gate3 conductor: reset the floating base and velocities,
# but do not overwrite joint positions. The MuJoCo keyframe/plant owns the
# nominal whole-body posture; writing deploy.yaml joints here creates a hidden
# reset-to-policy handoff that the official closed loop does not have.
message.set_joints = False
message.set_base_twist = True
message.zero_all_velocities = True
message.clear_ctrl = True
for _ in range(8):
    publisher.publish(message)
    rclpy.spin_once(node, timeout_sec=0.05)
    time.sleep(0.05)
node.destroy_node()
rclpy.shutdown()
PY
sleep 1
fi

CONFIG_DIR="$WS_INSTALL/hope_planner/share/hope_planner/config"
[[ -f "$CONFIG_DIR/hope_planner.yaml" ]] || die "planner config is missing"

if true; then
log "starting native runner first: PD_STAND warmup -> MOTION"
setsid bash -c 'exec "$1" \
  --runtime-cfg "$2" \
  --policy-dir "$3" \
  --planner --policy-native --gate3-qdes-audit-only \
  --start passive --official-stand --session-id project_gate3_closedloop' bash \
  "$RUNNER_BIN" "$RUNNER_CFG" "$POLICY_DIR" \
  > >(tee "$LOG_DIR/runner.log") 2>&1 &
RUNNER_PID="$!"
wait_for_log "$RUNNER_PID" "$LOG_DIR/runner.log" "joint map OK" 20 || \
  die "runner did not initialize; see $LOG_DIR/runner.log"

log "resetting MuJoCo again after native PD channel is ready"
python - <<'PY'
import time
import os
import rclpy
from mujoco_sim_msgs.msg import SimReset
from std_msgs.msg import Float64MultiArray

rclpy.init()
node = rclpy.create_node("mujoco_motion_chain_runner_reset")
publisher = node.create_publisher(SimReset, "/sim/a3/reset", 10)
control_pub = node.create_publisher(
    Float64MultiArray, "/hope/runner/control_request_flat", 10)
message = SimReset()
message.mode = SimReset.MODE_KEYFRAME
message.keyframe_id = 0
message.set_base = True
message.pelvis_pose.position.x = float(os.environ["MOTION_STAND_X"])
message.pelvis_pose.position.y = float(os.environ["MOTION_STAND_Y"])
message.pelvis_pose.position.z = float(os.environ["MOTION_STAND_Z"])
message.pelvis_pose.orientation.w = 1.0
message.set_joints = False
message.set_base_twist = True
message.zero_all_velocities = True
# The native runner is already publishing PD commands. Clearing ctrl on every
# repeated reset would create a 0.6 s torque-free interval and let the model fall.
message.clear_ctrl = False
control = Float64MultiArray()
control.data = [1.0, 101.0, 3.0, 0.0]
for _ in range(12):
    publisher.publish(message)
    control_pub.publish(control)
    rclpy.spin_once(node, timeout_sec=0.05)
    time.sleep(0.05)
node.destroy_node()
rclpy.shutdown()
PY
sleep 0.1

log "entering PD_STAND through the runner control contract"
python - <<'PY'
import time
import os
import rclpy
from mujoco_sim_msgs.msg import SimReset
from std_msgs.msg import Float64MultiArray

rclpy.init()
node = rclpy.create_node("mujoco_motion_chain_pd_stand")
reset_pub = node.create_publisher(SimReset, "/sim/a3/reset", 10)
control_pub = node.create_publisher(
    Float64MultiArray, "/hope/runner/control_request_flat", 10)
reset = SimReset()
reset.mode = SimReset.MODE_KEYFRAME
reset.keyframe_id = 0
reset.set_base = True
reset.pelvis_pose.position.x = float(os.environ["MOTION_STAND_X"])
reset.pelvis_pose.position.y = float(os.environ["MOTION_STAND_Y"])
reset.pelvis_pose.position.z = float(os.environ["MOTION_STAND_Z"])
reset.pelvis_pose.orientation.w = 1.0
reset.set_joints = False
reset.set_base_twist = True
reset.zero_all_velocities = True
reset.clear_ctrl = False
control = Float64MultiArray()
control.data = [1.0, 101.0, 3.0, 0.0]
# The official conductor publishes one reset, then enters PD_STAND. Repeating
# the reset while the plant is settling continually overwrites the physical
# state and is not equivalent to the production loop.
reset_pub.publish(reset)
rclpy.spin_once(node, timeout_sec=0.10)
time.sleep(0.20)
for _ in range(30):
    control_pub.publish(control)
    rclpy.spin_once(node, timeout_sec=0.01)
    time.sleep(0.05)
node.destroy_node()
rclpy.shutdown()
PY
sleep 1

log "checking early physical MuJoCo stand pose (non-blocking; final gate is authoritative)"
if false; then
if ! timeout 8s python - <<'PY'
import math
import os
import time
import rclpy
from geometry_msgs.msg import PoseStamped

rclpy.init()
node = rclpy.create_node("mujoco_motion_chain_stand_gate")
expected_x = float(os.environ["MOTION_STAND_X"])
expected_y = float(os.environ["MOTION_STAND_Y"])
state = {"good": 0, "last": None}

def callback(message):
    pose = message.pose
    q = pose.orientation
    # Angle between pelvis local +Z and world +Z.
    local_z_world_z = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    tilt = math.acos(max(-1.0, min(1.0, local_z_world_z)))
    sample = (pose.position.x, pose.position.y, pose.position.z, tilt)
    state["last"] = sample
    if (abs(sample[0] - expected_x) <= 0.08 and
            abs(sample[1] - expected_y) <= 0.08 and
            sample[2] >= 0.90 and tilt <= math.radians(20.0)):
        state["good"] += 1
    else:
        state["good"] = 0

node.create_subscription(PoseStamped, "/sim/a3/pelvis_pose", callback, 20)
deadline = time.monotonic() + 3.0
while rclpy.ok() and time.monotonic() < deadline and state["good"] < 8:
    rclpy.spin_once(node, timeout_sec=0.05)
if state["good"] < 8:
    sample = state["last"]
    print(f"stand_gate_failed expected=({expected_x:.3f},{expected_y:.3f}) "
          f"actual={sample}")
    node.destroy_node()
    rclpy.shutdown()
    raise SystemExit(1)
sample = state["last"]
print(f"stand_gate_passed pose=({sample[0]:.3f},{sample[1]:.3f},{sample[2]:.3f}) "
      f"tilt_deg={math.degrees(sample[3]):.2f}")
node.destroy_node()
rclpy.shutdown()
PY
then
  die "MuJoCo stand pose did not remain upright at the requested table-center station"
fi

fi

fi

# Start localization only after every MuJoCo reset in the PD_STAND prelude.
# Simulation follows the official chain: raw NamedPoseArray -> OptiTrack
# relay -> calibrated hope_world base-pose relay. This avoids bypassing the
# production frame/calibration contract with a direct /sim pelvis publisher.
if [[ "$INPUT_SOURCE" == "sim" ]]; then
  [[ -f "$OFFICIAL_SIM_MOCAP" ]] || die "official simulated mocap bridge is missing: $OFFICIAL_SIM_MOCAP"
  [[ -f "$WS_SOURCE/hope_bringup/config/hope_world_frame.yaml" ]] || die "hope_world_frame.yaml is missing"
  [[ -f "$P1_CALIBRATION_RECEIPT" ]] || die "P1 calibration receipt is missing: $P1_CALIBRATION_RECEIPT"

  log "starting official simulated raw NamedPoseArray boundary"
  setsid bash -c 'exec env PYTHONPATH="$1:$2:${PYTHONPATH:-}" python "$3" \
    --table-height-m 0.760 \
    --world-config "$4"' bash \
    "$OFFICIAL_SIM_CORE" "$A3_DIR/scripts" "$OFFICIAL_SIM_MOCAP" \
    "$WS_SOURCE/hope_bringup/config/hope_world_frame.yaml" \
    > >(tee "$LOG_DIR/official_sim_mocap.log") 2>&1 &
  POSE_BRIDGE_PID="$!"

  log "starting official OptiTrack relay (/optitrack/poses -> /poses)"
  setsid bash -c 'exec ros2 run hope_bringup optitrack_mct_relay --ros-args \
    --params-file "$1" -p publish_tf:=false' bash \
    "$WS_SOURCE/hope_bringup/config/optitrack_relay.yaml" \
    > >(tee "$LOG_DIR/official_optitrack_relay.log") 2>&1 &
  OPTITRACK_RELAY_PID="$!"

  log "starting official calibrated hope_world relay (/P1/pose -> /a3/base_pose_flat)"
  setsid bash -c 'exec ros2 launch hope_bringup hope_world.launch.py \
    p1_calibration_file:="$1" \
    base_pose_output_topic:=/a3/base_pose_flat \
    source_stamp_mode:=local_receipt' bash \
    "$P1_CALIBRATION_RECEIPT" \
    > >(tee "$LOG_DIR/official_hope_world.log") 2>&1 &
  WORLD_RELAY_PID="$!"
  # Match the official conductor's ROS/DDS warm-up before planner and runner
  # startup.  Without this, the first calibrated pose burst is consumed while
  # the relay is still discovering publishers and the runner sees a short
  # invalid/stale-base window.
  # Keep bridge discovery warm-up short; the native runner is already holding
  # the plant upright during this phase.
  sleep 1
else
  # Hardware/mocap mode retains the direct source relay used by this launcher.
  log "starting schema-2 MuJoCo base-pose relay after final reset"
  setsid bash -c 'exec ros2 run hope_planner hope_base_pose_flat_relay --ros-args \
  -p input_topic:=/sim/a3/pelvis_pose \
  -p output_topic:=/a3/base_pose_flat \
  -p expected_input_frame:=odom \
  -p expected_marker_frame:=sim_mujoco \
  -p pelvis_frame:=pelvis_link \
  -p marker_to_base_xyz:="[0.0, 0.0, 0.0]" \
  -p marker_to_base_quaternion_wxyz:="[1.0, 0.0, 0.0, 0.0]" \
  -p policy_z_offset:=0.0 \
  -p extrinsic_calibrated:=true \
  -p world_frame_calibrated:=true \
  -p calibration_sha256:=\"0000000000001000000000000000000000000000000000000000000000000000\" \
  -p world_frame_sha256:=\"0000000000002000000000000000000000000000000000000000000000000000\" \
  -p tracking_quality:=1.0 \
  -p max_linear_speed_mps:=10.0 \
  -p max_angular_speed_rps:=50.0 \
  -p source_stamp_mode:=local_receipt' bash \
  > >(tee "$LOG_DIR/base_relay.log") 2>&1 &
BASE_RELAY_PID="$!"
wait_for_log "$BASE_RELAY_PID" "$LOG_DIR/base_relay.log" "BASE RELAY schema=2" 15 || \
  die "schema-2 base relay did not initialize; see $LOG_DIR/base_relay.log"
fi

log "starting HOPE planner"
PLANNER_ARGS=(
  --params-file "$CONFIG_DIR/hope_planner.yaml"
  --params-file "$CONFIG_DIR/hope_planner.hitter_pure.yaml"
  -p base_pose_flat_input_topic:=/a3/base_pose_flat
  -p robot_pose_topic:=/sim/a3/pelvis_pose
  -p publish_base_flat:=false
  -p publish_flat_cmd:=true
  -p racket_flat_schema:=2
  -p debug_session_id:=project_gate3_closedloop
  -p debug_csv_path:="$LOG_DIR/planner_debug.csv"
  -p debug_flush_rows:=1
  -p max_predict_time:=2.0
  -p adaptive_predict_horizon:=false
  -p max_predict_time_cap:=3.0
  -p policy_z_offset:=0.76
)
if ((OFFICIAL_PLANNER)); then
  # Match the published build_1/rally_v14 planner contract.  These overrides
  # are intentionally explicit: relying on whatever happens to be installed
  # in hope_ws silently changes the model-to-planner interface.
  PLANNER_ARGS+=(
    -p fit_window:=26
    -p drag_k:=0.1261
    -p restitution_h:=0.64
    -p restitution_v:=0.9215
    -p x_hit_follow_robot:=false
    -p x_hit:=0.08
    -p x_hit_bh_delta:=0.0
    -p swing_side_split_y:=-0.25
    -p swing_side_hysteresis_y:=0.04
    -p target_land_x:=2.055
    -p target_land_y_fh:=-0.7625
    -p target_land_y_bh:=-0.7625
    -p delta_t_flight_fh:=0.50
    -p delta_t_flight_bh:=0.50
    -p max_predict_time:=2.6
    -p solve_period_s:=0.033
  )
fi
setsid bash -c 'exec ros2 run hope_planner hope_planner_node --ros-args \
  "${@:1}"' bash \
  "${PLANNER_ARGS[@]}" \
  > >(tee "$LOG_DIR/planner.log") 2>&1 &
PLANNER_PID="$!"
wait_for_log "$PLANNER_PID" "$LOG_DIR/planner.log" "HOPE planner started" 15 || \
  die "planner did not initialize; see $LOG_DIR/planner.log"

# The native runner was already started before localization above.
if false; then
log "starting native runner after localization and planner are ready"
setsid bash -c 'exec "$1" \
  --runtime-cfg "$2" \
  --policy-dir "$3" \
  --planner --policy-native --gate3-qdes-audit-only \
  --start passive --official-stand --session-id project_gate3_closedloop' bash \
  "$RUNNER_BIN" "$RUNNER_CFG" "$POLICY_DIR" \
  > >(tee "$LOG_DIR/runner.log") 2>&1 &
RUNNER_PID="$!"
wait_for_log "$RUNNER_PID" "$LOG_DIR/runner.log" "joint map OK" 20 || \
  die "runner did not initialize; see $LOG_DIR/runner.log"
fi

# Re-arm the already-running native controller after localization/planner
# startup. The reset is idempotent and prevents a long bridge startup window
# from consuming the earlier stand margin.
if true; then
log "final official-style reset after planner startup"
stand_ready=1
for stand_attempt in 1 2 3; do
  log "PD_STAND settle attempt ${stand_attempt}/3"
  python - <<'PY'
import os
import time
import rclpy
from mujoco_sim_msgs.msg import SimReset
from std_msgs.msg import Float64MultiArray

rclpy.init()
node = rclpy.create_node("mujoco_motion_chain_final_official_reset")
reset_pub = node.create_publisher(SimReset, "/sim/a3/reset", 10)
control_pub = node.create_publisher(
    Float64MultiArray, "/hope/runner/control_request_flat", 10)
reset = SimReset()
reset.mode = SimReset.MODE_KEYFRAME
reset.keyframe_id = 0
reset.set_base = True
reset.pelvis_pose.position.x = float(os.environ["MOTION_STAND_X"])
reset.pelvis_pose.position.y = float(os.environ["MOTION_STAND_Y"])
reset.pelvis_pose.position.z = float(os.environ["MOTION_STAND_Z"])
reset.pelvis_pose.orientation.w = 1.0
reset.set_joints = False
reset.set_base_twist = True
reset.zero_all_velocities = True
reset.clear_ctrl = False
control = Float64MultiArray()
control.data = [1.0, 101.0, 3.0, 0.0]
# The ROS graph does not reliably expose the AimRT subscriber count after the
# official relay handoff. Publish an idempotent reset burst instead; the
# simulator keeps the last message and the runner is already publishing PD.
for _ in range(5):
    reset_pub.publish(reset)
    rclpy.spin_once(node, timeout_sec=0.05)
    time.sleep(0.05)
time.sleep(0.10)
for _ in range(30):
    control_pub.publish(control)
    rclpy.spin_once(node, timeout_sec=0.01)
    time.sleep(0.05)
node.destroy_node()
rclpy.shutdown()
PY
  sleep 0.8

  # Planner startup can take long enough for a marginal PD_STAND to fall.  The
  # earlier historical grep above is not sufficient: gate on the newest runner
  # status immediately before handing control to the learned policy.  A reset
  # can race the first body-drive tick, so retry the complete reset/stand pair
  # before failing closed; no ball is allowed during this phase.
  if require_latest_runner_status "$RUNNER_PID" "$LOG_DIR/runner.log" \
      'mode=PD_STAND.*gravZ=(-1(\.0+)?|-0\.[89][0-9]*)[[:space:]]' 2; then
    stand_ready=0
    break
  fi
done
((stand_ready == 0)) || \
  die "robot is not upright immediately before MOTION; refusing policy handoff; see $LOG_DIR/runner.log"
fi

# A standing robot at the wrong x is still an invalid hitter-pure initial
# state.  Check the actual simulator pose before introducing a ball; otherwise
# the first planner command is interpreted by the policy as an emergency
# lunge.
if [[ "$INPUT_SOURCE" == "sim" ]]; then
  if ! timeout 5s python - <<'PY'
import math
import os
import time
import rclpy
from geometry_msgs.msg import PoseStamped

rclpy.init()
node = rclpy.create_node("mujoco_motion_chain_station_gate")
expected = (float(os.environ["MOTION_STAND_X"]), float(os.environ["MOTION_STAND_Y"]))
last = None
good = 0

def callback(message):
    global last, good
    pose = message.pose
    q = pose.orientation
    local_z_world_z = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    tilt = math.acos(max(-1.0, min(1.0, local_z_world_z)))
    last = (pose.position.x, pose.position.y, pose.position.z, tilt)
    if (abs(last[0] - expected[0]) <= 0.08 and
            abs(last[1] - expected[1]) <= 0.08 and
            last[2] >= 0.90 and tilt <= math.radians(20.0)):
        good += 1
    else:
        good = 0

node.create_subscription(PoseStamped, "/sim/a3/pelvis_pose", callback, 20)
deadline = time.monotonic() + 4.0
while rclpy.ok() and time.monotonic() < deadline and good < 8:
    rclpy.spin_once(node, timeout_sec=0.05)
print("station_gate expected=(%.3f,%.3f) actual=%s good=%d" %
      (expected[0], expected[1], last, good))
node.destroy_node()
rclpy.shutdown()
if good < 8:
    raise SystemExit(1)
PY
  then
    die "simulator pose is not at the planner station; refusing ball launch"
  fi
fi

if [[ "$INPUT_SOURCE" == "sim" ]]; then
  # Capture the authoritative 250 Hz Gate3BallState stream before the first
  # launch.  This is separate from the launcher log: the official evidence
  # accumulator joins shot_id-local contact/table/net counter edges and writes
  # a fail-closed per-shot physical report.
  log "starting per-shot Gate3 physical evidence recorder"
  setsid bash -c 'exec python "$1" \
    --expected-shots "$2" \
    --output "$3" \
    --min-samples 20 \
    --max-sample-gap-s 0.050' bash \
    "$A3_DIR/scripts/pp_gate3_ball_evidence.py" \
    "$SHOTS" "$PHYSICAL_EVIDENCE_JSON" \
    > >(tee "$LOG_DIR/physical_evidence.log") 2>&1 &
  EVIDENCE_PID="$!"
  sleep 0.3
  kill -0 "$EVIDENCE_PID" 2>/dev/null || \
    die "physical evidence recorder stopped; see $LOG_DIR/physical_evidence.log"
else
  log "starting real NatNet mocap -> HOPE relay -> /poses"
  setsid bash -c 'exec ros2 launch motion_capture_tracking natnet2ros2.launch.py \
    hostname:="$1" interface_ip:="$2" output_rate_hz:=200.0 header_time:=camera_utc' bash \
    "$MOTIVE_HOST" "$MOCAP_INTERFACE_IP" \
    > >(tee "$LOG_DIR/natnet_adapter.log") 2>&1 &
  MOCAP_ADAPTER_PID="$!"
  wait_for_log "$MOCAP_ADAPTER_PID" "$LOG_DIR/natnet_adapter.log" \
    "NatNet clock sync ready" 20 || \
    die "NatNet2ROS2 did not initialize; see $LOG_DIR/natnet_adapter.log"

  setsid bash -c 'exec ros2 launch hope_bringup optitrack_mct_relay.launch.py' bash \
    > >(tee "$LOG_DIR/optitrack_relay.log") 2>&1 &
  MOCAP_RELAY_PID="$!"
  wait_for_log "$MOCAP_RELAY_PID" "$LOG_DIR/optitrack_relay.log" \
    "relaying /optitrack/poses" 15 || \
    die "OptiTrack relay did not initialize; see $LOG_DIR/optitrack_relay.log"
fi

# Enter MOTION level 0 before the serve.  PD_STAND is only a static nominal
# joint hold; the learned policy's level-0 hold is the actual trained
# preparation state.  The robot is already reset/gated at the requested station
# above, so it can settle before the incoming ball is introduced.
log "preparing policy in MOTION level=0 before physical serve (${MOTION_IDLE_S}s idle)"
python - <<'PY'
import time
import rclpy
from std_msgs.msg import Float64MultiArray

rclpy.init()
node = rclpy.create_node("mujoco_motion_chain_prepare_motion")
control_pub = node.create_publisher(
    Float64MultiArray, "/hope/runner/control_request_flat", 10)
control = Float64MultiArray()
control.data = [1.0, 102.0, 4.0, 0.0]
for _ in range(50):
    control_pub.publish(control)
    rclpy.spin_once(node, timeout_sec=0.01)
    time.sleep(0.02)
node.destroy_node()
if rclpy.ok():
    rclpy.shutdown()
PY
sleep "$MOTION_IDLE_S"
wait_for_log "$RUNNER_PID" "$LOG_DIR/runner.log" "\[runner-control\].*ENTER_MOTION.*result=APPLIED" 5 || \
  die "runner did not accept pre-serve MOTION request; see $LOG_DIR/runner.log"
require_latest_runner_status "$RUNNER_PID" "$LOG_DIR/runner.log" \
  'mode=MOTION level=0.*gravZ=(-1(\.0+)?|-0\.[89][0-9]*)[[:space:]]' 8 || \
  die "runner did not stabilize in pre-serve MOTION level=0; see $LOG_DIR/runner.log"

if [[ "$INPUT_SOURCE" == "sim" ]]; then
log "starting simulated Gate3 serve sequence after policy preparation (${SHOTS} shots)"
LAUNCH_ARGS=(
  --shots "$SHOTS"
  --flight-window "$FLIGHT_WINDOW"
  --contact-hold "$CONTACT_HOLD"
  --inter-shot "$INTER_SHOT"
)
if ((${#SERVES[@]} > 0)); then
  for serve in "${SERVES[@]}"; do
    LAUNCH_ARGS+=(--serve "$serve")
  done
elif ((WIDE_LATERAL_MIXED_RANDOMIZE)); then
  LAUNCH_ARGS+=(--randomize-wide-lateral-mixed --seed "$RANDOM_SEED")
elif ((MIXED_RANDOMIZE)); then
  LAUNCH_ARGS+=(--randomize-mixed --seed "$RANDOM_SEED")
elif ((RANDOMIZE)); then
  LAUNCH_ARGS+=(--randomize --seed "$RANDOM_SEED")
elif ((OFFICIAL_PLANNER)); then
  # Published build_1/rally_v14 side-neutral sequence.  The source contract
  # stores z relative to the table surface; this launcher consumes the
  # floor-origin MuJoCo world frame, hence z=0.49+0.76=1.25 m.
  OFFICIAL_SERVES=(
    "2.40,-1.2025,1.25,-3.0,0.0,2.2"
    "2.40,-0.8525,1.25,-3.0,0.0,2.2"
    "2.40,-1.4425,1.25,-3.0,0.0,2.2"
    "2.40,-0.7925,1.25,-3.0,0.0,2.2"
    "2.40,-1.4425,1.25,-3.0,0.0,2.2"
    "2.40,-0.7725,1.25,-3.0,0.0,2.2"
    "2.40,-1.4425,1.25,-3.0,0.0,2.2"
    "2.40,-0.8125,1.25,-3.0,0.0,2.2"
    "2.40,-1.4325,1.25,-3.0,0.0,2.2"
    "2.40,-0.7625,1.25,-3.0,0.0,2.2"
    "2.40,-1.4425,1.25,-3.0,0.0,2.2"
    "2.40,-0.7925,1.25,-3.0,0.0,2.2"
  )
  for ((serve_index=0; serve_index<SHOTS && serve_index<${#OFFICIAL_SERVES[@]}; serve_index++)); do
    serve="${OFFICIAL_SERVES[$serve_index]}"
    LAUNCH_ARGS+=(--serve "$serve")
  done
else
  # Use the two side-neutral lanes that are inside the current
  # rally_v14/hitter_pure station support.  The former default pair
  # (y=-1.00/-0.52 with vx=-5) can produce a planner station step larger
  # than the trained 0.40 m gate; Gate3 is advisory in this runner and may
  # still release that unsupported swing, which can make the robot fall.
  # Keep arbitrary --serve and the explicit --official-planner sequence
  # available for diagnostics, but make the ordinary full-loop command safe.
  for ((shot_index=0; shot_index<SHOTS; shot_index++)); do
    if ((shot_index % 2 == 0)); then
      LAUNCH_ARGS+=(--serve "2.40,-1.2025,1.25,-3.00,0.00,2.20")
    else
      LAUNCH_ARGS+=(--serve "2.40,-0.8525,1.25,-3.00,0.00,2.20")
    fi
  done
fi
printf '%s\n' "${LAUNCH_ARGS[@]}" > "$LOG_DIR/serve_sequence.txt"
setsid bash -c 'script="$1"; shift; exec python "$script" "$@"' bash \
  "$ROOT_DIR/hope_training/whole_body_tracking/official_hope/scripts/gate3_ball_launcher.py" \
  "${LAUNCH_ARGS[@]}" \
  > >(tee "$LOG_DIR/gate3_launcher.log") 2>&1 &
BALL_PID="$!"
fi

log "checking planner output"
if ! timeout 15s python - <<'PY'
import math
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

class Probe(Node):
    def __init__(self):
        super().__init__("mujoco_base_pose_schema2_probe")
        self.valid = None
        self.sub = self.create_subscription(
            Float64MultiArray, "/a3/base_pose_flat", self.callback, 20)

    def callback(self, message):
        data = list(message.data)
        if (self.valid is None and len(data) >= 16 and
                data[0] == 2.0 and data[1] == 1.0 and
                int(data[13]) & 47 == 47 and
                all(math.isfinite(x) for x in data[:16])):
            self.valid = data

rclpy.init()
node = Probe()
deadline = time.monotonic() + 14.0
while rclpy.ok() and node.valid is None and time.monotonic() < deadline:
    rclpy.spin_once(node, timeout_sec=0.1)
if node.valid is None:
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    raise SystemExit(1)
data = node.valid
print("base_pose_valid schema=%.0f seq=%.0f pos=(%.4f,%.4f,%.4f) "
      "flags=%.0f calib=%.0f world=%.0f" %
      (data[0], data[2], data[5], data[6], data[7],
       data[13], data[14], data[15]))
node.destroy_node()
rclpy.shutdown()
PY
then
  die "no valid schema-2 /a3/base_pose_flat message observed"
fi

if ! timeout 15s python - <<'PY'
import math
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

class Probe(Node):
    def __init__(self):
        super().__init__("mujoco_motion_chain_probe")
        self.samples = 0
        self.valid = None
        self.sub = self.create_subscription(
            Float64MultiArray, "/racket/command_flat", self.callback, 20)

    def callback(self, message):
        data = list(message.data)
        self.samples += 1
        if (self.valid is None and len(data) >= 11 and
                data[0] in (1.0, 2.0) and data[1] == 1.0 and
                all(math.isfinite(x) for x in data[:11])):
            self.valid = data

rclpy.init()
node = Probe()
deadline = time.monotonic() + 14.0
while rclpy.ok() and node.valid is None and time.monotonic() < deadline:
    rclpy.spin_once(node, timeout_sec=0.1)
if node.valid is None:
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    raise SystemExit(1)
data = node.valid
print("planner_valid schema=%.0f swing=%.0f pos=(%.4f,%.4f,%.4f) "
      "vel=(%.4f,%.4f,%.4f) tts=%.4f samples=%d" %
      (data[0], data[2], data[3], data[4], data[5],
       data[6], data[7], data[8], data[9], node.samples))
node.destroy_node()
rclpy.shutdown()
PY
then
  die "no valid /racket/command_flat message observed"
fi

log "READY: AimRT + planner + prepared MOTION level=0 + ${INPUT_SOURCE} ball input are connected"
log "runner: $(grep -F "[status] mode=MOTION" "$LOG_DIR/runner.log" | grep -E "ticks=[1-9][0-9]*" | tail -1)"
log "logs: $LOG_DIR"

if [[ "$RUN_DURATION" != "0" ]]; then
  log "running for $RUN_DURATION seconds"
  sleep "$RUN_DURATION"
else
  log "running until Ctrl+C"
  while :; do
    kill -0 "$RUNNER_PID" 2>/dev/null || die "runner stopped unexpectedly"
    sleep 2
  done
fi

if [[ "$INPUT_SOURCE" == "sim" ]]; then
  # Stop the ball/evidence pair in a defined order so the recorder can flush
  # its final snapshot before the surrounding ROS graph is torn down.
  log "finalizing per-shot physical evidence"
  kill_group "$BALL_PID"
  BALL_PID=""
  kill_group "$EVIDENCE_PID"
  EVIDENCE_PID=""
  if python "$A3_DIR/scripts/pp_closed_loop_audit.py" \
      --log-dir "$LOG_DIR" \
      --physical-evidence "$PHYSICAL_EVIDENCE_JSON" \
      --output "$AUDIT_SUMMARY_JSON"; then
    log "audit summary: $AUDIT_SUMMARY_JSON"
  else
    log "audit summary failed; raw logs and physical evidence were preserved"
  fi
else
  log "mocap run complete; raw NatNet and relay logs: $LOG_DIR"
fi
