#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WBT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${WBT_DIR}/../.." && pwd)"

ROS_SETUP_CANDIDATES=()
if [ -n "${HOPE_ROS_WS_SETUP:-}" ]; then
  ROS_SETUP_CANDIDATES+=("${HOPE_ROS_WS_SETUP}")
else
  ROS_SETUP_CANDIDATES+=(
    "${HOPE_ROS_SETUP:-}"
    "${REPO_DIR}/hope_ws/install/setup.bash"
    "${REPO_DIR}/hope_ws/install/setup.sh"
    "${REPO_DIR}/hope_ws/install/local_setup.bash"
    "${REPO_DIR}/hope_ws/install/local_setup.sh"
  )
fi

ROS_UNDERLAY_CANDIDATES=()
if [ -n "${HOPE_ROS_UNDERLAY_SETUP:-}" ]; then
  ROS_UNDERLAY_CANDIDATES+=("${HOPE_ROS_UNDERLAY_SETUP}")
else
  while IFS= read -r candidate; do
    ROS_UNDERLAY_CANDIDATES+=("${candidate}")
  done < <(find /opt/ros -maxdepth 2 -type f \( -name setup.bash -o -name setup.sh \) 2>/dev/null | sort)
fi

ROS_UNDERLAY_SETUP=""
for candidate in "${ROS_UNDERLAY_CANDIDATES[@]}"; do
  [ -n "${candidate}" ] || continue
  if [ -f "${candidate}" ]; then
    ROS_UNDERLAY_SETUP="${candidate}"
    break
  fi
done

ROS_SETUP=""
for candidate in "${ROS_SETUP_CANDIDATES[@]}"; do
  if [[ "${candidate}" != /* ]]; then
    candidate="${REPO_DIR}/${candidate}"
  fi
  if [ -f "${candidate}" ]; then
    ROS_SETUP="${candidate}"
    break
  fi
done

BALL_TOPIC="${HOPE_BALL_TOPIC:-/ball/point}"
PREDICTED_STRIKE_TOPIC="${HOPE_PREDICTED_STRIKE_TOPIC:-/ball/predicted_strike}"
POST_BOUNCE_PREDICTED_STRIKE_TOPIC="${HOPE_POST_BOUNCE_PREDICTED_STRIKE_TOPIC:-/ball/post_bounce_predicted_strike}"
BALL_FRAME_ID="${HOPE_BALL_FRAME_ID:-world}"
BALL_UDP_HOST="${HOPE_BALL_TRUTH_UDP_HOST:-127.0.0.1}"
BALL_UDP_PORT="${HOPE_BALL_TRUTH_UDP_PORT:-19531}"
TRAJECTORY_UDP_HOST="${HOPE_TRAJECTORY_UDP_HOST:-127.0.0.1}"
TRAJECTORY_UDP_PORT="${HOPE_TRAJECTORY_UDP_PORT:-19532}"
TRAJECTORY_HORIZON="${HOPE_TRAJECTORY_HORIZON:-1.2}"
TRAJECTORY_DRAW_PERIOD="${HOPE_TRAJECTORY_DRAW_PERIOD:-0.03}"
TRAJECTORY_SAMPLE_STRIDE="${HOPE_TRAJECTORY_SAMPLE_STRIDE:-8}"
TRAJECTORY_DRAG_COEFFICIENT="${HOPE_TRAJECTORY_DRAG_COEFFICIENT:-0.09375}"
TRAJECTORY_GRAVITY_X="${HOPE_TRAJECTORY_GRAVITY_X:-0.0}"
TRAJECTORY_GRAVITY_Y="${HOPE_TRAJECTORY_GRAVITY_Y:-0.0}"
TRAJECTORY_GRAVITY_Z="${HOPE_TRAJECTORY_GRAVITY_Z:--9.81}"
TRAJECTORY_BALL_RADIUS="${HOPE_TRAJECTORY_BALL_RADIUS:-0.02}"
TRAJECTORY_BALL_MASS="${HOPE_TRAJECTORY_BALL_MASS:-0.0027}"
TRAJECTORY_TABLE_TANGENTIAL_RETENTION="${HOPE_TRAJECTORY_TABLE_TANGENTIAL_RETENTION:-0.649}"
TRAJECTORY_TABLE_NORMAL_RESTITUTION="${HOPE_TRAJECTORY_TABLE_NORMAL_RESTITUTION:-0.906}"
TRAJECTORY_TABLE_LENGTH="${HOPE_TRAJECTORY_TABLE_LENGTH:-2.74}"
TRAJECTORY_TABLE_WIDTH="${HOPE_TRAJECTORY_TABLE_WIDTH:-1.525}"
TRAJECTORY_TABLE_NET_X="${HOPE_TRAJECTORY_TABLE_NET_X:-1.37}"
TRAJECTORY_DT_INTEGRATE="${HOPE_TRAJECTORY_DT_INTEGRATE:-0.001}"
HIT_UDP_HOST="${HOPE_HIT_UDP_HOST:-127.0.0.1}"
HIT_UDP_PORT="${HOPE_HIT_UDP_PORT:-19533}"
LANDING_CANDIDATE_UDP_HOST="${HOPE_LANDING_CANDIDATE_UDP_HOST:-127.0.0.1}"
LANDING_CANDIDATE_UDP_PORT="${HOPE_LANDING_CANDIDATE_UDP_PORT:-19534}"
START_PLANNER=1
START_STRIKE_PREDICTOR=1
START_TRUTH_BRIDGE=1
START_TRAJECTORY_OVERLAY=1
START_HIT_BRIDGE=1
START_LANDING_CANDIDATE_BRIDGE=1
PLANNER_PID=""
DECISION_PID=""
STRIKE_PREDICTOR_PID=""
TRUTH_BRIDGE_PID=""
TRAJECTORY_OVERLAY_PID=""
HIT_BRIDGE_PID=""
LANDING_CANDIDATE_BRIDGE_PID=""

ARGS=()
for arg in "$@"; do
  case "${arg}" in
    --no-planner)
      START_PLANNER=0
      ;;
    --no-ball-truth-bridge)
      START_TRUTH_BRIDGE=0
      ;;
    --no-trajectory-overlay)
      START_TRAJECTORY_OVERLAY=0
      ARGS+=("--no-draw-trajectory")
      ;;
    --no-hit-overlay)
      START_HIT_BRIDGE=0
      ARGS+=("--no-hit-overlay")
      ;;
    --no-landing-candidate-overlay)
      START_LANDING_CANDIDATE_BRIDGE=0
      ARGS+=("--no-landing-candidate-overlay")
      ;;
    *)
      ARGS+=("${arg}")
      ;;
  esac
done

cleanup() {
  if [ -n "${DECISION_PID}" ]; then
    kill -- "-${DECISION_PID}" 2>/dev/null || kill "${DECISION_PID}" 2>/dev/null || true
    wait "${DECISION_PID}" 2>/dev/null || true
  fi
  if [ -n "${PLANNER_PID}" ]; then
    kill -- "-${PLANNER_PID}" 2>/dev/null || kill "${PLANNER_PID}" 2>/dev/null || true
    wait "${PLANNER_PID}" 2>/dev/null || true
  fi
  if [ -n "${TRUTH_BRIDGE_PID}" ]; then
    kill -- "-${TRUTH_BRIDGE_PID}" 2>/dev/null || kill "${TRUTH_BRIDGE_PID}" 2>/dev/null || true
    wait "${TRUTH_BRIDGE_PID}" 2>/dev/null || true
  fi
  if [ -n "${STRIKE_PREDICTOR_PID}" ]; then
    kill -- "-${STRIKE_PREDICTOR_PID}" 2>/dev/null || kill "${STRIKE_PREDICTOR_PID}" 2>/dev/null || true
    wait "${STRIKE_PREDICTOR_PID}" 2>/dev/null || true
  fi
  if [ -n "${TRAJECTORY_OVERLAY_PID}" ]; then
    kill -- "-${TRAJECTORY_OVERLAY_PID}" 2>/dev/null || kill "${TRAJECTORY_OVERLAY_PID}" 2>/dev/null || true
    wait "${TRAJECTORY_OVERLAY_PID}" 2>/dev/null || true
  fi
  if [ -n "${HIT_BRIDGE_PID}" ]; then
    kill -- "-${HIT_BRIDGE_PID}" 2>/dev/null || kill "${HIT_BRIDGE_PID}" 2>/dev/null || true
    wait "${HIT_BRIDGE_PID}" 2>/dev/null || true
  fi
  if [ -n "${LANDING_CANDIDATE_BRIDGE_PID}" ]; then
    kill -- "-${LANDING_CANDIDATE_BRIDGE_PID}" 2>/dev/null || kill "${LANDING_CANDIDATE_BRIDGE_PID}" 2>/dev/null || true
    wait "${LANDING_CANDIDATE_BRIDGE_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [ -z "${ROS_SETUP}" ]; then
  echo "[hope_ros_run] ROS setup not found. Tried:" >&2
  for candidate in "${ROS_SETUP_CANDIDATES[@]}"; do
    if [[ "${candidate}" != /* ]]; then
      candidate="${REPO_DIR}/${candidate}"
    fi
    echo "  - ${candidate}" >&2
  done
  exit 1
fi

if [ -z "${ROS_UNDERLAY_SETUP}" ]; then
  echo "[hope_ros_run] ROS underlay setup not found. Tried:" >&2
  for candidate in "${ROS_UNDERLAY_CANDIDATES[@]}"; do
    [ -n "${candidate}" ] || continue
    echo "  - ${candidate}" >&2
  done
  exit 1
fi

ROS_PREFIX="$(cd "$(dirname "${ROS_SETUP}")" && pwd)"
DECISION_NODE="${ROS_PREFIX}/decision/lib/decision/decision_node"
SOLVER_NODE="${ROS_PREFIX}/solver/lib/solver/solver_node"
SOLVER_CONFIG="${ROS_PREFIX}/solver/share/solver/config/solver.yaml"
TRUTH_BRIDGE_NODE="${ROS_PREFIX}/bringup/lib/bringup/ball_truth_udp_bridge"
STRIKE_PREDICTION_NODE="${ROS_PREFIX}/trajectory/lib/trajectory/strike_prediction_node"
TRAJECTORY_OVERLAY_NODE="${ROS_PREFIX}/trajectory/lib/trajectory/trajectory_overlay_udp_node"
HIT_BRIDGE_NODE="${ROS_PREFIX}/bringup/lib/bringup/hit_state_udp_bridge"
LANDING_CANDIDATE_BRIDGE_NODE="${ROS_PREFIX}/bringup/lib/bringup/landing_candidates_udp_bridge"

if [ "${START_PLANNER}" -eq 1 ] && [ ! -x "${DECISION_NODE}" ]; then
  echo "[hope_ros_run] decision executable not found: ${DECISION_NODE}" >&2
  exit 1
fi
if [ ! -x "${SOLVER_NODE}" ]; then
  echo "[hope_ros_run] solver executable not found: ${SOLVER_NODE}" >&2
  exit 1
fi
if [ ! -f "${SOLVER_CONFIG}" ]; then
  echo "[hope_ros_run] solver config not found: ${SOLVER_CONFIG}" >&2
  exit 1
fi
if [ "${START_TRUTH_BRIDGE}" -eq 1 ] && [ ! -x "${TRUTH_BRIDGE_NODE}" ]; then
  echo "[hope_ros_run] ball truth bridge executable not found: ${TRUTH_BRIDGE_NODE}" >&2
  exit 1
fi
if [ "${START_STRIKE_PREDICTOR}" -eq 1 ] && [ ! -x "${STRIKE_PREDICTION_NODE}" ]; then
  echo "[hope_ros_run] strike prediction executable not found: ${STRIKE_PREDICTION_NODE}" >&2
  exit 1
fi
if [ "${START_TRAJECTORY_OVERLAY}" -eq 1 ] && [ ! -x "${TRAJECTORY_OVERLAY_NODE}" ]; then
  echo "[hope_ros_run] trajectory overlay executable not found: ${TRAJECTORY_OVERLAY_NODE}" >&2
  exit 1
fi
if [ "${START_HIT_BRIDGE}" -eq 1 ] && [ ! -x "${HIT_BRIDGE_NODE}" ]; then
  echo "[hope_ros_run] hit state bridge executable not found: ${HIT_BRIDGE_NODE}" >&2
  exit 1
fi
if [ "${START_LANDING_CANDIDATE_BRIDGE}" -eq 1 ] && [ ! -x "${LANDING_CANDIDATE_BRIDGE_NODE}" ]; then
  echo "[hope_ros_run] landing candidate bridge executable not found: ${LANDING_CANDIDATE_BRIDGE_NODE}" >&2
  exit 1
fi

if [ "${START_PLANNER}" -eq 1 ]; then
  setsid bash -lc '
    set -eo pipefail
    set +u
    source "$1"
    source "$2"
    set -u
    shift 2
    exec "$@"
  ' _ "${ROS_UNDERLAY_SETUP}" "${ROS_SETUP}" \
    "${DECISION_NODE}" --ros-args \
    -p "pre_aim_strike_topic:=${PREDICTED_STRIKE_TOPIC}" \
    -p "strike_adjust_topic:=${POST_BOUNCE_PREDICTED_STRIKE_TOPIC}" &
  DECISION_PID=$!
  echo "[hope_ros_run] landing decision started: pid=${DECISION_PID}"
  sleep 1

  setsid bash -lc '
    set -eo pipefail
    set +u
    source "$1"
    source "$2"
    set -u
    shift 2
    exec "$@"
  ' _ "${ROS_UNDERLAY_SETUP}" "${ROS_SETUP}" \
    "${SOLVER_NODE}" --ros-args --params-file "${SOLVER_CONFIG}" \
    -p "pre_aim_strike_topic:=${PREDICTED_STRIKE_TOPIC}" \
    -p "strike_adjust_topic:=${POST_BOUNCE_PREDICTED_STRIKE_TOPIC}" &
  PLANNER_PID=$!
  echo "[hope_ros_run] solver started: pid=${PLANNER_PID}"
  sleep 2
else
  echo "[hope_ros_run] planner startup skipped."
fi

if [ "${START_TRUTH_BRIDGE}" -eq 1 ]; then
  setsid bash -lc '
    set -eo pipefail
    set +u
    source "$1"
    source "$2"
    set -u
    shift 2
    exec "$@"
  ' _ "${ROS_UNDERLAY_SETUP}" "${ROS_SETUP}" \
    "${TRUTH_BRIDGE_NODE}" --ros-args \
    -p "topic:=${BALL_TOPIC}" \
    -p "frame_id:=${BALL_FRAME_ID}" \
    -p "udp_host:=${BALL_UDP_HOST}" \
    -p "udp_port:=${BALL_UDP_PORT}" &
  TRUTH_BRIDGE_PID=$!
  echo "[hope_ros_run] ball truth bridge started: pid=${TRUTH_BRIDGE_PID}"
  sleep 1
else
  echo "[hope_ros_run] ball truth bridge startup skipped."
fi

if [ "${START_STRIKE_PREDICTOR}" -eq 1 ]; then
  setsid bash -lc '
    set -eo pipefail
    set +u
    source "$1"
    source "$2"
    set -u
    shift 2
    exec "$@"
  ' _ "${ROS_UNDERLAY_SETUP}" "${ROS_SETUP}" \
    "${STRIKE_PREDICTION_NODE}" --ros-args \
    -p "ball_topic:=${BALL_TOPIC}" \
    -p "pre_aim_strike_topic:=${PREDICTED_STRIKE_TOPIC}" \
    -p "strike_adjust_topic:=${POST_BOUNCE_PREDICTED_STRIKE_TOPIC}" &
  STRIKE_PREDICTOR_PID=$!
  echo "[hope_ros_run] strike prediction started: pid=${STRIKE_PREDICTOR_PID}"
  sleep 1
else
  echo "[hope_ros_run] strike prediction startup skipped."
fi

if [ "${START_TRAJECTORY_OVERLAY}" -eq 1 ]; then
  setsid bash -lc '
    set -eo pipefail
    set +u
    source "$1"
    source "$2"
    set -u
    shift 2
    exec "$@"
  ' _ "${ROS_UNDERLAY_SETUP}" "${ROS_SETUP}" \
    "${TRAJECTORY_OVERLAY_NODE}" --ros-args \
    -p "ball_topic:=${BALL_TOPIC}" \
    -p "udp_host:=${TRAJECTORY_UDP_HOST}" \
    -p "udp_port:=${TRAJECTORY_UDP_PORT}" \
    -p "horizon_s:=${TRAJECTORY_HORIZON}" \
    -p "draw_period_s:=${TRAJECTORY_DRAW_PERIOD}" \
    -p "sample_stride:=${TRAJECTORY_SAMPLE_STRIDE}" \
    -p "physics.drag_coefficient:=${TRAJECTORY_DRAG_COEFFICIENT}" \
    -p "physics.gravity_x:=${TRAJECTORY_GRAVITY_X}" \
    -p "physics.gravity_y:=${TRAJECTORY_GRAVITY_Y}" \
    -p "physics.gravity_z:=${TRAJECTORY_GRAVITY_Z}" \
    -p "physics.ball_radius:=${TRAJECTORY_BALL_RADIUS}" \
    -p "physics.ball_mass:=${TRAJECTORY_BALL_MASS}" \
    -p "physics.table_tangential_retention:=${TRAJECTORY_TABLE_TANGENTIAL_RETENTION}" \
    -p "physics.table_normal_restitution:=${TRAJECTORY_TABLE_NORMAL_RESTITUTION}" \
    -p "table.length:=${TRAJECTORY_TABLE_LENGTH}" \
    -p "table.width:=${TRAJECTORY_TABLE_WIDTH}" \
    -p "table.net_x:=${TRAJECTORY_TABLE_NET_X}" \
    -p "config.dt_integrate:=${TRAJECTORY_DT_INTEGRATE}" &
  TRAJECTORY_OVERLAY_PID=$!
  echo "[hope_ros_run] trajectory overlay started: pid=${TRAJECTORY_OVERLAY_PID}"
  sleep 1
else
  echo "[hope_ros_run] trajectory overlay startup skipped."
fi

if [ "${START_HIT_BRIDGE}" -eq 1 ]; then
  setsid bash -lc '
    set -eo pipefail
    set +u
    source "$1"
    source "$2"
    set -u
    shift 2
    exec "$@"
  ' _ "${ROS_UNDERLAY_SETUP}" "${ROS_SETUP}" \
    "${HIT_BRIDGE_NODE}" --ros-args \
    -p "hit_state_topic:=${HOPE_HIT_STATE_TOPIC:-/hit/state}" \
    -p "udp_host:=${HIT_UDP_HOST}" \
    -p "udp_port:=${HIT_UDP_PORT}" &
  HIT_BRIDGE_PID=$!
  echo "[hope_ros_run] hit state bridge started: pid=${HIT_BRIDGE_PID}"
  sleep 1
else
  echo "[hope_ros_run] hit state bridge startup skipped."
fi

if [ "${START_LANDING_CANDIDATE_BRIDGE}" -eq 1 ]; then
  setsid bash -lc '
    set -eo pipefail
    set +u
    source "$1"
    source "$2"
    set -u
    shift 2
    exec "$@"
  ' _ "${ROS_UNDERLAY_SETUP}" "${ROS_SETUP}" \
    "${LANDING_CANDIDATE_BRIDGE_NODE}" --ros-args \
    -p "candidates_topic:=${HOPE_LANDING_CANDIDATES_TOPIC:-/planner/landing_candidates}" \
    -p "udp_host:=${LANDING_CANDIDATE_UDP_HOST}" \
    -p "udp_port:=${LANDING_CANDIDATE_UDP_PORT}" &
  LANDING_CANDIDATE_BRIDGE_PID=$!
  echo "[hope_ros_run] landing candidate bridge started: pid=${LANDING_CANDIDATE_BRIDGE_PID}"
  sleep 1
else
  echo "[hope_ros_run] landing candidate bridge startup skipped."
fi

cd "${WBT_DIR}"
set +u
# shellcheck disable=SC1091
source "${WBT_DIR}/setup_train_env.sh"
set -u

export PYTHONPATH="${HOPE_WBT_PYTHONPATH}"

echo "[hope_ros_run] starting Isaac table-tennis sim"
echo "[hope_ros_run] publishing ball truth on ${BALL_TOPIC} via udp://${BALL_UDP_HOST}:${BALL_UDP_PORT}"
echo "[hope_ros_run] receiving trajectory overlay on udp://${TRAJECTORY_UDP_HOST}:${TRAJECTORY_UDP_PORT}"
echo "[hope_ros_run] receiving hit state on udp://${HIT_UDP_HOST}:${HIT_UDP_PORT}"
echo "[hope_ros_run] receiving landing candidates on udp://${LANDING_CANDIDATE_UDP_HOST}:${LANDING_CANDIDATE_UDP_PORT}"
"${HOPE_ISAAC_PYTHON}" "${SCRIPT_DIR}/play_table_tennis.py" \
  --publish-ball-truth \
  --ball-truth-topic "${BALL_TOPIC}" \
  --ball-truth-frame-id "${BALL_FRAME_ID}" \
  --ball-truth-udp-host "${BALL_UDP_HOST}" \
  --ball-truth-udp-port "${BALL_UDP_PORT}" \
  --trajectory-udp-host "${TRAJECTORY_UDP_HOST}" \
  --trajectory-udp-port "${TRAJECTORY_UDP_PORT}" \
  --hit-overlay-udp-host "${HIT_UDP_HOST}" \
  --hit-overlay-udp-port "${HIT_UDP_PORT}" \
  --landing-candidate-udp-host "${LANDING_CANDIDATE_UDP_HOST}" \
  --landing-candidate-udp-port "${LANDING_CANDIDATE_UDP_PORT}" \
  "${ARGS[@]}"
