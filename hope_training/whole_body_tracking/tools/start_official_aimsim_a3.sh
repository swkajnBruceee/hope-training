#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_official_aimsim_env.sh"

if [[ "${SIM_MODE}" != "sil" ]]; then
  echo "[official-aimsim] only SIL is supported by this project launcher; got ${SIM_MODE}" >&2
  exit 2
fi

"${AIMSIM_PYTHON}" -m aimsim.cli mujoco init-config \
  --user-config-path "${AIMSIM_USER_CONFIG}" \
  --robot "${MUJOCO_ROBOT}" >/dev/null

# The project-owned racket overlay is the default for the table-tennis SIL
# route. It leaves AimSim's installed resources untouched and can be disabled
# explicitly with AIMSIM_RACKET_MODEL=0 for upstream-model diagnostics.
if [[ "${AIMSIM_RACKET_MODEL:-1}" == "1" ]]; then
  RACKET_ROOT="${HOPE_PROJECT_ROOT}/hope_training/whole_body_tracking/assets/official_aimsim_racket"
  RACKET_PREPARE_ARGS=(
    "${AIMSIM_PYTHON}" "${SCRIPT_DIR}/prepare_official_aimsim_racket_model.py"
    --base-xml "${AIMSIM_PYTHON%/bin/python}/lib/python3.10/site-packages/mujoco_simulator/resources/${MUJOCO_ROBOT}/mjcf/${MUJOCO_ROBOT}.xml"
    --pingpong-model "${HOPE_PROJECT_ROOT}/agibot/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml"
    --robot-model-info "${AIMSIM_PYTHON%/bin/python}/lib/python3.10/site-packages/mujoco_simulator/app/cfg/${MUJOCO_ROBOT}/robot_model_info.yaml"
    --aimrt-config "${AIMSIM_USER_CONFIG}/mujoco/cfg/${MUJOCO_ROBOT}/mujoco_simulator_cfg_sil.yaml"
    --output-root "${RACKET_ROOT}"
  )
  if [[ "${AIMSIM_PINGPONG_TASK:-0}" == "1" ]]; then
    RACKET_PREPARE_ARGS+=(--with-pingpong-task)
    echo "[official-aimsim] enabling fixed-ball coordinate/contact calibration task"
  fi
  "${RACKET_PREPARE_ARGS[@]}" \
    >/dev/null
  export AIMSIM_MUJOCO_CONFIG="${RACKET_ROOT}/config_mujoco.yaml"
  export MUJOCO_APP_CFG_ROOT="${RACKET_ROOT}/mujoco_cfg"
  echo "[official-aimsim] using project-side right-racket model: ${RACKET_ROOT}"
fi

MC_SCRIPT="${AIMSIM_MOTION_CONTROL_ROOT}/scripts/motion_control/start_motion_control.sh"
MC_LOG_DIR="${AIMSIM_OFFICIAL_ROOT}/logs/motion_control"
mkdir -p "${MC_LOG_DIR}"

cleanup() {
  trap - TERM INT
  if [[ -n "${GET_UP_PID:-}" ]] && kill -0 "${GET_UP_PID}" 2>/dev/null; then
    kill -TERM "${GET_UP_PID}" 2>/dev/null || true
    wait "${GET_UP_PID}" 2>/dev/null || true
  fi
  if [[ -n "${MC_PID:-}" ]] && kill -0 "${MC_PID}" 2>/dev/null; then
    kill -TERM "${MC_PID}" 2>/dev/null || true
    wait "${MC_PID}" 2>/dev/null || true
  fi
  # The official start script owns this router in this validation session.
  if [[ -x "${AIMSIM_MOTION_CONTROL_ROOT}/bin/iox-roudi" ]]; then
    pkill -TERM -f "${AIMSIM_MOTION_CONTROL_ROOT}/bin/iox-roudi" 2>/dev/null || true
  fi
}
# `aimsim.cli mujoco start` intentionally backgrounds the simulator and then
# returns.  Do not bind cleanup to EXIT here: doing so tears down MOTION just
# after the simulator has started.  Explicit termination of this launcher
# still cleans up the controller and its session-owned router.
trap cleanup TERM INT

echo "[official-aimsim] starting official motion_control first"
(
  export AGIBOT_ROBOT_MODEL
  export LOG_PATH="${MC_LOG_DIR}"
  exec bash "${MC_SCRIPT}"
) >"${MC_LOG_DIR}/motion_control.stdout.log" 2>&1 &
MC_PID=$!

sleep 4
if ! kill -0 "${MC_PID}" 2>/dev/null; then
  echo "[official-aimsim] official motion_control exited during startup" >&2
  sed -n '1,240p' "${MC_LOG_DIR}/motion_control.stdout.log" >&2 || true
  exit 1
fi

# The simulator intentionally starts in PASSIVE (zero torque). This helper
# waits for the simulator-backed service, leaves PASSIVE long enough for the
# normal fall, then requests the official DAMPING -> GET_UP -> MOTION path.
# It is not a position reset and never requests PD_STAND.
echo "[official-aimsim] scheduling official PASSIVE -> DAMPING -> GET_UP -> MOTION"
(
  SIM_MODE=sil "${AIMSIM_PYTHON}" "${SCRIPT_DIR}/activate_official_aimsim_sil.py" \
    --action MOTION --ready-timeout-s 45 --get-up-timeout-s 45 --passive-settle-s 1.5
) >"${MC_LOG_DIR}/get_up.stdout.log" 2>&1 &
GET_UP_PID=$!

echo "[official-aimsim] starting official AimSim MuJoCo SIL"
"${AIMSIM_PYTHON}" -m aimsim.cli mujoco start \
  --user-config-path "${AIMSIM_USER_CONFIG}"

# The official CLI has now handed the simulator to its own process and
# returned.  Keep this launcher alive for as long as its MOTION child is
# alive; this preserves the control process in terminal-managed SIL sessions.
wait "${MC_PID}"
