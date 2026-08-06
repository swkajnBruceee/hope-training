#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/agibot/data/user_deploy/mc"
HAL_SCRIPT="/agibot/software/v0/scripts/hal_ethercat/start_hal_ethercat.sh"
HAL_LOG="$ROOT/hal_user_run.log"
MODEL_LOG="$ROOT/user_model3396.log"
HAL_PGID_FILE="$ROOT/.model3396_hal.pgid"
HAL_LAUNCHER_PID=""
HAL_PGID=""
MODEL_PID=""

process_group_alive() {
  [[ "$HAL_PGID" =~ ^[0-9]+$ ]] || return 1
  ps -eo pid=,pgid=,stat= | awk -v pgid="$HAL_PGID" \
    '$2 == pgid && $3 !~ /^Z/ {found=1} END {exit !found}'
}

stop_hal() {
  [[ "$HAL_PGID" =~ ^[0-9]+$ ]] || return 0
  (( HAL_PGID > 1 )) || return 0
  process_group_alive || return 0

  echo "[serve] stopping HAL process group PGID=$HAL_PGID"
  kill -TERM -- "-$HAL_PGID" 2>/dev/null || true
  for _ in {1..100}; do
    process_group_alive || break
    sleep 0.1
  done
  if process_group_alive; then
    echo "[serve] HAL process group did not exit after TERM; sending KILL" >&2
    kill -KILL -- "-$HAL_PGID" 2>/dev/null || true
  fi
  if [[ "$HAL_LAUNCHER_PID" =~ ^[0-9]+$ ]]; then
    if kill -0 "$HAL_LAUNCHER_PID" 2>/dev/null; then
      kill -TERM "$HAL_LAUNCHER_PID" 2>/dev/null || true
    fi
    wait "$HAL_LAUNCHER_PID" 2>/dev/null || true
  fi
  for _ in {1..20}; do
    process_group_alive || break
    sleep 0.1
  done
}

verify_no_runtime_processes() {
  local leftovers
  leftovers="$(ps -eo pid=,args= | awk -v self="$$" \
    '$1 != self && \
     (($2 ~ /(^|\/)a3_deploy_model3396$/) || \
      ($2 ~ /(^|\/)aimrt_main_hal$/) || \
      ($0 ~ /(^|\/)scripts\/run_model3396_real\.sh([[:space:]]|$)/)) {print}')"
  if [[ -n "$leftovers" ]]; then
    echo "[serve] runtime process cleanup failed:" >&2
    echo "$leftovers" >&2
    return 1
  fi
  echo "[serve] verified no a3_deploy_model3396, aimrt_main_hal, or run_model3396_real.sh remain"
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM HUP
  stop_hal
  if [[ "$MODEL_PID" =~ ^[0-9]+$ ]]; then
    wait "$MODEL_PID" 2>/dev/null || true
  fi
  verify_no_runtime_processes || rc=70
  rm -f "$HAL_PGID_FILE"
  echo "[serve] stopped; logs: $MODEL_LOG and $HAL_LOG"
  exit "$rc"
}

forward_signal() {
  local signal_name=$1
  if [[ "$MODEL_PID" =~ ^[0-9]+$ ]] && kill -0 "$MODEL_PID" 2>/dev/null; then
    echo "[serve] $signal_name received; waiting for controller reset"
    kill -s "$signal_name" "$MODEL_PID" 2>/dev/null || true
  fi
}

handle_runtime_signal() {
  local signal_name="$1"
  echo "[serve] outer script received $signal_name" >&2
  forward_signal "$signal_name"
}

trap cleanup EXIT
# The real TTY foreground process group delivers Ctrl+C directly to the model,
# whose existing signal handler performs the reset. Keep the wrapper alive so
# its EXIT cleanup can wait for that reset and then stop the HAL.
trap '' INT
trap 'handle_runtime_signal TERM' TERM
trap 'handle_runtime_signal HUP' HUP

[[ "$(id -un)" == "agi" ]] || {
  echo "[serve] run as user agi on the MDU" >&2; exit 2;
}
[[ -x "$ROOT/bin/a3_deploy_model3396" ]] || {
  echo "[serve] missing controller binary" >&2; exit 2;
}
[[ -x "$HAL_SCRIPT" ]] || {
  echo "[serve] missing HAL startup script" >&2; exit 2;
}
[[ -t 0 && -e /dev/tty ]] || {
  echo "[serve] a real TTY is required; model stdin will not be redirected to /dev/null" >&2
  exit 5
}
pgrep -f '(^|/)a3_deploy_model3396([[:space:]]|$)' >/dev/null 2>&1 && {
  echo "[serve] controller is already running" >&2; exit 3;
}
pgrep -x aimrt_main_hal >/dev/null 2>&1 && {
  echo "[serve] HAL is already running" >&2; exit 3;
}

if systemctl is-active --quiet agibot_pm.service; then
  echo "[serve] stopping official MC/PM"
  sudo -n systemctl stop agibot_pm.service
  for _ in {1..50}; do
    systemctl is-active --quiet agibot_pm.service || break
    sleep 0.1
  done
  systemctl is-active --quiet agibot_pm.service && {
    echo "[serve] official MC/PM did not stop" >&2; exit 3;
  }
fi

mkdir -p "$ROOT"
rm -f "$HAL_PGID_FILE"
: > "$HAL_LOG"
: > "$MODEL_LOG"
export LD_LIBRARY_PATH="$ROOT/bin:$ROOT/bin/runtime:/opt/ros/jazzy/lib:/opt/agibot/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export AMENT_PREFIX_PATH="/opt/ros/jazzy${AMENT_PREFIX_PATH:+:$AMENT_PREFIX_PATH}"

# HAL is isolated from terminal Ctrl+C. It is stopped only after the model has
# completed its return sequence and exited.
echo "[serve] starting HAL"
setsid bash "$HAL_SCRIPT" >"$HAL_LOG" 2>&1 < /dev/null &
HAL_LAUNCHER_PID=$!

for _ in {1..50}; do
  HAL_PGID="$(ps -o pgid= -p "$HAL_LAUNCHER_PID" 2>/dev/null | tr -d ' ')"
  [[ "$HAL_PGID" =~ ^[0-9]+$ ]] && break
  sleep 0.1
done
[[ "$HAL_PGID" =~ ^[0-9]+$ ]] || {
  echo "[serve] HAL launcher failed to publish a process group" >&2; exit 4;
}

ready=0
for _ in {1..600}; do
  grep -q 'ethercat controller start success' "$HAL_LOG" 2>/dev/null && {
    ready=1; break;
  }
  process_group_alive || {
    echo "[serve] HAL exited during startup" >&2
    tail -n 80 "$HAL_LOG" >&2 || true
    exit 4
  }
  sleep 0.1
done
(( ready == 1 )) || {
  echo "[serve] HAL readiness timeout" >&2
  tail -n 80 "$HAL_LOG" >&2 || true
  exit 4
}

# The HAL launcher waits for aimrt_main_hal, but the runtime may have a
# different process group. Capture the actual runtime PGID from the complete
# command line after readiness; the launcher PID is waited separately.
HAL_RUNTIME_PID=""
HAL_RUNTIME_PGID=""
HAL_RUNTIME_INFO="$(ps -eo pid=,pgid=,args= | awk \
  '$0 ~ /(^|[[:space:]])(\.\/)?aimrt_main_hal([[:space:]]|$)/ {print $1, $2; exit}')"
read -r HAL_RUNTIME_PID HAL_RUNTIME_PGID <<< "$HAL_RUNTIME_INFO"
[[ "$HAL_RUNTIME_PGID" =~ ^[0-9]+$ ]] || {
  echo "[serve] aimrt_main_hal did not publish a valid process group" >&2
  exit 4
}
HAL_PGID="$HAL_RUNTIME_PGID"
printf '%s\n' "$HAL_PGID" > "$HAL_PGID_FILE"
echo "[serve] HAL process group PGID=$HAL_PGID launcher PID=$HAL_LAUNCHER_PID"

cd "$ROOT"
echo "[match] HAL ready; serving once automatically, then entering rally mode"
echo "[match] SPACE=serve again, Ctrl+C=return to startup pose and exit"

./bin/a3_deploy_model3396 </dev/tty >"$MODEL_LOG" 2>&1 &
MODEL_PID=$!

while true; do
  set +e
  wait "$MODEL_PID"
  MODEL_RC=$?
  set -e
  kill -0 "$MODEL_PID" 2>/dev/null || break
done
MODEL_PID=""

if (( MODEL_RC == 0 )); then
  echo "[serve] controller reset complete"
else
  echo "[serve] controller exited with code $MODEL_RC" >&2
  tail -n 80 "$MODEL_LOG" >&2 || true
fi
exit "$MODEL_RC"
