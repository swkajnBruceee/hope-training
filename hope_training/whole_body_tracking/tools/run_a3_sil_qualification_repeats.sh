#!/usr/bin/env bash

# Run target-bound A3 MuJoCo standalone repeats against an already-running
# local simulator.  This is intentionally a SIL-only orchestrator: it invokes
# the project-side RobotIOBackend runner and official-model task recorder, but
# never starts hardware transport or a real robot.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_a3_sil_qualification_repeats.sh \
    --executor-contract <executor_contract.json> --target-spec <target_spec.json> \
  --command <canonical_command.npz> --runner <a3_strike_robotio_replay> \
  --backend-config <a3_aimrt_config.ros2_mujoco_sim.yaml> --out <fresh-dir> \
    [--repeats 10] [--task-duration-s 3.0] [--tail-s 0.5]

Start the A3 MuJoCo/AimRT simulator in another terminal first. The target must
already be immutable; every recorded task sample embeds and is checked against
its target hash and racket mount contract.
EOF
}

executor_contract=""
target_spec=""
command=""
runner=""
backend_config=""
out_dir=""
repeats=10
task_duration_s=3.0
tail_s=0.5

while [[ $# -gt 0 ]]; do
  case "$1" in
    --executor-contract) executor_contract=${2:?}; shift 2 ;;
    --target-spec) target_spec=${2:?}; shift 2 ;;
    --command) command=${2:?}; shift 2 ;;
    --runner) runner=${2:?}; shift 2 ;;
    --backend-config) backend_config=${2:?}; shift 2 ;;
    --out) out_dir=${2:?}; shift 2 ;;
    --repeats) repeats=${2:?}; shift 2 ;;
    --task-duration-s) task_duration_s=${2:?}; shift 2 ;;
    --tail-s) tail_s=${2:?}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 64 ;;
  esac
done

for required in executor_contract target_spec command runner backend_config out_dir; do
  if [[ -z ${!required} ]]; then
    echo "missing --${required//_/-}" >&2
    usage >&2
    exit 64
  fi
done
if ! [[ $repeats =~ ^[0-9]+$ ]] || (( repeats < 10 )); then
  echo "--repeats must be an integer of at least 10 for qualification" >&2
  exit 64
fi
for input in "$executor_contract" "$target_spec" "$command" "$runner" "$backend_config"; do
  if [[ ! -f $input ]]; then
    echo "required file is missing: $input" >&2
    exit 66
  fi
done
if [[ -e $out_dir ]]; then
  echo "--out must be a fresh path to avoid mixing repeat evidence: $out_dir" >&2
  exit 73
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
wbt_root=$(cd "$script_dir/.." && pwd)
project_root=$(cd "$wbt_root/../.." && pwd)
# shellcheck disable=SC1091
set +u  # The external colcon overlay reads optional variables such as COLCON_TRACE.
source "$script_dir/setup_a3_mujoco_sim_env.sh" >/dev/null
set -u

# Fail before touching the simulator when someone supplies the review packet
# rather than the separately frozen immutable target specification.
PYTHONPATH="$script_dir${PYTHONPATH:+:$PYTHONPATH}" python - "$target_spec" <<'PY'
import json
import sys
from pathlib import Path
from a3_strike_contract import verify_target_spec

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
try:
    digest = verify_target_spec(payload)
except (TypeError, ValueError) as exc:
    raise SystemExit(f"--target-spec is not an immutable A3 target specification: {exc}") from exc
if payload.get("source_target_sha256") != digest:
    raise SystemExit("target spec must declare its immutable source_target_sha256")
PY

mkdir -p "$out_dir"
backend_options="cfg_file_path=$backend_config,sync_mode=min_skew_pair,sync_hz=500,max_skew_ms=10,max_sample_age_ms=100,publish_enabled=true"
qualification_args=(
  --executor-contract "$executor_contract" --target-spec "$target_spec" --command "$command"
  --out "$out_dir/qualification_report.json"
)

for ((repeat_index = 1; repeat_index <= repeats; repeat_index++)); do
  run_dir=$(printf '%s/repeat_%02d' "$out_dir" "$repeat_index")
  mkdir -p "$run_dir"
  sampler_log="$run_dir/task_sampler.log"
  python "$script_dir/record_a3_racket_task_samples.py" \
    --output "$run_dir/actual_task_samples.npz" --target-spec "$target_spec" \
    --duration-s "$task_duration_s" --start-timeout-s 60 --skip-command-messages 150 \
    --stand-gate-file "$run_dir/pd_stand_gate.json" \
    >"$sampler_log" 2>&1 &
  sampler_pid=$!
  # Let DDS discovery complete before the runner starts its 3-second PD-STAND.
  sleep 1
  # Start the backend first. It writes a marker only after its state channels
  # are ready, then blocks. This eliminates the previous uncontrolled gap
  # between a keyframe reset and the first PD-STAND command.
  "$runner" --command "$command" --backend-config "$backend_options" --out "$run_dir" --tail-s "$tail_s" \
    --stand-gate-file "$run_dir/pd_stand_gate.json" \
    --stand-reset-ready-file "$run_dir/backend_ready_for_stand_reset" \
    --stand-reset-ack-file "$run_dir/stand_reset_ack.json" \
    >"$run_dir/runner.log" 2>&1 &
  runner_pid=$!
  reset_ready_deadline=$((SECONDS + 15))
  while [[ ! -f "$run_dir/backend_ready_for_stand_reset" && $SECONDS -lt $reset_ready_deadline ]]; do
    sleep 0.05
  done
  # The model root is a free joint.  Reset it to its documented `stand`
  # keyframe only after the runner has completed its transport startup, then
  # acknowledge the verified upright pose to release PD-STAND immediately.
  if [[ ! -f "$run_dir/backend_ready_for_stand_reset" ]] || ! python "$script_dir/reset_a3_mujoco_to_stand.py" \
      --ack-file "$run_dir/stand_reset_ack.json"; then
    kill "$runner_pid" 2>/dev/null || true
    kill "$sampler_pid" 2>/dev/null || true
    wait "$runner_pid" 2>/dev/null || true
    wait "$sampler_pid" 2>/dev/null || true
    echo "repeat $repeat_index: could not establish the A3 stand keyframe" >&2
    exit 1
  fi
  set +e
  wait "$runner_pid"
  runner_status=$?
  set -e
  if (( runner_status != 0 )); then
    kill "$sampler_pid" 2>/dev/null || true
    wait "$sampler_pid" 2>/dev/null || true
    echo "repeat $repeat_index: RobotIOBackend runner failed; see $run_dir/runner.log" >&2
    exit "$runner_status"
  fi
  if ! wait "$sampler_pid"; then
    echo "repeat $repeat_index: task sampler failed; see $sampler_log" >&2
    exit 1
  fi
  python "$script_dir/convert_a3_robotio_replay.py" \
    --raw-state-csv "$run_dir/raw_state.csv" --command-csv "$run_dir/command.csv" \
    --output "$run_dir/raw_state_sidecar.npz"
  qualification_args+=(--rollout-task-samples "$run_dir/actual_task_samples.npz")
  qualification_args+=(--rollout-state-samples "$run_dir/raw_state_sidecar.npz")
  echo "repeat $repeat_index/$repeats complete: $run_dir"
done

python "$script_dir/a3_standalone_qualification.py" "${qualification_args[@]}"
echo "qualification evidence: $out_dir"
