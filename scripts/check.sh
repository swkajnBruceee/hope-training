#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}" || exit 1

fail=0

step() {
  printf '\n[check] %s\n' "$1"
}

fail_msg() {
  printf '[check] ERROR: %s\n' "$1" >&2
  fail=1
}

step "validating local-only files are ignored"
git check-ignore -q hope_training/whole_body_tracking/setup_train_env.local.sh \
  || fail_msg "setup_train_env.local.sh is not ignored"
git check-ignore -q .env \
  || fail_msg ".env is not ignored"

step "checking tracked local-only files"
tracked_local="$(git ls-files \
  hope_training/whole_body_tracking/setup_train_env.local.sh \
  .env \
  '.env.*' \
  '*.local.yaml' \
  '*.local.yml' \
  '*.local.json')"
if [ -n "${tracked_local}" ]; then
  printf '%s\n' "${tracked_local}" >&2
  fail_msg "local-only files are tracked"
fi

step "checking tracked root-level generated data"
tracked_data="$(git ls-files '*.csv' '*.bag' '*.pt' '*.pth' '*.ckpt' '*.onnx' '*.rknn' '*.engine' '*.trt' '*.npz' ':!:agibot/**')"
if [ -n "${tracked_data}" ]; then
  printf '%s\n' "${tracked_data}" >&2
  fail_msg "generated data/model artifacts are tracked outside approved reference folders"
fi

step "compiling Python sources"
PYTHONPATH="${ROOT}/hope_ws/src/hope_planner:${ROOT}/hope_training/whole_body_tracking/source/whole_body_tracking" \
  python3 -m compileall -q scripts hope_ws/src/hope_planner/hope_planner hope_training/whole_body_tracking/scripts \
  || fail_msg "Python compile check failed"

step "running planner unit tests when dependencies are available"
if python3 -c "import pytest, numpy" >/dev/null 2>&1; then
  PYTHONPATH="${ROOT}/hope_ws/src/hope_planner" \
    python3 -m pytest -q hope_ws/src/hope_planner/test \
    || fail_msg "planner unit tests failed"
else
  printf '[check] skip: pytest and/or numpy is not installed in this shell\n'
fi

if [ "${fail}" -ne 0 ]; then
  printf '\n[check] failed\n' >&2
  exit 1
fi

printf '\n[check] ok\n'
