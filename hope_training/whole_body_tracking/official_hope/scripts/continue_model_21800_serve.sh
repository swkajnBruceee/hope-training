#!/usr/bin/env bash
set -euo pipefail

# Official ordinary continuation: restores actor/critic, optimizer and iteration=21800.
# model_21800 predates hope_exact_resume_state, so checkpoint_exact_resume must remain false.
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
checkpoint_path="${repo_dir}/checkpoints/model_21800.pt"
checkpoint_override="checkpoints/model_21800.pt"
additional_iterations="${1:-1000}"

if [[ ! -f "${checkpoint_path}" ]]; then
  echo "checkpoint not found: ${checkpoint_path}" >&2
  exit 1
fi
if [[ -z "${ISAAC_PYTHON:-}" || ! -x "${ISAAC_PYTHON}" ]]; then
  echo "ISAAC_PYTHON is not set; run 'source setup_train_env.sh' first" >&2
  exit 127
fi

cd "${repo_dir}"
exec env PYTHONPATH="${HOPE_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${ISAAC_PYTHON}" scripts/train.py \
  task=HOPEPingPongServeFT \
  algo=ppo \
  headless=true \
  checkpoint_path="${checkpoint_override}" \
  checkpoint_exact_resume=false \
  algo.runner.empirical_normalization=false \
  max_iterations="${additional_iterations}" \
  run_name=model21800_serve_continuation
