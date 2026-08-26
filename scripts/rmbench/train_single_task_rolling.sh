#!/usr/bin/env bash
# Train RollingWAM on one RMBench task (RMBench's per-task evaluation protocol).
# Defaults: constant LR, 1500 steps, batch_size 4 x grad_accum 1 x num_gpus.
# Every default is a plain Hydra override, so trailing overrides win.
#
# Usage:
#   bash scripts/rmbench/train_single_task_rolling.sh <task_name> [hydra_overrides...]
#   bash scripts/rmbench/train_single_task_rolling.sh put_back_block max_steps=1500
#   RMBENCH_NPROC=4 bash scripts/rmbench/train_single_task_rolling.sh put_back_block gradient_accumulation_steps=8

set -euo pipefail

TASK_NAME="${1:?Usage: bash $0 <task_name> [hydra_overrides...]}"
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

NPROC="${RMBENCH_NPROC:-8}"
TASK_DIR="${RMBENCH_DATA_ROOT:-/datasets/RMBench-data/rollingwam-rmbench}/${TASK_NAME}"
[[ -f "${TASK_DIR}/meta/info.json" ]] || { echo "No converted dataset at ${TASK_DIR}" >&2; exit 2; }

# train_zero2.sh names the run directory after the Hydra task config, which is
# shared by every RMBench task; tag it so per-task runs stay apart.
export RUN_ID="${RUN_ID:-${TASK_NAME}_$(date +%Y-%m-%d_%H-%M-%S)}"

DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${REPO_ROOT}/checkpoints}" \
DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}" \
bash scripts/train_zero2.sh "${NPROC}" \
  task=rmbench_rolling_3cam_384_1e-4 \
  "data.dataset_dirs=[${TASK_DIR}]" \
  lr_scheduler_type=constant \
  max_steps=1500 \
  batch_size=4 \
  gradient_accumulation_steps=1 \
  "$@"
