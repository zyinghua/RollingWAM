#!/usr/bin/env bash
# Evaluate the nine official RMBench tasks with one RollingWAM checkpoint.
#
# Usage:
#   bash scripts/rmbench/eval_all_tasks_rolling.sh \
#     <first_gpu_id> <ckpt_path> [dataset_stats_path] [hydra_overrides...]
#
# The Hydra `task=` choice describes the checkpoint/model configuration, not an
# RMBench task. Override its default with ROLLINGWAM_TASK_CONFIG when needed.

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash $0 <first_gpu_id> <ckpt_path> [dataset_stats_path] [hydra_overrides...]" >&2
  exit 2
fi

GPU_ID="$1"
CKPT="$2"
shift 2

if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
  echo "Error: first_gpu_id must be a non-negative integer, got: $GPU_ID" >&2
  exit 2
fi

if [[ $# -ge 1 && "$1" != *=* ]]; then
  DATASET_STATS="$1"
  shift
else
  DATASET_STATS="$(dirname "$(dirname "$(dirname "$CKPT")")")/dataset_stats.json"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_TASK="${ROLLINGWAM_TASK_CONFIG:-robotwin_selected_tasks_rolling_3cam_384_1e-4}"
MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${REPO_ROOT}/checkpoints}"
INSTRUCTION_TYPE="${RMBENCH_INSTRUCTION_TYPE:-unseen}"
EVAL_EPISODES="${RMBENCH_EVAL_NUM_EPISODES:-100}"
NUM_GPUS="${RMBENCH_NUM_GPUS:-1}"
TASKS_PER_GPU="${RMBENCH_TASKS_PER_GPU:-1}"
SKIP_WITHIN_REPLAN="${RMBENCH_SKIP_GET_OBS_WITHIN_REPLAN:-false}"
cd "$REPO_ROOT"

DIFFSYNTH_MODEL_BASE_PATH="$MODEL_BASE_PATH" \
DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}" \
python experiments/rmbench/run_rmbench_manager.py \
  "task=${MODEL_TASK}" \
  "ckpt=${CKPT}" \
  "EVALUATION.dataset_stats_path=${DATASET_STATS}" \
  "EVALUATION.rmbench_root=${REPO_ROOT}/third_party/RMBench" \
  EVALUATION.task_config=demo_clean \
  "EVALUATION.instruction_type=${INSTRUCTION_TYPE}" \
  "EVALUATION.eval_num_episodes=${EVAL_EPISODES}" \
  "EVALUATION.skip_get_obs_within_replan=${SKIP_WITHIN_REPLAN}" \
  "gpu_id=${GPU_ID}" \
  "MULTIRUN.num_gpus=${NUM_GPUS}" \
  "MULTIRUN.max_tasks_per_gpu=${TASKS_PER_GPU}" \
  "$@"
