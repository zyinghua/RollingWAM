#!/usr/bin/env bash
# Evaluate selected DOMINO task levels sequentially on one GPU.
# Usage:
#   bash scripts/domino/eval_selected_tasks_rolling.sh \
#     <gpu_id> <ckpt_path> [dataset_stats_path] [hydra_overrides...]
# Select a subset with DOMINO_TASK_SPECS="task_name:level ...".

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash $0 <gpu_id> <ckpt_path> [dataset_stats_path] [hydra_overrides...]" >&2
  exit 2
fi

GPU_ID="$1"
CKPT="$2"
shift 2
if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
  echo "Error: gpu_id must be a non-negative integer, got: $GPU_ID" >&2
  exit 2
fi

if [[ $# -ge 1 && "$1" != *=* ]]; then
  DATASET_STATS="$1"
  shift
else
  DATASET_STATS="$(dirname "$(dirname "$(dirname "$CKPT")")")/dataset_stats.json"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RAW_SPECS="${DOMINO_TASK_SPECS:-beat_block_hammer:2 place_bread_basket:1}"
RAW_SPECS="${RAW_SPECS//,/ }"
read -r -a TASK_SPECS <<< "$RAW_SPECS"
if [[ ${#TASK_SPECS[@]} -eq 0 ]]; then
  echo "Error: no DOMINO task:level entries were selected." >&2
  exit 2
fi

MODEL_TASK="${ROLLINGWAM_TASK_CONFIG:-domino_selected_tasks_rolling_3cam_384_1e-4}"
MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${REPO_ROOT}/checkpoints}"
INSTRUCTION_TYPE="${DOMINO_INSTRUCTION_TYPE:-unseen}"
RUN_TS="${DOMINO_EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${DOMINO_EVAL_OUTPUT_DIR:-./evaluate_results/domino/${RUN_TS}}"

if [[ "$INSTRUCTION_TYPE" != "seen" && "$INSTRUCTION_TYPE" != "unseen" ]]; then
  echo "Error: DOMINO_INSTRUCTION_TYPE must be seen or unseen, got: $INSTRUCTION_TYPE" >&2
  exit 2
fi
SEEN_SPECS=" "
for spec in "${TASK_SPECS[@]}"; do
  if [[ ! "$spec" =~ ^([A-Za-z0-9_]+):([123])$ ]]; then
    echo "Error: invalid task spec '$spec'; expected task_name:1, task_name:2, or task_name:3." >&2
    exit 2
  fi
  if [[ "$SEEN_SPECS" == *" $spec "* ]]; then
    echo "Error: duplicate DOMINO task spec: $spec" >&2
    exit 2
  fi
  SEEN_SPECS+="$spec "
done

for spec in "${TASK_SPECS[@]}"; do
  [[ "$spec" =~ ^([A-Za-z0-9_]+):([123])$ ]]
  task_name="${BASH_REMATCH[1]}"
  level="${BASH_REMATCH[2]}"
  task_config="demo_clean_dynamic_level${level}"
  if [[ ! -f "third_party/DOMINO/task_config/${task_config}.yml" ]]; then
    echo "Error: DOMINO evaluation config not found: ${task_config}.yml" >&2
    exit 2
  fi

  echo "=== Evaluating ${task_name}, dynamic level ${level} (DOMINO native 100 episodes) ==="
  DIFFSYNTH_MODEL_BASE_PATH="$MODEL_BASE_PATH" \
  DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}" \
  PYTHONDONTWRITEBYTECODE=1 \
  python -B experiments/domino/eval_domino_single.py \
    "task=${MODEL_TASK}" \
    "ckpt=${CKPT}" \
    "EVALUATION.dataset_stats_path=${DATASET_STATS}" \
    "EVALUATION.domino_root=${REPO_ROOT}/third_party/DOMINO" \
    "EVALUATION.task_name=${task_name}" \
    "EVALUATION.dynamic_level=${level}" \
    "EVALUATION.task_config=${task_config}" \
    "EVALUATION.instruction_type=${INSTRUCTION_TYPE}" \
    "EVALUATION.output_dir=${OUTPUT_DIR}" \
    "gpu_id=${GPU_ID}" \
    "$@"
done

echo "Completed ${#TASK_SPECS[@]} DOMINO task-level evaluations."
