#!/usr/bin/env bash
# Train RollingWAM jointly on selected DOMINO task levels.
# Usage:
#   bash scripts/domino/train_selected_tasks_rolling.sh [num_gpus] [hydra_overrides...]
# Defaults: 8 GPUs and 50 demos each for task specs in DOMINO_TASK_SPECS.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

NPROC="${DOMINO_NPROC:-8}"
if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
  NPROC="$1"
  shift
fi
if [[ ! "$NPROC" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: num_gpus must be a positive integer, got: $NPROC" >&2
  exit 2
fi

RAW_SPECS="${DOMINO_TASK_SPECS:-beat_block_hammer:2 place_bread_basket:1}"
RAW_SPECS="${RAW_SPECS//,/ }"
read -r -a TASK_SPECS <<< "$RAW_SPECS"
if [[ ${#TASK_SPECS[@]} -eq 0 ]]; then
  echo "Error: no DOMINO task:level entries were selected." >&2
  exit 2
fi

DATA_ROOT="${DOMINO_DATA_ROOT:-/datasets/DOMINO/rollingwam-domino}"
EXPECT_EPISODES="${DOMINO_EXPECT_EPISODES:-50}"
if [[ ! "$EXPECT_EPISODES" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: DOMINO_EXPECT_EPISODES must be a positive integer, got: $EXPECT_EPISODES" >&2
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

DATASET_DIRS=()
TOTAL_EPISODES=0
for spec in "${TASK_SPECS[@]}"; do
  [[ "$spec" =~ ^([A-Za-z0-9_]+):([123])$ ]]
  task_name="${BASH_REMATCH[1]}"
  level="${BASH_REMATCH[2]}"
  dataset_dir="${DATA_ROOT}/${task_name}_level${level}"
  info_path="${dataset_dir}/meta/info.json"
  if [[ ! -f "$info_path" ]]; then
    echo "Error: converted DOMINO dataset not found: ${dataset_dir}" >&2
    echo "Run scripts/domino/prepare_domino_data.sh first." >&2
    exit 2
  fi
  actual="$(PYTHONDONTWRITEBYTECODE=1 python -B -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["total_episodes"])' "$info_path")"
  if [[ ! "$actual" =~ ^[0-9]+$ || "$actual" -ne "$EXPECT_EPISODES" ]]; then
    echo "Error: ${dataset_dir} has ${actual} episodes; expected ${EXPECT_EPISODES}." >&2
    exit 2
  fi
  DATASET_DIRS+=("$dataset_dir")
  TOTAL_EPISODES=$((TOTAL_EPISODES + actual))
done

EXPECTED_TOTAL=$((EXPECT_EPISODES * ${#TASK_SPECS[@]}))
[[ "$TOTAL_EPISODES" -eq "$EXPECTED_TOTAL" ]] || { echo "Error: found ${TOTAL_EPISODES} demos; expected ${EXPECTED_TOTAL}." >&2; exit 2; }
DATASET_DIRS_CSV="$(IFS=,; echo "${DATASET_DIRS[*]}")"

if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  GPU_IDS=()
  for ((gpu = 0; gpu < NPROC; gpu++)); do GPU_IDS+=("$gpu"); done
  CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPU_IDS[*]}")"
fi

echo "Training on ${TOTAL_EPISODES} DOMINO demonstrations across ${#TASK_SPECS[@]} task levels."
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
PYTHONDONTWRITEBYTECODE=1 \
DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${REPO_ROOT}/checkpoints}" \
DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}" \
bash scripts/train_zero2.sh "$NPROC" \
  task=domino_selected_tasks_rolling_3cam_384_1e-4 \
  "data.dataset_root=${DATA_ROOT}" \
  "data.dataset_dirs=[${DATASET_DIRS_CSV}]" \
  "$@"
