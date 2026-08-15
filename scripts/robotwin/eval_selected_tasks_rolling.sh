#!/usr/bin/env bash
# Usage:
#   bash scripts/robotwin/eval_selected_tasks_rolling.sh <gpu_id> <ckpt_path> [dataset_stats_path] [hydra_overrides...]
# dataset_stats_path defaults to <run_dir>/dataset_stats.json derived from the checkpoint path.
#
# Template:
#   bash scripts/robotwin/eval_selected_tasks_rolling.sh 0 \
#     /workspace/RollingWAM/runs/robotwin_selected_tasks_rolling_3cam_384_1e-4/2026-08-09_03-29-17/checkpoints/weights/step_002305.pt \
#     /workspace/RollingWAM/runs/robotwin_selected_tasks_rolling_3cam_384_1e-4/2026-08-09_03-29-17/dataset_stats.json

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

# optional third positional arg: dataset stats; else derived from .../checkpoints/weights/<step>.pt
if [[ $# -ge 1 && "$1" != *=* ]]; then
  DATASET_STATS="$1"
  shift
else
  DATASET_STATS="$(dirname "$(dirname "$(dirname "$CKPT")")")/dataset_stats.json"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

for task_name in lift_pot beat_block_hammer place_dual_shoes stack_bowls_two blocks_ranking_size stack_blocks_three; do
  DIFFSYNTH_MODEL_BASE_PATH=/workspace/RollingWAM/checkpoints \
  DIFFSYNTH_SKIP_DOWNLOAD=true \
  python experiments/robotwin/eval_robotwin_single.py \
    task=robotwin_selected_tasks_rolling_3cam_384_1e-4 \
    "ckpt=${CKPT}" \
    "EVALUATION.dataset_stats_path=${DATASET_STATS}" \
    EVALUATION.robotwin_root=/workspace/RollingWAM/third_party/RoboTwin \
    EVALUATION.task_name="$task_name" \
    EVALUATION.task_config=demo_clean \
    EVALUATION.eval_num_episodes=100 \
    EVALUATION.skip_get_obs_within_replan=true \
    gpu_id="$GPU_ID" \
    "$@"
done
