#!/usr/bin/env bash
# Single window/chunk ablation run on the selected-task subset.
# actions_per_chunk and num_frames derive from window_blocks / chunk_latents in the yaml.
#
# Usage:
#   bash scripts/robotwin/train_window_ablation_8gpu.sh <window_blocks> <chunk_latents> <eval_num_inference_steps> [batch_size] [grad_accum]
# batch_size / grad_accum fall back to the task yaml defaults when omitted.
# Examples (aspc x W -> args):
#   16x5: bash scripts/robotwin/train_window_ablation_8gpu.sh 5 1 10
#   16x7: bash scripts/robotwin/train_window_ablation_8gpu.sh 7 1 14 2 4
#   32x3: bash scripts/robotwin/train_window_ablation_8gpu.sh 3 2 12 2 4

set -euo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "Usage: bash $0 <window_blocks> <chunk_latents> <eval_num_inference_steps> [batch_size] [grad_accum]" >&2
  exit 2
fi

OVERRIDES=(
  "model.rolling.window_blocks=$1"
  "model.rolling.chunk_latents=$2"
  "eval_num_inference_steps=$3"
)
if [[ $# -ge 4 ]]; then
  OVERRIDES+=("batch_size=$4")
fi
if [[ $# -ge 5 ]]; then
  OVERRIDES+=("gradient_accumulation_steps=$5")
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
DIFFSYNTH_MODEL_BASE_PATH=/workspace/RollingWAM/checkpoints \
DIFFSYNTH_SKIP_DOWNLOAD=true \
bash scripts/train_zero2.sh 8 \
  task=robotwin_selected_tasks_rolling_3cam_384_1e-4 \
  "${OVERRIDES[@]}"
