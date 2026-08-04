#!/usr/bin/env bash
# Usage:
#   bash scripts/robotwin/eval_selected_tasks_rolling.sh <gpu_id>
# Example:
#   bash scripts/robotwin/eval_selected_tasks_rolling.sh 0

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/robotwin/eval_selected_tasks_rolling.sh <gpu_id>" >&2
  exit 2
fi

GPU_ID="$1"
if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
  echo "Error: gpu_id must be a non-negative integer, got: $GPU_ID" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

for task_name in beat_block_hammer lift_pot put_object_cabinet; do
  DIFFSYNTH_MODEL_BASE_PATH=/workspace/RollingWAM/checkpoints \
  DIFFSYNTH_SKIP_DOWNLOAD=true \
  python experiments/robotwin/eval_robotwin_single.py \
    task=robotwin_selected_tasks_rolling_3cam_384_1e-4 \
    ckpt=/workspace/RollingWAM/runs/robotwin_selected_tasks_rolling_joint_3cam_384_1e-4/2026-08-02_03-17-47/checkpoints/weights/step_000955.pt \
    EVALUATION.dataset_stats_path=/workspace/RollingWAM/runs/robotwin_selected_tasks_rolling_joint_3cam_384_1e-4/2026-08-02_03-17-47/dataset_stats.json \
    EVALUATION.robotwin_root=/workspace/RollingWAM/third_party/RoboTwin \
    EVALUATION.task_name="$task_name" \
    EVALUATION.task_config=demo_clean \
    EVALUATION.eval_num_episodes=100 \
    EVALUATION.skip_get_obs_within_replan=true \
    gpu_id="$GPU_ID"
done
