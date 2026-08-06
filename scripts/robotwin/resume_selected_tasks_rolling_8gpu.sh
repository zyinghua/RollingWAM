#!/usr/bin/env bash
# Usage: bash scripts/robotwin/resume_selected_tasks_rolling_8gpu.sh <state_dir>

set -euo pipefail

RESUME_STATE="${1:?Pass checkpoints/state/step_xxxxxx}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
DIFFSYNTH_MODEL_BASE_PATH=/workspace/RollingWAM/checkpoints \
DIFFSYNTH_SKIP_DOWNLOAD=true \
bash scripts/train_zero2.sh 8 \
  task=robotwin_selected_tasks_rolling_3cam_384_1e-4 \
  "resume=$RESUME_STATE"
