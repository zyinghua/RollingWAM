#!/usr/bin/env bash

set -euo pipefail

RESUME_STATE="${1:?Pass checkpoints/state/step_xxxxxx}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
DIFFSYNTH_MODEL_BASE_PATH=/workspace/RollingWAM/checkpoints \
DIFFSYNTH_SKIP_DOWNLOAD=true \
bash scripts/train_zero2.sh 4 \
  task=robotwin_selected_tasks_rolling_3cam_384_1e-4_4gpu \
  "resume=$RESUME_STATE"
