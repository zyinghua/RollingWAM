#!/usr/bin/env bash
# Usage:
#   bash scripts/robotwin/train_selected_tasks_rolling_4gpu.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
DIFFSYNTH_MODEL_BASE_PATH=/workspace/RollingWAM/checkpoints \
DIFFSYNTH_SKIP_DOWNLOAD=true \
bash scripts/train_zero2.sh 4 \
  task=robotwin_selected_tasks_rolling_3cam_384_1e-4
