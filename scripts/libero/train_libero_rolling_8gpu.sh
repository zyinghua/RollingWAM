#!/usr/bin/env bash
# Trains one model on all four LIBERO suites together (spatial/object/goal/10);
# the data config lists all four suite datasets.
# Usage:
#   bash scripts/libero/train_libero_rolling_8gpu.sh [hydra_overrides...]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
DIFFSYNTH_MODEL_BASE_PATH=/workspace/RollingWAM/checkpoints \
DIFFSYNTH_SKIP_DOWNLOAD=true \
bash scripts/train_zero2.sh 8 \
  task=libero_rolling_2cam224_1e-4 \
  "$@"
