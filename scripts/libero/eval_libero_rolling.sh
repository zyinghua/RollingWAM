#!/usr/bin/env bash
# Evaluates a checkpoint on all four LIBERO suites via the multi-GPU manager.
# Usage:
#   bash scripts/libero/eval_libero_rolling.sh <ckpt_path> [hydra_overrides...]
# Example:
#   bash scripts/libero/eval_libero_rolling.sh runs/libero_rolling_2cam224_1e-4/<ts>/checkpoints/weights/step_XXXXXX.pt

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash $0 <ckpt_path> [hydra_overrides...]" >&2
  exit 2
fi
CKPT="$1"
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DIFFSYNTH_MODEL_BASE_PATH=/workspace/RollingWAM/checkpoints \
DIFFSYNTH_SKIP_DOWNLOAD=true \
python experiments/libero/run_libero_manager.py \
  task=libero_rolling_2cam224_1e-4 \
  "ckpt=${CKPT}" \
  "$@"
