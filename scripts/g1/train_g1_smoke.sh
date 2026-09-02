#!/usr/bin/env bash
# Train RollingWAM on both tasks in the Unitree G1 LeRobot v3 dataset.
# Usage:
#   bash scripts/g1/train_g1_smoke.sh [num_gpus] [hydra_overrides...]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

NPROC="${1:-8}"
if [[ $# -gt 0 ]]; then
  shift
fi
if [[ ! "$NPROC" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: num_gpus must be a positive integer, got: $NPROC" >&2
  exit 2
fi

if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  GPU_IDS=()
  for ((gpu = 0; gpu < NPROC; gpu++)); do
    GPU_IDS+=("$gpu")
  done
  CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPU_IDS[*]}")"
fi

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
PYTHONDONTWRITEBYTECODE=1 \
DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${REPO_ROOT}/checkpoints}" \
DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}" \
bash scripts/train_zero2.sh "$NPROC" \
  task=g1_pnp_pour_rolling_1cam_320_1e-4 \
  "$@"
