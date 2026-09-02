#!/usr/bin/env bash
# Precompute the two G1 dataset task prompts before the training smoke test.
# Usage:
#   bash scripts/g1/precompute_g1_text_embeds.sh [num_gpus] [hydra_overrides...]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

NPROC="${1:-1}"
if [[ $# -gt 0 ]]; then
  shift
fi
if [[ ! "$NPROC" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: num_gpus must be a positive integer, got: $NPROC" >&2
  exit 2
fi

PYTHONDONTWRITEBYTECODE=1 \
DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${REPO_ROOT}/checkpoints}" \
DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}" \
torchrun --standalone --nproc_per_node="$NPROC" scripts/precompute_text_embeds.py \
  task=g1_pnp_pour_rolling_1cam_320_1e-4 \
  "$@"
