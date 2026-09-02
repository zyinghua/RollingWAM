#!/usr/bin/env bash
# Serve a trained RollingWAM G1 policy over WebSocket.
# Usage:
#   bash scripts/g1/serve_g1_policy.sh <checkpoint.pt>

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/g1/serve_g1_policy.sh <checkpoint.pt>" >&2
  exit 2
fi

CHECKPOINT="$1"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
PYTHONDONTWRITEBYTECODE=1 \
DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${REPO_ROOT}/checkpoints}" \
DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}" \
python scripts/serve.py \
  --checkpoint "$CHECKPOINT" \
  --device cuda:0 \
  --num-steps 10 \
  --embodiment unitree_g1_sonic \
  --image-key ego_view \
  --state-key state \
  --action-key action \
  --fps 10 \
  --host 0.0.0.0 \
  --port 8000
