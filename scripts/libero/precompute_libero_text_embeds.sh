#!/usr/bin/env bash
# One-off: encodes every instruction of the four LIBERO suite datasets into the
# text-embed cache (training requirement; eval encodes prompts live).
# Usage:
#   bash scripts/libero/precompute_libero_text_embeds.sh [num_gpus]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

NPROC="${1:-4}"

DIFFSYNTH_MODEL_BASE_PATH=/workspace/RollingWAM/checkpoints \
DIFFSYNTH_SKIP_DOWNLOAD=true \
torchrun --standalone --nproc_per_node="$NPROC" scripts/precompute_text_embeds.py \
  task=libero_rolling_2cam224_1e-4
