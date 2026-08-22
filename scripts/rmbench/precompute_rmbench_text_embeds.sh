#!/usr/bin/env bash
# One-off: encodes every instruction of the converted RMBench task datasets into
# the text-embed cache (training requirement; eval encodes prompts live).
# Run AFTER scripts/rmbench/convert_rmbench_data.sh.
# Usage:
#   bash scripts/rmbench/precompute_rmbench_text_embeds.sh [num_gpus]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

NPROC="${1:-4}"

DIFFSYNTH_MODEL_BASE_PATH=/workspace/RollingWAM/checkpoints \
DIFFSYNTH_SKIP_DOWNLOAD=true \
torchrun --standalone --nproc_per_node="$NPROC" scripts/precompute_text_embeds.py \
  task=rmbench_rolling_3cam_384_1e-4
