#!/usr/bin/env bash
# Run on the model host. The simulator connects to this machine's address.
# Usage: bash scripts/robotwin/serve_robotwin_policy.sh <checkpoint.pt> [server_args...]
# Example:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/robotwin/serve_robotwin_policy.sh \
#     /runs/robotwin/checkpoints/weights/step_002500.pt --host 0.0.0.0 --port 8000
#   bash scripts/robotwin/serve_robotwin_policy.sh /models/step_002500.pt \
#     --config /models/config.yaml --dataset-stats /models/dataset_stats.json --num-steps 10

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ $# -eq 0 ]]; then
  echo "Usage: bash scripts/robotwin/serve_robotwin_policy.sh <checkpoint.pt> [server_args...]" >&2
  exit 2
fi
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  exec python experiments/robotwin/serve_robotwin_policy.py --help
fi

CHECKPOINT="$1"
shift
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONDONTWRITEBYTECODE=1
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${REPO_ROOT}/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
exec python experiments/robotwin/serve_robotwin_policy.py --checkpoint "$CHECKPOINT" "$@"
