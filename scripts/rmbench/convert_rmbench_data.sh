#!/usr/bin/env bash
# Convert the raw RMBench demo_clean data into per-task LeRobot datasets and
# validate each one. CPU-only; run after third_party/RMBench/data/_download.py.
#
# Usage:
#   bash scripts/rmbench/convert_rmbench_data.sh                    # all 9 official tasks
#   bash scripts/rmbench/convert_rmbench_data.sh put_back_block ... # subset
#
# Environment overrides:
#   RMBENCH_RAW_ROOT        raw data root      (default /datasets/RMBench-data/data)
#   RMBENCH_OUT_ROOT        converted-out root (default /datasets/RMBench-data/lerobot)
#   RMBENCH_CONFIG          raw config subdir  (default demo_clean; e.g. demo_clean_200)
#   RMBENCH_EXPECT_EPISODES episodes per task  (default 50; 200 for demo_clean_200)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RAW_ROOT="${RMBENCH_RAW_ROOT:-/datasets/RMBench-data/data}"
OUT_ROOT="${RMBENCH_OUT_ROOT:-/datasets/RMBench-data/lerobot}"
CONFIG="${RMBENCH_CONFIG:-demo_clean}"
EXPECT_EPISODES="${RMBENCH_EXPECT_EPISODES:-50}"

TASKS=(
  observe_and_pickup
  rearrange_blocks
  put_back_block
  swap_blocks
  swap_T
  battery_try
  blocks_ranking_try
  cover_blocks
  press_button
)
if [[ $# -gt 0 ]]; then
  TASKS=("$@")
fi

for task in "${TASKS[@]}"; do
  echo "=== ${task} ==="
  python tools/rmbench/raw_to_lerobot.py \
    --raw-root "${RAW_ROOT}" \
    --task "${task}" \
    --config "${CONFIG}" \
    --out "${OUT_ROOT}/${task}"
  python tools/rmbench/validate_dataset.py \
    --dataset "${OUT_ROOT}/${task}" \
    --expect-episodes "${EXPECT_EPISODES}"
done

echo "All requested tasks converted and validated under ${OUT_ROOT}."
