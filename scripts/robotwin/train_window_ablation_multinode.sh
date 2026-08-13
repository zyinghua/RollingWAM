#!/usr/bin/env bash
# Window-size ablation on a MULTI-NODE allocation: one W at a time, each trained
# as a single job spanning all nodes (world size = NNODES * 8). Every node in the
# allocation runs THIS SAME script with the same arguments; nodes rendezvous per
# run and proceed to the next W together.
#
# chunk_latents is 1 everywhere. batch_size 4 x grad_accum 1 x 32 GPUs = effective 128.
#   W:            1   2   3   4   5   6   7   8
#   eval steps:  12  12  12  12  15  12  14  16
#
# Required env on every node (map from your scheduler):
#   NNODES       total nodes in the job          (SLURM: $SLURM_NNODES)
#   NODE_RANK    this node's rank, 0-based       (SLURM: $SLURM_NODEID)
#   MASTER_ADDR  hostname/IP of rank-0 node      (SLURM: scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
#   MASTER_PORT  optional, default 29500
#
# Usage (identical command on every node):
#   bash scripts/robotwin/train_window_ablation_multinode.sh            # all 8 Ws, sequentially
#   bash scripts/robotwin/train_window_ablation_multinode.sh 5 7 8      # subset
#   DRY_RUN=1 NNODES=4 NODE_RANK=0 MASTER_ADDR=x bash scripts/robotwin/train_window_ablation_multinode.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

TASK=robotwin_selected_tasks_rolling_3cam_384_1e-4
DRY_RUN="${DRY_RUN:-0}"
BATCH_SIZE=4
GRAD_ACCUM=1

: "${NNODES:?Set NNODES (total nodes in the job)}"
: "${NODE_RANK:?Set NODE_RANK (0-based rank of this node)}"
: "${MASTER_ADDR:?Set MASTER_ADDR (rank-0 node hostname/IP)}"

WINDOWS=(1 2 3 4 5 6 7 8)
EVAL_STEPS=(12 12 12 12 15 12 14 16)

SELECTED=("${WINDOWS[@]}")
if [[ $# -gt 0 ]]; then
  SELECTED=("$@")
fi

index_of() {
  local w="$1"
  for i in "${!WINDOWS[@]}"; do
    if [[ "${WINDOWS[$i]}" == "$w" ]]; then
      echo "$i"
      return 0
    fi
  done
  return 1
}

for w in "${SELECTED[@]}"; do
  if ! i="$(index_of "$w")"; then
    echo "Error: unknown window size '$w' (choose from: ${WINDOWS[*]})" >&2
    exit 2
  fi

  world_size=$(( NNODES * 8 ))
  effective_batch=$(( BATCH_SIZE * GRAD_ACCUM * world_size ))
  echo "=========================================================="
  echo "[ablation] W=${w} chunk_latents=1 eval_steps=${EVAL_STEPS[$i]}" \
       "world=${world_size} bs=${BATCH_SIZE} accum=${GRAD_ACCUM}" \
       "(effective ${effective_batch}) node_rank=${NODE_RANK}  ($(date '+%F %T'))"
  echo "=========================================================="

  cmd=(
    bash scripts/train_zero2.sh 8
    "task=${TASK}"
    "model.rolling.window_blocks=${w}"
    "model.rolling.chunk_latents=1"
    "batch_size=${BATCH_SIZE}"
    "gradient_accumulation_steps=${GRAD_ACCUM}"
    "eval_num_inference_steps=${EVAL_STEPS[$i]}"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN: NNODES=${NNODES} NODE_RANK=${NODE_RANK} MASTER_ADDR=${MASTER_ADDR} ${cmd[*]}"
    continue
  fi

  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  DIFFSYNTH_MODEL_BASE_PATH=/workspace/RollingWAM/checkpoints \
  DIFFSYNTH_SKIP_DOWNLOAD=true \
  "${cmd[@]}"

  if [[ "${NODE_RANK}" == "0" ]]; then
    run_dir="$(ls -td "runs/${TASK}"/*/ 2>/dev/null | head -1 || true)"
    echo "[ablation] W=${w} finished ($(date '+%F %T')) -> ${run_dir:-<run dir not found>}"
  fi
done

echo "[ablation] node_rank=${NODE_RANK}: all requested window sizes finished."
