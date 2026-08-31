#!/usr/bin/env bash
# Time RollingWAM checkpoints for W=1 through 8.
#
# Manifest format (tab-separated, one row per window size):
#   <window_blocks>\t<checkpoint_path>\t<dataset_stats_path>
#
# Usage:
#   bash scripts/robotwin/eval_window_timing_ablation.sh \
#     <gpu_id> <manifest.tsv> [task_name] [episodes] [result_dir]

set -euo pipefail

usage() {
  echo "Usage: bash $0 <gpu_id> <manifest.tsv> [task_name=beat_block_hammer] [episodes=10] [result_dir]" >&2
}

fail() {
  echo "Error: $*" >&2
  exit 2
}

if [[ $# -lt 2 || $# -gt 5 ]]; then
  usage
  exit 2
fi

GPU_ID="$1"
MANIFEST="$2"
TASK_NAME="${3:-beat_block_hammer}"
EPISODES="${4:-10}"
DRY_RUN="${DRY_RUN:-0}"
SEED="${SEED:-0}"

[[ "$GPU_ID" =~ ^[0-9]+$ ]] || fail "gpu_id must be a non-negative integer, got: $GPU_ID"
[[ "$EPISODES" =~ ^[1-9][0-9]*$ ]] || fail "episodes must be a positive integer, got: $EPISODES"
[[ "$DRY_RUN" == "0" || "$DRY_RUN" == "1" ]] || fail "DRY_RUN must be 0 or 1, got: $DRY_RUN"
[[ "$SEED" =~ ^[0-9]+$ ]] || fail "SEED must be a non-negative integer, got: $SEED"
[[ -n "$TASK_NAME" ]] || fail "task_name must not be empty"
[[ -f "$MANIFEST" ]] || fail "manifest not found: $MANIFEST"

MANIFEST="$(cd "$(dirname "$MANIFEST")" && pwd -P)/$(basename "$MANIFEST")"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TIMING_INIT="$REPO_ROOT/experiments/robotwin/rollingwam_policy/__init__.py"

if ! grep -Eq '^[[:space:]]*from[[:space:]]+\.deploy_policy_timing[[:space:]]+import[[:space:]]+\*[[:space:]]*$' "$TIMING_INIT"; then
  fail "timing policy is not wired. Set $TIMING_INIT to: from .deploy_policy_timing import *"
fi

ROLLING_HYDRA_TASK="${ROLLING_HYDRA_TASK:-${HYDRA_TASK:-robotwin_selected_tasks_rolling_3cam_384_1e-4}}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-$REPO_ROOT/third_party/RoboTwin}"
MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$REPO_ROOT/checkpoints}"
PYTHON_BIN="${PYTHON_BIN:-python}"

[[ -d "$ROBOTWIN_ROOT" ]] || fail "RoboTwin root not found: $ROBOTWIN_ROOT"
[[ -d "$MODEL_BASE_PATH" ]] || fail "model base path not found: $MODEL_BASE_PATH"
ROBOTWIN_ROOT="$(cd "$ROBOTWIN_ROOT" && pwd -P)"
MODEL_BASE_PATH="$(cd "$MODEL_BASE_PATH" && pwd -P)"

declare -a CHECKPOINTS=()
declare -a DATASET_STATS=()
ROW_COUNT=0
LINE_NUMBER=0

while IFS=$'\t' read -r window checkpoint stats extra || [[ -n "${window}${checkpoint}${stats}${extra}" ]]; do
  LINE_NUMBER=$((LINE_NUMBER + 1))
  window="${window%$'\r'}"
  checkpoint="${checkpoint%$'\r'}"
  stats="${stats%$'\r'}"
  extra="${extra%$'\r'}"

  [[ -z "$window" || "$window" == \#* ]] && continue
  [[ "$window" =~ ^[1-8]$ ]] || fail "manifest line $LINE_NUMBER has invalid window size: $window"
  [[ -n "$checkpoint" && -n "$stats" && -z "$extra" ]] || \
    fail "manifest line $LINE_NUMBER must contain exactly three tab-separated fields"
  [[ -z "${CHECKPOINTS[$window]+x}" ]] || fail "manifest contains duplicate window size: $window"
  [[ -f "$checkpoint" ]] || fail "checkpoint for W=$window not found: $checkpoint"
  [[ -f "$stats" ]] || fail "dataset stats for W=$window not found: $stats"

  CHECKPOINTS[$window]="$(cd "$(dirname "$checkpoint")" && pwd -P)/$(basename "$checkpoint")"
  DATASET_STATS[$window]="$(cd "$(dirname "$stats")" && pwd -P)/$(basename "$stats")"
  ROW_COUNT=$((ROW_COUNT + 1))
done < "$MANIFEST"

[[ "$ROW_COUNT" -eq 8 ]] || fail "manifest must contain exactly 8 data rows; found $ROW_COUNT"
for window in {1..8}; do
  [[ -n "${CHECKPOINTS[$window]+x}" ]] || fail "manifest is missing window size: $window"
done

RUN_ID="$(date +%Y%m%d_%H%M%S)"
if [[ $# -ge 5 ]]; then
  RESULT_DIR="$5"
else
  RESULT_DIR="$REPO_ROOT/evaluate_results/robotwin_timing/rollingwam_${RUN_ID}"
fi
mkdir -p "$RESULT_DIR/rollingwam"
RESULT_DIR="$(cd "$RESULT_DIR" && pwd -P)"

cd "$REPO_ROOT"

echo "RollingWAM timing sweep: task=$TASK_NAME episodes=$EPISODES seed=$SEED gpu=$GPU_ID"
echo "Schedule: K=floor(16/W); total S=W*K; horizon=16*W"
echo "Results: $RESULT_DIR"

for window in {1..8}; do
  steady_steps=$((16 / window))
  total_steps=$((window * steady_steps))
  action_horizon=$((16 * window))
  checkpoint="${CHECKPOINTS[$window]}"
  stats="${DATASET_STATS[$window]}"
  run_name="w${window}_h${action_horizon}_k${steady_steps}"
  timing_json="$RESULT_DIR/rollingwam/$run_name.json"
  timing_log="$RESULT_DIR/rollingwam/$run_name.log"
  eval_name="timing_rolling_${run_name}_${RUN_ID}"

  command=(
    "$PYTHON_BIN" experiments/robotwin/eval_robotwin_single.py
    "task=$ROLLING_HYDRA_TASK"
    "ckpt=$checkpoint"
    "EVALUATION.dataset_stats_path=$stats"
    "EVALUATION.robotwin_root=$ROBOTWIN_ROOT"
    "EVALUATION.task_name=$TASK_NAME"
    EVALUATION.task_config=demo_clean
    "EVALUATION.eval_num_episodes=$EPISODES"
    "EVALUATION.num_inference_steps=$total_steps"
    EVALUATION.timing_enabled=true
    EVALUATION.skip_get_obs_within_replan=true
    EVALUATION.save_imagined_rollouts=false
    EVALUATION.compile_action_infer=false
    model.vae_encode_batch_size=1
    model.compile_vae_encode=false
    "seed=$SEED"
    "EVALUATION.output_dir=$eval_name"
    "gpu_id=$GPU_ID"
  )

  echo
  echo "[RollingWAM W=$window] horizon=$action_horizon total_steps=$total_steps steady_steps=$steady_steps checkpoint=$checkpoint"
  printf '  '
  printf '%q ' env \
    "DIFFSYNTH_MODEL_BASE_PATH=$MODEL_BASE_PATH" \
    DIFFSYNTH_SKIP_DOWNLOAD=true \
    "WAM_TIMING_RESULT_PATH=$timing_json" \
    "ROLLINGWAM_TIMING_EXPECTED_W=$window" \
    ROLLINGWAM_TIMING_EXPECTED_CHUNK_LATENTS=1 \
    "${command[@]}"
  printf '\n'

  if [[ "$DRY_RUN" == "0" ]]; then
    env \
      "DIFFSYNTH_MODEL_BASE_PATH=$MODEL_BASE_PATH" \
      DIFFSYNTH_SKIP_DOWNLOAD=true \
      "WAM_TIMING_RESULT_PATH=$timing_json" \
      "ROLLINGWAM_TIMING_EXPECTED_W=$window" \
      ROLLINGWAM_TIMING_EXPECTED_CHUNK_LATENTS=1 \
      "${command[@]}" 2>&1 | tee "$timing_log"

    [[ -f "$timing_json" ]] || fail "timing result was not written for W=$window: $timing_json"
  fi
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo
  echo "Dry run complete; no evaluations or summary were written."
  exit 0
fi

"$PYTHON_BIN" - "$RESULT_DIR" "$TASK_NAME" "$EPISODES" "$SEED" <<'PY'
import json
import sys
from pathlib import Path

result_dir = Path(sys.argv[1])
task_name = sys.argv[2]
episodes = int(sys.argv[3])
seed = int(sys.argv[4])
steady_step_schedule = [16, 8, 5, 4, 3, 2, 2, 2]


def require_equal(actual, expected, label):
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def read_result(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


runs = []
schedule = []
for window, steady_steps in enumerate(steady_step_schedule, start=1):
    total_steps = window * steady_steps
    action_horizon = 16 * window
    run_name = f"w{window}_h{action_horizon}_k{steady_steps}"
    result_path = result_dir / "rollingwam" / f"{run_name}.json"
    result = read_result(result_path)

    require_equal(result.get("policy"), "rollingwam", f"W={window} policy")
    require_equal(result.get("model", {}).get("window_blocks"), window, f"W={window} window_blocks")
    require_equal(result.get("model", {}).get("chunk_latents"), 1, f"W={window} chunk_latents")
    require_equal(result.get("model", {}).get("actions_per_chunk"), 16, f"W={window} actions_per_chunk")
    require_equal(result.get("model", {}).get("window_action_horizon"), action_horizon, f"W={window} horizon")
    require_equal(result.get("model", {}).get("emitted_actions_per_replan"), 16, f"W={window} emitted actions")
    require_equal(result.get("model", {}).get("executed_actions_per_replan"), 16, f"W={window} executed actions")
    require_equal(result.get("model", {}).get("num_inference_steps"), total_steps, f"W={window} total steps")
    require_equal(result.get("model", {}).get("initialization_denoising_steps"), total_steps, f"W={window} initialization steps")
    require_equal(result.get("model", {}).get("steady_denoising_steps_per_replan"), steady_steps, f"W={window} steady steps")
    require_equal(result.get("aggregate", {}).get("episodes"), episodes, f"W={window} episodes")

    schedule.append(
        {
            "window_blocks": window,
            "action_horizon": action_horizon,
            "nominal_total_step_budget": 16,
            "effective_total_inference_steps": total_steps,
            "steady_denoising_steps_per_replan": steady_steps,
        }
    )
    runs.append(
        {
            "window_blocks": window,
            "result_file": str(result_path.relative_to(result_dir)),
            "checkpoint": result.get("checkpoint"),
            "model": result.get("model"),
            "aggregate": result.get("aggregate"),
        }
    )

summary = {
    "schema_version": 1,
    "policy": "rollingwam",
    "unit": "ms",
    "task": task_name,
    "requested_episodes_per_window": episodes,
    "seed": seed,
    "executed_actions_per_replan": 16,
    "expected_chunk_latents": 1,
    "schedule_rule": "K=floor(16/W); total S=W*K; steady passes=S/W=K",
    "schedule": schedule,
    "runs": runs,
}
summary_path = result_dir / "summary.json"
temporary_path = summary_path.with_suffix(".json.tmp")
with temporary_path.open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
    handle.write("\n")
temporary_path.replace(summary_path)
print(f"Summary written to: {summary_path}")
PY

echo "RollingWAM timing sweep complete."
