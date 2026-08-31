#!/usr/bin/env bash
# Run RollingWAM, FastWAM, and a FastWAM-Joint timing proxy for W=1 through 8.
# The joint path reuses the FastWAM-trained checkpoint; it is not an accuracy evaluation.
#
# Manifest format (tab-separated, one row per window size):
#   <window_blocks>\t<checkpoint_path>\t<dataset_stats_path>
#
# Usage:
#   bash scripts/robotwin/eval_window_timing_ablation.sh \
#     <gpu_id> <manifest.tsv> [task_name] [episodes] [result_dir]
#
# Required environment:
#   FASTWAM_CKPT=/path/to/fastwam.pt  (reused by FastWAM and the joint proxy)
#
# Optional environment:
#   FASTWAM_ROOT=/workspace/FastWAM
#   FASTWAM_DATASET_STATS=/path/to/dataset_stats.json
#   DRY_RUN=1  Validate inputs and print commands without evaluating.

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
ROLLING_TIMING_INIT="$REPO_ROOT/experiments/robotwin/rollingwam_policy/__init__.py"

if [[ -n "${FASTWAM_ROOT:-}" ]]; then
  FASTWAM_ROOT_CANDIDATE="$FASTWAM_ROOT"
elif [[ -d /workspace/FastWAM ]]; then
  FASTWAM_ROOT_CANDIDATE=/workspace/FastWAM
else
  FASTWAM_ROOT_CANDIDATE="$REPO_ROOT/../ref/FastWAM"
fi
[[ -d "$FASTWAM_ROOT_CANDIDATE" ]] || fail "FastWAM root not found: $FASTWAM_ROOT_CANDIDATE"
FASTWAM_ROOT="$(cd "$FASTWAM_ROOT_CANDIDATE" && pwd -P)"
FASTWAM_TIMING_INIT="$FASTWAM_ROOT/experiments/robotwin/fastwam_policy/__init__.py"

if ! grep -Eq '^[[:space:]]*from[[:space:]]+\.deploy_policy_timing[[:space:]]+import[[:space:]]+\*[[:space:]]*$' "$ROLLING_TIMING_INIT"; then
  fail "timing policy is not wired. Set experiments/robotwin/rollingwam_policy/__init__.py to: from .deploy_policy_timing import *"
fi
if ! grep -Eq '^[[:space:]]*from[[:space:]]+\.deploy_policy_timing[[:space:]]+import[[:space:]]+\*[[:space:]]*$' "$FASTWAM_TIMING_INIT"; then
  fail "FastWAM timing policy is not wired. Set $FASTWAM_TIMING_INIT to: from .deploy_policy_timing import *"
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
if [[ $# -ge 5 ]]; then
  RESULT_DIR="$5"
else
  RESULT_DIR="$REPO_ROOT/evaluate_results/robotwin_timing/$RUN_ID"
fi
mkdir -p "$RESULT_DIR"
RESULT_DIR="$(cd "$RESULT_DIR" && pwd -P)"

ROLLING_HYDRA_TASK="${ROLLING_HYDRA_TASK:-${HYDRA_TASK:-robotwin_selected_tasks_rolling_3cam_384_1e-4}}"
FASTWAM_HYDRA_TASK="${FASTWAM_HYDRA_TASK:-robotwin_selected_tasks_uncond_3cam_384_1e-4}"
FASTWAM_JOINT_HYDRA_TASK="${FASTWAM_JOINT_HYDRA_TASK:-robotwin_joint_3cam_384_1e-4}"
ROLLING_ROBOTWIN_ROOT="${ROLLING_ROBOTWIN_ROOT:-${ROBOTWIN_ROOT:-$REPO_ROOT/third_party/RoboTwin}}"
FASTWAM_ROBOTWIN_ROOT="${FASTWAM_ROBOTWIN_ROOT:-$ROLLING_ROBOTWIN_ROOT}"
ROLLING_MODEL_BASE_PATH="${ROLLING_MODEL_BASE_PATH:-${DIFFSYNTH_MODEL_BASE_PATH:-$REPO_ROOT/checkpoints}}"
FASTWAM_MODEL_BASE_PATH="${FASTWAM_MODEL_BASE_PATH:-$FASTWAM_ROOT/checkpoints}"
PYTHON_BIN="${PYTHON_BIN:-python}"
FASTWAM_PYTHON_BIN="${FASTWAM_PYTHON_BIN:-$PYTHON_BIN}"

[[ -d "$ROLLING_ROBOTWIN_ROOT" ]] || fail "RollingWAM RoboTwin root not found: $ROLLING_ROBOTWIN_ROOT"
[[ -d "$FASTWAM_ROBOTWIN_ROOT" ]] || fail "FastWAM RoboTwin root not found: $FASTWAM_ROBOTWIN_ROOT"
[[ -d "$ROLLING_MODEL_BASE_PATH" ]] || fail "RollingWAM model base path not found: $ROLLING_MODEL_BASE_PATH"
[[ -d "$FASTWAM_MODEL_BASE_PATH" ]] || fail "FastWAM model base path not found: $FASTWAM_MODEL_BASE_PATH"

FASTWAM_CKPT="${FASTWAM_CKPT:-}"
[[ -n "$FASTWAM_CKPT" ]] || fail "FASTWAM_CKPT must point to the FastWAM checkpoint used by both FastWAM variants"
[[ -f "$FASTWAM_CKPT" ]] || fail "FastWAM checkpoint not found: $FASTWAM_CKPT"
FASTWAM_CKPT="$(cd "$(dirname "$FASTWAM_CKPT")" && pwd -P)/$(basename "$FASTWAM_CKPT")"

find_dataset_stats() {
  local directory
  local depth
  directory="$(dirname "$1")"
  for depth in {1..6}; do
    if [[ -f "$directory/dataset_stats.json" ]]; then
      printf '%s\n' "$directory/dataset_stats.json"
      return 0
    fi
    [[ "$directory" != / ]] || break
    directory="$(dirname "$directory")"
  done
  return 1
}

if [[ -n "${FASTWAM_DATASET_STATS:-}" ]]; then
  [[ -f "$FASTWAM_DATASET_STATS" ]] || fail "FastWAM dataset stats not found: $FASTWAM_DATASET_STATS"
  FASTWAM_DATASET_STATS="$(cd "$(dirname "$FASTWAM_DATASET_STATS")" && pwd -P)/$(basename "$FASTWAM_DATASET_STATS")"
else
  FASTWAM_DATASET_STATS="$(find_dataset_stats "$FASTWAM_CKPT")" || \
    fail "could not find dataset_stats.json above FASTWAM_CKPT; set FASTWAM_DATASET_STATS"
fi

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

cd "$REPO_ROOT"

mkdir -p "$RESULT_DIR/rollingwam" "$RESULT_DIR/fastwam" "$RESULT_DIR/fastwam_joint"

echo "Three-way timing sweep: task=$TASK_NAME episodes=$EPISODES seed=$SEED gpu=$GPU_ID"
echo "Schedule: K=floor(16/W); Rolling total S=W*K; both FastWAM variants use K; action_horizon=16*W; joint_video_frames=4*W+1"
echo "Results: $RESULT_DIR"

for window in {1..8}; do
  steady_steps=$((16 / window))
  rolling_inference_steps=$((window * steady_steps))
  fastwam_inference_steps=$steady_steps
  action_horizon=$((16 * window))
  num_video_frames=$((4 * window + 1))
  checkpoint="${CHECKPOINTS[$window]}"
  stats="${DATASET_STATS[$window]}"
  run_name="w${window}_h${action_horizon}_k${steady_steps}"
  rolling_json="$RESULT_DIR/rollingwam/$run_name.json"
  rolling_log="$RESULT_DIR/rollingwam/$run_name.log"
  fastwam_json="$RESULT_DIR/fastwam/$run_name.json"
  fastwam_log="$RESULT_DIR/fastwam/$run_name.log"
  fastwam_joint_json="$RESULT_DIR/fastwam_joint/$run_name.json"
  fastwam_joint_log="$RESULT_DIR/fastwam_joint/$run_name.log"
  rolling_eval_name="timing_rolling_${run_name}_${RUN_ID}"
  fastwam_eval_name="timing_fastwam_${run_name}_${RUN_ID}"
  fastwam_joint_eval_name="timing_fastwam_joint_${run_name}_${RUN_ID}"

  rolling_command=(
    "$PYTHON_BIN" experiments/robotwin/eval_robotwin_single.py
    "task=$ROLLING_HYDRA_TASK"
    "ckpt=$checkpoint"
    "EVALUATION.dataset_stats_path=$stats"
    "EVALUATION.robotwin_root=$ROLLING_ROBOTWIN_ROOT"
    "EVALUATION.task_name=$TASK_NAME"
    EVALUATION.task_config=demo_clean
    "EVALUATION.eval_num_episodes=$EPISODES"
    "EVALUATION.num_inference_steps=$rolling_inference_steps"
    EVALUATION.timing_enabled=true
    EVALUATION.skip_get_obs_within_replan=true
    EVALUATION.save_imagined_rollouts=false
    EVALUATION.compile_action_infer=false
    model.vae_encode_batch_size=1
    model.compile_vae_encode=false
    "seed=$SEED"
    "EVALUATION.output_dir=$rolling_eval_name"
    "gpu_id=$GPU_ID"
  )

  fastwam_command=(
    "$FASTWAM_PYTHON_BIN" experiments/robotwin/eval_robotwin_single.py
    "task=$FASTWAM_HYDRA_TASK"
    "ckpt=$FASTWAM_CKPT"
    "EVALUATION.dataset_stats_path=$FASTWAM_DATASET_STATS"
    "EVALUATION.robotwin_root=$FASTWAM_ROBOTWIN_ROOT"
    "EVALUATION.task_name=$TASK_NAME"
    EVALUATION.task_config=demo_clean
    "EVALUATION.eval_num_episodes=$EPISODES"
    "EVALUATION.action_horizon=$action_horizon"
    EVALUATION.replan_steps=16
    "EVALUATION.num_inference_steps=$fastwam_inference_steps"
    EVALUATION.timing_enabled=true
    EVALUATION.skip_get_obs_within_replan=true
    "seed=$SEED"
    "EVALUATION.output_dir=$fastwam_eval_name"
    "gpu_id=$GPU_ID"
  )

  fastwam_joint_command=(
    "$FASTWAM_PYTHON_BIN" experiments/robotwin/eval_robotwin_single.py
    "task=$FASTWAM_JOINT_HYDRA_TASK"
    "ckpt=$FASTWAM_CKPT"
    "EVALUATION.dataset_stats_path=$FASTWAM_DATASET_STATS"
    "EVALUATION.robotwin_root=$FASTWAM_ROBOTWIN_ROOT"
    "EVALUATION.task_name=$TASK_NAME"
    EVALUATION.task_config=demo_clean
    "EVALUATION.eval_num_episodes=$EPISODES"
    "EVALUATION.action_horizon=$action_horizon"
    "EVALUATION.num_video_frames=$num_video_frames"
    EVALUATION.replan_steps=16
    "EVALUATION.num_inference_steps=$fastwam_inference_steps"
    EVALUATION.timing_enabled=true
    EVALUATION.skip_get_obs_within_replan=true
    "seed=$SEED"
    "EVALUATION.output_dir=$fastwam_joint_eval_name"
    "gpu_id=$GPU_ID"
  )

  echo
  echo "[RollingWAM W=$window] horizon=$action_horizon total_steps=$rolling_inference_steps steady_steps=$steady_steps checkpoint=$checkpoint"
  printf '  (cd %q && ' "$REPO_ROOT"
  printf '%q ' env \
    "DIFFSYNTH_MODEL_BASE_PATH=$ROLLING_MODEL_BASE_PATH" \
    DIFFSYNTH_SKIP_DOWNLOAD=true \
    "WAM_TIMING_RESULT_PATH=$rolling_json" \
    "ROLLINGWAM_TIMING_EXPECTED_W=$window" \
    ROLLINGWAM_TIMING_EXPECTED_CHUNK_LATENTS=1 \
    "${rolling_command[@]}"
  printf ')\n'

  if [[ "$DRY_RUN" == "0" ]]; then
    (
      cd "$REPO_ROOT"
      env \
        "DIFFSYNTH_MODEL_BASE_PATH=$ROLLING_MODEL_BASE_PATH" \
        DIFFSYNTH_SKIP_DOWNLOAD=true \
        "WAM_TIMING_RESULT_PATH=$rolling_json" \
        "ROLLINGWAM_TIMING_EXPECTED_W=$window" \
        ROLLINGWAM_TIMING_EXPECTED_CHUNK_LATENTS=1 \
        "${rolling_command[@]}"
    ) 2>&1 | tee "$rolling_log"

    [[ -f "$rolling_json" ]] || fail "RollingWAM timing result was not written for W=$window: $rolling_json"
  fi

  echo
  echo "[FastWAM equivalent W=$window] horizon=$action_horizon steps=$fastwam_inference_steps checkpoint=$FASTWAM_CKPT"
  printf '  (cd %q && ' "$FASTWAM_ROOT"
  printf '%q ' env \
    "DIFFSYNTH_MODEL_BASE_PATH=$FASTWAM_MODEL_BASE_PATH" \
    DIFFSYNTH_SKIP_DOWNLOAD=true \
    "WAM_TIMING_RESULT_PATH=$fastwam_json" \
    FASTWAM_TIMING_EXPECTED_MODEL_CLASS=FastWAM \
    "${fastwam_command[@]}"
  printf ')\n'

  if [[ "$DRY_RUN" == "0" ]]; then
    (
      cd "$FASTWAM_ROOT"
      env \
        "DIFFSYNTH_MODEL_BASE_PATH=$FASTWAM_MODEL_BASE_PATH" \
        DIFFSYNTH_SKIP_DOWNLOAD=true \
        "WAM_TIMING_RESULT_PATH=$fastwam_json" \
        FASTWAM_TIMING_EXPECTED_MODEL_CLASS=FastWAM \
        "${fastwam_command[@]}"
    ) 2>&1 | tee "$fastwam_log"

    [[ -f "$fastwam_json" ]] || fail "FastWAM timing result was not written for W=$window: $fastwam_json"
  fi

  echo
  echo "[FastWAM-Joint proxy W=$window] horizon=$action_horizon video_frames=$num_video_frames steps=$fastwam_inference_steps checkpoint=$FASTWAM_CKPT"
  printf '  (cd %q && ' "$FASTWAM_ROOT"
  printf '%q ' env \
    "DIFFSYNTH_MODEL_BASE_PATH=$FASTWAM_MODEL_BASE_PATH" \
    DIFFSYNTH_SKIP_DOWNLOAD=true \
    "WAM_TIMING_RESULT_PATH=$fastwam_joint_json" \
    FASTWAM_TIMING_EXPECTED_MODEL_CLASS=FastWAMJoint \
    "${fastwam_joint_command[@]}"
  printf ')\n'

  if [[ "$DRY_RUN" == "0" ]]; then
    (
      cd "$FASTWAM_ROOT"
      env \
        "DIFFSYNTH_MODEL_BASE_PATH=$FASTWAM_MODEL_BASE_PATH" \
        DIFFSYNTH_SKIP_DOWNLOAD=true \
        "WAM_TIMING_RESULT_PATH=$fastwam_joint_json" \
        FASTWAM_TIMING_EXPECTED_MODEL_CLASS=FastWAMJoint \
        "${fastwam_joint_command[@]}"
    ) 2>&1 | tee "$fastwam_joint_log"

    [[ -f "$fastwam_joint_json" ]] || fail "FastWAM-Joint timing result was not written for W=$window: $fastwam_joint_json"
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


def optional_mean(result):
    value = result.get("aggregate", {}).get("steady_state", {}).get("mean_ms")
    return None if value is None else float(value)


def ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


runs = []
schedule = []
for window, steady_steps in enumerate(steady_step_schedule, start=1):
    rolling_total_steps = window * steady_steps
    action_horizon = 16 * window
    num_video_frames = 4 * window + 1
    run_name = f"w{window}_h{action_horizon}_k{steady_steps}"
    rolling_path = result_dir / "rollingwam" / f"{run_name}.json"
    fastwam_path = result_dir / "fastwam" / f"{run_name}.json"
    fastwam_joint_path = result_dir / "fastwam_joint" / f"{run_name}.json"
    rolling = read_result(rolling_path)
    fastwam = read_result(fastwam_path)
    fastwam_joint = read_result(fastwam_joint_path)

    require_equal(rolling.get("policy"), "rollingwam", f"W={window} RollingWAM policy")
    require_equal(rolling.get("model", {}).get("window_blocks"), window, f"W={window} RollingWAM window_blocks")
    require_equal(rolling.get("model", {}).get("chunk_latents"), 1, f"W={window} RollingWAM chunk_latents")
    require_equal(rolling.get("model", {}).get("actions_per_chunk"), 16, f"W={window} RollingWAM actions_per_chunk")
    require_equal(rolling.get("model", {}).get("window_action_horizon"), action_horizon, f"W={window} RollingWAM horizon")
    require_equal(rolling.get("model", {}).get("executed_actions_per_replan"), 16, f"W={window} RollingWAM executed actions")
    require_equal(rolling.get("model", {}).get("num_inference_steps"), rolling_total_steps, f"W={window} RollingWAM total steps")
    require_equal(rolling.get("model", {}).get("steady_denoising_steps_per_replan"), steady_steps, f"W={window} RollingWAM steady steps")

    require_equal(fastwam.get("policy"), "fastwam", f"W={window} FastWAM policy")
    require_equal(fastwam.get("model", {}).get("class"), "FastWAM", f"W={window} FastWAM class")
    require_equal(fastwam.get("model", {}).get("action_horizon"), action_horizon, f"W={window} FastWAM horizon")
    require_equal(fastwam.get("model", {}).get("executed_actions_per_replan"), 16, f"W={window} FastWAM executed actions")
    require_equal(fastwam.get("model", {}).get("num_inference_steps"), steady_steps, f"W={window} FastWAM inference steps")
    require_equal(fastwam.get("model", {}).get("denoising_steps_per_replan"), steady_steps, f"W={window} FastWAM denoising steps")

    require_equal(fastwam_joint.get("policy"), "fastwam", f"W={window} FastWAM-Joint policy")
    require_equal(fastwam_joint.get("model", {}).get("class"), "FastWAMJoint", f"W={window} FastWAM-Joint class")
    require_equal(fastwam_joint.get("model", {}).get("action_horizon"), action_horizon, f"W={window} FastWAM-Joint horizon")
    require_equal(fastwam_joint.get("model", {}).get("num_video_frames"), num_video_frames, f"W={window} FastWAM-Joint video frames")
    require_equal(fastwam_joint.get("model", {}).get("executed_actions_per_replan"), 16, f"W={window} FastWAM-Joint executed actions")
    require_equal(fastwam_joint.get("model", {}).get("num_inference_steps"), steady_steps, f"W={window} FastWAM-Joint inference steps")
    require_equal(fastwam_joint.get("model", {}).get("denoising_steps_per_replan"), steady_steps, f"W={window} FastWAM-Joint denoising steps")
    require_equal(fastwam_joint.get("checkpoint"), fastwam.get("checkpoint"), f"W={window} shared FastWAM checkpoint")

    require_equal(rolling.get("aggregate", {}).get("episodes"), episodes, f"W={window} RollingWAM episodes")
    require_equal(fastwam.get("aggregate", {}).get("episodes"), episodes, f"W={window} FastWAM episodes")
    require_equal(fastwam_joint.get("aggregate", {}).get("episodes"), episodes, f"W={window} FastWAM-Joint episodes")

    rolling_mean = optional_mean(rolling)
    fastwam_mean = optional_mean(fastwam)
    fastwam_joint_mean = optional_mean(fastwam_joint)

    schedule.append(
        {
            "window_blocks": window,
            "equivalent_action_horizon": action_horizon,
            "fastwam_joint_num_video_frames": num_video_frames,
            "nominal_rolling_total_step_budget": 16,
            "rolling_total_inference_steps": rolling_total_steps,
            "denoising_steps_per_steady_replan": steady_steps,
            "fastwam_inference_steps": steady_steps,
            "fastwam_joint_inference_steps": steady_steps,
        }
    )
    runs.append(
        {
            "window_blocks": window,
            "equivalent_action_horizon": action_horizon,
            "denoising_steps_per_steady_replan": steady_steps,
            "rollingwam": {
                "result_file": str(rolling_path.relative_to(result_dir)),
                "model": rolling.get("model"),
                "aggregate": rolling.get("aggregate"),
            },
            "fastwam": {
                "result_file": str(fastwam_path.relative_to(result_dir)),
                "model": fastwam.get("model"),
                "aggregate": fastwam.get("aggregate"),
            },
            "fastwam_joint": {
                "result_file": str(fastwam_joint_path.relative_to(result_dir)),
                "model": fastwam_joint.get("model"),
                "aggregate": fastwam_joint.get("aggregate"),
                "weights": "FastWAM-trained checkpoint; joint inference timing proxy only",
            },
            "steady_state_comparison": {
                "rollingwam_mean_ms": rolling_mean,
                "fastwam_mean_ms": fastwam_mean,
                "fastwam_joint_mean_ms": fastwam_joint_mean,
                "rollingwam_speedup_vs_fastwam": ratio(fastwam_mean, rolling_mean),
                "rollingwam_speedup_vs_fastwam_joint": ratio(fastwam_joint_mean, rolling_mean),
                "fastwam_joint_slowdown_vs_fastwam": ratio(fastwam_joint_mean, fastwam_mean),
                "rollingwam_steady_ms_per_denoiser_call": (
                    None if rolling_mean is None else rolling_mean / steady_steps
                ),
                "fastwam_steady_ms_per_denoiser_call": (
                    None if fastwam_mean is None else fastwam_mean / steady_steps
                ),
                "fastwam_joint_steady_ms_per_denoiser_call": (
                    None if fastwam_joint_mean is None else fastwam_joint_mean / steady_steps
                ),
            },
        }
    )

summary = {
    "schema_version": 1,
    "comparison": "rollingwam_vs_fastwam_vs_fastwam_joint_proxy",
    "unit": "ms",
    "task": task_name,
    "requested_episodes_per_window": episodes,
    "seed": seed,
    "executed_actions_per_replan": 16,
    "expected_chunk_latents": 1,
    "schedule_rule": "K=floor(16/W); RollingWAM total S=W*K and steady S/W=K; FastWAM variants use S=K",
    "fastwam_joint_weights": "FastWAM-trained checkpoint; timing proxy only",
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

echo "Timing sweep complete."
