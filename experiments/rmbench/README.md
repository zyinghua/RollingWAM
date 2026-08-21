# RollingWAM on RMBench

The official RMBench source is located under `third_party/RMBench`; unrelated
policy baselines are omitted. Evaluation reuses
`experiments/robotwin/rollingwam_policy` through a runtime-created policy
symlink.

## 1. Download benchmark assets

From the RollingWAM container:

```bash
cd /workspace/RollingWAM/third_party/RMBench
bash script/_download_assets.sh
```

## 2. Single-task evaluation

Use the `task=` Hydra choice that matches the checkpoint's RollingWAM training
configuration. The `EVALUATION.task_name` value is the RMBench simulator task.

```bash
RMBENCH_EVAL_NUM_EPISODES=1 \
bash scripts/rmbench/eval_single_task_rolling.sh \
  0 observe_and_pickup \
  /path/to/step_xxxxxx.pt \
  /path/to/dataset_stats.json
```

Results are written under
`evaluate_results/rmbench/<checkpoint>/<run>/<task>/<instruction_type>/` and
include both `_result.txt` and machine-readable `result.json`.

## 3. Official nine-task evaluation

```bash
bash scripts/rmbench/eval_all_tasks_rolling.sh \
  0 /path/to/step_xxxxxx.pt /path/to/dataset_stats.json
```

For eight GPUs, one 5B policy worker per GPU:

```bash
RMBENCH_NUM_GPUS=8 \
bash scripts/rmbench/eval_all_tasks_rolling.sh \
  0 /path/to/step_xxxxxx.pt /path/to/dataset_stats.json
```

Useful environment overrides are:

- `ROLLINGWAM_TASK_CONFIG`: checkpoint/model Hydra task choice.
- `RMBENCH_NUM_GPUS`: consecutive GPUs beginning at the first CLI argument.
- `RMBENCH_EVAL_NUM_EPISODES`: episodes per task; default `100`.
- `RMBENCH_INSTRUCTION_TYPE`: `seen` or `unseen`; default `unseen`.
- `RMBENCH_SKIP_GET_OBS_WITHIN_REPLAN`: default `false` for faithful videos.

The manager evaluates the official M(1) and M(n) task lists using only
`demo_clean`, then writes per-task and tier-level success/reward summaries.
