# RollingWAM on SageMaker

Bolt-on SageMaker training, ported from the FastWAM SageMaker setup. Nothing
under `src/`, `configs/` or `scripts/` is modified — SageMaker paths are
injected as Hydra overrides, so `scripts/train.py` runs unchanged. Data and
pretrained weights are expected in S3 (see the `channels:` block in
`configs/<target>.yaml`); the RoboTwin dataset, dataset_stats.json, trimmed
text-embed cache and Wan/ActionDiT weights reuse the FastWAM mirrors already
uploaded under `s3://tri-ml-sandbox-16011-us-west-2-datasets/junjie/...`.

## Targets

Every submission names its target explicitly — nothing is ever implied.

| target | when | task |
|---|---|---|
| `robotwin_full` | **the standard training setup — use this unless you specifically want the ablation** | `robotwin_rolling_3cam_384_1e-4` |
| `robotwin_selected` | 6-task window-size ablation only; needs two extra uploads (below) | `robotwin_selected_tasks_rolling_3cam_384_1e-4` |
| `robotwin_smoke` | pre-flight check, 4 steps on the full-RoboTwin task | `robotwin_rolling_3cam_384_1e-4` |

## 0. Offline checks (no AWS, no torch)

```bash
python3 sagemaker/selftest.py
```

Run after every change to `sagemaker/`. `DRY_RUN=1` on any submit prints the
assembled job without submitting (and never builds).

## 1. Build + push the image (needs docker + ECR access)

```bash
python sagemaker/launch_sm.py --config robotwin_full --build-only
```

The image is target-independent, so any `--config` builds the same one. Re-run
only when code or dependencies change; layer caching makes incremental builds
fast. Submitting without `SKIP_BUILD=1` also does this automatically.

## 2. Smoke test (1 node, ~15 min)

```bash
SKIP_BUILD=1 bash sagemaker/run_sm.sh robotwin_smoke 1 smoke
```

Runs 4 steps + 2 checkpoint saves against the full-RoboTwin task with a pinned
instruction, and exits. Do this before any real job.

## 3. Train

```bash
# usage: run_sm.sh <CONFIG> <INSTANCE_COUNT> <NAME> [hydra overrides...]

# Full RoboTwin, 1 node:
SKIP_BUILD=1 bash sagemaker/run_sm.sh robotwin_full 1 full-run

# Selected-tasks W-ablation: one W per job, 4 nodes = 32 GPUs,
# batch 4 x accum 1 x 32 = effective 128 (matches the DGX plan):
SKIP_BUILD=1 bash sagemaker/run_sm.sh robotwin_selected 4 w5 \
    model.rolling.window_blocks=5 batch_size=4 gradient_accumulation_steps=1 \
    eval_num_inference_steps=15
```

- Overrides after `<NAME>` go verbatim to `scripts/train.py` and win over the
  target YAML's (merged per key by `sagemaker/entry.py`).
- `batch_size` is per GPU; effective = `batch_size × grad_accum × nodes × 8`.
- `eval_num_inference_steps` must be divisible by `model.rolling.window_blocks`
  (the trainer raises otherwise). The ablation pairs: W 1..8 →
  12 12 12 12 15 12 14 16.
- **wandb**: the trainer defaults to `mode: offline` (files land under
  output_dir → S3). For live logging pass `wandb.mode=online` and put
  `WANDB_API_KEY` in `.env` at the repo root — `run_sm.sh` sources it and the
  launcher forwards it into the job.
- Queues: `cv-spot` (p5en), `vla-spot` (p5), `cv-spot-p6` (b200); `--spot` is
  implied for spot queues.
- ⚠️ checkpoint volume: every `save_every` writes weights (~few GB) plus the
  full per-rank ZeRO-2 state under `checkpoints/state/step_*`, never pruned,
  all synced to S3 — set `save_every` deliberately before a long run.

### Prerequisite for `robotwin_selected`

The per-task text-embed caches are NOT on S3 yet. Selected-tasks mode derives
episode→task membership from the per-task layout and reads embeddings from it,
so upload the six task dirs from the training box first (command in the header
of `sagemaker/configs/robotwin_selected.yaml`).

Normalization stats are computed on the fly from the 6 tasks, as on the DGX.
⚠️ That works on **one node only**: rank 0 broadcasts stats in memory for the
train set, but the val set is handed `<output_dir>/dataset_stats.json` and every
rank reads it from its own local disk — only `algo-1` wrote it, and there is no
shared filesystem. For multi-node, add `~data.val` to the submit command (val
falls back to the train dataset, whose stats are already in memory), or
precompute the stats file, upload it, and pass
`data.train/val.pretrained_norm_stats`.

### Spot interruptions / resume

SageMaker restores `/opt/ml/checkpoints` from S3 at job (re)start. The trainer
never auto-resumes on its own, so `entry.py` scans the job's output_dir for
`checkpoints/state/step_*` and injects `resume=<newest>` when found (full
optimizer/scheduler/dataloader state; the `[entry] auto-resume ...` log line
confirms it). A step dir only qualifies if it holds `trainer_state.json` AND
one ZeRO optimizer shard per rank — this skips checkpoints truncated by an
interruption mid-save/mid-sync, and also means resume only engages when the
instance count matches the original run (a mismatched world size is skipped
with a log line and training starts fresh). Disable auto-resume entirely with
`ROLLINGWAM_SM_AUTO_RESUME=false` via `--env`.

⚠️ The `<NAME>` arg becomes part of the job name, and the job name keys the S3
checkpoint prefix — resubmitting with the SAME name reuses that prefix and
therefore RESUMES the earlier run instead of starting fresh. Use a new name per
independent run (e.g. `w5-r2`).

## 4. Monitor

Batch-queue jobs run under a renamed training job (`AWSBatch<name><hash>`):

```bash
aws batch list-service-jobs --job-queue <fss-queue> \
    --filters name=JOB_NAME,values=<job> --profile <profile> --region us-west-2
aws sagemaker describe-training-job --training-job-name <resolved-name> \
    --profile <profile> --region us-west-2   # logs are in CloudWatch
```

Outputs land in `s3://tri-ml-sandbox-16011-us-west-2-datasets/rollingwam/sagemaker/<user>/<job>/`.
Set `TRI_OWNER_EMAIL` (and `SM_USER` if `$USER` isn't your TRI handle) before
submitting — they tag and prefix the job.

## Adding a new target

Copy a YAML in `configs/`, point `channels` at your S3 prefixes and adjust the
`overrides`. No Python changes. Keep episode data on a `File` channel and the
text-embed cache on `FastFile` (rationale in the YAML comments — a FastFile
episode channel stalls ~90 min in LeRobot's per-file metadata storm).
