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
| `robotwin_selected` | 6-task window-size ablation only; needs an in-repo task index (below) | `robotwin_selected_tasks_rolling_3cam_384_1e-4` |
| `robotwin_smoke` | pre-flight check, 4 steps on the full-RoboTwin task | `robotwin_rolling_3cam_384_1e-4` |
| `libero` | all four LIBERO suites together; needs two uploads (below) | `libero_rolling_2cam224_1e-4` |

## 0. Dry run (no AWS calls, nothing submitted)

```bash
DRY_RUN=1 bash sagemaker/run_sm.sh robotwin_full 1 test
```

Assembles the channels, overrides and estimator and prints them without
submitting. It never triggers a build. Use it after any change to `sagemaker/`.

## 1. Build + push the image (needs docker + ECR access)

```bash
python sagemaker/launch_sm.py --config robotwin_full --build-only
```

The image is target-independent, so any `--config` builds the same one. Re-run
only when code or dependencies change; layer caching makes incremental builds
fast. Submitting without `SKIP_BUILD=1` also does this automatically.

⚠️ **Run this once before any `SKIP_BUILD=1` submit.** FastWAM's image lives in
a different ECR repo (`fastwam`), so `rollingwam:latest` does not exist until
this step runs. The ECR repo itself is created automatically. A `SKIP_BUILD=1`
submit beforehand queues, then fails at image pull.

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

# Selected-tasks W-ablation: one W per job, 1 node = 8 GPUs. The task yaml's
# batch 8 x accum 2 x 8 = effective 128, so only W and S need overriding:
SKIP_BUILD=1 bash sagemaker/run_sm.sh robotwin_selected 1 w5 \
    model.rolling.window_blocks=5 eval_num_inference_steps=15

# On MORE than one node you must add ~data.val (on-the-fly stats are not
# readable across hosts) and rescale the batch to keep effective 128:
SKIP_BUILD=1 bash sagemaker/run_sm.sh robotwin_selected 4 w5 \
    model.rolling.window_blocks=5 batch_size=4 gradient_accumulation_steps=1 \
    eval_num_inference_steps=15 '~data.val' 
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

### Prerequisite for `robotwin_selected` (one small file)

Embeddings come from junjie's **flat trimmed cache** — the same prefix the
full-RoboTwin targets use, no per-task upload. But selected-tasks mode labels
each episode by intersecting its prompt hash against the set of cache filenames
belonging to each task, so it needs a per-task *view* of that cache.

Rather than uploading real per-task directories (tens of thousands of small S3
objects, walked at startup over a FastFile mount), the labels live in the repo
at `configs/data/robotwin_selected_tasks_text_embeds_cache_index.json`, ride
into the image with the rest of the code, and `entry.py` rebuilds the directory
view inside the container as symlinks into the flat mount — names only, no file
contents, no S3 reads. Generate it once on the training box:

```bash
python3 sagemaker/tools/make_task_index.py --cache-root /datasets/robotwin2.0-fastwam/text_embeds_cache --tasks lift_pot beat_block_hammer place_dual_shoes stack_bowls_two blocks_ranking_size stack_blocks_three
```

It writes to that path by default. Re-run it whenever `selected_task_names`
changes, then rebuild the image so the new index is baked in.

Normalization stats are computed on the fly from the 6 tasks, as on the DGX.
That works on **one node only**: rank 0 broadcasts stats in memory for the
train set, but the val set is handed `<output_dir>/dataset_stats.json` and every
rank reads it from its own local disk — only `algo-1` wrote it, and there is no
shared filesystem. For multi-node, add `~data.val` to the submit command (val
falls back to the train dataset, whose stats are already in memory).

### LIBERO

```bash
SKIP_BUILD=1 bash sagemaker/run_sm.sh libero 1 libero-run
```

`batch_size: 16 × accum 1 × 8 GPUs = 128` effective, straight from the task
yaml. Simpler than the RoboTwin targets because `configs/data/libero_2cam.yaml`
defines no `val:` block: `build_datasets` sets `val_ds = train_ds`, so no
`dataset_stats.json` is ever read from disk. Normalization is computed on the
fly and stays in memory — that holds at any node count, so no stats channel and
no `~data.val`. There is also no per-task cache layout (LIBERO uses the flat
one), and training imports nothing from the LIBERO simulator, so the stock
image works — mujoco is only needed for evaluation.

Two uploads first, since FastWAM mirrored no LIBERO data to S3 (exact commands
in the header of `sagemaker/configs/libero.yaml`): the four converted suite
datasets from `/datasets/libero-fastwam/libero_mujoco3.3.2`, and the text-embed cache produced by
`scripts/libero/precompute_libero_text_embeds.sh`.

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
