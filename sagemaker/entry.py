#!/usr/bin/env python
"""SageMaker container entry point (``SAGEMAKER_PROGRAM``).

Runs ONCE PER HOST — the estimator deliberately does not enable SageMaker's
``torch_distributed``. RollingWAM's trainer reads
``accelerator.state.deepspeed_plugin.deepspeed_config`` unconditionally
(src/rollingwam/trainer.py), so the Accelerate DeepSpeed plugin must be active;
that only happens when Accelerate is launched with the repo's ``--config_file``.
So this entry reconstructs the repo's own launch line::

    accelerate launch --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml \
        --num_processes <G*N> --num_machines <N> --machine_rank <R> \
        --main_process_ip algo-1 --main_process_port 29500 \
        scripts/train.py <overrides...>

which is exactly what ``scripts/train_zero2.sh`` does locally, with the
multi-node arguments derived from SageMaker's ``resourceconfig.json``.

Nothing in the RollingWAM package is modified: every SageMaker-specific path is
injected as a Hydra override.

Launcher -> container contract (hyperparameters):
  train_argv    shlex-joined user CLI:   ``task=<name> [hydra overrides ...]``
  sm_overrides  shlex-joined path overrides, with placeholders (see expand())
  sm_env        JSON dict of env vars, values may contain the same placeholders

Auto-resume: SageMaker restores ``/opt/ml/checkpoints`` from ``checkpoint_s3_uri``
at job (re)start — after a spot interruption the previous run's checkpoints are
back on disk, but the trainer never scans for them (``cfg.resume`` defaults to
null). So when the resolved ``output_dir`` already contains
``checkpoints/state/step_*`` dirs and no ``resume=`` override was given, the
newest COMPLETE one is injected as ``resume=<dir>`` (see
``_is_complete_state_dir``). Disable with ``ROLLINGWAM_SM_AUTO_RESUME=false``.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sm_env  # noqa: E402

# The repo root inside the image (code is baked in; see sagemaker/Dockerfile).
REPO_DIR = os.environ.get("ROLLINGWAM_REPO_DIR", sm_env.SM_CODE_DIR)
ACCELERATE_CONFIG = "scripts/accelerate_configs/accelerate_zero2_ds.yaml"
TRAIN_SCRIPT = "scripts/train.py"


def expand(template: str, *, task: str) -> str:
    """Resolve the launcher's placeholders against this container's real paths.

    ``{channel:NAME}``  -> /opt/ml/input/data/NAME   (a mounted input channel)
    ``{checkpoint_dir}``-> /opt/ml/checkpoints        (synced to checkpoint_s3_uri)
    ``{pretrained_dir}``-> /opt/ml/pretrained         (weights pulled via aws s3 cp)
    ``{task}``          -> the Hydra task name
    ``{job}``           -> the SageMaker training job name

    Templates (rather than absolute paths) keep the launcher free of any
    knowledge of SageMaker's in-container layout.
    """
    out = template
    while "{channel:" in out:
        head, _, rest = out.partition("{channel:")
        name, _, tail = rest.partition("}")
        out = head + sm_env.channel_path(name) + tail
    return (
        out.replace("{checkpoint_dir}", sm_env.SM_CHECKPOINT_DIR)
        .replace("{pretrained_dir}", sm_env.SM_PRETRAINED_DIR)
        .replace("{task}", task)
        .replace("{job}", sm_env.job_name())
    )


def override_key(token: str) -> str:
    """Hydra override key, ignoring the ``+`` / ``++`` / ``~`` prefixes.

    ``+data.train.foo=1`` and ``data.train.foo=2`` target the same key; Hydra
    rejects the pair, so they must be deduplicated before launch.
    """
    return token.lstrip("+~").split("=", 1)[0]


def merge_overrides(base: list[str], user: list[str]) -> list[str]:
    """Merge Hydra overrides, letting ``user`` win per key, preserving order."""
    merged: dict[str, str] = {}
    for token in [*base, *user]:
        merged[override_key(token)] = token
    return list(merged.values())


def resolve_task(user_argv: list[str]) -> str:
    """Extract ``task=<name>`` from the user CLI (for {task} in path templates)."""
    for token in user_argv:
        if override_key(token) == "task":
            return token.split("=", 1)[1]
    return "train"


def _is_complete_state_dir(path: Path, world_size: int) -> bool:
    """True when a ``step_*`` dir is a fully saved AND fully synced checkpoint.

    Two failure modes to exclude: (a) an interruption mid-save on a host, and
    (b) a partial S3 sync — each host uploads its own /opt/ml/checkpoints
    independently, so any subset of another host's files can be missing on
    restore. trainer_state.json (written last, main process) covers (a); one
    ZeRO ``zero_pp_rank_*_optim_states.pt`` shard per rank covers (b), and as a
    side effect rejects a checkpoint from a run with a different world size,
    which DeepSpeed could not load anyway.
    """
    if not (path / "trainer_state.json").is_file():
        return False
    shards = list(path.rglob("zero_pp_rank_*_optim_states.pt"))
    return len(shards) == world_size


def find_auto_resume(overrides: list[str], world_size: int) -> str | None:
    """Newest complete state checkpoint under the job's output_dir, if any.

    Returns the ``.../checkpoints/state/step_NNNNNN`` dir with the highest step
    that passes ``_is_complete_state_dir``, or None when auto-resume is
    disabled, a ``resume=`` override is already present, or no complete state
    checkpoint exists (the normal fresh-start case).
    """
    if os.environ.get(f"{sm_env.ENV_PREFIX}_AUTO_RESUME", "true").lower() == "false":
        return None
    output_dir: str | None = None
    for token in overrides:
        key = override_key(token)
        if key == "resume":
            return None  # an explicit resume choice always wins
        if key == "output_dir":
            output_dir = token.split("=", 1)[1]
    if not output_dir:
        return None

    state_root = Path(output_dir) / "checkpoints" / "state"
    if not state_root.is_dir():
        return None
    steps = []
    for path in state_root.iterdir():
        match = re.fullmatch(r"step_(\d+)", path.name)
        if match and path.is_dir():
            steps.append((int(match.group(1)), path))
    for _, path in sorted(steps, reverse=True):
        if _is_complete_state_dir(path, world_size):
            return str(path)
        print(
            f"[entry] skipping incomplete/mismatched checkpoint {path.name} "
            f"(needs trainer_state.json + {world_size} optimizer shards)",
            flush=True,
        )
    return None


def main() -> None:
    if not sm_env.is_sagemaker():
        raise SystemExit(
            "sagemaker/entry.py is the in-container entry point; it found no "
            "SageMaker environment. Run scripts/train.py directly for local training."
        )

    hp = sm_env.read_hyperparameters()
    if "train_argv" not in hp:
        raise SystemExit(
            "Missing the 'train_argv' hyperparameter. launch_sm.py packs the "
            "training CLI via shlex.join(['task=<name>', *overrides])."
        )

    user_argv = shlex.split(hp["train_argv"])
    task = resolve_task(user_argv)

    # Env from the launcher's target config (e.g. DIFFSYNTH_MODEL_BASE_PATH).
    for key, value in json.loads(hp.get("sm_env", "{}")).items():
        os.environ[key] = expand(str(value), task=task)

    # Pretrained weights that are NOT mounted as a channel get pulled from S3.
    pretrained_s3 = os.environ.get(f"{sm_env.ENV_PREFIX}_PRETRAINED_S3")
    if pretrained_s3:
        sm_env.sync_from_s3(pretrained_s3, sm_env.SM_PRETRAINED_DIR)

    sm_overrides = [
        expand(token, task=task) for token in shlex.split(hp.get("sm_overrides", ""))
    ]
    overrides = merge_overrides(sm_overrides, user_argv)

    dist = sm_env.distributed_env()
    resume_dir = find_auto_resume(overrides, dist["nproc_per_node"] * dist["num_machines"])
    if resume_dir:
        overrides.append(f"resume={resume_dir}")
        print(f"[entry] auto-resume from restored checkpoint: {resume_dir}", flush=True)

    cmd = [
        "accelerate", "launch",
        "--config_file", ACCELERATE_CONFIG,
        "--num_processes", str(dist["nproc_per_node"] * dist["num_machines"]),
    ]
    if dist["num_machines"] > 1:
        cmd += [
            "--num_machines", str(dist["num_machines"]),
            "--machine_rank", str(dist["machine_rank"]),
            "--main_process_ip", dist["main_process_ip"],
            "--main_process_port", str(dist["main_process_port"]),
            # `standard` is the launcher proven live on this cluster (FastWAM's
            # multi-node fix); `nossh` hung at rendezvous in practice.
            "--deepspeed_multinode_launcher",
            os.environ.get(f"{sm_env.ENV_PREFIX}_DS_LAUNCHER", "standard"),
        ]
    cmd += [TRAIN_SCRIPT, *overrides]

    os.makedirs(sm_env.SM_CHECKPOINT_DIR, exist_ok=True)
    os.chdir(REPO_DIR)

    # Make the bolt-on subclasses (rollingwam_sm.*) importable by the training
    # processes accelerate spawns, so a target YAML can select one via Hydra's
    # `_target_` without touching src/rollingwam. This directory is added rather
    # than the repo root: a top-level `sagemaker` package would shadow the AWS
    # SageMaker SDK.
    sm_dir = str(Path(__file__).resolve().parent)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [p for p in (sm_dir, os.environ.get("PYTHONPATH", "")) if p]
    )

    print(f"[entry] cwd={REPO_DIR}", flush=True)
    print(f"[entry] channels={sm_env.channel_map()}", flush=True)
    print(f"[entry] dist={dist}", flush=True)
    print(f"[entry] {shlex.join(cmd)}", flush=True)

    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
