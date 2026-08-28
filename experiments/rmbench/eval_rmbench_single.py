"""RMBench single-task evaluation entrypoint (Hydra).

Features:
- Read `configs/sim_rmbench.yaml`.
- Check or create the symlink:
  `RMBench/policy/rollingwam_policy -> experiments/robotwin/rollingwam_policy`.
- Forward config overrides to the official RMBench entrypoint
  `script/eval_policy.py` and save logs.

Common arguments:
- `ckpt`: path to the WAM checkpoint (required).
- `EVALUATION.task_name`: task name to evaluate (required).
- `gpu_id`: sets `CUDA_VISIBLE_DEVICES`.

Examples:
1) Minimal run
   python experiments/rmbench/eval_rmbench_single.py \
     ckpt=/path/to/ckpt.pt \
     EVALUATION.task_name=observe_and_pickup

2) Run with more evaluation overrides
   python experiments/rmbench/eval_rmbench_single.py \
     ckpt=/path/to/ckpt.pt \
     EVALUATION.task_name=observe_and_pickup \
     EVALUATION.eval_num_episodes=1 \
     gpu_id=0
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_NAME = "rollingwam_policy"


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _resolve_optional_path(path_value: Any, *, base: Path) -> Path | None:
    if path_value is None:
        return None
    text = str(path_value).strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    return _resolve_path(text, base=base)


def _resolve_dataset_stats_path(cfg: DictConfig, ckpt_path: Path) -> Path:
    explicit = _resolve_optional_path(cfg.EVALUATION.dataset_stats_path, base=PROJECT_ROOT)
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)

    for parent in list(ckpt_path.parents)[:4]:
        candidates.append((parent / "dataset_stats.json").resolve())

    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    raise FileNotFoundError(
        "Failed to locate dataset_stats.json. Tried explicit "
        "EVALUATION.dataset_stats_path and checkpoint parent directories. "
        "Please pass EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )


def _resolve_ckpt_tag(ckpt_path: Path) -> str:
    parts = ckpt_path.resolve().parts
    if "runs" in parts:
        runs_idx = parts.index("runs")
        if runs_idx + 2 >= len(parts):
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        task_name = parts[runs_idx + 1]
        date_dir = parts[runs_idx + 2]
        if task_name == "" or date_dir == "":
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        return f"{task_name}_{date_dir}"
    return ckpt_path.stem


def _ensure_policy_symlink(rmbench_root: Path, policy_source_dir: Path) -> Path:
    policy_root = rmbench_root / "policy"
    if not policy_root.is_dir():
        raise FileNotFoundError(f"RMBench policy directory not found: {policy_root}")

    policy_target = policy_root / POLICY_NAME
    source_resolved = policy_source_dir.resolve()

    if not policy_target.exists() and not policy_target.is_symlink():
        try:
            policy_target.symlink_to(source_resolved, target_is_directory=True)
            return policy_target
        except FileExistsError:
            # Parallel manager workers can race to create the same link. The
            # validation below decides whether the winner created the right one.
            pass

    if policy_target.is_symlink():
        target_resolved = policy_target.resolve()
        if target_resolved != source_resolved:
            raise RuntimeError(
                f"Policy symlink conflict: {policy_target} -> {target_resolved}, "
                f"expected -> {source_resolved}"
            )
        return policy_target

    raise RuntimeError(
        f"Path already exists and is not a symlink: {policy_target}. "
        "Please handle it manually to avoid overriding existing policy files."
    )


def _format_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, (int, float)):
        return str(value)
    return repr(str(value))


def _append_override(overrides: list[str], key: str, value: Any, *, skip_none: bool = True) -> None:
    if skip_none and value is None:
        return
    overrides.extend([f"--{key}", _format_override_value(value)])


def _validate_rmbench_root(rmbench_root: Path, task_name: str) -> None:
    required = [
        rmbench_root / "script" / "eval_policy.py",
        rmbench_root / "script" / "test_render.py",
        rmbench_root / "task_config" / "demo_clean.yml",
        rmbench_root / "description" / "task_instruction" / f"{task_name}.json",
        rmbench_root / "envs" / f"{task_name}.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        formatted = "\n  - ".join(missing)
        raise FileNotFoundError(
            "RMBench is incomplete or the task name is invalid. Missing:\n"
            f"  - {formatted}"
        )

    assets = [
        rmbench_root / "assets" / "embodiments" / "aloha-agilex" / "config.yml",
        rmbench_root / "assets" / "objects",
    ]
    missing_assets = [str(path) for path in assets if not path.exists()]
    if missing_assets:
        formatted = "\n  - ".join(missing_assets)
        raise FileNotFoundError(
            "RMBench assets are missing. Run `bash script/_download_assets.sh` "
            f"from {rmbench_root}. Missing:\n  - {formatted}"
        )


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_rmbench.yaml")
def main(cfg: DictConfig):
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must not be None.")
    if cfg.EVALUATION.task_name is None:
        raise ValueError("`EVALUATION.task_name` must not be None.")
    if str(cfg.EVALUATION.task_config) != "demo_clean":
        raise ValueError("RMBench evaluation must use EVALUATION.task_config=demo_clean.")
    if str(cfg.EVALUATION.instruction_type) not in {"seen", "unseen"}:
        raise ValueError("EVALUATION.instruction_type must be 'seen' or 'unseen'.")
    if str(cfg.EVALUATION.policy_name) != POLICY_NAME:
        raise ValueError(f"EVALUATION.policy_name must be {POLICY_NAME!r}.")

    ckpt_path = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt_tag = _resolve_ckpt_tag(ckpt_path)

    rmbench_root = _resolve_path(str(cfg.EVALUATION.rmbench_root), base=PROJECT_ROOT)
    if not rmbench_root.exists():
        raise FileNotFoundError(f"RMBench root not found: {rmbench_root}")
    _validate_rmbench_root(rmbench_root, str(cfg.EVALUATION.task_name))

    policy_source_dir = (PROJECT_ROOT / "experiments" / "robotwin" / POLICY_NAME).resolve()
    if not policy_source_dir.is_dir():
        raise FileNotFoundError(f"Policy source directory not found: {policy_source_dir}")

    _ensure_policy_symlink(rmbench_root=rmbench_root, policy_source_dir=policy_source_dir)

    output_dir = _resolve_path(str(cfg.EVALUATION.output_dir), base=PROJECT_ROOT)
    run_ts = output_dir.name
    if run_ts == "":
        raise ValueError(f"Invalid EVALUATION.output_dir (missing run_ts): {output_dir}")
    run_output_dir = (
        PROJECT_ROOT
        / "evaluate_results"
        / "rmbench"
        / ckpt_tag
        / run_ts
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_output_dir / (
        f"eval_{str(cfg.EVALUATION.task_name)}_{str(cfg.EVALUATION.instruction_type)}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    rmbench_eval_base = (
        PROJECT_ROOT
        / "evaluate_results"
        / "rmbench"
        / ckpt_tag
        / run_ts
        / str(cfg.EVALUATION.task_name)
        / str(cfg.EVALUATION.instruction_type)
    )

    sim_cfg_path = (PROJECT_ROOT / "configs" / "sim_rmbench.yaml").resolve()
    sim_task = HydraConfig.get().runtime.choices.get("task")

    dataset_stats_path = _resolve_dataset_stats_path(cfg, ckpt_path)

    overrides: list[str] = []
    _append_override(overrides, "task_name", cfg.EVALUATION.task_name)
    _append_override(overrides, "task_config", cfg.EVALUATION.task_config)
    _append_override(overrides, "ckpt_setting", str(ckpt_path))
    _append_override(overrides, "seed", cfg.seed)
    _append_override(overrides, "policy_name", cfg.EVALUATION.policy_name)
    _append_override(overrides, "instruction_type", cfg.EVALUATION.instruction_type)
    _append_override(overrides, "eval_num_episodes", cfg.EVALUATION.eval_num_episodes)

    _append_override(overrides, "sim_cfg_path", str(sim_cfg_path))
    _append_override(overrides, "sim_task", sim_task)
    _append_override(overrides, "eval_output_dir", str(rmbench_eval_base))
    _append_override(overrides, "mixed_precision", cfg.mixed_precision)
    _append_override(overrides, "device", cfg.EVALUATION.device)
    _append_override(overrides, "dataset_stats_path", str(dataset_stats_path))
    _append_override(overrides, "num_inference_steps", cfg.EVALUATION.num_inference_steps)
    _append_override(overrides, "compile_action_infer", cfg.EVALUATION.get("compile_action_infer", False))
    _append_override(overrides, "vae_encode_batch_size", cfg.model.get("vae_encode_batch_size", 1))
    _append_override(overrides, "compile_vae_encode", cfg.model.get("compile_vae_encode", False))
    _append_override(overrides, "text_cfg_scale", cfg.EVALUATION.text_cfg_scale)
    _append_override(overrides, "negative_prompt", cfg.EVALUATION.negative_prompt)
    _append_override(overrides, "timing_enabled", cfg.EVALUATION.timing_enabled)
    _append_override(
        overrides,
        "skip_get_obs_within_replan",
        cfg.EVALUATION.skip_get_obs_within_replan,
    )
    _append_override(
        overrides,
        "save_imagined_rollouts",
        cfg.EVALUATION.save_imagined_rollouts,
    )

    cmd = [
        sys.executable,
        "-u",
        "script/eval_policy.py",
        "--config",
        f"policy/{POLICY_NAME}/deploy_policy.yml",
        "--overrides",
        *overrides,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    env["PYTHONUNBUFFERED"] = "1"

    with open(log_file, "w", encoding="utf-8") as log_f:
        process = subprocess.Popen(
            cmd,
            cwd=str(rmbench_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
            log_f.flush()
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"RMBench evaluation failed with return code {return_code}. Log: {log_file}"
        )

    print(f"Evaluation finished successfully. Log saved to: {log_file}")
    OmegaConf.save(
        config=cfg,
        f=str(
            run_output_dir
            / (
                f"eval_config_{str(cfg.EVALUATION.task_name)}_"
                f"{str(cfg.EVALUATION.instruction_type)}.yaml"
            )
        ),
    )


if __name__ == "__main__":
    main()
