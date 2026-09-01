"""Evaluate one DOMINO task with a RollingWAM checkpoint.

This wrapper validates the DOMINO dynamic-task setup, exposes the dedicated
RollingWAM DOMINO policy through DOMINO's policy directory, and delegates to
DOMINO's native evaluation entrypoint. DOMINO therefore remains responsible
for dynamic episode initialization, success checks, and manipulation metrics.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_NAME = "rollingwam_policy"
NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
NATIVE_EVAL_NUM_EPISODES = 100


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _resolve_optional_path(path_value: Any, *, base: Path) -> Path | None:
    if path_value is None:
        return None
    text = str(path_value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return _resolve_path(text, base=base)


def _resolve_dataset_stats_path(cfg: DictConfig, ckpt_path: Path) -> Path:
    explicit = _resolve_optional_path(cfg.EVALUATION.dataset_stats_path, base=PROJECT_ROOT)
    candidates = ([explicit] if explicit is not None else []) + [
        (parent / "dataset_stats.json").resolve()
        for parent in list(ckpt_path.parents)[:4]
    ]

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen and resolved.is_file():
            return resolved
        seen.add(resolved)

    raise FileNotFoundError(
        "Failed to locate dataset_stats.json. Pass "
        "EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )


def _ensure_policy_symlink(domino_root: Path, policy_source_dir: Path) -> Path:
    policy_root = domino_root / "policy"
    if not policy_root.is_dir():
        raise FileNotFoundError(f"DOMINO policy directory not found: {policy_root}")

    policy_target = policy_root / POLICY_NAME
    source_resolved = policy_source_dir.resolve()
    if not policy_target.exists() and not policy_target.is_symlink():
        try:
            policy_target.symlink_to(source_resolved, target_is_directory=True)
            return policy_target
        except FileExistsError:
            pass  # Another evaluator may have created it concurrently.

    if policy_target.is_symlink():
        target_resolved = policy_target.resolve()
        if target_resolved == source_resolved:
            return policy_target

        legacy_source = (
            PROJECT_ROOT / "experiments" / "robotwin" / POLICY_NAME
        ).resolve()
        if target_resolved == legacy_source:
            temporary_link = policy_root / f".{POLICY_NAME}.{os.getpid()}.tmp"
            try:
                temporary_link.symlink_to(source_resolved, target_is_directory=True)
                os.replace(temporary_link, policy_target)
            finally:
                temporary_link.unlink(missing_ok=True)
            return policy_target

        raise RuntimeError(
            f"Policy symlink conflict: {policy_target} -> {target_resolved}, "
            f"expected -> {source_resolved}"
        )
    raise RuntimeError(
        f"Path already exists and is not a symlink: {policy_target}. "
        "Handle it manually so existing policy files are not overwritten."
    )


def _resolve_task_config(cfg: DictConfig, dynamic_level: int) -> str:
    configured = cfg.EVALUATION.task_config
    if configured is None or str(configured).strip().lower() in {"", "none", "null"}:
        return f"demo_clean_dynamic_level{dynamic_level}"
    task_config = str(configured).strip()
    if not NAME_PATTERN.fullmatch(task_config):
        raise ValueError(
            "EVALUATION.task_config must be a config stem such as "
            "'demo_clean_dynamic_level2' (without .yml)."
        )
    return task_config


def _validate_domino_setup(
    domino_root: Path,
    *,
    task_name: str,
    task_config: str,
    dynamic_level: int,
) -> None:
    required_files = [
        domino_root / "script" / "eval_policy.py",
        domino_root / "script" / "test_render.py",
        domino_root / "task_config" / f"{task_config}.yml",
        domino_root / "description" / "task_instruction" / f"{task_name}.json",
        domino_root / "envs" / f"{task_name}.py",
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "DOMINO is incomplete, or the task/config name is invalid. Missing:\n  - "
            + "\n  - ".join(missing)
        )

    config_path = domino_root / "task_config" / f"{task_config}.yml"
    task_args = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(task_args, dict):
        raise ValueError(f"DOMINO task config must contain a mapping: {config_path}")
    if task_args.get("use_dynamic") is not True:
        raise ValueError(f"DOMINO evaluation config must set use_dynamic: true: {config_path}")
    if int(task_args.get("dynamic_level", -1)) != dynamic_level:
        raise ValueError(
            f"EVALUATION.dynamic_level={dynamic_level}, but {config_path.name} sets "
            f"dynamic_level={task_args.get('dynamic_level')!r}."
        )

    camera = task_args.get("camera", {})
    data_type = task_args.get("data_type", {})
    if (
        camera.get("head_camera_type") != "D435"
        or camera.get("wrist_camera_type") != "D435"
        or camera.get("collect_head_camera") is not True
        or camera.get("collect_wrist_camera") is not True
        or data_type.get("rgb") is not True
        or data_type.get("qpos") is not True
    ):
        raise ValueError(
            f"{config_path.name} must enable D435 head/wrist RGB and qpos for the "
            "RollingWAM three-camera policy."
        )

    required_assets = [
        domino_root / "assets" / "embodiments" / "aloha-agilex" / "config.yml",
        domino_root / "assets" / "objects",
    ]
    missing_assets = [str(path) for path in required_assets if not path.exists()]
    if missing_assets:
        raise FileNotFoundError(
            "DOMINO assets are missing. Run `bash script/_download_assets.sh` from "
            f"{domino_root}. Missing:\n  - " + "\n  - ".join(missing_assets)
        )


def _format_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, (int, float)):
        return str(value)
    return repr(str(value))


def _append_override(
    overrides: list[str], key: str, value: Any, *, skip_none: bool = True
) -> None:
    if skip_none and value is None:
        return
    overrides.extend([f"--{key}", _format_override_value(value)])


def _validate_metrics(
    metrics_path: Path,
    *,
    requested_episodes: int,
    allow_manifest_shortfall: bool,
) -> None:
    if not metrics_path.is_file():
        raise FileNotFoundError(f"DOMINO did not produce its metrics file: {metrics_path}")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    actual = int(payload.get("total_episodes", -1))
    if allow_manifest_shortfall:
        valid = 0 < actual <= requested_episodes
    else:
        valid = actual == requested_episodes
    if not valid:
        qualifier = "1..requested" if allow_manifest_shortfall else "requested"
        raise RuntimeError(
            f"DOMINO metrics contain {actual} episodes; expected {qualifier} "
            f"({requested_episodes}): {metrics_path}"
        )


def _native_metrics_paths(
    domino_root: Path, *, task_name: str, task_config: str
) -> set[Path]:
    result_root = domino_root / "eval_result" / task_name / POLICY_NAME / task_config
    if not result_root.is_dir():
        return set()
    return {path.resolve() for path in result_root.rglob("_metrics.json")}


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="sim_domino.yaml",
)
def main(cfg: DictConfig) -> None:
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must not be None.")
    if cfg.EVALUATION.task_name is None:
        raise ValueError("`EVALUATION.task_name` must not be None.")

    task_name = str(cfg.EVALUATION.task_name).strip()
    if not NAME_PATTERN.fullmatch(task_name):
        raise ValueError(f"Invalid DOMINO task name: {task_name!r}")
    try:
        dynamic_level = int(cfg.EVALUATION.dynamic_level)
    except (TypeError, ValueError) as exc:
        raise ValueError("EVALUATION.dynamic_level must be one of 1, 2, or 3.") from exc
    if dynamic_level not in {1, 2, 3}:
        raise ValueError("EVALUATION.dynamic_level must be one of 1, 2, or 3.")

    task_config = _resolve_task_config(cfg, dynamic_level)
    instruction_type = str(cfg.EVALUATION.instruction_type)
    if instruction_type not in {"seen", "unseen"}:
        raise ValueError("EVALUATION.instruction_type must be 'seen' or 'unseen'.")
    if str(cfg.EVALUATION.policy_name) != POLICY_NAME:
        raise ValueError(f"EVALUATION.policy_name must be {POLICY_NAME!r}.")
    gpu_id = int(cfg.gpu_id)
    if gpu_id < 0:
        raise ValueError("gpu_id must be a non-negative integer.")

    ckpt_path = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    dataset_stats_path = _resolve_dataset_stats_path(cfg, ckpt_path)

    domino_root = _resolve_path(str(cfg.EVALUATION.domino_root), base=PROJECT_ROOT)
    if not domino_root.is_dir():
        raise FileNotFoundError(f"DOMINO root not found: {domino_root}")
    _validate_domino_setup(
        domino_root,
        task_name=task_name,
        task_config=task_config,
        dynamic_level=dynamic_level,
    )

    policy_source_dir = PROJECT_ROOT / "experiments" / "domino" / POLICY_NAME
    if not policy_source_dir.is_dir():
        raise FileNotFoundError(f"Policy source directory not found: {policy_source_dir}")
    _ensure_policy_symlink(domino_root, policy_source_dir)

    manifest_path = _resolve_optional_path(
        cfg.EVALUATION.get("episode_manifest"), base=domino_root
    )
    if manifest_path is not None and not manifest_path.is_file():
        raise FileNotFoundError(f"Episode manifest not found: {manifest_path}")

    run_output_dir = _resolve_path(str(cfg.EVALUATION.output_dir), base=PROJECT_ROOT)
    policy_output_dir = (
        run_output_dir / task_name / f"level{dynamic_level}" / instruction_type
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_output_dir / (
        f"eval_{task_name}_level{dynamic_level}_{instruction_type}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    sim_task = HydraConfig.get().runtime.choices.get("task")
    if sim_task is None:
        raise ValueError("Hydra did not resolve the model task configuration.")
    sim_cfg_path = (PROJECT_ROOT / "configs" / "sim_domino.yaml").resolve()

    overrides: list[str] = []
    _append_override(overrides, "task_name", task_name)
    _append_override(overrides, "task_config", task_config)
    _append_override(overrides, "ckpt_setting", str(ckpt_path))
    _append_override(overrides, "seed", cfg.seed)
    _append_override(overrides, "policy_name", cfg.EVALUATION.policy_name)
    _append_override(overrides, "instruction_type", instruction_type)
    _append_override(
        overrides,
        "episode_manifest",
        str(manifest_path) if manifest_path else None,
    )
    _append_override(overrides, "eval_output_dir", str(policy_output_dir))

    _append_override(overrides, "sim_cfg_path", str(sim_cfg_path))
    _append_override(overrides, "sim_task", sim_task)
    _append_override(overrides, "mixed_precision", cfg.mixed_precision)
    _append_override(overrides, "device", cfg.EVALUATION.device)
    _append_override(overrides, "dataset_stats_path", str(dataset_stats_path))
    _append_override(overrides, "num_inference_steps", cfg.EVALUATION.num_inference_steps)
    _append_override(
        overrides,
        "compile_action_infer",
        cfg.EVALUATION.get("compile_action_infer", False),
    )
    _append_override(
        overrides, "vae_encode_batch_size", cfg.model.get("vae_encode_batch_size", 1)
    )
    _append_override(
        overrides, "compile_vae_encode", cfg.model.get("compile_vae_encode", False)
    )
    _append_override(overrides, "text_cfg_scale", cfg.EVALUATION.text_cfg_scale)
    _append_override(overrides, "negative_prompt", cfg.EVALUATION.negative_prompt)
    _append_override(overrides, "timing_enabled", cfg.EVALUATION.timing_enabled)
    _append_override(
        overrides,
        "action_source_hz",
        cfg.EVALUATION.get("action_source_hz"),
    )
    _append_override(
        overrides,
        "action_target_hz",
        cfg.EVALUATION.get("action_target_hz"),
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
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    metrics_before = _native_metrics_paths(
        domino_root, task_name=task_name, task_config=task_config
    )
    native_result_file: Path | None = None
    with log_file.open("w", encoding="utf-8") as log_f:
        process = subprocess.Popen(
            cmd,
            cwd=str(domino_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            if line.startswith("Data has been saved to "):
                result_text = line.removeprefix("Data has been saved to ").strip()
                result_file = Path(result_text)
                if not result_file.is_absolute():
                    result_file = domino_root / result_file
                native_result_file = result_file.resolve()
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
            log_f.flush()
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"DOMINO evaluation failed with return code {return_code}. Log: {log_file}"
        )

    metrics_path = (
        native_result_file.parent / "_metrics.json"
        if native_result_file is not None
        else None
    )
    if metrics_path is None or not metrics_path.is_file():
        metrics_after = _native_metrics_paths(
            domino_root, task_name=task_name, task_config=task_config
        )
        new_metrics = sorted(
            metrics_after - metrics_before,
            key=lambda path: path.stat().st_mtime_ns,
        )
        if len(new_metrics) != 1:
            raise FileNotFoundError(
                "Could not identify the metrics produced by DOMINO's native "
                f"evaluator. New candidates: {new_metrics}. Log: {log_file}"
            )
        metrics_path = new_metrics[0]

    _validate_metrics(
        metrics_path,
        requested_episodes=NATIVE_EVAL_NUM_EPISODES,
        allow_manifest_shortfall=manifest_path is not None,
    )
    cfg.EVALUATION.task_config = task_config
    OmegaConf.save(
        config=cfg,
        f=str(
            run_output_dir
            / f"eval_config_{task_name}_level{dynamic_level}_{instruction_type}.yaml"
        ),
    )
    print(f"Evaluation finished successfully. Results: {metrics_path.parent}")
    print(f"Log saved to: {log_file}")


if __name__ == "__main__":
    main()
