#!/usr/bin/env python
"""Checkpoint-backed synthetic Rolling-WAM replan latency benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf


sys.dont_write_bytecode = True

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "evaluate_results" / "latency"
DEFAULT_TASK = "robotwin_selected_tasks_rolling_3cam_384_1e-4"

sys.path.insert(0, str(REPO_ROOT / "src"))
from rollingwam.utils.config_resolvers import register_default_resolvers


register_default_resolvers()


@dataclass(frozen=True)
class RunSpec:
    task: str
    image_height: int
    image_width: int
    context_len: int
    text_dim: int
    action_dim: int
    proprio_dim: int
    window_blocks: int
    chunk_latents: int
    actions_per_chunk: int
    num_inference_steps: int
    steady_denoising_steps: int
    model_seed: int
    synthetic_seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure checkpoint-loaded Rolling-WAM replan latency on synthetic RoboTwin inputs."
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--expected-window-blocks", type=int, default=5)
    parser.add_argument("--expected-chunk-latents", type=int, default=1)
    parser.add_argument("--synthetic-seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.warmup < 0 or args.iters <= 0:
        parser.error("--warmup must be >= 0 and --iters must be > 0")
    if args.num_inference_steps <= 0:
        parser.error("--num-inference-steps must be > 0")
    if args.expected_window_blocks <= 0 or args.expected_chunk_latents <= 0:
        parser.error("expected rolling dimensions must be positive")
    return args


def load_config(task: str) -> Any:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="train", overrides=[f"task={task}"])


def normalize_saved_rolling(model: torch.nn.Module, saved: Any) -> dict[str, Any]:
    if not isinstance(saved, dict):
        raise TypeError(
            f"Checkpoint rolling configuration must be a mapping, got {type(saved).__name__}"
        )
    expected = set(model.ROLLING_KEYS)
    unexpected = set(saved) - expected
    if unexpected:
        raise ValueError(f"Unexpected checkpoint rolling keys: {sorted(unexpected)}")
    missing = expected - set(saved)
    compatible = {
        key: model.ROLLING_LEGACY_DEFAULTS.get(key, getattr(model, key))
        for key in missing
    }
    return {**compatible, **saved}


def build_model(
    cfg: Any,
    checkpoint: Path,
    device: torch.device,
) -> torch.nn.Module:
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model_cfg.load_text_encoder = False
    model_cfg.skip_dit_load_from_pretrain = True
    model_cfg.action_dit_pretrained_path = None
    model_cfg.compile_training_denoise = False
    model_cfg.vae_encode_batch_size = 1
    model_cfg.compile_vae_encode = False
    model = instantiate(model_cfg, model_dtype=torch.bfloat16, device=str(device))

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(
            f"Checkpoint payload must be a mapping, got {type(payload).__name__}"
        )
    required = {"mot", "proprio_encoder", "rolling", "scheduler"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Checkpoint is missing required components: {sorted(missing)}")
    if model.proprio_encoder is None:
        raise RuntimeError("Configured model has no proprio encoder")

    scheduler = payload["scheduler"]
    expected_scheduler = {
        "shift": model.train_video_scheduler.shift,
        "num_train_timesteps": model.train_video_scheduler.num_train_timesteps,
    }
    if not isinstance(scheduler, dict) or any(
        scheduler.get(key) != value for key, value in expected_scheduler.items()
    ):
        raise ValueError(
            f"Checkpoint scheduler {scheduler!r} does not match config {expected_scheduler!r}"
        )

    model.mot.load_state_dict(payload["mot"], strict=True)
    model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
    model.configure_rolling(**normalize_saved_rolling(model, payload["rolling"]))
    del payload
    return model.to(device).eval()


def make_run_spec(model: torch.nn.Module, cfg: Any, args: argparse.Namespace) -> RunSpec:
    height, width = (int(value) for value in cfg.data.train.video_size)
    window_blocks = int(model.window_blocks)
    chunk_latents = int(model.chunk_latents)
    actions_per_chunk = int(model.actions_per_chunk)
    if window_blocks != int(args.expected_window_blocks):
        raise ValueError(
            f"Checkpoint restored W={window_blocks}; expected {args.expected_window_blocks}"
        )
    if chunk_latents != int(args.expected_chunk_latents):
        raise ValueError(
            f"Checkpoint restored chunk_latents={chunk_latents}; "
            f"expected {args.expected_chunk_latents}"
        )
    if args.num_inference_steps % window_blocks:
        raise ValueError(
            f"num_inference_steps={args.num_inference_steps} must be divisible by W={window_blocks}"
        )
    return RunSpec(
        task=str(args.task),
        image_height=height,
        image_width=width,
        context_len=int(cfg.data.train.context_len),
        text_dim=int(model.text_dim),
        action_dim=int(model.action_expert.action_dim),
        proprio_dim=int(model.proprio_dim),
        window_blocks=window_blocks,
        chunk_latents=chunk_latents,
        actions_per_chunk=actions_per_chunk,
        num_inference_steps=int(args.num_inference_steps),
        steady_denoising_steps=int(args.num_inference_steps) // window_blocks,
        model_seed=int(cfg.seed),
        synthetic_seed=int(args.synthetic_seed),
    )


def prepare_inputs(spec: RunSpec, device: torch.device) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(spec.synthetic_seed)
    image = (
        torch.rand(
            (1, 3, spec.image_height, spec.image_width),
            generator=generator,
            dtype=torch.float32,
        )
        .mul_(2.0)
        .sub_(1.0)
        .to(device=device, dtype=torch.bfloat16)
    )
    return {
        "new_frames": image.unsqueeze(2),
        "context": torch.zeros(
            (1, spec.context_len, spec.text_dim),
            device=device,
            dtype=torch.bfloat16,
        ),
        "context_mask": torch.ones(
            (1, spec.context_len), device=device, dtype=torch.bool
        ),
        "proprio": torch.zeros((1, spec.proprio_dim), dtype=torch.float32),
    }


def rolling_kwargs(spec: RunSpec, inputs: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        "new_frames": inputs["new_frames"],
        "prompt": None,
        "context": inputs["context"],
        "context_mask": inputs["context_mask"],
        "proprio": inputs["proprio"],
        "negative_prompt": "",
        "text_cfg_scale": 1.0,
        "seed": spec.model_seed,
        "num_inference_steps": spec.num_inference_steps,
        "compile_action_infer": False,
    }


def timed_call(
    model: torch.nn.Module,
    spec: RunSpec,
    inputs: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], float]:
    torch.cuda.synchronize(model.device)
    start = time.perf_counter()
    with torch.no_grad():
        output = model.rolling_act(**rolling_kwargs(spec, inputs))
    torch.cuda.synchronize(model.device)
    return output, (time.perf_counter() - start) * 1000.0


def validate_output(output: dict[str, Any], spec: RunSpec) -> None:
    action = output.get("action")
    video = output.get("video")
    expected_action = (spec.actions_per_chunk, spec.action_dim)
    if not isinstance(action, torch.Tensor) or tuple(action.shape) != expected_action:
        raise RuntimeError(
            f"Action output has shape {getattr(action, 'shape', None)}, expected {expected_action}"
        )
    if not torch.isfinite(action).all():
        raise RuntimeError("Action output contains non-finite values")
    if not isinstance(video, torch.Tensor) or int(video.shape[2]) != spec.chunk_latents:
        raise RuntimeError(
            f"Video output has shape {getattr(video, 'shape', None)}, "
            f"expected temporal length {spec.chunk_latents}"
        )
    if not torch.isfinite(video).all():
        raise RuntimeError("Video output contains non-finite values")


def metric(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "std_ms": float(array.std(ddof=0)),
        "min_ms": float(array.min()),
        "median_ms": float(np.median(array)),
        "max_ms": float(array.max()),
    }


def write_results(
    output_dir: Path,
    settings: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    initialization = metric([row["initialization_ms"] for row in records])
    steady = metric([row["steady_replan_ms"] for row in records])
    summary = {
        "model": "Rolling-WAM",
        "initialization_ms": initialization,
        "steady_replan_ms": steady,
        "steady_vs_initialization_speedup": (
            initialization["mean_ms"] / steady["mean_ms"]
        ),
    }
    payload = {"settings": settings, "results": [summary], "records": records}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "latency_results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Replan Latency",
        "",
        f"- GPU: {settings['gpu']}",
        f"- Checkpoint: `{settings['checkpoint']}`",
        f"- Setting: BF16, batch 1, {settings['input_size'][0]} × "
        f"{settings['input_size'][1]}, W={settings['window_blocks']}, "
        f"chunk latents={settings['chunk_latents']}, "
        f"S={settings['num_inference_steps']}",
        f"- Protocol: {settings['warmup']} warm-up initialization/steady pairs "
        f"followed by {settings['iterations']} synchronized measurement pairs",
        "",
        "| Model | Initialization | Steady replan |",
        "|---|---:|---:|",
        f"| Rolling-WAM | {initialization['mean_ms']:.3f} ± "
        f"{initialization['std_ms']:.3f} | {steady['mean_ms']:.3f} ± "
        f"{steady['std_ms']:.3f} |",
    ]
    (output_dir / "latency_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this latency benchmark")
    if args.gpu_id < 0 or args.gpu_id >= torch.cuda.device_count():
        raise ValueError(
            f"Invalid --gpu-id {args.gpu_id}; found {torch.cuda.device_count()} device(s)"
        )
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)

    cfg = load_config(args.task)
    checkpoint = args.checkpoint.expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir
        or (DEFAULT_RESULTS_ROOT / f"rollingwam_robotwin_{timestamp}")
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}", flush=True)
    print(f"Loading checkpoint-backed Rolling-WAM on {device} ...", flush=True)
    setup_start = time.perf_counter()
    model = build_model(cfg, checkpoint, device)
    spec = make_run_spec(model, cfg, args)
    inputs = prepare_inputs(spec, device)
    torch.cuda.synchronize(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Loaded {parameter_count / 1e9:.3f}B parameters in "
        f"{time.perf_counter() - setup_start:.1f}s",
        flush=True,
    )
    print(
        f"W={spec.window_blocks}, chunk_latents={spec.chunk_latents}, "
        f"actions_per_chunk={spec.actions_per_chunk}, S={spec.num_inference_steps}, "
        f"steady passes={spec.steady_denoising_steps}",
        flush=True,
    )

    for index in range(args.warmup):
        model.rolling_reset()
        init_output, _ = timed_call(model, spec, inputs)
        steady_output, _ = timed_call(model, spec, inputs)
        validate_output(init_output, spec)
        validate_output(steady_output, spec)
        print(f"Warm-up pair {index + 1}/{args.warmup}", flush=True)

    records: list[dict[str, Any]] = []
    for index in range(args.iters):
        model.rolling_reset()
        init_output, init_ms = timed_call(model, spec, inputs)
        steady_output, steady_ms = timed_call(model, spec, inputs)
        validate_output(init_output, spec)
        validate_output(steady_output, spec)
        record = {
            "iteration": index,
            "initialization_ms": init_ms,
            "steady_replan_ms": steady_ms,
            "action_shape": list(steady_output["action"].shape),
            "video_shape": list(steady_output["video"].shape),
        }
        records.append(record)
        print(
            f"Measurement pair {index + 1}/{args.iters}: "
            f"initialization={init_ms:.3f} ms, steady={steady_ms:.3f} ms",
            flush=True,
        )

    settings = {
        "gpu": torch.cuda.get_device_name(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "checkpoint": str(checkpoint),
        "checkpoint_loaded": True,
        "dtype": "bfloat16",
        "batch_size": 1,
        "warmup": int(args.warmup),
        "iterations": int(args.iters),
        "input_size": [spec.image_height, spec.image_width],
        "synthetic_inputs": True,
        "timer_boundary": "cuda_sync; rolling_act; cuda_sync",
        **asdict(spec),
    }
    summary = write_results(output_dir, settings, records)
    initialization = summary["initialization_ms"]
    steady = summary["steady_replan_ms"]
    print(
        f"Rolling-WAM initialization: {initialization['mean_ms']:.3f} ± "
        f"{initialization['std_ms']:.3f} ms (n={args.iters})",
        flush=True,
    )
    print(
        f"Rolling-WAM steady replan: {steady['mean_ms']:.3f} ± "
        f"{steady['std_ms']:.3f} ms (n={args.iters})",
        flush=True,
    )
    print(f"Report: {output_dir / 'latency_report.md'}", flush=True)


if __name__ == "__main__":
    main()
