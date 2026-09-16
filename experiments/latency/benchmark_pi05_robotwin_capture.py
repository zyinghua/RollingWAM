#!/usr/bin/env python3
"""Benchmark the PyTorch pi0.5 graph on captured RoboTwin observations."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


OPENPI_ROOT = Path(
    os.environ.get("OPENPI_ROOT", "/workspace/RollingWAM/third_party/openpi")
).expanduser()
OPENPI_SRC = OPENPI_ROOT / "src"
if str(OPENPI_SRC) not in sys.path:
    sys.path.insert(0, str(OPENPI_SRC))

import torch  # noqa: E402

from openpi import transforms as pi_transforms  # noqa: E402
from openpi.models import model as model_lib  # noqa: E402
from openpi.models.pi0_config import Pi0Config  # noqa: E402
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch  # noqa: E402
from openpi.training.config import ModelTransformFactory  # noqa: E402


LOGGER = logging.getLogger("pi05_policy.deploy_policy_timing")
ACTION_HORIZON = 16
NUM_DENOISING_STEPS = 10


def _stats_ms(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean_ms": None,
            "std_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean_ms": float(array.mean()),
        "std_ms": float(array.std()),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
    }


def _instruction_text(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(f"Expected scalar instruction, got shape {value.shape}")
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _load_capture(path: Path) -> dict[str, np.ndarray]:
    required = {"head_rgb", "left_rgb", "right_rgb", "state", "instruction"}
    with np.load(path, allow_pickle=False) as data:
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Observation capture is missing keys: {sorted(missing)}")
        capture = {key: np.asarray(data[key]) for key in sorted(required)}

    num_observations = int(capture["state"].shape[0])
    if num_observations < 2:
        raise ValueError(f"At least two captured replans are required, got {num_observations}")
    for key, value in capture.items():
        if value.ndim == 0 or int(value.shape[0]) != num_observations:
            raise ValueError(
                f"Capture field {key!r} has shape {value.shape}; expected leading "
                f"replan dimension {num_observations}"
            )
    for key in ("head_rgb", "left_rgb", "right_rgb"):
        value = capture[key]
        if value.ndim != 4 or value.shape[-1] != 3 or value.dtype != np.uint8:
            raise ValueError(f"Capture field {key!r} must be uint8 [R,H,W,3], got {value.shape}")
    if capture["state"].shape[1:] != (14,):
        raise ValueError(f"RoboTwin qpos must be [R,14], got {capture['state'].shape}")
    return capture


def _tree_to_torch_batch(value: Any, device: torch.device) -> Any:
    if isinstance(value, dict):
        return {key: _tree_to_torch_batch(item, device) for key, item in value.items()}
    array = np.asarray(value)
    if not array.flags.writeable:
        array = array.copy()
    return torch.from_numpy(array).unsqueeze(0).to(device=device)


def _prepare_observation(
    transform: Any,
    capture: dict[str, np.ndarray],
    index: int,
    device: torch.device,
) -> model_lib.Observation:
    canonical = {
        "image": {
            "base_0_rgb": capture["head_rgb"][index],
            "left_wrist_0_rgb": capture["left_rgb"][index],
            "right_wrist_0_rgb": capture["right_rgb"][index],
        },
        "image_mask": {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        },
        "state": np.asarray(capture["state"][index], dtype=np.float32),
        "prompt": _instruction_text(capture["instruction"][index]),
    }
    transformed = transform(canonical)
    observation = model_lib.Observation.from_dict(
        _tree_to_torch_batch(transformed, device)
    )
    torch.cuda.synchronize(device)
    return observation


def _time_replan(
    model: PI0Pytorch,
    device: torch.device,
    observation: model_lib.Observation,
) -> tuple[float, np.ndarray]:
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        actions = model.sample_actions(
            device,
            observation,
            num_steps=NUM_DENOISING_STEPS,
        )
        actions_host = actions.detach().to(device="cpu", dtype=torch.float32).numpy()
    torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1000.0, actions_host


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Time an eager PyTorch pi0.5 graph on captured RoboTwin observations."
    )
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="beat_block_hammer")
    parser.add_argument("--measured-replans", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    capture_path = args.capture.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not capture_path.is_file():
        raise FileNotFoundError(f"Observation capture not found: {capture_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU is visible; expose exactly one free GPU")

    device = torch.device("cuda:0")
    capture = _load_capture(capture_path)
    available_replans = int(capture["state"].shape[0])
    measured_replans = (
        available_replans if args.measured_replans is None else args.measured_replans
    )
    if not 2 <= measured_replans <= available_replans:
        raise ValueError(
            f"--measured-replans must be in [2, {available_replans}], got "
            f"{measured_replans}"
        )

    torch.manual_seed(args.seed)
    config = Pi0Config(
        pi05=True,
        action_horizon=ACTION_HORIZON,
        action_dim=32,
        dtype="bfloat16",
        pytorch_compile_mode=None,
    )
    model = PI0Pytorch(config).eval().to(device=device)
    transform = pi_transforms.compose(ModelTransformFactory()(config).inputs)

    if config.pytorch_compile_mode is not None:
        raise RuntimeError("The controlled benchmark requires torch.compile to be disabled")

    warmup_observation = _prepare_observation(transform, capture, 0, device)
    _, warmup_actions = _time_replan(model, device, warmup_observation)
    expected_shape = (1, ACTION_HORIZON, 32)
    if warmup_actions.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected pi0.5 action shape {warmup_actions.shape}; expected {expected_shape}"
        )

    replan_ms: list[float] = []
    for index in range(measured_replans):
        observation = _prepare_observation(transform, capture, index, device)
        torch.manual_seed(args.seed + index + 1)
        elapsed_ms, actions = _time_replan(model, device, observation)
        if actions.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected pi0.5 action shape {actions.shape}; expected {expected_shape}"
            )
        replan_ms.append(elapsed_ms)

    initialization_ms = replan_ms[0]
    steady_ms = replan_ms[1:]
    steady_stats = _stats_ms(steady_ms)
    LOGGER.info(
        "Replan timing | initialization %.3f ms | "
        "steady-state mean %.3f ms min %.3f ms max %.3f ms "
        "(n=%d) | total replans=%d",
        initialization_ms,
        steady_stats["mean_ms"],
        steady_stats["min_ms"],
        steady_stats["max_ms"],
        len(steady_ms),
        len(replan_ms),
    )

    raw_camera_resolution = {
        key.removesuffix("_rgb"): [
            int(capture[key].shape[1]),
            int(capture[key].shape[2]),
        ]
        for key in ("head_rgb", "left_rgb", "right_rgb")
    }
    episode = {
        "episode": 0,
        "initialization_ms": initialization_ms,
        "steady_state_ms": steady_ms,
        "steady_state": steady_stats,
        "total_replans": len(replan_ms),
    }
    result = {
        "schema_version": 2,
        "policy": "pi0.5 PyTorch architecture timing proxy",
        "unit": "ms",
        "task_name": args.task,
        "observation_capture": str(capture_path),
        "replay_only": True,
        "weights": "untrained; randomly initialized",
        "model": {
            "implementation": "OpenPI PI0Pytorch",
            "framework": f"PyTorch {torch.__version__}",
            "dtype": "bfloat16 with native float32 submodules",
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "action_horizon": ACTION_HORIZON,
            "action_dim": 32,
            "emitted_actions_per_replan": ACTION_HORIZON,
            "num_inference_steps": NUM_DENOISING_STEPS,
            "raw_camera_resolution": raw_camera_resolution,
            "native_camera_resolution": [224, 224],
            "camera_views": 3,
            "images_per_replan": 3,
        },
        "timing": {
            "untimed_warmup_calls": 1,
            "preprocessing_timed": False,
            "timed_region": (
                "eager model call + CUDA synchronization + host action materialization"
            ),
            "initialization_is_schema_label_only": True,
            "torch_compile": False,
            "cuda_graphs": False,
            "tensorrt": False,
        },
        "protocol": {
            "environment": "RoboTwin beat_block_hammer demo_clean seed 100000",
            "same_stationary_trajectory_as_wam": True,
            "visual_history": "one current image from each of three RoboTwin cameras",
        },
        "episodes": [episode],
        "aggregate": {
            "episodes": 1,
            "initialization": _stats_ms([initialization_ms]),
            "steady_state": steady_stats,
            "total_replans": len(replan_ms),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary_path.replace(output_path)
    LOGGER.info("Timing JSON saved to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] [%(name)s] %(message)s",
        force=True,
    )
    main()
