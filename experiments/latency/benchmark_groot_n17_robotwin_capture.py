#!/usr/bin/env python3
"""Benchmark the GR00T N1.7 compute graph on captured RoboTwin observations.

The released N1.7 configuration is used, but model weights are deliberately
skipped because its Cosmos-Reason2 backbone is gated.  The public
Qwen3-VL-2B configuration has the same operator dimensions and is used only to
instantiate the backbone.  This is a latency proxy, not a policy-quality run.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


GROOT_ROOT = Path(
    os.environ.get("GROOT_ROOT", "/tmp/Isaac-GR00T-N1.7")
).expanduser()
if str(GROOT_ROOT) not in sys.path:
    sys.path.insert(0, str(GROOT_ROOT))

# GR00T provides this opt-in path for constructing the complete model without
# resolving checkpoint tensors.  It is set before importing the package.
os.environ.setdefault("GROOT_HF_LOCAL_FIRST", "1")
os.environ.setdefault("GROOT_SKIP_HF_MODEL_WEIGHTS", "1")
os.environ.setdefault("GROOT_PATCH_MISTRAL", "1")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import torch  # noqa: E402

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config  # noqa: E402
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7  # noqa: E402
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import (  # noqa: E402
    Gr00tN1d7Processor,
)


PUBLIC_BACKBONE = "Qwen/Qwen3-VL-2B-Instruct"
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


def _load_capture(path: Path) -> dict[str, np.ndarray]:
    required = {"head_rgb", "left_rgb", "right_rgb", "state", "instruction"}
    with np.load(path, allow_pickle=False) as data:
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Observation capture is missing keys: {sorted(missing)}")
        capture = {key: np.asarray(data[key]) for key in sorted(required)}

    replans = int(capture["state"].shape[0])
    if replans < 2:
        raise ValueError(f"At least two captured replans are required, got {replans}")
    for key, value in capture.items():
        if value.ndim == 0 or int(value.shape[0]) != replans:
            raise ValueError(
                f"Capture field {key!r} has shape {value.shape}; expected leading "
                f"replan dimension {replans}"
            )
    for key in ("head_rgb", "left_rgb", "right_rgb"):
        value = capture[key]
        if value.ndim != 4 or value.shape[-1] != 3 or value.dtype != np.uint8:
            raise ValueError(f"Capture field {key!r} must be uint8 [R,H,W,3], got {value.shape}")
    if capture["state"].shape[1:] != (14,):
        raise ValueError(f"RoboTwin qpos must be [R,14], got {capture['state'].shape}")
    return capture


def _instruction(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _raw_observation(capture: dict[str, np.ndarray], index: int) -> dict[str, Any]:
    qpos = np.asarray(capture["state"][index], dtype=np.float32)

    def current_frame(image: np.ndarray) -> np.ndarray:
        return image[None, None, ...]

    return {
        "video.top_camera-images-rgb_320_240": current_frame(capture["head_rgb"][index]),
        "video.left_camera-images-rgb_320_240": current_frame(capture["left_rgb"][index]),
        "video.right_camera-images-rgb_320_240": current_frame(capture["right_rgb"][index]),
        "state.left_wrist_eef": np.empty((1, 1, 0), dtype=np.float32),
        "state.right_wrist_eef": np.empty((1, 1, 0), dtype=np.float32),
        "state.left_gripper_pos": qpos[6:7].reshape(1, 1, 1),
        "state.right_gripper_pos": qpos[13:14].reshape(1, 1, 1),
        "state.left_joint_pos": qpos[0:6].reshape(1, 1, 6),
        "state.right_joint_pos": qpos[7:13].reshape(1, 1, 6),
        "annotation.task": [_instruction(capture["instruction"][index])],
    }


def _prepare_inputs(
    model: Gr00tN1d7,
    processor: Gr00tN1d7Processor,
    capture: dict[str, np.ndarray],
    index: int,
) -> tuple[Any, Any]:
    processed = processor.process_observation(
        _raw_observation(capture, index), EmbodimentTag.XDOF
    )
    return model.prepare_input(processed)


def _time_replan(model: Gr00tN1d7, backbone_inputs: Any, action_inputs: Any) -> tuple[float, np.ndarray]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        backbone_outputs = model.backbone(backbone_inputs)
        prediction = model.action_head.get_action(backbone_outputs, action_inputs)
        actions_host = prediction["action_pred"].float().cpu().numpy()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0, actions_host


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--config-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task", default="beat_block_hammer")
    parser.add_argument("--measured-replans", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    capture_path = args.capture.expanduser().resolve()
    config_dir = args.config_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not capture_path.is_file():
        raise FileNotFoundError(f"Observation capture not found: {capture_path}")
    for filename in ("config.json", "processor_config.json", "statistics.json"):
        if not config_dir.joinpath(filename).is_file():
            raise FileNotFoundError(f"Required N1.7 config file not found: {config_dir / filename}")
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU is visible")

    capture = _load_capture(capture_path)
    available_replans = int(capture["state"].shape[0])
    measured_replans = available_replans if args.measured_replans is None else args.measured_replans
    if not 2 <= measured_replans <= available_replans:
        raise ValueError(
            f"--measured-replans must be in [2, {available_replans}], got {measured_replans}"
        )

    torch.manual_seed(args.seed)
    config = Gr00tN1d7Config.from_pretrained(config_dir)
    config.model_name = PUBLIC_BACKBONE
    config.action_horizon = ACTION_HORIZON
    config.num_inference_timesteps = NUM_DENOISING_STEPS
    config.use_flash_attention = False
    model = Gr00tN1d7(
        config,
        transformers_loading_kwargs={
            "trust_remote_code": True,
            "attn_implementation": "sdpa",
        },
    ).eval().to(device="cuda", dtype=torch.bfloat16)
    model.action_head.num_inference_timesteps = NUM_DENOISING_STEPS

    processor = Gr00tN1d7Processor.from_pretrained(
        config_dir,
        model_name=PUBLIC_BACKBONE,
        transformers_loading_kwargs={"trust_remote_code": True},
    )
    processor.eval()

    xdof_config = processor.modality_configs[EmbodimentTag.XDOF.value]
    xdof_config["video"].delta_indices = [0]
    xdof_config["action"].delta_indices = list(range(ACTION_HORIZON))
    processor.max_action_horizon = ACTION_HORIZON

    assert len(model.backbone.model.language_model.layers) == 16
    assert len(model.action_head.model.transformer_blocks) == 32
    assert len(model.action_head.vl_self_attention.transformer_blocks) == 4
    assert model.action_head.action_horizon == ACTION_HORIZON
    assert model.config.action_horizon == ACTION_HORIZON
    assert not model.config.use_flash_attention
    assert model.backbone.model.config._attn_implementation == "sdpa"

    warmup_backbone, warmup_action = _prepare_inputs(model, processor, capture, 0)
    _, warmup_output = _time_replan(model, warmup_backbone, warmup_action)
    expected_shape = (1, ACTION_HORIZON, 132)
    if warmup_output.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected GR00T action shape {warmup_output.shape}; expected {expected_shape}"
        )

    replan_ms: list[float] = []
    input_shapes: dict[str, list[int]] | None = None
    for index in range(measured_replans):
        backbone_inputs, action_inputs = _prepare_inputs(model, processor, capture, index)
        if input_shapes is None:
            input_shapes = {
                key: list(value.shape)
                for key, value in backbone_inputs.items()
                if isinstance(value, torch.Tensor)
            }
            if input_shapes.get("image_grid_thw") != [3, 3]:
                raise RuntimeError(
                    "Expected exactly three current camera images, got "
                    f"image_grid_thw={input_shapes.get('image_grid_thw')}"
                )
            if list(action_inputs["action_mask"].shape) != [1, ACTION_HORIZON]:
                raise RuntimeError(
                    f"Unexpected action_mask shape {tuple(action_inputs['action_mask'].shape)}"
                )
        elapsed_ms, actions = _time_replan(model, backbone_inputs, action_inputs)
        if actions.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected GR00T action shape {actions.shape}; expected {expected_shape}"
            )
        replan_ms.append(elapsed_ms)

    initialization_ms = replan_ms[0]
    steady_ms = replan_ms[1:]
    steady_stats = _stats_ms(steady_ms)
    line = (
        "[INFO] [groot_n17_policy.deploy_policy_timing] Replan timing | "
        f"initialization {initialization_ms:.3f} ms | "
        f"steady-state mean {steady_stats['mean_ms']:.3f} ms "
        f"min {steady_stats['min_ms']:.3f} ms max {steady_stats['max_ms']:.3f} ms "
        f"(n={len(steady_ms)}) | total replans={len(replan_ms)}"
    )
    print(line, flush=True)

    episode = {
        "episode": 0,
        "initialization_ms": initialization_ms,
        "steady_state_ms": steady_ms,
        "steady_state": steady_stats,
        "total_replans": len(replan_ms),
    }
    result = {
        "schema_version": 1,
        "policy": "GR00T N1.7 architecture timing proxy",
        "unit": "ms",
        "task_name": args.task,
        "observation_capture": str(capture_path),
        "replay_only": True,
        "weights": (
            "untrained; backbone zero-initialized by GR00T's skip-weight path and "
            "action head randomly initialized"
        ),
        "model": {
            "configuration": str(config_dir / "config.json"),
            "backbone_architecture_source": PUBLIC_BACKBONE,
            "dtype": "bfloat16",
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "retained_vlm_layers": 16,
            "action_dit_layers": 32,
            "vl_self_attention_layers": 4,
            "predicted_action_horizon": ACTION_HORIZON,
            "comparison_execution_horizon": ACTION_HORIZON,
            "action_dim": 132,
            "num_inference_steps": NUM_DENOISING_STEPS,
            "raw_camera_resolution": [240, 320],
            "native_visual_history_slots": 1,
            "camera_views": 3,
            "images_per_replan": 3,
            "native_image_target": [256, 256],
            "input_shapes": input_shapes,
        },
        "timing": {
            "untimed_warmup_calls": 1,
            "preprocessing_timed": False,
            "timed_region": (
                "backbone + ten-step action head + CUDA synchronization + "
                "full host action materialization"
            ),
            "initialization_is_schema_label_only": True,
            "attention_implementation": "PyTorch SDPA",
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


if __name__ == "__main__":
    main()
