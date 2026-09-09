#!/usr/bin/env python3
"""Serve the existing RoboTwin evaluation policy on a separate model host."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

logger = logging.getLogger(__name__)
PROTOCOL = "rollingwam.robotwin.v1"
CAMERA_KEYS = ("head_camera", "left_camera", "right_camera")


class RobotWinPolicyServerAdapter:
    """Expose the local policy's exact preprocessing and rolling inference over RPC."""

    def __init__(
        self,
        policy: Any,
        *,
        checkpoint: str,
        config: str,
        dataset_stats: str,
    ) -> None:
        self.policy = policy
        self.checkpoint = str(checkpoint)
        self.config = str(config)
        self.dataset_stats = str(dataset_stats)
        self.actions_per_chunk = int(policy.model.actions_per_chunk)
        self.window_blocks = int(policy.model.window_blocks)
        if self.window_blocks < 1:
            raise ValueError("window_blocks must be positive.")
        self.execute_horizon = (
            self.actions_per_chunk if policy.replan_steps is None else int(policy.replan_steps)
        )
        if int(policy.model.action_expert.action_dim) != 14 or policy.model.proprio_dim != 14:
            raise ValueError("RoboTwin evaluation requires 14D action and proprioception.")
        if self.actions_per_chunk < 1 or not 0 < self.execute_horizon <= self.actions_per_chunk:
            raise ValueError("The execution horizon must be within the predicted action chunk.")
        if policy.replan_steps is not None and self.window_blocks != 1:
            raise ValueError("replan_steps requires a window_blocks=1 checkpoint.")
        if policy.num_inference_steps < 1 or policy.num_inference_steps % self.window_blocks:
            raise ValueError("num_inference_steps must be positive and divisible by window_blocks.")

    @classmethod
    def from_checkpoint(cls, args: argparse.Namespace) -> "RobotWinPolicyServerAdapter":
        # Keep model dependencies out of argument parsing and --help.
        import torch
        from omegaconf import OmegaConf

        from experiments.robotwin.rollingwam_policy.deploy_policy import (
            WorldActionRobotWinPolicy,
            _mixed_precision_to_model_dtype,
        )
        from rollingwam.serving.rollingwam_policy import (
            _flat_dimension,
            _load_training_config,
            _resolve_stats_path,
        )

        checkpoint = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"RollingWAM checkpoint not found: {checkpoint}")
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {args.device!r} requested, but CUDA is unavailable.")
        cfg, config_path = _load_training_config(checkpoint, args.config, args.task_config)
        stats_path = _resolve_stats_path(checkpoint, args.dataset_stats)
        data_cfg = cfg.data.train
        if tuple(data_cfg.video_size) != (384, 320) or data_cfg.get("concat_multi_camera") != "robotwin":
            raise ValueError("RoboTwin serving requires video_size=[384,320] and concat_multi_camera=robotwin.")
        processor_cfg = OmegaConf.create(OmegaConf.to_container(data_cfg.processor, resolve=True))
        if len(processor_cfg.shape_meta.images) != 3 or processor_cfg.num_output_cameras != 3:
            raise ValueError("RoboTwin serving requires the three-camera processor.")
        for field in ("state", "action"):
            entries = processor_cfg.shape_meta[field]
            if len(entries) != 1 or _flat_dimension(entries[0].raw_shape, name=field) != 14:
                raise ValueError(f"RoboTwin serving requires one merged 14D {field} field.")
        if processor_cfg.get("action_state_transforms") is not None:
            raise ValueError("RoboTwin serving requires action_state_transforms=null.")

        model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        model_cfg.load_text_encoder = True
        model_cfg.skip_dit_load_from_pretrain = True
        model_cfg.action_dit_pretrained_path = None
        model_cfg.compile_vae_encode = args.compile_vae_encode
        model_cfg.vae_encode_batch_size = args.vae_encode_batch_size
        policy = WorldActionRobotWinPolicy(
            model_cfg=model_cfg,
            processor_cfg=processor_cfg,
            checkpoint_path=str(checkpoint),
            dataset_stats_path=stats_path,
            device=args.device,
            model_dtype=_mixed_precision_to_model_dtype(args.mixed_precision),
            seed=args.seed,
            num_inference_steps=args.num_steps,
            text_cfg_scale=args.text_cfg_scale,
            negative_prompt=args.negative_prompt,
            timing_enabled=False,
            replan_steps=args.replan_steps,
            compile_action_infer=args.compile_action_infer,
        )
        return cls(
            policy,
            checkpoint=str(checkpoint),
            config=str(config_path) if config_path else f"task:{args.task_config}",
            dataset_stats=str(stats_path),
        )

    def server_metadata(self) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL,
            "method": "RollingWAM",
            "actions_per_chunk": self.actions_per_chunk,
            "execute_horizon": self.execute_horizon,
            "action_dim": 14,
            "state_dim": 14,
            "window_blocks": self.window_blocks,
            "seed": self.policy.seed,
            "checkpoint": self.checkpoint,
            "config": self.config,
            "dataset_stats": self.dataset_stats,
            "num_inference_steps": self.policy.num_inference_steps,
            "text_cfg_scale": self.policy.text_cfg_scale,
            "negative_prompt": self.policy.negative_prompt,
            "compile_action_infer": self.policy.compile_action_infer,
            "camera_keys": list(CAMERA_KEYS),
            "action_space": "unnormalized_qpos",
        }

    def reset(self) -> None:
        self.policy.reset()

    def infer(self, request: Mapping[str, Any]) -> dict[str, Any]:
        import numpy as np

        if not isinstance(request, Mapping):
            raise TypeError("RoboTwin requests must be mappings.")
        operation = request.get("op")
        if operation == "reset":
            seed = request.get("seed", self.policy.seed)
            if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
                raise TypeError("reset seed must be an integer or null.")
            self.policy.seed = seed
            self.reset()
            return {"op": "reset", "ok": True}
        if operation != "infer":
            raise ValueError(f"Unsupported RoboTwin operation: {operation!r}.")

        instruction = request.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Inference requires a non-empty instruction string.")
        observation = request.get("observation")
        if not isinstance(observation, Mapping):
            raise TypeError("Inference requires an observation mapping.")
        try:
            images = observation["observation"]
            state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Observation must contain camera observations and joint_action.vector.") from exc
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("joint_action.vector must contain 14 finite values.")
        if not isinstance(images, Mapping):
            raise TypeError("Camera observations must be a mapping.")
        validated_images = {}
        for camera in CAMERA_KEYS:
            try:
                image = np.asarray(images[camera]["rgb"])
            except (KeyError, TypeError) as exc:
                raise ValueError(f"Observation is missing {camera}.rgb.") from exc
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) < 1:
                raise ValueError(f"{camera}.rgb must be a non-empty uint8 HWC RGB image.")
            validated_images[camera] = {"rgb": image}
        observation = {
            "observation": validated_images,
            "joint_action": {"vector": state.copy()},
        }

        start = time.perf_counter()
        action = np.asarray(self.policy._infer_action_chunk(observation, instruction), dtype=np.float32)
        if action.shape != (self.actions_per_chunk, 14) or not np.isfinite(action).all():
            raise ValueError(f"Model action must be finite with shape [{self.actions_per_chunk},14].")
        return {
            "op": "infer",
            "action": np.ascontiguousarray(action[: self.execute_horizon]),
            "server_policy_s": time.perf_counter() - start,
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="RollingWAM checkpoint on the model host.")
    parser.add_argument("--config", help="Saved training config.yaml; otherwise found above the checkpoint.")
    parser.add_argument("--task-config", help="Fallback Hydra task when no saved config.yaml exists.")
    parser.add_argument("--dataset-stats", help="dataset_stats.json; otherwise found above the checkpoint.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--num-steps", type=int, default=10, help="Must be divisible by checkpoint window_blocks.")
    parser.add_argument("--seed", type=int, default=42, help="Initial model seed; the evaluation client's seed overrides it at each episode reset.")
    parser.add_argument("--text-cfg-scale", type=float, default=1.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--compile-action-infer", action="store_true")
    parser.add_argument("--compile-vae-encode", action="store_true")
    parser.add_argument("--vae-encode-batch-size", type=int, default=1)
    parser.add_argument("--replan-steps", type=int, help="Execute a shorter prefix; requires a W=1 checkpoint.")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.num_steps < 1:
        parser.error("--num-steps must be positive")
    if args.vae_encode_batch_size < 0:
        parser.error("--vae-encode-batch-size must be non-negative")
    if args.replan_steps is not None and args.replan_steps < 1:
        parser.error("--replan-steps must be positive")
    return args


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    from rollingwam.serving.websocket_policy_server import WebsocketPolicyServer

    adapter = RobotWinPolicyServerAdapter.from_checkpoint(args)
    metadata = adapter.server_metadata()
    logger.info("RoboTwin policy ready | %s", metadata)
    # Synchronous model warm-up can block the event loop longer than keepalive deadlines.
    WebsocketPolicyServer(
        adapter, host=args.host, port=args.port, metadata=metadata, ping_interval=None,
    ).serve_forever()


if __name__ == "__main__":
    main()
