"""RollingWAM policy adapter for OmniRobot's native G1 websocket contract."""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from experiments.g1.action_layout import ACTION_DIM, STATE_DIM, split_action, validate_state
from rollingwam.datasets.dataset_utils import (
    CenterCrop,
    Normalize,
    ResizeSmallestSideAspectPreserving,
)
from rollingwam.datasets.lerobot.text_cache import DEFAULT_PROMPT
from rollingwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from rollingwam.utils.config_resolvers import register_default_resolvers

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs"

DEFAULT_TASK_CONFIG = "g1_pnp_pour_rolling_1cam_320_1e-4"
DEFAULT_EMBODIMENT = "unitree_g1_sonic"
IMAGE_KEY = "ego_view"
STATE_KEY = "state"
ACTION_KEY = "action"

register_default_resolvers()


def _model_dtype(mixed_precision: str) -> torch.dtype:
    precision = str(mixed_precision).strip().lower()
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    raise ValueError(
        f"Unsupported mixed precision {mixed_precision!r}; expected no, fp16, or bf16."
    )


def _find_ancestor_file(checkpoint_path: Path, filename: str) -> Path | None:
    for parent in checkpoint_path.parents:
        candidate = parent / filename
        if candidate.is_file():
            return candidate
    return None


def _resolve_stats_path(checkpoint_path: Path, dataset_stats_path: str | None) -> Path:
    if dataset_stats_path:
        path = Path(dataset_stats_path).expanduser().resolve()
    else:
        path = _find_ancestor_file(checkpoint_path, "dataset_stats.json")
        if path is None:
            raise FileNotFoundError(
                "Could not derive dataset_stats.json from the checkpoint. "
                "Pass --dataset-stats explicitly."
            )
    if not path.is_file():
        raise FileNotFoundError(f"Dataset statistics not found: {path}")
    return path


def _compose_task_config(task_config: str) -> DictConfig:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_ROOT)):
        return compose(config_name="train", overrides=[f"task={task_config}"])


def _load_training_config(
    checkpoint_path: Path,
    config_path: str | None,
    task_config: str,
) -> tuple[DictConfig, Path | None]:
    if config_path:
        resolved = Path(config_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Training config not found: {resolved}")
        return OmegaConf.load(resolved), resolved

    run_config = _find_ancestor_file(checkpoint_path, "config.yaml")
    if run_config is not None and run_config.is_file():
        return OmegaConf.load(run_config), run_config

    logger.warning(
        "No config.yaml found above %s; composing current task config %s. "
        "Pass --config to avoid configuration drift.",
        checkpoint_path,
        task_config,
    )
    return _compose_task_config(task_config), None


def _load_complete_g1_checkpoint(model: Any, checkpoint_path: Path) -> int | None:
    """Strictly load a modern checkpoint; partial weights are unsafe on hardware."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint payload must be a dict, got {type(payload).__name__}.")

    required = {"mot", "proprio_encoder", "rolling", "scheduler"}
    missing = required - set(payload)
    if missing:
        raise ValueError(
            "G1 real-robot serving requires a complete modern RollingWAM checkpoint; "
            f"missing keys: {sorted(missing)}. Refusing to serve partially initialized weights."
        )
    if model.proprio_encoder is None:
        raise ValueError("The configured G1 model has no proprio encoder.")

    rolling = payload["rolling"]
    expected_rolling_keys = set(model.ROLLING_KEYS)
    if not isinstance(rolling, dict) or set(rolling) != expected_rolling_keys:
        got = sorted(rolling) if isinstance(rolling, dict) else type(rolling).__name__
        raise ValueError(
            "Checkpoint rolling configuration is incomplete: "
            f"expected {sorted(expected_rolling_keys)}, got {got}."
        )

    scheduler = payload["scheduler"]
    expected_scheduler = {
        "shift": model.train_video_scheduler.shift,
        "num_train_timesteps": model.train_video_scheduler.num_train_timesteps,
    }
    if not isinstance(scheduler, dict) or any(
        scheduler.get(key) != value for key, value in expected_scheduler.items()
    ):
        raise ValueError(
            f"Checkpoint scheduler {scheduler!r} does not match config {expected_scheduler!r}."
        )

    model.mot.load_state_dict(payload["mot"], strict=True)
    model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
    model.configure_rolling(**rolling)
    step = payload.get("step")
    del payload
    return None if step is None else int(step)


class RollingWAMG1Policy:
    """Stateful one-observation-to-one-action-chunk G1 inference adapter.

    The returned 78-D action is kept in the dataset's native SONIC layout:
    ``[motion token 64, left hand 7, right hand 7]``. Robot actuation and all
    safety limits remain client-side.
    """

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        video_size: tuple[int, int],
        num_inference_steps: int = 10,
        text_cfg_scale: float = 1.0,
        negative_prompt: str = "",
        seed: int | None = None,
        compile_action_infer: bool = False,
        embodiment: str = DEFAULT_EMBODIMENT,
        default_instruction: str = "",
        control_hz: float = 10.0,
    ) -> None:
        self.model = model
        self.processor = processor
        self.video_size = tuple(int(v) for v in video_size)
        self.num_inference_steps = int(num_inference_steps)
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.seed = seed
        self.compile_action_infer = bool(compile_action_infer)
        self.embodiment = str(embodiment)
        self.default_instruction = str(default_instruction)
        self.control_hz = float(control_hz)
        self._lock = threading.RLock()
        self._active_instruction: str | None = None

        if len(self.video_size) != 2 or min(self.video_size) < 1:
            raise ValueError(f"video_size must be positive [H,W], got {self.video_size}.")
        if self.control_hz <= 0:
            raise ValueError(f"control_hz must be positive, got {self.control_hz}.")
        if self.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive.")
        if self.num_inference_steps % int(self.model.window_blocks) != 0:
            raise ValueError(
                f"num_inference_steps ({self.num_inference_steps}) must be divisible by "
                f"checkpoint window_blocks ({self.model.window_blocks})."
            )
        if int(self.model.actions_per_chunk) < 1:
            raise ValueError("Checkpoint actions_per_chunk must be positive.")
        if int(self.model.action_expert.action_dim) != ACTION_DIM:
            raise ValueError(
                f"G1 requires a {ACTION_DIM}D action model, got "
                f"{self.model.action_expert.action_dim}."
            )
        if self.model.proprio_dim is None or int(self.model.proprio_dim) != STATE_DIM:
            raise ValueError(
                f"G1 requires a {STATE_DIM}D proprio encoder, got {self.model.proprio_dim}."
            )
        if self.processor.action_state_transforms is not None:
            raise ValueError(
                "The G1 serving adapter currently requires action_state_transforms=null, "
                "matching the G1 training config."
            )

        image_meta = self.processor.shape_meta["images"]
        state_meta = self.processor.shape_meta["state"]
        action_meta = self.processor.shape_meta["action"]
        if len(image_meta) != 1 or len(state_meta) != 1 or len(action_meta) != 1:
            raise ValueError("G1 serving requires exactly one image, state, and action field.")
        if int(state_meta[0]["raw_shape"]) != STATE_DIM:
            raise ValueError("Processor G1 state metadata does not describe 43 raw values.")
        if int(action_meta[0]["raw_shape"]) != ACTION_DIM:
            raise ValueError("Processor G1 action metadata does not describe 78 raw values.")

        self._image_meta = image_meta[0]
        self._state_meta = state_meta[0]
        self._action_meta = action_meta[0]
        self._final_resize = ResizeSmallestSideAspectPreserving(
            args={"img_h": self.video_size[0], "img_w": self.video_size[1]}
        )
        self._final_crop = CenterCrop(
            args={"img_h": self.video_size[0], "img_w": self.video_size[1]}
        )
        self._final_normalize = Normalize(args={"mean": 0.5, "std": 0.5})
        self.reset()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        dataset_stats_path: str | None = None,
        config_path: str | None = None,
        task_config: str = DEFAULT_TASK_CONFIG,
        device: str = "cuda",
        mixed_precision: str = "bf16",
        num_inference_steps: int = 10,
        text_cfg_scale: float = 1.0,
        negative_prompt: str = "",
        seed: int | None = None,
        compile_action_infer: bool = False,
        compile_vae_encode: bool = False,
        vae_encode_batch_size: int = 1,
        embodiment: str = DEFAULT_EMBODIMENT,
        default_instruction: str = "",
        control_hz: float = 10.0,
    ) -> "RollingWAMG1Policy":
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"RollingWAM checkpoint not found: {checkpoint}")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {device!r} requested, but CUDA is unavailable.")

        cfg, loaded_config = _load_training_config(checkpoint, config_path, task_config)
        stats_path = _resolve_stats_path(checkpoint, dataset_stats_path)

        model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        model_cfg.load_text_encoder = True
        model_cfg.skip_dit_load_from_pretrain = True
        model_cfg.action_dit_pretrained_path = None
        model_cfg.compile_vae_encode = bool(compile_vae_encode)
        model_cfg.vae_encode_batch_size = int(vae_encode_batch_size)

        model = instantiate(
            model_cfg,
            model_dtype=_model_dtype(mixed_precision),
            device=device,
        )
        checkpoint_step = _load_complete_g1_checkpoint(model, checkpoint)
        model = model.to(device).eval()

        processor_cfg = OmegaConf.create(
            OmegaConf.to_container(cfg.data.train.processor, resolve=True)
        )
        processor = instantiate(processor_cfg).eval()
        processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats_path)))

        logger.info(
            "Loaded G1 RollingWAM | checkpoint=%s | step=%s | config=%s | stats=%s",
            checkpoint,
            checkpoint_step,
            loaded_config or f"task:{task_config}",
            stats_path,
        )
        return cls(
            model=model,
            processor=processor,
            video_size=tuple(cfg.data.train.video_size),
            num_inference_steps=num_inference_steps,
            text_cfg_scale=text_cfg_scale,
            negative_prompt=negative_prompt,
            seed=seed,
            compile_action_infer=compile_action_infer,
            embodiment=embodiment,
            default_instruction=default_instruction,
            control_hz=control_hz,
        )

    def _preprocess_image(self, image: Any) -> torch.Tensor:
        image_np = np.asarray(image)
        if image_np.dtype != np.uint8:
            raise TypeError(f"{IMAGE_KEY} must be uint8 RGB, got {image_np.dtype}.")
        if image_np.ndim != 3 or image_np.shape[-1] != 3:
            raise ValueError(
                f"{IMAGE_KEY} must be HWC RGB with shape [H,W,3], got {image_np.shape}."
            )

        frame = torch.from_numpy(np.ascontiguousarray(image_np)).permute(2, 0, 1).unsqueeze(0)
        transforms = self.processor.val_transforms
        if isinstance(transforms, Mapping):
            transforms = transforms[self._image_meta["key"]]
        if transforms is not None:
            for transform in transforms:
                frame = transform(frame)

        expected = (1, *tuple(int(v) for v in self._image_meta["shape"]))
        if tuple(frame.shape) != expected:
            raise ValueError(
                f"Configured G1 image transforms must produce {expected}, got {tuple(frame.shape)}."
            )

        frame = self._final_resize(frame)
        frame = self._final_crop(frame)
        frame = self._final_normalize(frame)
        if tuple(frame.shape) != (1, 3, *self.video_size):
            raise ValueError(
                f"Final G1 frame must be [1,3,{self.video_size[0]},{self.video_size[1]}], "
                f"got {tuple(frame.shape)}."
            )
        return frame.unsqueeze(2).to(
            device=self.model.device,
            dtype=self.model.torch_dtype,
        )

    def _normalize_state(self, state: Any) -> torch.Tensor:
        state_np = np.asarray(state, dtype=np.float32)
        validate_state(state_np)
        if state_np.ndim != 1:
            raise ValueError(f"{STATE_KEY} must have shape [{STATE_DIM}], got {state_np.shape}.")
        if not np.isfinite(state_np).all():
            raise ValueError(f"{STATE_KEY} contains non-finite values.")

        key = self._state_meta["key"]
        batch = {
            "state": {
                key: torch.from_numpy(np.ascontiguousarray(state_np)).unsqueeze(0)
            }
        }
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        return batch["state"][key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim != 2 or tuple(action.shape) != (
            int(self.model.actions_per_chunk),
            ACTION_DIM,
        ):
            raise ValueError(
                "Model action must have shape "
                f"[{self.model.actions_per_chunk},{ACTION_DIM}], got {tuple(action.shape)}."
            )
        key = self._action_meta["key"]
        normalizer = self.processor.normalizer.normalizers["action"][key]
        action_np = normalizer.backward(
            action.unsqueeze(0).to(device="cpu", dtype=torch.float32)
        )[0].numpy()
        split_action(action_np)
        if not np.isfinite(action_np).all():
            raise ValueError("Model produced non-finite G1 actions.")
        return np.asarray(action_np, dtype=np.float32)

    def _resolve_instruction(self, obs: Mapping[str, Any]) -> str:
        instruction = obs.get("text", self.default_instruction)
        if instruction is None or (isinstance(instruction, str) and not instruction.strip()):
            instruction = self.default_instruction
        if not isinstance(instruction, str):
            raise TypeError(f"text must be a string, got {type(instruction).__name__}.")
        instruction = instruction.strip()
        if not instruction:
            raise ValueError(
                "A non-empty task instruction is required in request['text'] or "
                "via --default-instruction."
            )
        return instruction

    def _validate_embodiment(self, obs: Mapping[str, Any]) -> None:
        requested = obs.get("embodiment_tag")
        if requested is not None and str(requested).lower() != self.embodiment.lower():
            raise ValueError(
                f"This server only hosts {self.embodiment!r}, got embodiment_tag={requested!r}."
            )

    @torch.inference_mode()
    def infer(self, obs: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Infer one native 4x78 G1 action chunk from one latest observation."""
        if not isinstance(obs, Mapping):
            raise TypeError(f"Observation must be a mapping, got {type(obs).__name__}.")
        with self._lock:
            self._validate_embodiment(obs)
            try:
                image = obs["images"][IMAGE_KEY]
                state = obs["states"][STATE_KEY]
            except (KeyError, TypeError) as exc:
                raise KeyError(
                    f"Observation must contain images.{IMAGE_KEY} and states.{STATE_KEY}."
                ) from exc

            instruction = self._resolve_instruction(obs)
            if self._active_instruction is not None and instruction != self._active_instruction:
                logger.info("Instruction changed; resetting RollingWAM stream state.")
                self._reset_unlocked()

            new_frames = self._preprocess_image(image)
            proprio = self._normalize_state(state)
            prompt = DEFAULT_PROMPT.format(task=instruction)
            prediction = self.model.rolling_act(
                new_frames=new_frames,
                prompt=prompt,
                proprio=proprio,
                negative_prompt=self.negative_prompt,
                text_cfg_scale=self.text_cfg_scale,
                seed=self.seed,
                num_inference_steps=self.num_inference_steps,
                compile_action_infer=self.compile_action_infer,
            )
            self._active_instruction = instruction
            return {ACTION_KEY: self._denormalize_action(prediction["action"])}

    def _reset_unlocked(self) -> None:
        self.model.rolling_reset()
        self._active_instruction = None

    def reset(self) -> None:
        """Clear all rolling state. Call at every episode boundary."""
        with self._lock:
            self._reset_unlocked()

    def server_metadata(self) -> dict[str, Any]:
        horizon = int(self.model.actions_per_chunk)
        return {
            "embodiments": {
                self.embodiment: {
                    "image_keys": [IMAGE_KEY],
                    "state_keys": {STATE_KEY: STATE_DIM},
                    "action_keys": {ACTION_KEY: ACTION_DIM},
                    "action_horizon": horizon,
                }
            },
            "default_embodiment": self.embodiment,
            "image": {"dtype": "uint8", "layout": "HWC", "channels": "RGB"},
            "state": {"dtype": "float32"},
            "action": {
                "dtype": "float32",
                "shape": "(action_horizon, dim_per_key)",
                "space": "unnormalized",
            },
            "control": {
                "fps": self.control_hz,
                "execute_horizon": horizon,
                "replan_after_actions": horizon,
            },
        }
