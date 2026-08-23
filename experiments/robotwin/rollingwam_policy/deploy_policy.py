import atexit
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rollingwam.datasets.lerobot.processors.rollingwam_processor import RollingWAMProcessor
from rollingwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from rollingwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from rollingwam.utils.config_resolvers import register_default_resolvers
from rollingwam.utils.video_io import save_mp4

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

register_default_resolvers()


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _parse_optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(value)



def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_sim_cfg_name(sim_cfg_path: Optional[str], sim_cfg_name: Optional[str]) -> str:
    configs_root = (PROJECT_ROOT / "configs").resolve()
    if not _is_none_like(sim_cfg_path):
        cfg_path = Path(str(sim_cfg_path)).expanduser().resolve()
        try:
            relative = cfg_path.relative_to(configs_root)
        except ValueError as exc:
            raise ValueError(
                f"`sim_cfg_path` must be under {configs_root}, got: {cfg_path}"
            ) from exc
        return relative.as_posix()

    if _is_none_like(sim_cfg_name):
        return "sim_robotwin.yaml"
    return str(sim_cfg_name)


def _compose_sim_cfg(
    sim_cfg_path: Optional[str],
    sim_cfg_name: Optional[str],
    sim_task: Optional[str],
) -> DictConfig:
    config_name = _resolve_sim_cfg_name(sim_cfg_path=sim_cfg_path, sim_cfg_name=sim_cfg_name)
    configs_root = (PROJECT_ROOT / "configs").resolve()
    overrides = []
    if not _is_none_like(sim_task):
        overrides.append(f"task={str(sim_task)}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def _resolve_dataset_stats_path(dataset_stats_path: Optional[str]) -> Path:
    if _is_none_like(dataset_stats_path):
        raise FileNotFoundError(
            "`dataset_stats_path` is required. "
            "Please pass it from eval entrypoint overrides."
        )
    resolved = Path(str(dataset_stats_path)).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset stats path not found: {resolved}")
    return resolved


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize(size_wh, resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


class WorldActionRobotWinPolicy:
    def __init__(
        self,
        model_cfg: DictConfig,
        processor_cfg: DictConfig,
        checkpoint_path: str,
        dataset_stats_path: Path,
        device: str,
        model_dtype: torch.dtype,
        seed: Optional[int],
        num_inference_steps: int,
        text_cfg_scale: float,
        negative_prompt: str,
        timing_enabled: bool,
        save_imagined_rollouts: bool = False,
        imagined_dir: Optional[Path] = None,
        replan_steps: Optional[int] = None,
    ) -> None:
        model_cfg_copy = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
        model_cfg_copy.load_text_encoder = True

        self.model = instantiate(model_cfg_copy, model_dtype=model_dtype, device=device)
        self.model.load_checkpoint(checkpoint_path)
        self.model = self.model.to(device).eval()

        self.processor: RollingWAMProcessor = instantiate(processor_cfg).eval()
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)

        self.seed = seed
        self.num_inference_steps = int(num_inference_steps)
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.timing_enabled = bool(timing_enabled)

        self.replan_steps = None if replan_steps is None else int(replan_steps)
        if self.replan_steps is not None:
            if int(self.model.window_blocks) != 1:
                raise ValueError(
                    f"replan_steps requires a window_blocks=1 checkpoint (got "
                    f"W={self.model.window_blocks}): the rolling window advances one "
                    "full chunk per replan, so partial execution would desynchronize "
                    "it from the environment."
                )
            if not 0 < self.replan_steps <= int(self.model.actions_per_chunk):
                raise ValueError(
                    f"replan_steps must be in [1, {self.model.actions_per_chunk}], "
                    f"got {self.replan_steps}."
                )
            if save_imagined_rollouts:
                raise ValueError(
                    "save_imagined_rollouts assumes full-chunk execution; consecutive "
                    "chunks overlap in time under replan_steps, so the stitched video "
                    "would be wrong. Disable one of the two."
                )

        self.pending_actions: deque[np.ndarray] = deque()
        self.episode_count = 0
        self.step_count = 0
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}
        self._replan_times: list[float] = []
        if self.timing_enabled:
            atexit.register(self._log_replan_timing)

        if save_imagined_rollouts and imagined_dir is None:
            raise ValueError(
                "save_imagined_rollouts requires `eval_output_dir` so imagined videos "
                "land next to the simulator recordings."
            )
        self.save_imagined_rollouts = bool(save_imagined_rollouts)
        self.imagined_dir = imagined_dir
        self._imagined_anchor: Optional[torch.Tensor] = None
        self._imagined_latents: list[torch.Tensor] = []
        self._imagined_episode_idx = 0
        if self.save_imagined_rollouts:
            self.imagined_dir.mkdir(parents=True, exist_ok=True)
            atexit.register(self._flush_imagined_rollout)

        logger.info(
            "Initialized WorldActionRobotWinPolicy | ckpt=%s | stats=%s | chunk=%d actions | S=%d | exec=%s",
            checkpoint_path,
            dataset_stats_path,
            self.model.actions_per_chunk,
            self.num_inference_steps,
            "full" if self.replan_steps is None else self.replan_steps,
        )

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
        state_key = state_meta[0]["key"]

        state_batch = {"state": {state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}")

        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action key in shape_meta['action'].")

        action_key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
        return denorm.numpy()

    def _build_robotwin_image_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        obs_data = observation["observation"]
        head = _resize_rgb(obs_data["head_camera"]["rgb"], (320, 256))
        left = _resize_rgb(obs_data["left_camera"]["rgb"], (160, 128))
        right = _resize_rgb(obs_data["right_camera"]["rgb"], (160, 128))
        bottom = np.concatenate([left, right], axis=1)
        image = np.concatenate([head, bottom], axis=0)  # [384, 320, 3]

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(
            device=self.model.device,
            dtype=self.model.torch_dtype,
        )
        image_tensor = image_tensor * (2.0 / 255.0) - 1.0
        return image_tensor

    def _infer_action_chunk(self, observation: Dict[str, Any], instruction: str) -> np.ndarray:
        """Condition the rolling window on the latest observation and emit one action chunk."""
        image_tensor = self._build_robotwin_image_tensor(observation)
        state_vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        proprio = self._normalize_state(state_vector)
        new_frames = image_tensor.unsqueeze(2)  # [1, 3, 1, H, W]

        prompt = DEFAULT_PROMPT.format(task=instruction)
        infer_t0 = time.perf_counter() if self.timing_enabled else 0.0
        with torch.no_grad():
            pred = self.model.rolling_act(
                new_frames=new_frames,
                prompt=prompt,
                proprio=proprio,
                negative_prompt=self.negative_prompt,
                text_cfg_scale=self.text_cfg_scale,
                seed=self.seed,
                num_inference_steps=self.num_inference_steps,
            )
        if self.timing_enabled:
            infer_elapsed = time.perf_counter() - infer_t0
            self._timing_rollout["infer_s"] += infer_elapsed
            self._replan_times.append(infer_elapsed)

        if self.save_imagined_rollouts:
            if self._imagined_anchor is None:
                with torch.no_grad():
                    self._imagined_anchor = self.model._encode_video_latents(new_frames).detach()
            self._imagined_latents.append(pred["video"].detach().to(device="cpu"))

        action_chunk = self._denormalize_action(pred["action"])[0]  # [aspc, D]
        return action_chunk

    def _flush_imagined_rollout(self) -> None:
        """Decode the episode's emitted chunk latents in one pass and save the mp4.

        Each frame covers several action steps (dataset temporal subsampling), so frames
        are repeated to match the simulator recording's one-frame-per-action pacing."""
        if not self._imagined_latents:
            return

        anchor = self._imagined_anchor
        stream = torch.cat(
            [anchor] + [lat.to(device=anchor.device) for lat in self._imagined_latents],
            dim=2,
        )
        with torch.no_grad():
            frames = self.model._decode_latents(stream)

        frames_per_chunk = int(self.model.vae.temporal_downsample_factor) * int(self.model.chunk_latents)
        repeat = max(1, self.model.actions_per_chunk // frames_per_chunk)
        paced_frames = [frame for frame in frames for _ in range(repeat)]

        video_path = self.imagined_dir / f"episode{self._imagined_episode_idx}.mp4"
        save_mp4(paced_frames, str(video_path), fps=10)
        logger.info("Saved imagined rollout | %s | chunks=%d", video_path, len(self._imagined_latents))

        self._imagined_episode_idx += 1
        self._imagined_anchor = None
        self._imagined_latents = []

    def _fill_action_queue(self, observation: Dict[str, Any], instruction: str) -> None:
        action_chunk = self._infer_action_chunk(observation=observation, instruction=instruction)
        # rolling executes one full chunk per replan; replan_steps (W=1 only) truncates it
        if self.replan_steps is not None:
            action_chunk = action_chunk[: self.replan_steps]
        for i in range(action_chunk.shape[0]):
            self.pending_actions.append(np.asarray(action_chunk[i], dtype=np.float32))

    def should_request_observation(self) -> bool:
        return not self.pending_actions

    def step(self, task_env, observation: Optional[Dict[str, Any]]) -> None:
        if not self.pending_actions:
            if observation is None:
                raise ValueError(
                    "Observation is required when action queue is empty "
                    "(replan step for rollingwam)."
                )
            instruction = task_env.get_instruction()
            self._fill_action_queue(observation=observation, instruction=instruction)

        if not self.pending_actions:
            logger.warning("No action generated; skip current eval step.")
            return

        action = self.pending_actions.popleft()
        sim_t0 = time.perf_counter() if self.timing_enabled else 0.0
        task_env.take_action(action, action_type="qpos")
        if self.timing_enabled:
            self._timing_rollout["sim_s"] += time.perf_counter() - sim_t0
        self.step_count += 1

    def reset_timing_rollout(self) -> None:
        self._timing_rollout["infer_s"] = 0.0
        self._timing_rollout["sim_s"] = 0.0

    def get_timing_rollout(self) -> Dict[str, float]:
        return {
            "infer_s": float(self._timing_rollout["infer_s"]),
            "sim_s": float(self._timing_rollout["sim_s"]),
        }

    def _log_replan_timing(self) -> None:
        """Episode summary; RoboTwin resets before each episode, so the last one flushes at exit."""
        if not self._replan_times:
            return
        # the first replan runs the full init phase (S passes); steady replans run S/W
        init_s, steady = self._replan_times[0], self._replan_times[1:]
        logger.info(
            "Replan timing | init %.3fs | steady mean %.3fs min %.3fs max %.3fs (n=%d)",
            init_s,
            sum(steady) / len(steady) if steady else float("nan"),
            min(steady) if steady else float("nan"),
            max(steady) if steady else float("nan"),
            len(steady),
        )
        self._replan_times = []

    def reset(self) -> None:
        self.pending_actions.clear()
        if self.save_imagined_rollouts:
            self._flush_imagined_rollout()
        self._log_replan_timing()
        self.model.rolling_reset()
        self.episode_count += 1
        self.step_count = 0
        self.reset_timing_rollout()


def encode_obs(observation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return observation


def get_model(usr_args: Dict[str, Any]):
    sim_cfg_path = usr_args.get("sim_cfg_path")
    sim_cfg_name = usr_args.get("sim_cfg_name")
    sim_task = usr_args.get("sim_task")
    cfg = _compose_sim_cfg(
        sim_cfg_path=sim_cfg_path,
        sim_cfg_name=sim_cfg_name,
        sim_task=sim_task,
    )

    checkpoint_path = usr_args.get("ckpt_setting")
    if _is_none_like(checkpoint_path):
        raise ValueError("`ckpt_setting` is required and must be a valid checkpoint path.")

    device = str(usr_args.get("device") or cfg.EVALUATION.get("device") or "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; fallback device to cpu.")
        device = "cpu"

    mixed_precision = str(usr_args.get("mixed_precision") or cfg.get("mixed_precision", "bf16"))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    dataset_stats_path = _resolve_dataset_stats_path(
        dataset_stats_path=usr_args.get("dataset_stats_path"),
    )

    seed = _parse_optional_int(usr_args.get("seed"))
    num_inference_steps = int(
        usr_args.get("num_inference_steps", cfg.EVALUATION.get("num_inference_steps", 10))
    )
    text_cfg_scale = float(usr_args.get("text_cfg_scale", cfg.EVALUATION.get("text_cfg_scale", 1.0)))
    negative_prompt = str(usr_args.get("negative_prompt", cfg.EVALUATION.get("negative_prompt", "")))
    timing_enabled = _parse_bool(
        usr_args.get("timing_enabled", cfg.EVALUATION.get("timing_enabled", False))
    )
    replan_steps = _parse_optional_int(
        usr_args.get("replan_steps", cfg.EVALUATION.get("replan_steps"))
    )

    save_imagined_rollouts = _parse_bool(
        usr_args.get("save_imagined_rollouts", cfg.EVALUATION.get("save_imagined_rollouts", False))
    )
    imagined_dir: Optional[Path] = None
    if save_imagined_rollouts:
        eval_output_dir = usr_args.get("eval_output_dir")
        if _is_none_like(eval_output_dir):
            raise ValueError("save_imagined_rollouts requires `eval_output_dir` to be set.")
        imagined_dir = Path(str(eval_output_dir)).expanduser() / "imagined_rollouts"

    policy = WorldActionRobotWinPolicy(
        model_cfg=cfg.model,
        processor_cfg=cfg.data.train.processor,
        checkpoint_path=str(checkpoint_path),
        dataset_stats_path=dataset_stats_path,
        device=device,
        model_dtype=model_dtype,
        seed=seed,
        num_inference_steps=num_inference_steps,
        text_cfg_scale=text_cfg_scale,
        negative_prompt=negative_prompt,
        timing_enabled=timing_enabled,
        save_imagined_rollouts=save_imagined_rollouts,
        imagined_dir=imagined_dir,
        replan_steps=replan_steps,
    )
    return policy


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):
    obs = encode_obs(observation)
    model.step(TASK_ENV, obs)


def reset_model(model):
    model.reset()
