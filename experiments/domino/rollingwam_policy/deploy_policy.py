"""DOMINO-specific RollingWAM policy with control-rate adaptation."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.robotwin.rollingwam_policy.deploy_policy import (  # noqa: E402
    WorldActionRobotWinPolicy,
    _compose_sim_cfg,
    _is_none_like,
    _mixed_precision_to_model_dtype,
    _parse_bool,
    _parse_optional_int,
    _resolve_dataset_stats_path,
    encode_obs,
)
from experiments.domino.action_time_resampling import (  # noqa: E402
    resample_absolute_action_chunk,
    resampled_action_count,
    resampled_path_indices,
)
from rollingwam.utils.video_io import save_mp4  # noqa: E402

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _parse_optional_float(value: Any) -> Optional[float]:
    if _is_none_like(value):
        return None
    return float(value)


class WorldActionDominoPolicy(WorldActionRobotWinPolicy):
    """RollingWAM policy adapted from demonstration time to DOMINO's clock."""

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
        compile_action_infer: bool = False,
        action_source_hz: Optional[float] = None,
        action_target_hz: Optional[float] = None,
    ) -> None:
        self.action_source_hz = (
            None if action_source_hz is None else float(action_source_hz)
        )
        self.action_target_hz = (
            None if action_target_hz is None else float(action_target_hz)
        )
        if (self.action_source_hz is None) != (self.action_target_hz is None):
            raise ValueError(
                "action_source_hz and action_target_hz must either both be set "
                "or both be disabled."
            )
        if self.action_source_hz is not None:
            resampled_action_count(
                1,
                source_hz=self.action_source_hz,
                target_hz=self.action_target_hz,
            )
            if replan_steps is not None:
                raise ValueError(
                    "replan_steps cannot be combined with DOMINO action-rate "
                    "resampling: truncation would discard part of the predicted path."
                )

        super().__init__(
            model_cfg=model_cfg,
            processor_cfg=processor_cfg,
            checkpoint_path=checkpoint_path,
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
            compile_action_infer=compile_action_infer,
        )

        if self.action_source_hz is None:
            self.executed_actions_per_chunk = int(self.model.actions_per_chunk)
            action_rate = "native"
        else:
            self.executed_actions_per_chunk = resampled_action_count(
                int(self.model.actions_per_chunk),
                source_hz=self.action_source_hz,
                target_hz=self.action_target_hz,
            )
            action_rate = f"{self.action_source_hz:g}->{self.action_target_hz:g} Hz"

        logger.info(
            "Initialized WorldActionDominoPolicy | chunk=%d predicted/%d executed "
            "actions | action_rate=%s",
            self.model.actions_per_chunk,
            self.executed_actions_per_chunk,
            action_rate,
        )

    def _fill_action_queue(
        self,
        observation: Dict[str, Any],
        instruction: str,
    ) -> None:
        if self.action_source_hz is None:
            super()._fill_action_queue(observation=observation, instruction=instruction)
            return

        action_chunk = self._infer_action_chunk(
            observation=observation,
            instruction=instruction,
        )
        current_qpos = np.asarray(
            observation["joint_action"]["vector"],
            dtype=action_chunk.dtype,
        )
        action_chunk = resample_absolute_action_chunk(
            current_qpos,
            action_chunk,
            source_hz=self.action_source_hz,
            target_hz=self.action_target_hz,
        )
        for action in action_chunk:
            self.pending_actions.append(np.asarray(action, dtype=np.float32))

    def _flush_imagined_rollout(self) -> None:
        """Save imagined frames on the same clock as executed DOMINO commands."""
        if self.action_source_hz is None:
            super()._flush_imagined_rollout()
            return
        if not self._imagined_latents:
            return

        anchor = self._imagined_anchor
        if anchor is None:
            raise RuntimeError("Imagined rollout is missing its observation anchor.")
        stream = torch.cat(
            [anchor] + [lat.to(device=anchor.device) for lat in self._imagined_latents],
            dim=2,
        )
        with torch.no_grad():
            frames = self.model._decode_latents(stream)

        frames_per_chunk = int(self.model.vae.temporal_downsample_factor) * int(
            self.model.chunk_latents
        )
        expected_frames = 1 + len(self._imagined_latents) * frames_per_chunk
        if len(frames) != expected_frames:
            raise RuntimeError(
                "Decoded imagined-rollout frame count does not match the rolling "
                f"chunks: got {len(frames)}, expected {expected_frames}."
            )

        frame_indices = resampled_path_indices(
            frames_per_chunk,
            source_hz=(
                self.action_source_hz
                * frames_per_chunk
                / int(self.model.actions_per_chunk)
            ),
            target_hz=self.action_target_hz,
        )
        paced_frames = [frames[0]]
        for chunk_index in range(len(self._imagined_latents)):
            chunk_start = chunk_index * frames_per_chunk
            chunk_path = frames[chunk_start : chunk_start + frames_per_chunk + 1]
            paced_frames.extend(chunk_path[index] for index in frame_indices)

        video_path = self.imagined_dir / f"episode{self._imagined_episode_idx}.mp4"
        save_mp4(
            paced_frames,
            str(video_path),
            fps=max(1, int(round(self.action_target_hz))),
        )
        logger.info(
            "Saved DOMINO imagined rollout | %s | chunks=%d",
            video_path,
            len(self._imagined_latents),
        )

        self._imagined_episode_idx += 1
        self._imagined_anchor = None
        self._imagined_latents = []


def get_model(usr_args: Dict[str, Any]) -> WorldActionDominoPolicy:
    cfg = _compose_sim_cfg(
        sim_cfg_path=usr_args.get("sim_cfg_path"),
        sim_cfg_name=usr_args.get("sim_cfg_name"),
        sim_task=usr_args.get("sim_task"),
    )
    cfg.model.vae_encode_batch_size = int(
        usr_args.get(
            "vae_encode_batch_size",
            cfg.model.get("vae_encode_batch_size", 1),
        )
    )
    cfg.model.compile_vae_encode = _parse_bool(
        usr_args.get(
            "compile_vae_encode",
            cfg.model.get("compile_vae_encode", False),
        )
    )

    checkpoint_path = usr_args.get("ckpt_setting")
    if _is_none_like(checkpoint_path):
        raise ValueError("`ckpt_setting` is required and must be a valid checkpoint path.")

    device = str(usr_args.get("device") or cfg.EVALUATION.get("device") or "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; fallback device to cpu.")
        device = "cpu"

    mixed_precision = str(
        usr_args.get("mixed_precision") or cfg.get("mixed_precision", "bf16")
    )
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    dataset_stats_path = _resolve_dataset_stats_path(
        dataset_stats_path=usr_args.get("dataset_stats_path")
    )

    seed = _parse_optional_int(usr_args.get("seed"))
    num_inference_steps = int(
        usr_args.get(
            "num_inference_steps",
            cfg.EVALUATION.get("num_inference_steps", 10),
        )
    )
    text_cfg_scale = float(
        usr_args.get(
            "text_cfg_scale",
            cfg.EVALUATION.get("text_cfg_scale", 1.0),
        )
    )
    negative_prompt = str(
        usr_args.get(
            "negative_prompt",
            cfg.EVALUATION.get("negative_prompt", ""),
        )
    )
    timing_enabled = _parse_bool(
        usr_args.get(
            "timing_enabled",
            cfg.EVALUATION.get("timing_enabled", False),
        )
    )
    replan_steps = _parse_optional_int(
        usr_args.get("replan_steps", cfg.EVALUATION.get("replan_steps"))
    )
    action_source_hz = _parse_optional_float(
        usr_args.get("action_source_hz", cfg.EVALUATION.get("action_source_hz"))
    )
    action_target_hz = _parse_optional_float(
        usr_args.get("action_target_hz", cfg.EVALUATION.get("action_target_hz"))
    )

    save_imagined_rollouts = _parse_bool(
        usr_args.get(
            "save_imagined_rollouts",
            cfg.EVALUATION.get("save_imagined_rollouts", False),
        )
    )
    imagined_dir: Optional[Path] = None
    if save_imagined_rollouts:
        eval_output_dir = usr_args.get("eval_output_dir")
        if _is_none_like(eval_output_dir):
            raise ValueError(
                "save_imagined_rollouts requires `eval_output_dir` to be set."
            )
        imagined_dir = Path(str(eval_output_dir)).expanduser() / "imagined_rollouts"

    return WorldActionDominoPolicy(
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
        action_source_hz=action_source_hz,
        action_target_hz=action_target_hz,
        compile_action_infer=_parse_bool(
            usr_args.get(
                "compile_action_infer",
                cfg.EVALUATION.get("compile_action_infer", False),
            )
        ),
    )


def eval(
    task_env: Any,
    model: WorldActionDominoPolicy,
    observation: Optional[Dict[str, Any]],
) -> None:
    model.step(task_env, encode_obs(observation))


def reset_model(model: WorldActionDominoPolicy) -> None:
    model.reset()
