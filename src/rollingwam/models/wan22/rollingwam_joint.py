from typing import Any, Optional

import torch

from rollingwam.utils.logging_config import get_logger

from .rollingwam import RollingWAM

logger = get_logger(__name__)


class RollingWAMJoint(RollingWAM):
    """RollingWAM variant that rolls video and actions jointly at inference: window
    actions attend all video (including the noisy window) and only their own chunk;
    deployment denoises both windows and emits the front video chunk alongside actions."""

    @torch.no_grad()
    def _build_rolling_attention_mask(
        self,
        ctx_chunks: int,
        win_chunks: int,
        tokens_per_frame: int,
        actions_per_chunk: int,
        device: torch.device,
        video_rollout: bool = True,
    ) -> torch.Tensor:
        mask = super()._build_rolling_attention_mask(
            ctx_chunks=ctx_chunks,
            win_chunks=win_chunks,
            tokens_per_frame=tokens_per_frame,
            actions_per_chunk=actions_per_chunk,
            device=device,
            video_rollout=video_rollout,
        )
        video_chunks = ctx_chunks + (win_chunks if video_rollout else 0)
        video_seq_len = (1 + video_chunks * self.chunk_latents) * tokens_per_frame

        a_chunk = torch.arange(win_chunks, device=device).repeat_interleave(actions_per_chunk)
        mask[video_seq_len:, :video_seq_len] = True
        mask[video_seq_len:, video_seq_len:] = a_chunk.view(-1, 1) == a_chunk.view(1, -1)
        return mask

    @torch.no_grad()
    def rolling_act(
        self,
        new_frames: torch.Tensor,
        prompt: Optional[str] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        seed: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
    ) -> dict[str, Any]:
        """One control step: rolls video and actions jointly. Returns
        {'action': [aspc, action_dim], 'video': predicted front chunk latents}."""
        return self._rolling_act(
            new_frames=new_frames,
            prompt=prompt,
            context=context,
            context_mask=context_mask,
            proprio=proprio,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            seed=seed,
            num_inference_steps=num_inference_steps,
            video_rollout=True,
        )
