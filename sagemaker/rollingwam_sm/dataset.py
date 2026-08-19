"""RobotVideoDataset variant for read-only, optionally padding-trimmed caches.

Why
---
Two SageMaker-specific problems with the stock ``_get_cached_text_context``:

1. The flat-cache branch does ``cache_dir.mkdir(parents=True, exist_ok=True)``
   on the cache directory — a WRITE. A FastFile channel is a read-only
   mountpoint-s3 mount, so the stock path fails before reading anything.
2. The stock cache stores every T5 context padded to ``context_len`` (128), so
   a 40-token instruction still costs 1.05 MB. Measured over the RoboTwin
   prompts (FastWAM port): mean 40.2 real tokens, i.e. ~69 % padding — 969 GB
   padded vs 305 GB trimmed for the full cache. The mirrored
   ``text_embeds_cache_trimmed`` S3 prefix stores the trimmed layout.

Correctness
-----------
``RobotVideoDataset._get`` immediately does::

    context[~context_mask] = 0.0
    context_mask = torch.ones_like(context_mask)

so the pad positions are zeroed and the mask is discarded. Re-padding a trimmed
context with **zeros** (and a False mask) is therefore bit-identical to the
stock path, and the tensor handed to the model is unchanged.

Both layouts are accepted, so the same code runs against the stock padded cache
(e.g. the per-task selected-tasks caches) and a trimmed one. Selected-tasks
mode is preserved: the per-task filename index built by ``__init__``
(``_selected_text_embedding_cache_paths``) is consulted first, exactly like the
base class.
"""

from __future__ import annotations

from pathlib import Path

import torch

from rollingwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset
from rollingwam.datasets.lerobot.text_cache import text_embedding_cache_filename


class TrimmedTextCacheRobotVideoDataset(RobotVideoDataset):
    """Read-only text-embedding cache lookup; accepts padded or trimmed files."""

    def _get_cached_text_context(self, prompt: str):
        cache_filename = text_embedding_cache_filename(prompt, context_len=self.context_len)

        # Same lookup order as the base class: the per-task index built in
        # __init__ for selected-tasks mode wins; otherwise the flat cache dir —
        # WITHOUT the base class's mkdir (the cache mount is read-only here).
        if self._selected_text_embedding_cache_paths is not None:
            cache_path = self._selected_text_embedding_cache_paths.get(cache_filename)
            searched_location = (
                f"selected task cache directories "
                f"({len(self._selected_text_embedding_cache_paths)} files indexed)"
            )
        else:
            if self.text_embedding_cache_dir is None:
                raise ValueError("text_embedding_cache_dir is not set.")
            cache_path = Path(self.text_embedding_cache_dir) / cache_filename
            searched_location = str(self.text_embedding_cache_dir)

        if cache_path is None or not cache_path.exists():
            raise FileNotFoundError(
                f"Missing text embedding cache {cache_filename} in {searched_location}. "
                "Run scripts/precompute_text_embeds.py first."
            )

        payload = torch.load(str(cache_path), map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()

        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != context_mask.shape[0]:
            raise ValueError(
                f"Cached `context` and `mask` disagree on length: {context.shape[0]} vs "
                f"{context_mask.shape[0]} in {cache_path}"
            )
        if context.shape[0] > self.context_len:
            raise ValueError(
                f"Cached context_len {context.shape[0]} exceeds expected {self.context_len} "
                f"in {cache_path}"
            )

        # The `t5_len<N>` filename component records the context length the
        # embedding is FOR, not how many rows are stored, so padded and trimmed
        # caches are drop-in interchangeable.
        if context.shape[0] < self.context_len:
            stored = context.shape[0]
            padded = torch.zeros(
                (self.context_len, context.shape[1]), dtype=context.dtype
            )
            padded[:stored] = context
            padded_mask = torch.zeros(self.context_len, dtype=torch.bool)
            padded_mask[:stored] = context_mask
            context, context_mask = padded, padded_mask

        return context, context_mask
