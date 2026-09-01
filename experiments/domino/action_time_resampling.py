"""Utilities for adapting absolute-action trajectories between control rates."""

from __future__ import annotations

import math

import numpy as np


def resampled_action_count(
    num_source_actions: int,
    *,
    source_hz: float,
    target_hz: float,
) -> int:
    """Return the nearest positive number of target-rate commands."""
    if num_source_actions <= 0:
        raise ValueError(
            f"num_source_actions must be positive, got {num_source_actions}."
        )
    if not math.isfinite(source_hz) or source_hz <= 0:
        raise ValueError(f"source_hz must be finite and positive, got {source_hz}.")
    if not math.isfinite(target_hz) or target_hz <= 0:
        raise ValueError(f"target_hz must be finite and positive, got {target_hz}.")

    # floor(x + 0.5) gives conventional nearest-integer rounding rather than
    # Python's ties-to-even behavior. Keep at least the final command.
    return max(1, int(math.floor(num_source_actions * target_hz / source_hz + 0.5)))


def resampled_path_indices(
    num_source_steps: int,
    *,
    source_hz: float,
    target_hz: float,
) -> np.ndarray:
    """Map target steps onto a source path that includes its phase-zero item."""
    if num_source_steps <= 0:
        raise ValueError(
            f"num_source_steps must be positive, got {num_source_steps}."
        )
    output_count = resampled_action_count(
        num_source_steps,
        source_hz=source_hz,
        target_hz=target_hz,
    )
    target_phase = (
        np.arange(1, output_count + 1, dtype=np.float64) * source_hz / target_hz
    )
    np.clip(target_phase, 0.0, float(num_source_steps), out=target_phase)
    target_phase[-1] = float(num_source_steps)
    indices = np.floor(target_phase + 0.5).astype(np.int64)
    np.clip(indices, 0, num_source_steps, out=indices)
    indices[-1] = num_source_steps
    return indices


def resample_absolute_action_chunk(
    current_action: np.ndarray,
    action_chunk: np.ndarray,
    *,
    source_hz: float,
    target_hz: float,
) -> np.ndarray:
    """Time-resample an absolute-action path while preserving its endpoint.

    ``current_action`` is the robot state at phase zero. ``action_chunk`` contains
    consecutive absolute targets sampled at ``source_hz``. The returned targets
    span the complete source trajectory at ``target_hz``; this is intentionally
    different from taking only the first N actions of the chunk.
    """
    current = np.asarray(current_action)
    chunk = np.asarray(action_chunk)
    if current.ndim != 1:
        raise ValueError(f"current_action must be 1-D, got shape {current.shape}.")
    if chunk.ndim != 2:
        raise ValueError(f"action_chunk must be 2-D, got shape {chunk.shape}.")
    if chunk.shape[0] == 0:
        raise ValueError("action_chunk must contain at least one action.")
    if chunk.shape[1] != current.shape[0]:
        raise ValueError(
            "current_action and action_chunk dimensions differ: "
            f"{current.shape[0]} != {chunk.shape[1]}."
        )

    output_count = resampled_action_count(
        chunk.shape[0], source_hz=source_hz, target_hz=target_hz
    )
    if output_count == chunk.shape[0] and source_hz == target_hz:
        return chunk.copy()

    path = np.concatenate([current[None], chunk], axis=0)
    source_phase = np.arange(path.shape[0], dtype=np.float64)
    target_phase = (
        np.arange(1, output_count + 1, dtype=np.float64) * source_hz / target_hz
    )
    np.clip(target_phase, 0.0, float(chunk.shape[0]), out=target_phase)
    target_phase[-1] = float(chunk.shape[0])
    output = np.stack(
        [
            np.interp(target_phase, source_phase, path[:, dimension])
            for dimension in range(path.shape[1])
        ],
        axis=1,
    ).astype(chunk.dtype, copy=False)

    # Make endpoint preservation exact even when interpolation uses float64.
    output[-1] = chunk[-1]
    return output
