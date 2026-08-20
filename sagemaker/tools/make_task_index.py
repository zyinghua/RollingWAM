#!/usr/bin/env python3
"""Build the task-label index for the SageMaker selected-tasks target.

Run this ON THE TRAINING BOX, where the per-task text-embedding cache lives.

    python3 sagemaker/tools/make_task_index.py \
        --cache-root /datasets/robotwin2.0-fastwam/text_embeds_cache \
        --tasks lift_pot beat_block_hammer place_dual_shoes \
                stack_bowls_two blocks_ranking_size stack_blocks_three

It writes into the repo by default, so the index is versioned with the task
list and baked into the training image — no S3 channel, no upload. Rebuild the
image after regenerating it.

Why this exists
---------------
Selected-tasks mode labels each episode by intersecting the episode's prompt
hash against the set of cache filenames under ``<cache_root>/<task_name>/``.
The file SET is the label; the file contents are just embeddings. Shipping the
real directories to S3 would mean tens of thousands of small objects, so we
ship the sets as one small JSON instead and rebuild the directory view as
symlinks inside the container (see ``build_task_cache_tree`` in entry.py).

Only filenames are read — never file contents — so this is fast even on NFS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_ENCODER_ID = "wan22ti2v5b"
# In the repo, so it versions with `selected_task_names` and rides in the image.
DEFAULT_OUT = (
    Path(__file__).resolve().parents[2]
    / "configs/data/robotwin_selected_tasks_text_embeds_cache_index.json"
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-root", required=True,
                   help="Directory holding one subdirectory per task.")
    p.add_argument("--tasks", nargs="+", required=True,
                   help="Task names, matching `selected_task_names` in the data config.")
    p.add_argument("--context-len", type=int, default=128)
    p.add_argument("--encoder-id", default=DEFAULT_ENCODER_ID)
    p.add_argument("--out", default=str(DEFAULT_OUT),
                   help=f"Output path (default: {DEFAULT_OUT}, baked into the image).")
    args = p.parse_args()

    root = Path(args.cache_root)
    if not root.is_dir():
        raise SystemExit(f"--cache-root is not a directory: {root}")

    pattern = f"*.t5_len{args.context_len}.{args.encoder_id}.pt"
    index: dict[str, list[str]] = {}
    for task in args.tasks:
        task_dir = root / task
        if not task_dir.is_dir():
            raise SystemExit(f"Missing task cache directory: {task_dir}")
        names = sorted({path.name for path in task_dir.rglob(pattern)})
        if not names:
            raise SystemExit(f"No {pattern} files under {task_dir}")
        index[task] = names
        print(f"{task}: {len(names)} entries")

    # The per-task sets must be disjoint: a filename in two tasks would make an
    # episode match both, and selection refuses ambiguous matches.
    seen: dict[str, str] = {}
    for task, names in index.items():
        for name in names:
            if name in seen:
                raise SystemExit(
                    f"Filename {name} appears in both {seen[name]!r} and {task!r}; "
                    "the per-task sets must be disjoint or episode selection is ambiguous."
                )
            seen[name] = task

    Path(args.out).write_text(json.dumps(index, indent=0, sort_keys=True))
    print(f"\nwrote {args.out}: {len(index)} tasks, {sum(map(len, index.values()))} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
