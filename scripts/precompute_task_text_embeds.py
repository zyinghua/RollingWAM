"""Per-task text embedding caches for selected-task training.

Selected-task training identifies an episode's task by looking up the episode's
instruction inside ``<cache_root>/<task_name>/``, so each task needs its own
cache directory (``scripts/precompute_text_embeds.py`` writes one flat directory
for whole-dataset training instead).

The released archive carries no task labels, so a task's instructions are derived
from its aligned 550-episode block and mapped to a task name with RoboTwin's
instruction templates (``description/task_instruction/<task>.json``). Templates
are not unique across tasks, so ``encode`` refuses a weak or contested match.

Modes:
- ``list``     inspect every block: matched task, scores, instruction samples.
- ``validate`` diff derived filenames against existing cache directories.
- ``encode``   encode the requested tasks into ``<cache_root>/<task_name>/``.

``list`` and ``validate`` read metadata only (no GPU, no model download).

Examples:
  python scripts/precompute_task_text_embeds.py task=robotwin_selected_tasks_rolling_3cam_384_1e-4 +mode=list
  python scripts/precompute_task_text_embeds.py task=robotwin_selected_tasks_rolling_3cam_384_1e-4 +mode=validate
  python scripts/precompute_task_text_embeds.py task=robotwin_selected_tasks_rolling_3cam_384_1e-4 +mode=encode +tasks=[click_alarmclock]
"""

import json
import logging
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, ListConfig
from tqdm import tqdm

from rollingwam.datasets.lerobot.base_lerobot_dataset import (
    ROBOTWIN_FASTWAM_EPISODES_PER_TASK,
    ROBOTWIN_FASTWAM_TOTAL_EPISODES,
)
from rollingwam.datasets.lerobot3.lerobot_dataset import LeRobotDatasetMetadata as LeRobotV30Metadata
from rollingwam.datasets.lerobot.text_cache import (
    DEFAULT_PROMPT,
    DEFAULT_TEXT_ENCODER_ID,
    text_embedding_cache_filename,
)
from rollingwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from rollingwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
from rollingwam.utils.config_resolvers import register_default_resolvers
from rollingwam.utils.logging_config import get_logger, setup_logging

register_default_resolvers()
logger = get_logger(__name__)

DEFAULT_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
DEFAULT_TOKENIZER_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
DEFAULT_BATCH_SIZE = 16
MODES = ("list", "validate", "encode")
V21_META_FILES = ("episodes.jsonl", "episodes_stats.jsonl", "tasks.jsonl")
# Templates overlap between related tasks, so a match must be strong and not a near-tie:
# the true task's own templates generate its instructions, so a clearly lower runner-up is
# evidence the winner is right, whereas a tie means the winner was picked arbitrarily.
DEFAULT_MIN_MATCH_SCORE = 0.5
DEFAULT_RUNNER_UP_MARGIN = 0.9
SAMPLE_INSTRUCTIONS = 2


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse bool value: {value!r}")


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, ListConfig)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _model_id_to_enc_id(model_id: str) -> str:
    base = str(model_id).split("/")[-1]
    return re.sub(r"[^a-z0-9]+", "", base.lower()) or "textenc"


def _atomic_torch_save(payload: dict[str, torch.Tensor], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    torch.save(payload, str(tmp_path))
    os.replace(tmp_path, output_path)


# ---------------------------------------------------------------- archive metadata


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path} line {line_number} must contain a JSON object.")
        rows.append(row)
    return rows


def _flatten_numbers(value: Any) -> list[int]:
    if isinstance(value, list):
        return [number for item in value for number in _flatten_numbers(item)]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [int(value)]
    return []


def _load_archive(dataset_dir: Path):
    """Read local v2.1 or v3.0 metadata without decoding video or downloading files."""
    meta_root = dataset_dir / "meta"
    if all((meta_root / name).is_file() for name in V21_META_FILES):
        episodes = {
            row["episode_index"]: row for row in _read_jsonl(meta_root / "episodes.jsonl")
        }
        tasks_by_index = {
            row["task_index"]: row["task"] for row in _read_jsonl(meta_root / "tasks.jsonl")
        }
        episode_stats = {
            row["episode_index"]: row.get("stats", {})
            for row in _read_jsonl(meta_root / "episodes_stats.jsonl")
        }
    elif (meta_root / "tasks.parquet").is_file() and (meta_root / "episodes").is_dir():
        metadata = LeRobotV30Metadata(repo_id=str(dataset_dir), root=dataset_dir)
        episodes = metadata.episodes
        tasks_by_index = metadata.tasks
        episode_stats = metadata.episodes_stats
    else:
        raise FileNotFoundError(
            f"Unsupported or incomplete LeRobot metadata under {meta_root}; expected the "
            "v2.1 JSONL files or v3.0 tasks/episode parquet metadata."
        )
    if len(episodes) != ROBOTWIN_FASTWAM_TOTAL_EPISODES:
        raise ValueError(
            "Per-task derivation requires the released RoboTwin layout with "
            f"{ROBOTWIN_FASTWAM_TOTAL_EPISODES} episodes, got {len(episodes)}."
        )
    return episodes, tasks_by_index, episode_stats


def _episode_instructions(
    episode_index: int,
    episode_stats: dict[int, dict[str, Any]],
    tasks_by_index: dict[int, str],
) -> tuple[set[str], bool]:
    """Instructions an episode's frames can carry, from its ``task_index`` stats.

    The runtime prompt comes from the per-frame ``task_index``, and the stats only
    record min/max, so the inclusive range is used to stay a superset. The coarse
    ``tasks`` field is deliberately not used as a fallback: it also holds coarse and
    quality annotations, which are shared across tasks and would poison selection.
    """
    stats = episode_stats.get(episode_index)
    task_index_stats = stats.get("task_index") if isinstance(stats, dict) else None
    if not isinstance(task_index_stats, dict):
        raise ValueError(
            f"Episode {episode_index} has no `task_index` statistics in "
            "episode metadata; cannot resolve its instruction."
        )

    low = _flatten_numbers(task_index_stats.get("min"))
    high = _flatten_numbers(task_index_stats.get("max"))
    if not low or not high:
        raise ValueError(f"Episode {episode_index} has empty `task_index` min/max statistics.")

    indices = range(min(low), max(high) + 1)
    unknown = [index for index in indices if index not in tasks_by_index]
    if unknown:
        raise ValueError(
            f"Episode {episode_index} references task indices absent from task metadata: {unknown[:5]}."
        )
    return {tasks_by_index[index] for index in indices}, len(indices) > 1


def _load_task_blocks(dataset_dir: Path) -> list[dict[str, Any]]:
    """Split the released archive into its aligned per-task episode blocks."""
    episodes, tasks_by_index, episode_stats = _load_archive(dataset_dir)

    blocks = []
    multi_instruction_episodes = 0
    for block_index in range(len(episodes) // ROBOTWIN_FASTWAM_EPISODES_PER_TASK):
        start = block_index * ROBOTWIN_FASTWAM_EPISODES_PER_TASK
        instructions: set[str] = set()
        for episode_index in range(start, start + ROBOTWIN_FASTWAM_EPISODES_PER_TASK):
            if episode_index not in episodes:
                raise ValueError(f"Episode {episode_index} missing from episode metadata.")
            episode_instructions, spans_multiple = _episode_instructions(
                episode_index, episode_stats, tasks_by_index
            )
            if not episode_instructions:
                raise ValueError(f"Episode {episode_index} resolved to no instruction.")
            multi_instruction_episodes += int(spans_multiple)
            instructions |= episode_instructions
        blocks.append({"index": block_index, "start": start, "instructions": sorted(instructions)})

    if multi_instruction_episodes:
        logger.warning(
            "%d episode(s) span more than one task_index; their whole index range was cached.",
            multi_instruction_episodes,
        )
    return blocks


# ---------------------------------------------------------------- task identification


def _template_patterns(task_instruction_dir: Path) -> dict[str, list[re.Pattern]]:
    """Compile RoboTwin instruction templates, treating {A}/{a} as wildcards."""
    patterns: dict[str, list[re.Pattern]] = {}
    for path in sorted(task_instruction_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        compiled = []
        for key in ("seen", "unseen"):
            for template in payload.get(key, []):
                escaped = re.escape(str(template).strip())
                wildcarded = re.sub(r"\\\{[A-Za-z]\\\}", ".+?", escaped)
                compiled.append(re.compile(f"^{wildcarded}$", re.IGNORECASE))
        if compiled:
            patterns[path.stem] = compiled
    if not patterns:
        raise FileNotFoundError(f"No instruction templates found under {task_instruction_dir}.")
    return patterns


def _match_blocks_to_tasks(
    blocks: list[dict[str, Any]],
    patterns: dict[str, list[re.Pattern]],
) -> dict[int, dict[str, Any]]:
    """Score each block against every task's templates and keep the ranking."""
    matches = {}
    for block in blocks:
        scores = {}
        for task_name, task_patterns in patterns.items():
            hits = sum(
                1
                for instruction in block["instructions"]
                if any(pattern.match(instruction) for pattern in task_patterns)
            )
            if hits:
                scores[task_name] = hits / len(block["instructions"])
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        matches[block["index"]] = {
            "task_name": ranked[0][0] if ranked else None,
            "score": ranked[0][1] if ranked else 0.0,
            "runner_up_name": ranked[1][0] if len(ranked) > 1 else None,
            "runner_up": ranked[1][1] if len(ranked) > 1 else 0.0,
        }
    return matches


def _match_is_confident(match: dict[str, Any], min_score: float, margin: float) -> bool:
    return (
        match["task_name"] is not None
        and match["score"] >= min_score
        and match["runner_up"] < margin * match["score"]
    )


def _blocks_by_task(
    blocks: list[dict[str, Any]],
    matches: dict[int, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[int]]]:
    """Invert the mapping, collecting (rather than raising on) contested task names."""
    candidates: dict[str, list[dict[str, Any]]] = {}
    for block in blocks:
        task_name = matches[block["index"]]["task_name"]
        if task_name is not None:
            candidates.setdefault(task_name, []).append({**block, **matches[block["index"]]})

    by_task = {name: entries[0] for name, entries in candidates.items() if len(entries) == 1}
    contested = {
        name: [entry["index"] for entry in entries]
        for name, entries in candidates.items()
        if len(entries) > 1
    }
    return by_task, contested


def _shared_instructions(blocks: list[dict[str, Any]]) -> dict[str, list[int]]:
    owners: dict[str, list[int]] = {}
    for block in blocks:
        for instruction in block["instructions"]:
            owners.setdefault(instruction, []).append(block["index"])
    return {
        instruction: block_indices
        for instruction, block_indices in owners.items()
        if len(block_indices) > 1
    }


def _derived_filenames(instructions: list[str], context_len: int) -> set[str]:
    return {
        text_embedding_cache_filename(
            DEFAULT_PROMPT.format(task=instruction), context_len=context_len
        )
        for instruction in instructions
    }


# ---------------------------------------------------------------- modes


def _run_list(
    blocks: list[dict[str, Any]],
    matches: dict[int, dict[str, Any]],
    contested: dict[str, list[int]],
    min_score: float,
    margin: float,
):
    logger.info(
        "%-6s %-9s %-6s %-26s %-6s %-26s %-6s %s",
        "block", "episodes", "instr", "task", "score", "runner-up", "score", "sample",
    )
    for block in blocks:
        match = matches[block["index"]]
        flag = "" if _match_is_confident(match, min_score, margin) else "  <-- LOW CONFIDENCE"
        logger.info(
            "%-6d %-9s %-6d %-26s %-6.2f %-26s %-6.2f %s%s",
            block["index"],
            f"{block['start']}+{ROBOTWIN_FASTWAM_EPISODES_PER_TASK}",
            len(block["instructions"]),
            match["task_name"] or "<unmatched>",
            match["score"],
            match["runner_up_name"] or "-",
            match["runner_up"],
            "; ".join(block["instructions"][:SAMPLE_INSTRUCTIONS])[:60],
            flag,
        )

    if contested:
        logger.warning("Task names matched by more than one block (not encodable): %s", contested)
    shared = _shared_instructions(blocks)
    if shared:
        logger.warning(
            "%d instruction(s) appear in multiple blocks; encoding a task that owns one of "
            "them would make task selection ambiguous.",
            len(shared),
        )
        for instruction, block_indices in list(shared.items())[:5]:
            logger.warning("  blocks %s share: %r", block_indices, instruction)
    else:
        logger.info("No instruction is shared across blocks; task identity is unambiguous.")


def _run_validate(
    blocks: list[dict[str, Any]],
    by_task: dict[str, dict[str, Any]],
    task_names: list[str],
    cache_root: Path,
    context_len: int,
) -> bool:
    ok = True
    for task_name in task_names:
        cache_dir = cache_root / task_name
        if task_name not in by_task:
            logger.error("%-26s no confident block match; run `+mode=list`.", task_name)
            ok = False
            continue
        if not cache_dir.is_dir():
            logger.info("%-26s not encoded yet (%s)", task_name, cache_dir)
            continue

        block = by_task[task_name]
        expected = _derived_filenames(block["instructions"], context_len)
        pattern = f"*.t5_len{context_len}.{DEFAULT_TEXT_ENCODER_ID}.pt"
        actual = {path.name for path in cache_dir.rglob(pattern)}
        missing, extra = expected - actual, actual - expected

        if not missing and not extra:
            logger.info(
                "%-26s MATCH (%d files, block %d, score %.2f)",
                task_name, len(expected), block["index"], block["score"],
            )
        else:
            logger.warning(
                "%-26s derived=%d existing=%d missing=%d extra=%d (block %d)",
                task_name, len(expected), len(actual), len(missing), len(extra), block["index"],
            )
            ok = False
    return ok


def _load_text_encoder(model_cfg: DictConfig, context_len: int, device: str):
    model_id = str(model_cfg.get("model_id", DEFAULT_MODEL_ID))
    enc_id = _model_id_to_enc_id(model_id)
    if enc_id != DEFAULT_TEXT_ENCODER_ID:
        raise ValueError(
            f"Encoder id {enc_id!r} does not match the id the dataset looks up "
            f"({DEFAULT_TEXT_ENCODER_ID!r}); the caches would never be found."
        )
    _, text_config, _, tokenizer_config = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=str(model_cfg.get("tokenizer_model_id", DEFAULT_TOKENIZER_MODEL_ID)),
        redirect_common_files=bool(model_cfg.get("redirect_common_files", True)),
    )
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()
    text_encoder = _load_registered_model(
        text_config.path, "wan_video_text_encoder", torch_dtype=torch.bfloat16, device=device
    ).eval()
    tokenizer = HuggingfaceTokenizer(
        name=tokenizer_config.path, seq_len=context_len, clean="whitespace"
    )
    return text_encoder, tokenizer


def _run_encode(
    blocks: list[dict[str, Any]],
    by_task: dict[str, dict[str, Any]],
    task_names: list[str],
    cache_root: Path,
    context_len: int,
    model_cfg: DictConfig,
    *,
    min_score: float,
    margin: float,
    overwrite: bool,
    force: bool,
    device: str,
):
    pattern = f"*.t5_len{context_len}.{DEFAULT_TEXT_ENCODER_ID}.pt"
    shared = _shared_instructions(blocks)
    pending = {}

    for task_name in task_names:
        if task_name not in by_task:
            raise ValueError(
                f"Task {task_name!r} has no confident block match. Run `+mode=list` to see "
                "the block table, scores and contested names."
            )
        block = by_task[task_name]
        logger.info(
            "%s -> block %d (score %.2f, runner-up %s %.2f)",
            task_name, block["index"], block["score"],
            block["runner_up_name"] or "-", block["runner_up"],
        )
        if not _match_is_confident(block, min_score, margin) and not force:
            raise ValueError(
                f"Task {task_name!r} matched block {block['index']} weakly "
                f"(score {block['score']:.2f} < {min_score:.2f} or runner-up "
                f"{block['runner_up_name']!r} at {block['runner_up']:.2f} too close). "
                "Verify with `+mode=list`, then pass `+force=true` to override."
            )

        overlapping = [
            instruction for instruction in block["instructions"] if instruction in shared
        ]
        if overlapping and not force:
            raise ValueError(
                f"Task {task_name!r} owns {len(overlapping)} instruction(s) that also appear in "
                f"other blocks (e.g. {overlapping[0]!r}); caching them would make task selection "
                "ambiguous at training time. Pass `+force=true` only if you understand why."
            )

        cache_dir = cache_root / task_name
        expected = _derived_filenames(block["instructions"], context_len)
        if cache_dir.is_dir():
            unexpected = {path.name for path in cache_dir.rglob(pattern)} - expected
            if unexpected and not force:
                raise ValueError(
                    f"{cache_dir} already holds {len(unexpected)} file(s) outside this task's "
                    "derived set (stale or foreign cache). Inspect with `+mode=validate`, then "
                    "clean it up or pass `+force=true`."
                )
        pending[task_name] = (block, cache_dir, expected)

    text_encoder, tokenizer = _load_text_encoder(model_cfg, context_len, device)

    for task_name, (block, cache_dir, expected) in pending.items():
        cache_dir.mkdir(parents=True, exist_ok=True)
        prompts = [
            DEFAULT_PROMPT.format(task=instruction) for instruction in block["instructions"]
        ]
        if not overwrite:
            prompts = [
                prompt
                for prompt in prompts
                if not (
                    cache_dir / text_embedding_cache_filename(prompt, context_len=context_len)
                ).exists()
            ]
        logger.info(
            "Encoding %d/%d prompts for %s into %s",
            len(prompts), len(block["instructions"]), task_name, cache_dir,
        )

        over_length = 0
        with torch.no_grad():
            for start in tqdm(
                range(0, len(prompts), DEFAULT_BATCH_SIZE),
                desc=task_name, unit="batch", dynamic_ncols=True,
            ):
                batch_prompts = prompts[start : start + DEFAULT_BATCH_SIZE]
                ids, mask = tokenizer(batch_prompts, return_mask=True, add_special_tokens=True)
                ids = ids.to(device)
                mask = mask.to(device=device, dtype=torch.bool)
                over_length += int(mask.all(dim=1).sum().item())
                context = text_encoder(ids, mask)
                for i, prompt in enumerate(batch_prompts):
                    payload = {
                        "context": context[i].detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
                        "mask": mask[i].detach().to(device="cpu", dtype=torch.bool).contiguous(),
                    }
                    _atomic_torch_save(
                        payload,
                        cache_dir / text_embedding_cache_filename(prompt, context_len=context_len),
                    )
        if over_length:
            logger.warning(
                "%d prompt(s) for %s filled the whole %d-token window and may be truncated.",
                over_length, task_name, context_len,
            )

        written = {path.name for path in cache_dir.rglob(pattern)}
        incomplete = expected - written
        if incomplete:
            raise RuntimeError(
                f"{cache_dir} is missing {len(incomplete)} of {len(expected)} derived files "
                "after encoding; the cache is incomplete and would silently drop episodes."
            )
        logger.info("%s complete: %d files in %s", task_name, len(expected), cache_dir)


# ---------------------------------------------------------------- config resolution


def _resolve_dataset_dir(data_cfg: DictConfig) -> Path:
    dataset_dirs = data_cfg.train.get("dataset_dirs")
    if not dataset_dirs:
        raise ValueError("`data.train.dataset_dirs` is required.")
    if len(dataset_dirs) != 1:
        raise ValueError(
            "Per-task derivation expects exactly one dataset dir (the released RoboTwin "
            f"archive), got {len(dataset_dirs)}."
        )
    dataset_dir = Path(str(dataset_dirs[0]))
    meta_root = dataset_dir / "meta"
    has_v21 = all((meta_root / name).is_file() for name in V21_META_FILES)
    has_v30 = (meta_root / "tasks.parquet").is_file() and (meta_root / "episodes").is_dir()
    if not has_v21 and not has_v30:
        raise FileNotFoundError(
            f"No complete LeRobot v2.1 or v3.0 metadata found under {meta_root}. "
            "Check `data.train.dataset_dirs` and that the archive is mounted."
        )
    return dataset_dir


def _resolve_cache_root(data_cfg: DictConfig) -> Path:
    cache_root = data_cfg.train.get("task_text_embedding_cache_root") or data_cfg.get(
        "text_embedding_cache_root"
    )
    if not cache_root:
        raise ValueError(
            "No per-task cache root found. Use a selected-task data config, or pass "
            "data.train.task_text_embedding_cache_root=/path/to/text_embeds_cache."
        )
    return Path(str(cache_root))


def _resolve_context_len(data_cfg: DictConfig) -> int:
    context_lens = {
        int(node["context_len"])
        for node in (data_cfg.get("train"), data_cfg.get("val"))
        if node is not None and node.get("context_len") is not None
    }
    if not context_lens:
        raise ValueError("No `context_len` found under `data.train`/`data.val`.")
    if len(context_lens) != 1:
        raise ValueError(f"Dataset nodes disagree on context_len: {sorted(context_lens)}.")
    return next(iter(context_lens))


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    setup_logging(log_level=logging.INFO)

    mode = str(cfg.get("mode", "list"))
    if mode not in MODES:
        raise ValueError(f"`mode` must be one of {list(MODES)}, got {mode!r}.")
    data_cfg = cfg.get("data")
    if data_cfg is None:
        raise ValueError("`data` is required; pass a task config (e.g. task=robotwin_...).")

    # Everything cheap is validated before the archive scan.
    min_score = float(cfg.get("min_match_score", DEFAULT_MIN_MATCH_SCORE))
    margin = float(cfg.get("runner_up_margin", DEFAULT_RUNNER_UP_MARGIN))
    force = _to_bool(cfg.get("force", False))
    overwrite = _to_bool(cfg.get("overwrite", False))
    robotwin_root = Path(
        str(cfg.get("robotwin_root", Path(__file__).resolve().parents[1] / "third_party/RoboTwin"))
    )
    patterns = _template_patterns(robotwin_root / "description" / "task_instruction")
    dataset_dir = _resolve_dataset_dir(data_cfg)
    cache_root = _resolve_cache_root(data_cfg)
    context_len = _resolve_context_len(data_cfg)

    task_names: list[str] = []
    device = "cpu"
    if mode != "list":
        task_names = _as_list(cfg.get("tasks"))
        if not task_names and mode == "validate":
            task_names = _as_list(data_cfg.get("selected_task_names"))
        if not task_names:
            raise ValueError("No tasks given. Pass `+tasks=[task_a,task_b]`.")
        unknown = [name for name in task_names if name not in patterns]
        if unknown:
            raise ValueError(
                f"Unknown RoboTwin task name(s) {unknown}; no instruction template exists for them."
            )
    if mode == "encode":
        if cfg.get("model") is None:
            raise ValueError("`model` is required for encoding; pass a task config.")
        cache_root.mkdir(parents=True, exist_ok=True)
        if not os.access(cache_root, os.W_OK):
            raise PermissionError(f"Cache root is not writable: {cache_root}")
        if torch.cuda.is_available():
            device = "cuda"
        elif _to_bool(cfg.get("allow_cpu", False)):
            logger.warning("No GPU found; encoding on CPU will be very slow.")
        else:
            raise RuntimeError("No GPU available. Pass `+allow_cpu=true` to encode on CPU anyway.")

    logger.info(
        "mode=%s dataset=%s cache_root=%s context_len=%d device=%s",
        mode, dataset_dir, cache_root, context_len, device,
    )

    blocks = _load_task_blocks(dataset_dir)
    matches = _match_blocks_to_tasks(blocks, patterns)
    by_task, contested = _blocks_by_task(blocks, matches)
    logger.info(
        "Derived %d blocks; %d task names matched uniquely, %d contested.",
        len(blocks), len(by_task), len(contested),
    )

    if mode == "list":
        _run_list(blocks, matches, contested, min_score, margin)
        return
    if mode == "validate":
        if not _run_validate(blocks, by_task, task_names, cache_root, context_len):
            sys.exit(1)
        return

    _run_encode(
        blocks, by_task, task_names, cache_root, context_len, cfg.model,
        min_score=min_score, margin=margin, overwrite=overwrite, force=force, device=device,
    )


if __name__ == "__main__":
    main()
