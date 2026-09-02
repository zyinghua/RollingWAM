# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Minimal local, read-only reader for the LeRobot v3.0 layout.

LeRobot v3.0 stores multiple episodes in chunked parquet and video files. This
module exposes the subset of the dataset API required by the training pipeline;
dataset creation, conversion, Hub upload, and streaming are intentionally out
of scope.
"""

import json
from itertools import accumulate
from pathlib import Path
from typing import Any, Callable

import datasets
import numpy as np
import packaging.version
import pandas as pd
import pyarrow.dataset as pa_dataset
import torch

from rollingwam.datasets.lerobot.constants import HF_LEROBOT_HOME
from rollingwam.datasets.lerobot.lerobot.datasets.utils import (
    get_hf_features_from_features,
    hf_transform_to_torch,
    unflatten_dict,
)
from rollingwam.datasets.lerobot.lerobot.datasets.video_utils import (
    VideoFrame,
    decode_video_frames,
    get_safe_default_codec,
)

CODEBASE_VERSION = packaging.version.parse("3.0")


def _load_json(path: Path):
    with path.open() as file:
        return json.load(file)


def _load_parquet_dataset(path: Path) -> datasets.Dataset:
    paths = sorted(path.glob("*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found under {path}.")
    return datasets.Dataset.from_parquet([str(item) for item in paths])


def _load_tasks(path: Path) -> tuple[dict[int, str], dict[str, int]]:
    frame = pd.read_parquet(path)
    if "task_index" not in frame.columns:
        raise ValueError(f"LeRobot v3 task metadata has no `task_index` column: {path}")

    if "task" in frame.columns:
        pairs = zip(frame["task_index"].tolist(), frame["task"].tolist(), strict=True)
    else:
        pairs = zip(frame["task_index"].tolist(), frame.index.tolist(), strict=True)

    tasks: dict[int, str] = {}
    for raw_index, raw_task in pairs:
        task_index = int(raw_index)
        task = str(raw_task)
        if task_index in tasks and tasks[task_index] != task:
            raise ValueError(f"Duplicate LeRobot task index {task_index} in {path}.")
        tasks[task_index] = task

    task_to_task_index = {task: task_index for task_index, task in tasks.items()}
    if len(task_to_task_index) != len(tasks):
        raise ValueError(f"Duplicate LeRobot task text in {path}.")
    return tasks, task_to_task_index


def _load_episode_metadata(
    path: Path,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    rows = _load_parquet_dataset(path)
    episodes: dict[int, dict[str, Any]] = {}
    episodes_stats: dict[int, dict[str, Any]] = {}
    for row in rows:
        episode_index = int(row["episode_index"])
        if episode_index in episodes:
            raise ValueError(f"Duplicate LeRobot episode index {episode_index} under {path}.")

        episodes[episode_index] = {
            key: value for key, value in row.items() if not key.startswith("stats/")
        }
        flat_stats = {
            key.removeprefix("stats/"): value
            for key, value in row.items()
            if key.startswith("stats/")
        }
        episodes_stats[episode_index] = unflatten_dict(flat_stats) if flat_stats else {}

    return episodes, episodes_stats


def _episode_data_index(episode_rows: dict[int, dict[str, Any]], episodes: list[int]):
    lengths = [int(episode_rows[episode_index]["length"]) for episode_index in episodes]
    cumulative = list(accumulate(lengths))
    return {
        "from": torch.LongTensor([0] + cumulative[:-1]),
        "to": torch.LongTensor(cumulative),
    }


class LeRobotDatasetMetadata:
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
    ):
        self.repo_id = repo_id
        self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
        self.info = _load_json(self.root / "meta" / "info.json")
        for feature in self.info["features"].values():
            feature["shape"] = tuple(feature["shape"])

        version = packaging.version.parse(self.info["codebase_version"])
        if version != CODEBASE_VERSION:
            raise ValueError(
                f"Expected LeRobot codebase_version v3.0, got {version} at {self.root}."
            )

        self.tasks, self.task_to_task_index = _load_tasks(
            self.root / "meta" / "tasks.parquet"
        )
        self.episodes, self.episodes_stats = _load_episode_metadata(
            self.root / "meta" / "episodes"
        )
        stats_path = self.root / "meta" / "stats.json"
        self.stats = _load_json(stats_path) if stats_path.exists() else None

        if len(self.episodes) != self.total_episodes:
            raise ValueError(
                f"LeRobot metadata declares {self.total_episodes} episodes but "
                f"{len(self.episodes)} rows were loaded from {self.root / 'meta' / 'episodes'}."
            )

    @property
    def _version(self) -> packaging.version.Version:
        return packaging.version.parse(self.info["codebase_version"])

    def get_data_file_path(self, episode_index: int) -> Path:
        episode = self.episodes[int(episode_index)]
        return Path(
            self.info["data_path"].format(
                chunk_index=episode["data/chunk_index"],
                file_index=episode["data/file_index"],
            )
        )

    def get_video_file_path(self, episode_index: int, video_key: str) -> Path:
        episode = self.episodes[int(episode_index)]
        return Path(
            self.info["video_path"].format(
                video_key=video_key,
                chunk_index=episode[f"videos/{video_key}/chunk_index"],
                file_index=episode[f"videos/{video_key}/file_index"],
            )
        )

    @property
    def fps(self) -> int:
        return int(self.info["fps"])

    @property
    def features(self) -> dict[str, dict]:
        return self.info["features"]

    @property
    def image_keys(self) -> list[str]:
        return [key for key, feature in self.features.items() if feature["dtype"] == "image"]

    @property
    def video_keys(self) -> list[str]:
        return [key for key, feature in self.features.items() if feature["dtype"] == "video"]

    @property
    def camera_keys(self) -> list[str]:
        return [
            key
            for key, feature in self.features.items()
            if feature["dtype"] in {"image", "video"}
        ]

    @property
    def total_episodes(self) -> int:
        return int(self.info["total_episodes"])

    @property
    def total_frames(self) -> int:
        return int(self.info["total_frames"])


class LeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        video_backend: str | None = None,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.tolerance_s = tolerance_s
        self.video_backend = video_backend or get_safe_default_codec()
        self.during_training = True
        self.meta = LeRobotDatasetMetadata(repo_id, root=self.root)

        selected_episodes = (
            list(range(self.meta.total_episodes)) if episodes is None else episodes
        )
        self.episodes = sorted(int(episode_index) for episode_index in selected_episodes)
        missing_episodes = [
            episode_index
            for episode_index in self.episodes
            if episode_index not in self.meta.episodes
        ]
        if missing_episodes:
            raise ValueError(
                f"Requested LeRobot episodes are absent from metadata: {missing_episodes[:10]}."
            )

        self.hf_dataset = self._load_hf_dataset()
        self.episode_data_index = _episode_data_index(self.meta.episodes, self.episodes)
        self._episode_id_to_nested_id = {
            episode_id: nested_id for nested_id, episode_id in enumerate(self.episodes)
        }
        self.delta_indices = self._get_delta_indices(delta_timestamps) if delta_timestamps else None

    def _load_hf_dataset(self) -> datasets.Dataset:
        paths = sorted((self.root / "data").glob("*/*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No parquet files found under {self.root / 'data'}.")

        features = get_hf_features_from_features(self.meta.features)
        filter_expression = pa_dataset.field("episode_index").isin(self.episodes)
        dataset = datasets.Dataset.from_parquet(
            [str(path) for path in paths],
            features=features,
            columns=list(features),
            filters=filter_expression,
        )
        dataset.set_transform(hf_transform_to_torch)

        expected_frames = sum(
            int(self.meta.episodes[episode_index]["length"])
            for episode_index in self.episodes
        )
        if len(dataset) != expected_frames:
            raise ValueError(
                f"Selected LeRobot episodes describe {expected_frames} frames but "
                f"{len(dataset)} rows were loaded from {self.root / 'data'}."
            )
        return dataset

    def _get_delta_indices(self, delta_timestamps: dict[str, list[float]]):
        return {
            key: [round(timestamp * self.fps) for timestamp in timestamps]
            for key, timestamps in delta_timestamps.items()
        }

    @property
    def fps(self) -> int:
        return self.meta.fps

    @property
    def num_frames(self) -> int:
        return len(self.hf_dataset)

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    @property
    def features(self) -> dict[str, dict]:
        return self.meta.features

    @property
    def hf_features(self) -> datasets.Features:
        return self.hf_dataset.features

    def _get_query_indices(self, idx: int, episode_index: int):
        nested_id = self._episode_id_to_nested_id[episode_index]
        episode_start = int(self.episode_data_index["from"][nested_id])
        episode_end = int(self.episode_data_index["to"][nested_id])
        query_indices = {
            key: [max(episode_start, min(episode_end - 1, idx + delta)) for delta in deltas]
            for key, deltas in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": torch.BoolTensor(
                [
                    (idx + delta < episode_start) or (idx + delta >= episode_end)
                    for delta in deltas
                ]
            )
            for key, deltas in self.delta_indices.items()
        }
        return query_indices, padding

    def _query_data(self, query_indices: dict[str, list[int]]):
        result = {}
        selected_cache = {}
        for key, indices in query_indices.items():
            if key in self.meta.video_keys:
                continue
            cache_key = tuple(indices)
            if cache_key not in selected_cache:
                selected_cache[cache_key] = self.hf_dataset.select(indices)
            selected = selected_cache[cache_key]
            result[key] = torch.stack(list(selected[key]))
        return result

    def _query_timestamps(self, current_timestamp: float, query_indices):
        timestamps = {}
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                values = self.hf_dataset.select(query_indices[key])["timestamp"]
                timestamps[key] = torch.stack(list(values)).tolist()
            else:
                timestamps[key] = [current_timestamp]
        return timestamps

    def _query_videos(self, query_timestamps, episode_index: int):
        episode = self.meta.episodes[episode_index]
        result = {}
        for video_key, timestamps in query_timestamps.items():
            start_timestamp = float(episode[f"videos/{video_key}/from_timestamp"])
            shifted = [start_timestamp + timestamp for timestamp in timestamps]
            video_path = self.root / self.meta.get_video_file_path(episode_index, video_key)
            result[video_key] = decode_video_frames(
                video_path,
                shifted,
                self.tolerance_s,
                self.video_backend,
            ).squeeze(0)
        return result

    def get_episode_data(self, nested_episode_index: int):
        start = int(self.episode_data_index["from"][nested_episode_index])
        end = int(self.episode_data_index["to"][nested_episode_index])
        selected = self.hf_dataset.select(range(start, end))
        result = {}
        for key in self.meta.features:
            if key in self.meta.video_keys:
                continue
            values = list(selected[key])
            if values and isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values)
            else:
                result[key] = np.asarray(values)
        return result

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx: int):
        item = self.hf_dataset[idx]
        episode_index = int(item["episode_index"])
        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, episode_index)
            item = {**item, **padding, **self._query_data(query_indices)}

        if self.meta.video_keys and self.during_training:
            timestamps = self._query_timestamps(float(item["timestamp"]), query_indices)
            item = {**self._query_videos(timestamps, episode_index), **item}

        if self.image_transforms is not None:
            for camera_key in self.meta.camera_keys:
                item[camera_key] = self.image_transforms(item[camera_key])

        task_index = torch.as_tensor(item["task_index"])
        if task_index.ndim == 0:
            resolved_task_index = int(task_index)
        else:
            item["chunked_task_index"] = task_index
            resolved_task_index = int(task_index[0])
        item["task"] = self.meta.tasks[resolved_task_index]
        item["step_is_qualified"] = True
        return item


class MultiLeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs: list[str],
        episodes: dict[str, list[int]] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerances_s: dict[str, float] | None = None,
        video_backend: str | None = None,
    ):
        super().__init__()
        self.dataset_dirs = dataset_dirs
        self.ds_names = list(dataset_dirs)
        tolerances_s = tolerances_s or dict.fromkeys(dataset_dirs, 1e-4)
        self._datasets = [
            LeRobotDataset(
                dataset_dir,
                root=Path(dataset_dir),
                episodes=episodes[dataset_dir] if episodes else None,
                image_transforms=image_transforms,
                delta_timestamps=delta_timestamps,
                tolerance_s=tolerances_s[dataset_dir],
                video_backend=video_backend,
            )
            for dataset_dir in dataset_dirs
        ]

        fps_values = {dataset.fps for dataset in self._datasets}
        if len(fps_values) != 1:
            raise ValueError(f"All LeRobot datasets must share one FPS, got {sorted(fps_values)}.")

        common_features = set(self._datasets[0].features)
        for dataset in self._datasets[1:]:
            common_features.intersection_update(dataset.features)
        if not common_features:
            raise ValueError("LeRobot datasets have no common features.")
        self.disabled_features = set().union(
            *(set(dataset.features) - common_features for dataset in self._datasets)
        )

    def set_during_training(self, during_training: bool):
        for dataset in self._datasets:
            dataset.during_training = during_training

    @property
    def repo_id_to_index(self):
        return {repo_id: index for index, repo_id in enumerate(self.ds_names)}

    @property
    def fps(self) -> int:
        return self._datasets[0].fps

    @property
    def num_frames(self) -> int:
        return sum(dataset.num_frames for dataset in self._datasets)

    @property
    def num_episodes(self) -> int:
        return sum(dataset.num_episodes for dataset in self._datasets)

    @property
    def video(self) -> bool:
        return bool(self._datasets[0].meta.video_keys)

    @property
    def features(self):
        return {
            key: feature
            for dataset in self._datasets
            for key, feature in dataset.hf_features.items()
            if key not in self.disabled_features
        }

    @property
    def camera_keys(self) -> list[str]:
        return [
            key
            for key, feature in self.features.items()
            if isinstance(feature, (datasets.Image, VideoFrame))
        ]

    def get_episode_data(self, episode_index: int):
        for dataset in self._datasets:
            if episode_index < dataset.num_episodes:
                return dataset.get_episode_data(episode_index)
            episode_index -= dataset.num_episodes
        raise IndexError("Episode index out of bounds.")

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx: int):
        offset = 0
        for dataset_index, dataset in enumerate(self._datasets):
            if idx < offset + dataset.num_frames:
                item = dataset[idx - offset]
                item["dataset_index"] = torch.tensor(dataset_index)
                for key in self.disabled_features:
                    item.pop(key, None)
                return item
            offset += dataset.num_frames
        raise IndexError(f"Index {idx} out of bounds for {self.num_frames} frames.")
