"""Adapter from the common robot dataset pipeline to LeRobot v3.0."""

from rollingwam.datasets.lerobot.base_lerobot_dataset import (
    BaseLerobotDataset as _BaseLerobotDataset,
)

from .lerobot_dataset import LeRobotDatasetMetadata, MultiLeRobotDataset


class BaseLerobotDataset(_BaseLerobotDataset):
    metadata_cls = LeRobotDatasetMetadata
    multi_dataset_cls = MultiLeRobotDataset
