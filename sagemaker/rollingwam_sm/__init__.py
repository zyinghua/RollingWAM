"""Bolt-on RollingWAM subclasses used only by the SageMaker path.

Selected through Hydra's ``_target_``, so `src/rollingwam` stays untouched:

    data.train._target_=rollingwam_sm.dataset.TrimmedTextCacheRobotVideoDataset

This package lives under ``sagemaker/`` but is deliberately NOT named
``sagemaker`` — a top-level package by that name would shadow the AWS SageMaker
SDK. ``sagemaker/entry.py`` puts this directory on ``PYTHONPATH`` for the
training process.
"""
