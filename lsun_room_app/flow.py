"""Aqueduct Flow."""
from pathlib import Path
from typing import Optional, Union

import numpy as np
from aqueduct import (
    BaseTask,
    BaseTaskHandler,
    Flow,
    FlowStep,
)
from PIL import Image

from .lsun_room_pipeline import LsunRoomEstimator
from .producer import default_producer


class Task(BaseTask):
    """Aqueduct task that stores input and prediction."""

    def __init__(
            self: 'Task',
            image: bytes,
            with_blend: bool
    ) -> None:
        super().__init__()
        self.input: Union[bytes, np.ndarray, Image.Image] = image
        self.pred: Union[np.ndarray, bytes] = None
        self.with_blend: bool = with_blend


class ImageLoaderHandler(BaseTaskHandler):
    def __init__(self: 'ImageLoaderHandler') -> None:
        self._model = default_producer.get_data_loader()

    def handle(self: 'ImageLoaderHandler', *tasks: Task) -> None:
        for task in tasks:
            # bytes -> Image.Image
            task.input = self._model.process(task.input)


class LsunRoomEstimatorHandler(BaseTaskHandler):
    def __init__(
            self: 'LsunRoomEstimatorHandler',
            weights_path: Optional[Path] = None
    ) -> None:
        self._model: Optional[LsunRoomEstimator] = None
        self._weights_path = weights_path

    def on_start(self: 'LsunRoomEstimatorHandler') -> None:
        self._model = default_producer.get_lsun_room_model(
            self._weights_path)

    def handle(self: 'LsunRoomEstimatorHandler', *tasks: Task) -> None:
        preds = self._model.process_list(
            data=[(task.input, task.with_blend) for task in tasks],
        )
        for pred, task in zip(preds, tasks):
            task.pred = pred
            task.input = None


class LsunRoomProcessHandler(BaseTaskHandler):
    def __init__(self: 'LsunRoomProcessHandler') -> None:
        self._model = default_producer.get_post_proc()

    def handle(self: 'LsunRoomProcessHandler', *tasks: Task) -> None:
        for task in tasks:
            task.pred = self._model.process(task.pred)


def get_flow(weights_path: Path) -> Flow:
    return Flow(
        FlowStep(ImageLoaderHandler(), nprocs=4),
        FlowStep(LsunRoomEstimatorHandler(weights_path), nprocs=1),
        FlowStep(LsunRoomProcessHandler(), nprocs=4),
        metrics_enabled=False,
        mp_start_method='spawn',
    )
