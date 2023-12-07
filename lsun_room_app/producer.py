"""Producer for pipeline steps objects."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .lsun_room_pipeline import DataLoader, LsunRoomEstimator, PostProcess


@dataclass
class ModelInfo:
    name: str


PipelineModelsInfo = ModelInfo('LsunRoomEstimation')


class ModelsProducer:
    """Returns models for pipeline steps."""

    def __init__(self: 'ModelsProducer', model_info: ModelInfo) -> None:
        self._model_info = model_info

    def get_lsun_room_model(
            self: 'ModelsProducer',
            weights_path: Optional[Path] = None
    ) -> LsunRoomEstimator:
        return LsunRoomEstimator(weights_path=weights_path)

    def get_data_loader(self: 'ModelsProducer') -> DataLoader:
        return DataLoader()

    def get_post_proc(self: 'ModelsProducer') -> PostProcess:
        return PostProcess()


default_producer = ModelsProducer(PipelineModelsInfo)
