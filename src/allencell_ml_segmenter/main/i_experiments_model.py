from abc import ABC, abstractmethod
from pathlib import Path

from allencell_ml_segmenter.core.publisher import Publisher


class IExperimentsModel(Publisher):

    """
    Interface for implementing and testing ExperimentsModel
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_experiment_name(self) -> str:
        pass

    @abstractmethod
    def set_experiment_name(self, name: str) -> None:
        pass

    @abstractmethod
    def get_checkpoint(self) -> str:
        pass

    @abstractmethod
    def set_checkpoint(self, checkpoint: str):
        pass

    @abstractmethod
    def get_experiments(self):
        pass

    @abstractmethod
    def refresh_experiments(self):
        pass

    @abstractmethod
    def refresh_checkpoints(self, experiment: str):
        pass

    @abstractmethod
    def get_user_experiments_path(self):
        pass

    @abstractmethod
    def get_model_checkpoints_path(
        self, experiment_name: str, checkpoint: str
    ) -> Path:
        pass
