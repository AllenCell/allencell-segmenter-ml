from abc import ABC, abstractmethod
from pathlib import Path


class IExperimentsModel(ABC):

    """
    Interface for implementing and testing ExperimentsModel
    """

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
