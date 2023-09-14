from abc import ABC, abstractmethod


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
