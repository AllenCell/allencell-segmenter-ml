from abc import abstractmethod
from pathlib import Path
from typing import List, Optional

from allencell_ml_segmenter.core.publisher import Publisher


class IExperimentsModel(Publisher):
    """
    Interface for implementing and testing ExperimentsModel
    """

    def __init__(self):
        super().__init__()

        # state
        self._experiment_name: Optional[str] = None


    def set_experiment_name_selection(self, name: Optional[str]) -> None:
        """
        Sets experiment name
        """
        self._experiment_name_selection = name

    def get_experiment_name_selection(self) -> Optional[str]:
        """
        Gets experiment name
        """
        return self._experiment_name_selection

    def get_experiment_name(self) -> Optional[str]:
        """
        Gets experiment name
        """
        return self._experiment_name

    def set_experiment_name(self, name: Optional[str]) -> None:
        """
        Sets experiment name

        name (str): name of cyto-dl experiment
        """
        self._experiment_name = name

    @abstractmethod
    def get_checkpoint(self) -> str:
        pass

    @abstractmethod
    def get_experiments(self) -> List[str]:
        pass

    @abstractmethod
    def refresh_experiments(self):
        pass

    @abstractmethod
    def get_user_experiments_path(self):
        pass

    @abstractmethod
    def get_model_checkpoints_path(
        self, experiment_name: str, checkpoint: str
    ) -> Path:
        pass
