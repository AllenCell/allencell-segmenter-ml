from abc import abstractmethod
from pathlib import Path
from typing import Optional

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.publisher import Publisher


class IExperimentsModel(Publisher):
    """
    Interface for implementing and testing ExperimentsModel
    """

    def __init__(self) -> None:
        super().__init__()

        # state
        self._experiment_name: Optional[str] = None

    def get_experiment_name(self) -> Optional[str]:
        """
        Gets experiment name
        """
        return self._experiment_name

    def apply_experiment_name(self, name: Optional[str]) -> None:
        """
        Sets experiment name

        name (str): name of cyto-dl experiment
        """
        print(f"applying {name} experiment")
        self._experiment_name = name
        self.dispatch(Event.ACTION_EXPERIMENT_APPLIED)

    @abstractmethod
    def get_best_ckpt(self) -> Optional[Path]:
        pass

    @abstractmethod
    def get_experiments(self) -> list[str]:
        pass

    @abstractmethod
    def refresh_experiments(self) -> None:
        pass

    @abstractmethod
    def get_user_experiments_path(self) -> Optional[Path]:
        pass

    @abstractmethod
    def get_model_checkpoints_path(
        self, experiment_name: Optional[str], checkpoint: Optional[str]
    ) -> Path:
        pass

    @abstractmethod
    def get_metrics_csv_path(self) -> Optional[Path]:
        pass

    @abstractmethod
    def get_latest_metrics_csv_version(self) -> int:
        pass

    @abstractmethod
    def get_latest_metrics_csv_path(self) -> Optional[Path]:
        pass

    @abstractmethod
    def get_csv_path(self) -> Optional[Path]:
        pass

    @abstractmethod
    def get_cache_dir(self) -> Optional[Path]:
        pass

    @abstractmethod
    def get_channel_selection_path(self) -> Optional[Path]:
        pass
