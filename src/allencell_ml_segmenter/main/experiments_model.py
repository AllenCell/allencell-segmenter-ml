from pathlib import Path
from typing import Optional
from allencell_ml_segmenter.config.i_user_settings import IUserSettings
from allencell_ml_segmenter.utils.experiment_utils import ExperimentUtils

import copy

from allencell_ml_segmenter.main.i_experiments_model import IExperimentsModel


class ExperimentsModel(IExperimentsModel):
    def __init__(self, config: IUserSettings) -> None:
        super().__init__()
        self.user_settings: IUserSettings = config

        # options
        self.experiments: list[str] = []
        self.refresh_experiments()

    def refresh_experiments(self) -> None:
        # TODO: make a FileUtils method for this?
        self.experiments = []
        user_exp_path: Optional[Path] = (
            self.user_settings.get_user_experiments_path()
        )
        if user_exp_path is not None:
            for experiment in user_exp_path.iterdir():
                if (
                    experiment.is_dir()
                    and self._is_cyto_dl_experiment(experiment)
                    and experiment not in self.experiments
                    and not experiment.name.startswith(".")
                ):
                    self.experiments.append(experiment.name)
            self.experiments.sort()

    """
    Returns a defensive copy of Experiments list.
    """

    def _is_cyto_dl_experiment(self, experiment: Path) -> bool:
        # Heuristic for checking if dir is a cyto-dl experiment
        csv_path: Path = experiment / "data" / "train.csv"
        checkpoints_path: Path = experiment / "checkpoints"
        return checkpoints_path.exists() or csv_path.exists()

    def get_experiments(self) -> list[str]:
        return copy.deepcopy(self.experiments)

    def get_user_settings(self) -> IUserSettings:
        return self.user_settings

    def get_user_experiments_path(self) -> Optional[Path]:
        return self.get_user_settings().get_user_experiments_path()

    # TODO: possibly refactor to get rid of exp name and checkpoint params?
    def get_model_checkpoints_path(
        self, experiment_name: Optional[str], checkpoint: Optional[str]
    ) -> Path:
        """
        Gets checkpoints for model path
        """
        if experiment_name is None:
            raise ValueError(
                "Experiment name cannot be None in order to get model_checkpoint_path"
            )

        if checkpoint is None:
            raise ValueError(
                "Checkpoint cannot be None in order to get model_checkpoint_path"
            )

        user_exp_path: Optional[Path] = self.get_user_experiments_path()
        if user_exp_path is None:
            raise ValueError("User experiments path cannot be None")

        return user_exp_path / experiment_name / "checkpoints" / checkpoint

    def _get_exp_path(self) -> Optional[Path]:
        user_exp_path: Optional[Path] = self.get_user_experiments_path()
        exp_name: Optional[str] = self.get_experiment_name()
        if exp_name is not None and user_exp_path is not None:
            return user_exp_path / exp_name
        return None

    def get_csv_path(self) -> Optional[Path]:
        exp_path: Optional[Path] = self._get_exp_path()
        if exp_path is not None:
            return exp_path / "data"
        return None

    def get_metrics_csv_path(self) -> Optional[Path]:
        exp_path: Optional[Path] = self._get_exp_path()
        if exp_path is not None:
            return exp_path / "csv"
        return None

    def get_cache_dir(self) -> Optional[Path]:
        exp_path: Optional[Path] = self._get_exp_path()
        if exp_path is not None:
            return exp_path / "cache"
        return None

    def get_latest_metrics_csv_version(self) -> int:
        """
        Returns version number of the most recent version directory within
        the cyto-dl CSV folder (self._csv_path) or -1 if no version directories
        exist
        """
        last_version: int = -1
        csv_path: Optional[Path] = self.get_metrics_csv_path()
        if csv_path is not None and csv_path.exists():
            for child in csv_path.glob("version_*"):
                if child.is_dir():
                    version_str: str = child.name.split("_")[-1]
                    try:
                        last_version = (
                            int(version_str)
                            if int(version_str) > last_version
                            else last_version
                        )
                    except ValueError:
                        continue
        return last_version

    def get_latest_metrics_csv_path(self) -> Optional[Path]:
        version: int = self.get_latest_metrics_csv_version()
        csv_path: Optional[Path] = self.get_metrics_csv_path()
        return (
            csv_path / f"version_{version}" / "metrics.csv"
            if version >= 0 and csv_path is not None
            else None
        )

    def get_train_config_path(
        self, experiment_name: Optional[str] = None
    ) -> Path:
        if experiment_name is not None:
            # user is getting a config for an existing experiment
            user_exp_path: Optional[Path] = self.get_user_experiments_path()
            if user_exp_path is None:
                raise ValueError(
                    "user_exp_path cannot be None if experiment_name is also None in get_train_config_path"
                )
            return user_exp_path / experiment_name / "train_config.yaml"
        else:
            user_exp_path = self._get_exp_path()
            # get config for currently selected experiment
            if user_exp_path is None:
                raise ValueError(
                    "user_exp_path cannot be None if experiment_name is also None in get_train_config_path"
                )
            return user_exp_path / "train_config.yaml"

    def get_current_epoch(self) -> Optional[int]:
        ckpt: Optional[Path] = self.get_best_ckpt()
        if not ckpt:
            return None
        # assumes checkpoint format: path/to/checkpoint/epoch_001.ckpt

        return int(ckpt.name.split(".")[0].split("_")[-1])

    def get_best_ckpt(self) -> Optional[Path]:
        user_exp_path: Optional[Path] = (
            self.user_settings.get_user_experiments_path()
        )
        if user_exp_path is None:
            # need user_exp_path in order to get experiments list
            return None

        selected_experiment: Optional[str] = self.get_experiment_name()
        if selected_experiment is None:
            return None
        return ExperimentUtils.get_best_ckpt(
            user_exp_path, selected_experiment
        )

    def get_channel_selection_path(self) -> Optional[Path]:
        csv_path: Optional[Path] = self.get_csv_path()
        if csv_path is None:
            return None
        return csv_path / "selected_channels.json"
