from pathlib import Path
from typing import Optional
from allencell_ml_segmenter.config.i_user_settings import IUserSettings

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

    def _get_exp_path(self) -> Path:
        user_exp_path: Optional[Path] = self.get_user_experiments_path()
        exp_name: Optional[str] = self.get_experiment_name()
        if user_exp_path is None or exp_name is None:
            raise ValueError("Experiment path or name undefined")
        return user_exp_path / exp_name

    def get_csv_path(self) -> Path:
        return self._get_exp_path() / "data"

    def get_metrics_csv_path(self) -> Path:
        return self._get_exp_path() / "csv"

    def get_cache_dir(self) -> Path:
        return self._get_exp_path() / "cache"

    def get_latest_metrics_csv_version(self) -> int:
        """
        Returns version number of the most recent version directory within
        the cyto-dl CSV folder (self._csv_path) or -1 if no version directories
        exist
        """
        last_version: int = -1
        if self.get_metrics_csv_path().exists():
            for child in self.get_metrics_csv_path().glob("version_*"):
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
        return (
            self.get_metrics_csv_path() / f"version_{version}" / "metrics.csv"
            if version >= 0
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
            # get config for currently selected experiment
            return self._get_exp_path() / "train_config.yaml"

    def get_current_epoch(self) -> Optional[int]:
        ckpt: Optional[str] = self.get_best_ckpt()
        if not ckpt:
            return None
        # assumes checkpoint format: epoch_001.ckpt
        return int(ckpt.split(".")[0].split("_")[-1])

    def get_best_ckpt(
        self, experiment_name: Optional[str] = None
    ) -> Optional[str]:
        # If no experiment name is specified for this function, assume we want the to use the current experiment selected in this model.
        if experiment_name is None:
            experiment_name = self.get_experiment_name()

        user_exp_path: Optional[Path] = (
            self.user_settings.get_user_experiments_path()
        )
        if (
            experiment_name is None or user_exp_path is None
        ):  # need to re-check experiemnt_name since self.get_experiment_name above can return None
            return None

        checkpoints_path = user_exp_path / experiment_name / "checkpoints"
        if not checkpoints_path.exists():
            return None

        files: list[Path] = [
            entry
            for entry in checkpoints_path.iterdir()
            if entry.is_file() and not "last" in entry.name.lower()
        ]
        if not files:
            return None

        files.sort(key=lambda file: file.stat().st_mtime)
        return files[-1].name

    def get_channel_selection_path(self) -> Optional[Path]:
        return (
            self.get_csv_path() / "selected_channels.json"
            if self.get_csv_path() is not None
            else None
        )
