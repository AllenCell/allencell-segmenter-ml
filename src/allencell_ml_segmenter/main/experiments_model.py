from pathlib import Path
from allencell_ml_segmenter.config.i_user_settings import IUserSettings

import copy

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.main.i_experiments_model import IExperimentsModel


class ExperimentsModel(IExperimentsModel):
    def __init__(self, config: IUserSettings) -> None:
        super().__init__()
        self.user_settings = config

        # options
        self.experiments = {}
        self.refresh_experiments()

        # state
        self._experiment_name: str = None
        self._checkpoint: str = None

    def get_experiment_name(self) -> str:
        """
        Gets experiment name
        """
        return self._experiment_name

    def set_experiment_name(self, name: str) -> None:
        """
        Sets experiment name

        name (str): name of cyto-dl experiment
        """
        self._experiment_name = name
        if self._experiment_name:
            # if a experiment name is set
            self.dispatch(Event.ACTION_EXPERIMENT_SELECTED)

    def get_checkpoint(self) -> str:
        """
        Gets checkpoint
        """
        return self._checkpoint

    def set_checkpoint(self, checkpoint: str) -> None:
        """
        Sets checkpoint

        checkpoint (str): name of checkpoint to use
        """
        self._checkpoint = checkpoint

    def refresh_experiments(self) -> None:
        for (
            experiment
        ) in self.user_settings.get_user_experiments_path().iterdir():
            if (
                experiment not in self.experiments
                and not experiment.name.startswith(".")
            ):
                self.experiments[experiment.name] = set()
                self.refresh_checkpoints(experiment.name)

    def refresh_checkpoints(self, experiment: str) -> None:
        checkpoints_path = (
            Path(self.user_settings.get_user_experiments_path())
            / experiment
            / "checkpoints"
        )
        if checkpoints_path.exists() and len([checkpoints_path.iterdir()]) > 0:
            for checkpoint in checkpoints_path.iterdir():
                if checkpoint.suffix == ".ckpt":
                    self.experiments[experiment].add(checkpoint.name)

    """
    Returns a defensive copy of Experiments dict.
    """

    def get_experiments(self) -> dict:
        return copy.deepcopy(self.experiments)

    def get_user_settings(self) -> IUserSettings:
        return self.user_settings

    def get_user_experiments_path(self) -> Path:
        return self.get_user_settings().get_user_experiments_path()

    def get_model_test_images_path(self, experiment_name: str) -> Path:
        return (
            self.get_user_settings().get_user_experiments_path()
            / experiment_name
            / "test_images"
            if experiment_name
            else None
        )

    def get_model_checkpoints_path(
        self, experiment_name: str, checkpoint: str
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
        return (
            self.get_user_experiments_path()
            / experiment_name
            / "checkpoints"
            / checkpoint
        )

    def get_csv_path(self) -> Path:
        return (
            self.get_user_experiments_path()
            / self.get_experiment_name()
            / "csv"
        )

    def get_train_config_path(self, experiment_name: str) -> Path:
        return (
            self.get_user_experiments_path()
            / experiment_name
            / "train_config.yaml"
            if experiment_name
            else None
        )
