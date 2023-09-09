from pathlib import Path

from allencell_ml_segmenter.config.cyto_dl_config import CytoDlConfig
import copy


class ExperimentsModel:
    def __init__(self, config: CytoDlConfig) -> None:
        self.config = config
        self.experiments = {}
        self.refresh_experiments()

    def refresh_experiments(self) -> None:
        for experiment in Path(self.config._user_experiments_path).iterdir():
            if experiment not in self.experiments:
                self.experiments[experiment.name] = set()
                self.refresh_checkpoints(experiment.name)

    def refresh_checkpoints(self, experiment: str) -> None:
        checkpoints_path = Path(self.config._user_experiments_path) / experiment / "checkpoints"
        if (
            checkpoints_path.exists()
            and len([checkpoints_path.iterdir()]) > 0
        ):
            for checkpoint in checkpoints_path.iterdir():
                self.experiments[experiment].add(checkpoint.name)

    """
    Returns a defensive copy of Experiments dict.
    """

    def get_experiments(self) -> dict:
        return copy.deepcopy(self.experiments)

    def get_cyto_dl_config(self) -> CytoDlConfig:
        return self.config
