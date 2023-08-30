import os

from allencell_ml_segmenter.config.cyto_dl_config import CytoDlConfig
import copy

class ExperimentsModel():
    def __init__(self, config: CytoDlConfig) -> None:
        self.config = config
        self.experiments = {}
        self.refresh_experiments()

    def refresh_experiments(self) -> dict:
        self.experiments = {}
        for experiment in os.listdir(self.config._user_experiments_path):
            checkpoints_path = os.path.join(self.config._user_experiments_path, experiment, "checkpoints")
            if os.path.exists(checkpoints_path) and len(os.listdir(checkpoints_path)) > 0:
                self.experiments[experiment] = []
                for checkpoint in os.listdir(checkpoints_path):
                    self.experiments[experiment].append(checkpoint)

    """
    Returns a defensive copy of Experiments dict.
    """
    def get_experiments(self) -> dict:
        return copy.deepcopy(self.experiments)
    
    def get_cyto_dl_config(self) -> CytoDlConfig:
        return self.config