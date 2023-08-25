import os

from allencell_ml_segmenter.config.cyto_dl_config import CytoDlConfig
import copy

class ExperimentsModel():
    def __init__(self, config: CytoDlConfig) -> None:
        self.config = config
        self.experiments = {}
        self.refreshExperiments()

    def refreshExperiments(self) -> dict:
        self.experiments = {}
        for filename in os.listdir(self.config._user_experiments_path):
            self.experiments[filename] = []
            checkpoints_path = os.path.join(self.config._user_experiments_path, filename, "checkpoints")
            if os.path.exists(checkpoints_path):
                for checkpoint in os.listdir(checkpoints_path):
                    self.experiments[filename].append(checkpoint)

    """
    Returns a defensive copy of Experiments dict.
    """
    def get_experiments(self) -> dict:
        return copy.deepcopy(self.experiments)
    
    def get_cyto_dl_config(self) -> str:
        return self.config