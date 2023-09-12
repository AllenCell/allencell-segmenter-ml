

from allencell_ml_segmenter.main.i_experiments_model import IExperimentsModel


class FakeExperimentModel(IExperimentsModel):
    
    def get_experiments(self):
        return {'0_exp': {}, '1_exp': {}, '2_exp': {'0.ckpt', '1.ckpt'}}
    
    def refresh_experiments(self):
        pass

    def refresh_checkpoints(self, experiment: str):
        pass