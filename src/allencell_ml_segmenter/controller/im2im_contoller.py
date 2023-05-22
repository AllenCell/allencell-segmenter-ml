from allencell_ml_segmenter.model.pub_sub import Subscriber, Event
from allencell_ml_segmenter.model.sample_model import SampleModel
import hydra
from aics_im2im.train import entry_point_call
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig


class Im2imContoller(Subscriber):
    def __init__(self, application, model: SampleModel):
        super().__init__()
        self._model = model
        self._model.subscribe(self)
        self._application = application

    def handle_event(self, event: Event):
        # TODO change to switch
        if event == Event.TRAINING:
            if self._model.get_model_training():
                self.start_training_test()

    def start_training_test(self):
        GlobalHydra.instance().clear()
        hydra.initialize(version_base="1.3", config_path="../../../../aics-im2im/configs")

        cfg = hydra.compose(config_name="train",
                            overrides=["trainer=cpu", "experiment=im2im/segmentation.yaml", "hydra.runtime.cwd=."],
                            return_hydra_config=True)
        HydraConfig().cfg = cfg
        entry_point_call(cfg)
        print(cfg.paths.output_dir)