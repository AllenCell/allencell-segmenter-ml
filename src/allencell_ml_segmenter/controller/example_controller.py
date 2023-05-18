from qtpy.QtWidgets import QWidget
import napari
import hydra
from omegaconf import OmegaConf
from aics_im2im.train import entry_point_call
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from allencell_ml_segmenter.view.test_view import TestView

class TestController():
    def __init__(self, application) -> None:
        # add all ui elements here
        self.application = application
        self._view: TestView = TestView()
        # called on init

    def index(self):
        # called when loading new controller
        self.load_view()
        self._connect_slots()

    def _on_click(self) -> None:
        print("napari has", len(self.viewer.layers), "layers")

    def _connect_slots(self):
        self._view.widget.btn.clicked.connect(self.start_training_test)

    def load_view(self):
        """
        Loads the given view
        :param: view: the View to load
        """
        return self.application.view_manager.load_view(self._view)

    def start_training_test(self):
        GlobalHydra.instance().clear()
        hydra.initialize(version_base="1.3", config_path="../../../../aics-im2im/configs")

        cfg = hydra.compose(config_name="train", overrides=["trainer=cpu", "experiment=im2im/segmentation.yaml", "hydra.runtime.cwd=."], return_hydra_config=True)
        HydraConfig().cfg = cfg
        # OmegaConf.resolve(cfg)
        entry_point_call(cfg)
        print(cfg.paths.output_dir)

