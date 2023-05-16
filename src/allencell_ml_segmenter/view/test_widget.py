from qtpy.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
    QLabel
)
import hydra
from omegaconf import OmegaConf
from aics_im2im.train import entry_point_call
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig

class TestWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        self.btn: QPushButton = QPushButton("Start Training")
        self.layout().addWidget(self.btn)

        self.btn.clicked.connect(self.start_training_test)

    def start_training_test(self):
        GlobalHydra.instance().clear()
        hydra.initialize(version_base="1.3", config_path="../../../../aics-im2im/configs")

        cfg = hydra.compose(config_name="train", overrides=["trainer=cpu", "experiment=im2im/segmentation.yaml", "hydra.runtime.cwd=."], return_hydra_config=True)
        HydraConfig().cfg = cfg
        # OmegaConf.resolve(cfg)
        entry_point_call(cfg)
        print(cfg.paths.output_dir)

