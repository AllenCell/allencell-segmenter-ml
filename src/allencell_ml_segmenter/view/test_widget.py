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



