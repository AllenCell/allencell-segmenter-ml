import napari
from PyQt5.QtWidgets import QTabWidget
from qtpy.QtWidgets import (
    QVBoxLayout,
    QSizePolicy,
)

from allencell_ml_segmenter.prediction.view import PredictionView
from allencell_ml_segmenter.sample.sample_view import SampleView


class MainTabWidget(QTabWidget):
    """
    Adopted and redesigned copy of MainWidget.
    """

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer: napari.Viewer = viewer

        # basic styling
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        # initialize the tabs
        self.addTab(PredictionView(), "Prediction")
        self.addTab(SampleView(), "Training")
