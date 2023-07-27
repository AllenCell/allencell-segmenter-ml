import napari
from PyQt5.QtWidgets import QTabWidget
from qtpy.QtWidgets import (
    QVBoxLayout,
    QSizePolicy,
)

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.view import View
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.prediction.view import PredictionView
from allencell_ml_segmenter.sample.sample_view import SampleView
from allencell_ml_segmenter.training.view import TrainingView
from allencell_ml_segmenter.training.view import TrainingView
from allencell_ml_segmenter.core.subscriber import Subscriber


class MainMeta(type(QTabWidget), type(Subscriber)):
    """
    Metaclass for MainTabWidget
    """

    pass


class MainTabWidget(QTabWidget, Subscriber, metaclass=MainMeta):
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

        # main model
        self._model: MainModel = MainModel()
        self._model.subscribe(
            Event.ACTION_CHANGE_VIEW, self, self.handle_event
        )

        # keep track of views
        self._view_to_index = dict()

        # initialize the tabs
        self._prediction_view = PredictionView(self._model)
        self._initialize_view(self._prediction_view, "Prediction")

        training_view = TrainingView(self._model)
        self._initialize_view(training_view, "Training")

    def handle_event(self, event: Event) -> None:
        """
        Handle event function for the main widget, which handles MainEvents.

        inputs:
            event - MainEvent
        """
        self._set_view(self._model.get_current_view())

    def _set_view(self, view: View) -> None:
        """
        Set the current views, must be initialized first
        """
        self.setCurrentIndex(self._view_to_index[view])

    def _initialize_view(self, view: View, title: str) -> None:
        # QTabWidget count method keeps track of how many child widgets have been added
        self._view_to_index[view] = self.count()
        self.addTab(view, title)
