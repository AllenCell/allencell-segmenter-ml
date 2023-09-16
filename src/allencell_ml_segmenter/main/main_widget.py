from typing import Dict

import napari
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QVBoxLayout,
    QSizePolicy,
    QTabWidget,
)
from allencell_ml_segmenter.config.cyto_dl_config import CytoDlConfig
from allencell_ml_segmenter.constants import (
    CYTO_DL_HOME_PATH,
    USER_EXPERIMENTS_PATH,
)
from allencell_ml_segmenter.core.aics_widget import AicsWidget

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.view import View
from allencell_ml_segmenter.main.experiments_model import ExperimentsModel
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.prediction.view import PredictionView
from allencell_ml_segmenter.training.view import TrainingView


class MainWidget(AicsWidget):
    """
    Holds the pertinent view at the moment to be displayed to the user.
    """

    def __init__(self, viewer: napari.Viewer, config: CytoDlConfig = None):
        super().__init__()
        self.viewer: napari.Viewer = viewer

        # basic styling
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        self._model: MainModel = MainModel()
        self._model.subscribe(
            Event.ACTION_CHANGE_VIEW, self, self.handle_change_view
        )

        # keep track of views
        self._view_container: QTabWidget = QTabWidget()
        self.layout().addWidget(self._view_container, Qt.AlignTop)
        self.layout().addStretch(100)

        self._view_to_index: Dict[View, int] = dict()

        # initialize the tabs
        self._prediction_view: PredictionView = PredictionView(self._model)
        self._initialize_view(self._prediction_view, "Prediction")

        if config is None:
            config = CytoDlConfig(CYTO_DL_HOME_PATH, USER_EXPERIMENTS_PATH)
        experiment_model = ExperimentsModel(config)

        training_view: TrainingView = TrainingView(
            main_model=self._model,
            viewer=self.viewer,
            experiments_model=experiment_model,
        )
        self._initialize_view(training_view, "Training")

        self._view_container.currentChanged.connect(self._tab_changed)

    def handle_change_view(self, event: Event) -> None:
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
        self._view_container.setCurrentIndex(self._view_to_index[view])

    def _initialize_view(self, view: View, title: str) -> None:
        # QTabWidget count method keeps track of how many child widgets have been added
        self._view_to_index[view] = self._view_container.count()
        self._view_container.addTab(view, title)

    def _tab_changed(self, index: int) -> None:
        """
        Resize bottom edge of QTabWidget to fit the current view.
        """
        for i in range(self._view_container.count()):
            if i == index:
                self._view_container.widget(i).setSizePolicy(
                    QSizePolicy.Preferred, QSizePolicy.Maximum
                )
            else:
                self._view_container.widget(i).setSizePolicy(
                    QSizePolicy.Ignored, QSizePolicy.Ignored
                )
