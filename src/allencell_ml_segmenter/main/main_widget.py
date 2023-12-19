from typing import Dict

from allencell_ml_segmenter.main.viewer import Viewer

import napari
from allencell_ml_segmenter.curation.curation_widget import CurationWidget
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QVBoxLayout,
    QSizePolicy,
    QTabWidget,
)
from allencell_ml_segmenter.config.cyto_dl_config import UserConfig

from allencell_ml_segmenter.core.aics_widget import AicsWidget

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.view import View
from allencell_ml_segmenter.main.experiments_model import ExperimentsModel
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.prediction.view import PredictionView
from allencell_ml_segmenter.services.training_service import TrainingService
from allencell_ml_segmenter.training.model_selection_widget import (
    ModelSelectionWidget,
)
from allencell_ml_segmenter.training.training_model import TrainingModel
from allencell_ml_segmenter.training.view import TrainingView
from allencell_ml_segmenter.config.cyto_dl_config import UserConfig

from qtpy.QtWidgets import QFileDialog
from qtpy.QtWidgets import QMessageBox
from qtpy.QtCore import QSettings

CYTO_DL_HOME_PATH = "/Users/chrishu/dev/code/test2/cyto-dl"
EXPERIMENTS_HOME = "experimentshome"

class MainWidget(AicsWidget):
    """
    Holds the pertinent view at the moment to be displayed to the user.
    """

    def __init__(self, viewer: napari.Viewer, config: UserConfig = None):
        super().__init__()
        self.viewer: Viewer = Viewer(viewer)

        # basic styling
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        self._model: MainModel = MainModel()
        self._model.subscribe(
            Event.ACTION_CHANGE_VIEW, self, self.handle_change_view
        )

        if config:  # Passed in by test cases
            self._experiments_model = ExperimentsModel(config)
        else:
            self._experiments_model = ExperimentsModel(
                self._get_user_config()
            )

        self._training_model: TrainingModel = TrainingModel(
            main_model=self._model, experiments_model=self._experiments_model
        )
        self._training_service: TrainingService = TrainingService(
            training_model=self._training_model,
            experiments_model=self._experiments_model,
        )

        # Model selection which applies to all views
        model_selection_widget: ModelSelectionWidget = ModelSelectionWidget(
            self._experiments_model
        )
        model_selection_widget.setObjectName("modelSelection")
        self.layout().addWidget(model_selection_widget, Qt.AlignTop)

        # keep track of views
        self._view_container: QTabWidget = QTabWidget()
        self.layout().addWidget(self._view_container, Qt.AlignCenter)
        self.layout().addStretch(100)

        self._view_to_index: Dict[View, int] = dict()

        # initialize the tabs
        self._prediction_view: PredictionView = PredictionView(self._model)
        self._initialize_view(self._prediction_view, "Prediction")

        self._training_view: TrainingView = TrainingView(
            main_model=self._model,
            viewer=self.viewer,
            experiments_model=self._experiments_model,
            training_model=self._training_model,
        )
        self._initialize_view(self._training_view, "Training")

        self._curation_view: CurationWidget = CurationWidget(
            self.viewer, self._model, self._experiments_model
        )
        self._initialize_view(self._curation_view, "Curation")

        self._view_container.currentChanged.connect(self._tab_changed)

    def _get_user_config(self) -> UserConfig:
        settings = QSettings("AICS", "Segmenter ML")
        experiments_home_path = settings.value(EXPERIMENTS_HOME)
        if experiments_home_path is None:
            message_dialog = QMessageBox(parent = self, text = "Please select a location to store your Segmenter ML data.")
            message_dialog.exec()
            directory_dialog = QFileDialog(parent = self)
            directory_dialog.setFileMode(QFileDialog.Directory)
            experiments_home_path = directory_dialog.getExistingDirectory();
            settings.setValue(EXPERIMENTS_HOME, experiments_home_path)
        return UserConfig(CYTO_DL_HOME_PATH, experiments_home_path)
    
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
