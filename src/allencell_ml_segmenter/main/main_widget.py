from typing import Dict
from allencell_ml_segmenter.config.i_user_settings import IUserSettings

from allencell_ml_segmenter.main.viewer import Viewer

import napari
from allencell_ml_segmenter.curation.curation_widget import CurationWidget
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QVBoxLayout,
    QSizePolicy,
    QTabWidget,
)
from allencell_ml_segmenter.config.user_settings import UserSettings

from allencell_ml_segmenter.core.aics_widget import AicsWidget

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.view import MainWindow
from allencell_ml_segmenter.main.experiments_model import ExperimentsModel
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.main.main_service import MainService
from allencell_ml_segmenter.main.i_viewer import IViewer
from allencell_ml_segmenter.prediction.model import PredictionModel
from allencell_ml_segmenter.prediction.view import PredictionView
from allencell_ml_segmenter.services.prediction_service import (
    PredictionService,
)
from allencell_ml_segmenter.services.training_service import TrainingService
from allencell_ml_segmenter.training.model_selection_widget import (
    ModelSelectionWidget,
)
from allencell_ml_segmenter.training.training_model import TrainingModel
from allencell_ml_segmenter.training.view import TrainingView
from allencell_ml_segmenter.curation.curation_model import CurationModel
from allencell_ml_segmenter._style import Style
from allencell_ml_segmenter.curation.curation_service import CurationService


class MainWidget(AicsWidget):
    """
    Holds the pertinent view at the moment to be displayed to the user.
    """

    def __init__(self, viewer: napari.Viewer, settings: IUserSettings = None):
        super().__init__()
        self.user_settings: IUserSettings = settings
        if self.user_settings is None:
            self.user_settings = UserSettings()
        self.viewer: IViewer = Viewer(viewer)

        # basic styling
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        self._model: MainModel = MainModel()
        self._model.subscribe(
            Event.ACTION_CHANGE_VIEW, self, self._handle_change_view
        )
        self._model.subscribe(
            Event.ACTION_NEW_MODEL, self, self._handle_new_model
        )

        if self.user_settings.get_user_experiments_path() is None:
            self.user_settings.prompt_for_user_experiments_home(parent=self)

        # init models
        self._experiments_model = ExperimentsModel(self.user_settings)

        self._training_model: TrainingModel = TrainingModel(
            main_model=self._model, experiments_model=self._experiments_model
        )

        self._prediction_model: PredictionModel = PredictionModel()
        self._curation_model: CurationModel = CurationModel(
            self._experiments_model,
            self._model,
        )

        # init services
        self._main_service: MainService = MainService(
            self._model, self._experiments_model
        )
        self._curation_service: CurationService = CurationService(
            self._curation_model,
            self._experiments_model,
        )
        self._training_service: TrainingService = TrainingService(
            training_model=self._training_model,
            experiments_model=self._experiments_model,
        )
        self._prediction_service: PredictionService = PredictionService(
            prediction_model=self._prediction_model,
            experiments_model=self._experiments_model,
        )

        # keep track of windows
        self._window_container: QTabWidget = QTabWidget()
        self._window_container.setDisabled(True)
        self._window_to_index: Dict[MainWindow, int] = dict()

        self._experiments_model.subscribe(
            Event.ACTION_EXPERIMENT_APPLIED,
            self,
            self._handle_experiment_applied,
        )

        # initialize the tabs
        self._curation_view: CurationWidget = CurationWidget(
            self.viewer, self._curation_model
        )
        self._initialize_window(self._curation_view, "Curation")
        self._training_view: TrainingView = TrainingView(
            main_model=self._model,
            viewer=self.viewer,
            experiments_model=self._experiments_model,
            training_model=self._training_model,
        )
        self._initialize_window(self._training_view, "Training")
        self._prediction_view: PredictionView = PredictionView(
            main_model=self._model,
            prediction_model=self._prediction_model,
            viewer=self.viewer,
        )
        self._initialize_window(self._prediction_view, "Prediction")
        self._window_container.currentChanged.connect(self._tab_changed)

        # Model selection which applies to all views
        model_selection_widget: ModelSelectionWidget = ModelSelectionWidget(
            self._model, self._experiments_model, self.user_settings
        )
        model_selection_widget.setObjectName("modelSelection")

        self.layout().addWidget(model_selection_widget, Qt.AlignTop)
        self.layout().addWidget(self._window_container, Qt.AlignCenter)
        self.layout().addStretch(100)
        self.setStyleSheet(Style.get_stylesheet("core.qss"))

    def _handle_experiment_applied(self, _: Event) -> None:
        """
        Handle the experiment applied event.
        """
        self._window_container.setDisabled(
            self._experiments_model.get_experiment_name() is None
        )

    def _handle_new_model(self, _: Event) -> None:
        """
        Handle the new model radio button toggled event.

        inputs:
            is_new_model - bool
        """
        self._window_container.setTabEnabled(0, self._model.is_new_model())
        self._window_container.setTabEnabled(1, self._model.is_new_model())
        self._set_window(
            self._curation_view
            if self._model.is_new_model()
            else self._prediction_view
        )

    def _handle_change_view(self, event: Event) -> None:
        """
        Handle event function for the main widget, which handles MainEvents.

        inputs:
            event - MainEvent
        """
        self._set_window(self._model.get_current_view())

    def _set_window(self, window: MainWindow) -> None:
        """
        Set the current Window (prediction/training/curation), must be initialized first
        """
        self._window_container.setCurrentIndex(self._window_to_index[window])

    def _initialize_window(self, window: MainWindow, title: str) -> None:
        # QTabWidget count method keeps track of how many child widgets have been added
        self._window_to_index[window] = self._window_container.count()
        self._window_container.addTab(window, title)

    def _tab_changed(self, index: int) -> None:
        """
        Resize bottom edge of QTabWidget to fit the current view.
        """
        for i in range(self._window_container.count()):
            if i == index:
                self._window_container.widget(i).setSizePolicy(
                    QSizePolicy.Preferred, QSizePolicy.Maximum
                )
            else:
                self._window_container.widget(i).setSizePolicy(
                    QSizePolicy.Ignored, QSizePolicy.Ignored
                )

                # call the focus_changed method of the view that we are switching to
        list(self._window_to_index.keys())[index].focus_changed()
