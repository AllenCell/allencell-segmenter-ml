from typing import Dict
from allencell_ml_segmenter.config.i_user_settings import IUserSettings
from allencell_ml_segmenter.core.extractor_factory import ExtractorFactory

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
from allencell_ml_segmenter.core.view import View
from allencell_ml_segmenter.main.experiments_model import ExperimentsModel
from allencell_ml_segmenter.main.main_model import MainModel
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


class MainWidget(AicsWidget):
    """
    Holds the pertinent view at the moment to be displayed to the user.
    """

    def __init__(self, viewer: napari.Viewer, settings: IUserSettings = None):
        super().__init__()
        self.user_settings: IUserSettings = settings
        if self.user_settings is None:
            self.user_settings = UserSettings()
        self.viewer: Viewer = Viewer(viewer)

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
            path = self.user_settings.prompt_for_user_experiments_home(
                parent=self
            )
            self.user_settings.set_user_experiments_path(path)

        # init models
        self._experiments_model = ExperimentsModel(self.user_settings)

        self._training_model: TrainingModel = TrainingModel(
            main_model=self._model, experiments_model=self._experiments_model
        )

        self._prediction_model: PredictionModel = PredictionModel()

        # init services
        self._training_service: TrainingService = TrainingService(
            training_model=self._training_model,
            experiments_model=self._experiments_model,
            extractor_factory=ExtractorFactory(),
        )
        self._prediction_service: PredictionService = PredictionService(
            prediction_model=self._prediction_model,
            experiments_model=self._experiments_model,
        )

        # Model selection which applies to all views
        model_selection_widget: ModelSelectionWidget = ModelSelectionWidget(
            self._model, self._experiments_model
        )
        model_selection_widget.setObjectName("modelSelection")
        self.layout().addWidget(model_selection_widget, Qt.AlignTop)

        # keep track of views
        self._view_container: QTabWidget = QTabWidget()
        self.layout().addWidget(self._view_container, Qt.AlignCenter)
        self.layout().addStretch(100)

        self._view_to_index: Dict[View, int] = dict()

        # initialize the tabs
        self._curation_view: CurationWidget = CurationWidget(
            self.viewer, self._model, self._experiments_model
        )
        self._initialize_view(self._curation_view, "Curation")
        self._training_view: TrainingView = TrainingView(
            main_model=self._model,
            viewer=self.viewer,
            experiments_model=self._experiments_model,
            training_model=self._training_model,
        )
        self._initialize_view(self._training_view, "Training")
        self._prediction_view: PredictionView = PredictionView(
            main_model=self._model,
            prediction_model=self._prediction_model,
            viewer=self.viewer,
        )
        self._initialize_view(self._prediction_view, "Prediction")
        self._view_container.currentChanged.connect(self._tab_changed)
        self._model.set_new_model(False)

    def _handle_new_model(self, _: Event) -> None:
        """
        Handle the new model radio button toggled event.

        inputs:
            is_new_model - bool
        """
        self._view_container.setTabEnabled(0, self._model.is_new_model())
        self._view_container.setTabEnabled(1, self._model.is_new_model())
        self._set_view(
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
