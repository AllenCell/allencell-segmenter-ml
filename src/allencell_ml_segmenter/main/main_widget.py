from typing import Optional, Dict
from allencell_ml_segmenter.config.i_user_settings import IUserSettings

from allencell_ml_segmenter.main.viewer import Viewer

import napari  # type: ignore
from allencell_ml_segmenter.curation.curation_widget import CurationWidget
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QVBoxLayout,
    QSizePolicy,
    QTabWidget,
    QWidget,
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
from allencell_ml_segmenter.thresholding.thresholding_model import (
    ThresholdingModel,
)
from allencell_ml_segmenter.thresholding.thresholding_service import (
    ThresholdingService,
)
from allencell_ml_segmenter.training.model_selection_widget import (
    ModelSelectionWidget,
)
from allencell_ml_segmenter.training.training_model import TrainingModel
from allencell_ml_segmenter.training.view import TrainingView
from allencell_ml_segmenter.curation.curation_model import CurationModel
from allencell_ml_segmenter._style import Style
from allencell_ml_segmenter.curation.curation_service import CurationService
from allencell_ml_segmenter.thresholding.thresholding_view import (
    ThresholdingView,
)
from allencell_ml_segmenter.core.file_input_model import FileInputModel


class MainWidget(AicsWidget):
    """
    Holds the pertinent view at the moment to be displayed to the user.
    """

    def __init__(
        self, viewer: napari.Viewer, settings: Optional[IUserSettings] = None
    ):
        super().__init__()
        self.user_settings: IUserSettings
        if settings is None:
            self.user_settings = UserSettings()
        else:
            self.user_settings = settings

        self.viewer: IViewer = Viewer(viewer)

        # basic styling
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding
        )
        layout: QVBoxLayout = QVBoxLayout()
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)

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
        self._prediction_file_input_model: FileInputModel = FileInputModel()
        self._prediction_model: PredictionModel = PredictionModel()
        self._curation_model: CurationModel = CurationModel(
            self._experiments_model,
            self._model,
        )

        self._thresholding_model: ThresholdingModel = ThresholdingModel()

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
            file_input_model=self._prediction_file_input_model,
            experiments_model=self._experiments_model,
        )

        self._thresholding_service: ThresholdingService = ThresholdingService(
            thresholding_model=self._thresholding_model,
            experiments_model=self._experiments_model,
            main_model=self._model,
            viewer=self.viewer,
        )

        # keep track of windows
        self._window_container: QTabWidget = QTabWidget()
        self._window_container.setDisabled(True)
        self._window_to_index: dict[MainWindow, int] = dict()

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
            file_input_model=self._prediction_file_input_model,
            viewer=self.viewer,
        )
        self._initialize_window(self._prediction_view, "Prediction")
        self._window_container.currentChanged.connect(self._tab_changed)

        self._thresholding_view = ThresholdingView(
            main_model=self._model,
            thresholding_model=self._thresholding_model,
            experiments_model=self._experiments_model,
            viewer=self.viewer,
        )
        self._initialize_window(self._thresholding_view, "Thresholding")

        # Model selection which applies to all views
        model_selection_widget: ModelSelectionWidget = ModelSelectionWidget(
            self._model, self._experiments_model, self.user_settings
        )
        model_selection_widget.setObjectName("modelSelection")

        layout.addWidget(model_selection_widget, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self._window_container, Qt.AlignmentFlag.AlignCenter)
        layout.addStretch(100)
        self.setStyleSheet(Style.get_stylesheet("core.qss"))

        # events for auto window switching
        self._model.subscribe(
            Event.PROCESS_TRAINING_COMPLETE,
            self,
            self._disable_non_prediction_tabs,
        )

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
        self._window_container.setTabEnabled(3, True)  # TODO remove

    def _disable_non_prediction_tabs(self, _: Event) -> None:
        """
        Handle existing model selection (disable tabs).

        inputs:
            is_new_model - bool
        """
        self._window_container.setTabEnabled(0, False)
        self._window_container.setTabEnabled(1, False)
        self._window_container.setTabEnabled(3, True)  # TODO remove

    def _handle_change_view(self, event: Event) -> None:
        """
        Handle event function for the main widget, which handles MainEvents.

        inputs:
            event - MainEvent
        """
        curr_view: Optional[MainWindow] = self._model.get_current_view()
        if curr_view is not None:
            self._set_window(curr_view)

    def _set_window(self, window: MainWindow) -> None:
        """
        Set the current Window (prediction/training/curation), must be initialized first
        """
        self._window_container.setCurrentIndex(self._window_to_index[window])

    def _initialize_window(self, window: MainWindow, title: str) -> None:
        # QTabWidget count method keeps track of how many child widgets have been added
        self._window_to_index[window] = self._window_container.count()
        self._window_container.addTab(window, title)  # type: ignore

    def _tab_changed(self, index: int) -> None:
        """
        Resize bottom edge of QTabWidget to fit the current view.
        """
        for i in range(self._window_container.count()):
            widget: Optional[QWidget] = self._window_container.widget(i)
            if widget is not None and i == index:
                widget.setSizePolicy(
                    QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum
                )
            elif widget is not None:
                widget.setSizePolicy(
                    QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored
                )

                # call the focus_changed method of the view that we are switching to
        list(self._window_to_index.keys())[index].focus_changed()
