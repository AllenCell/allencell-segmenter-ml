from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QSizePolicy,
    QFrame,
    QGridLayout,
    QComboBox,
    QRadioButton,
)
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.main.i_experiments_model import IExperimentsModel
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.training.experiment_info_widget import (
    ExperimentInfoWidget,
)

from allencell_ml_segmenter.widgets.label_with_hint_widget import LabelWithHint


class ModelSelectionWidget(QWidget):
    """
    A widget for segmentation model selection.
    """

    TITLE_TEXT: str = "Segmentation model"

    def __init__(
        self,
        main_model: MainModel,
        experiments_model: IExperimentsModel,
    ):
        super().__init__()

        self._main_model: MainModel = main_model
        self._experiments_model: IExperimentsModel = experiments_model

        # widget skeleton
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        title: LabelWithHint = LabelWithHint(ModelSelectionWidget.TITLE_TEXT)
        title.setObjectName("title")
        # TODO: hints for widget titles?
        self.layout().addWidget(title)

        frame: QFrame = QFrame()
        frame.setLayout(QVBoxLayout())
        frame.setObjectName("frame")
        self.layout().addWidget(frame)

        # model selection components
        top_grid_layout: QGridLayout = QGridLayout()

        self._radio_new_model: QRadioButton = QRadioButton()
        self._radio_new_model.toggled.connect(self._model_radio_handler)
        top_grid_layout.addWidget(self._radio_new_model, 0, 0)

        self.experiment_info_widget = ExperimentInfoWidget(
            self._experiments_model
        )
        label_new_model: LabelWithHint = LabelWithHint("Start a new model")
        top_grid_layout.addWidget(label_new_model, 0, 1)
        top_grid_layout.addWidget(self.experiment_info_widget, 0, 2)

        self._radio_existing_model: QRadioButton = QRadioButton()
        self._radio_existing_model.toggled.connect(self._model_radio_handler)
        top_grid_layout.addWidget(self._radio_existing_model, 1, 0)

        label_existing_model: LabelWithHint = LabelWithHint("Existing model")
        top_grid_layout.addWidget(label_existing_model, 1, 1)

        self._combo_box_existing_models: QComboBox = QComboBox()
        self._combo_box_existing_models.setCurrentIndex(-1)
        self._combo_box_existing_models.setPlaceholderText("Select an option")
        self._combo_box_existing_models.setEnabled(False)
        self._combo_box_existing_models.setMinimumWidth(306)

        self._refresh_experiment_options()
        self._combo_box_existing_models.currentTextChanged.connect(
            self._model_combo_handler
        )
        self._experiments_model.subscribe(
            Event.ACTION_REFRESH, self, self._handle_process_event
        )

        top_grid_layout.addWidget(self._combo_box_existing_models, 1, 2)

        self._combo_box_existing_models.setEnabled(False)
        self.experiment_info_widget.set_enabled(False)

        frame.layout().addLayout(top_grid_layout)
        main_model.subscribe(
            Event.ACTION_NEW_MODEL, self, self._handle_new_model_selection
        )

    def _model_combo_handler(self, experiment_name: str) -> None:
        """
        Triggered when the user selects a model from the _combo_box_existing_models.
        Sets the model path in the model.
        """
        if experiment_name == "":
            self._experiments_model.set_experiment_name(None)
        else:
            self._experiments_model.set_experiment_name(experiment_name)

    def _model_radio_handler(self) -> None:
        self._main_model.set_new_model(self._radio_new_model.isChecked())
        if self._radio_new_model.isChecked():
            """
            Triggered when the user selects the "start a new model" radio button.
            Enables and disables relevent controls.
            """
            self._combo_box_existing_models.setCurrentIndex(-1)
            self._combo_box_existing_models.setEnabled(False)
            self.experiment_info_widget.set_enabled(True)

            self._experiments_model.set_experiment_name(None)

        if self._radio_existing_model.isChecked():
            """
            Triggered when the user selects the "existing model" radio button.
            Enables and disables relevent controls.
            """
            self._experiments_model.set_experiment_name(None)
            self._combo_box_existing_models.setEnabled(True)
            self.experiment_info_widget.set_enabled(False)
            self.experiment_info_widget.clear()

    def _handle_new_model_selection(self, _: Event = None) -> None:
        self._radio_existing_model.setChecked(
            not self._main_model.is_new_model()
        )
        self._radio_new_model.setChecked(self._main_model.is_new_model())

    def _handle_process_event(self, _: Event = None) -> None:
        """
        Refreshes the experiments in the _combo_box_existing_models.
        """
        if self._radio_new_model.isChecked():
            self._refresh_experiment_options()

    def _refresh_experiment_options(self):
        self._experiments_model.refresh_experiments()
        self._combo_box_existing_models.clear()
        self._combo_box_existing_models.addItems(
            self._experiments_model.get_experiments()
        )
