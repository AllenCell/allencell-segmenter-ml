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
        experiments_model: IExperimentsModel,
    ):
        super().__init__()

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
        frame.layout().addWidget(LabelWithHint("Select a model:"))

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
        label_existing_model_checkpoint: LabelWithHint = LabelWithHint(
            "Checkpoint"
        )
        top_grid_layout.addWidget(label_existing_model_checkpoint, 2, 1)

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
            Event.ACTIOIN_REFRESH, self, self._process_event_handler
        )

        top_grid_layout.addWidget(self._combo_box_existing_models, 1, 2)

        self._combo_box_existing_models_checkpoint: QComboBox = QComboBox()
        self._combo_box_existing_models_checkpoint.setCurrentIndex(-1)
        self._combo_box_existing_models_checkpoint.setPlaceholderText(
            "Select an option"
        )
        self._combo_box_existing_models_checkpoint.setEnabled(False)
        self._combo_box_existing_models_checkpoint.setMinimumWidth(306)
        self._combo_box_existing_models_checkpoint.currentTextChanged.connect(
            lambda path_text: self._experiments_model.set_checkpoint(path_text)
        )
        self._combo_box_existing_models.setEnabled(False)
        self._combo_box_existing_models_checkpoint.setEnabled(False)
        self.experiment_info_widget.set_enabled(False)

        top_grid_layout.addWidget(
            self._combo_box_existing_models_checkpoint, 2, 2
        )

        frame.layout().addLayout(top_grid_layout)

    def _model_combo_handler(self, experiment_name: str) -> None:
        """
        Triggered when the user selects a model from the _combo_box_existing_models.
        Sets the model path in the model.
        """
        if experiment_name == "":
            self._experiments_model.set_experiment_name(None)
        else:
            self._experiments_model.set_experiment_name(experiment_name)
            self._refresh_checkpoint_options()

    def _model_radio_handler(self) -> None:
        if self._radio_new_model.isChecked():
            """
            Triggered when the user selects the "start a new model" radio button.
            Enables and disables relevent controls.
            """
            self._combo_box_existing_models.setCurrentIndex(-1)
            self._combo_box_existing_models.setEnabled(False)
            self.experiment_info_widget.set_enabled(True)

            self._combo_box_existing_models_checkpoint.setEnabled(False)
            self._combo_box_existing_models_checkpoint.clear()

            self._experiments_model.set_experiment_name(None)
            self._experiments_model.set_checkpoint(None)

        if self._radio_existing_model.isChecked():
            """
            Triggered when the user selects the "existing model" radio button.
            Enables and disables relevent controls.
            """
            self._experiments_model.set_experiment_name(None)
            self._combo_box_existing_models.setEnabled(True)
            self.experiment_info_widget.set_enabled(False)
            self.experiment_info_widget.clear()

    def _process_event_handler(self, _: Event = None) -> None:
        """
        Refreshes the experiments in the _combo_box_existing_models.
        """
        if self._radio_new_model.isChecked():#TODO is firing twice on complete, only needs to run once
            self._refresh_experiment_options()
        if (
            self._radio_existing_model.isChecked()
            and self._experiments_model.get_experiment_name() is not None
        ):
            self._refresh_checkpoint_options()

    def _refresh_experiment_options(self):
        self._experiments_model.refresh_experiments()
        self._combo_box_existing_models.clear()
        self._combo_box_existing_models.addItems(
            self._experiments_model.get_experiments().keys()
        )

    def _refresh_checkpoint_options(self):
        # update and enable checkpoint combo box
        self._experiments_model.refresh_checkpoints(
            self._experiments_model.get_experiment_name()
        )
        self._combo_box_existing_models_checkpoint.clear()
        self._combo_box_existing_models_checkpoint.addItems(
            self._experiments_model.get_experiments()[
                self._experiments_model.get_experiment_name()
            ]
        )
        self._combo_box_existing_models_checkpoint.setCurrentIndex(-1)
        self._combo_box_existing_models_checkpoint.setEnabled(True)
