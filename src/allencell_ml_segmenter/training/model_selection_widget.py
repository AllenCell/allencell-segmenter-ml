from qtpy.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QSizePolicy,
    QFrame,
    QGridLayout,
    QComboBox,
    QRadioButton,
    QLineEdit,
    QPushButton,
    QStackedWidget,
    QLabel,
)
from qtpy.QtCore import Qt
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.main.i_experiments_model import IExperimentsModel
from allencell_ml_segmenter.main.main_model import MainModel

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
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        self._title: LabelWithHint = LabelWithHint(
            ModelSelectionWidget.TITLE_TEXT,
        )
        self._title.setObjectName("title")

        self._model_name_label = QLabel()
        self._model_name_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # layout for model labels
        label_widget_layout = QHBoxLayout()
        layout.addLayout(label_widget_layout)
        label_widget_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        label_widget_layout.setSpacing(0)
        label_widget_layout.addWidget(self._title)
        label_widget_layout.addWidget(
            self._model_name_label, alignment=Qt.AlignLeft
        )
        label_widget_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        # TODO: hints for widget titles?

        frame: QFrame = QFrame()
        frame.setLayout(QVBoxLayout())
        frame.setObjectName("frame")
        self.layout().addWidget(frame)

        # existing model selection components must be initialized before the new/existing model radios
        self._combo_box_existing_models: QComboBox = QComboBox()
        self._combo_box_existing_models.setCurrentIndex(-1)
        self._combo_box_existing_models.setPlaceholderText("Select an option")
        self._combo_box_existing_models.setEnabled(False)
        self._combo_box_existing_models.setMinimumWidth(306)

        self._refresh_experiment_options()
        self._combo_box_existing_models.currentTextChanged.connect(
            self._model_combo_handler
        )

        # model selection components
        top_grid_layout: QGridLayout = QGridLayout()

        self._radio_new_model: QRadioButton = QRadioButton()
        self._radio_new_model.toggled.connect(self._model_radio_handler)
        # initialize the radio button and combos / tabs to match the model state
        self._radio_new_model.setChecked(self._main_model.is_new_model())
        top_grid_layout.addWidget(self._radio_new_model, 0, 0)

        self._experiment_name_input: QLineEdit = QLineEdit()
        self._experiment_name_input.setPlaceholderText("Name your model")
        self._experiment_name_input.textChanged.connect(
            self._experiment_name_input_handler
        )

        label_new_model: LabelWithHint = LabelWithHint("Start a new model")
        top_grid_layout.addWidget(label_new_model, 0, 1)
        top_grid_layout.addWidget(self._experiment_name_input, 0, 2)

        self._radio_existing_model: QRadioButton = QRadioButton()
        self._radio_existing_model.toggled.connect(self._model_radio_handler)
        # initialize the radio button and combos / tabs to match the model state
        self._radio_existing_model.setChecked(
            not self._main_model.is_new_model()
        )
        top_grid_layout.addWidget(self._radio_existing_model, 1, 0)
        top_grid_layout.addWidget(
            LabelWithHint("Select an existing model"), 1, 1
        )
        top_grid_layout.addWidget(self._combo_box_existing_models, 1, 2)

        self._apply_change_stacked_widget = QStackedWidget()

        self._experiments_model.subscribe(
            Event.ACTION_EXPERIMENT_SELECTED,
            self,
            self._handle_experiment_selected,
        )
        apply_model_layout = QVBoxLayout()
        apply_model_layout.addLayout(top_grid_layout)

        apply_model_widget = QWidget()
        apply_model_widget.setLayout(apply_model_layout)
        self._apply_change_stacked_widget.addWidget(apply_model_widget)
        self._apply_btn: QPushButton = QPushButton("Apply")
        self._apply_btn.clicked.connect(self._handle_apply_model)
        apply_model_layout.addWidget(self._apply_btn)

        # Disabled 4/6/24 - no needed for MVP V1 - chrishu
        # change_model_layout = QVBoxLayout()
        # self._model_name_label = QLabel("Model name")
        # change_model_layout.addWidget(self._model_name_label)

        # self._change_model_btn: QPushButton = QPushButton("Change model")
        # self._change_model_btn.clicked.connect(self._handle_change_model)
        # change_model_layout.addWidget(self._change_model_btn)
        #
        # change_model_widget = QWidget()
        # change_model_widget.setLayout(change_model_layout)
        # self._apply_change_stacked_widget.addWidget(change_model_widget)

        frameLayout = QVBoxLayout()
        frameLayout.addWidget(self._apply_change_stacked_widget)
        frame.layout().addLayout(frameLayout)

        self._experiments_model.subscribe(
            Event.ACTION_REFRESH, self, self._handle_process_event
        )
        # initialize the rest of the UI to match the radio button's state
        self._model_radio_handler()

    def _handle_experiment_selected(self, _: Event = None) -> None:
        experiment_selected = (
            self._experiments_model.get_experiment_name_selection() is not None
        )
        self._apply_btn.setEnabled(experiment_selected)

    def _handle_apply_model(self):
        self._experiments_model.apply_experiment_name(
            self._experiments_model.get_experiment_name_selection()
        )
        self._title.set_value_text(
            '    ' + self._experiments_model.get_experiment_name()
        )
        # Change button Disabled 4/6/24 - no needed for MVP V1 - chrishu
        # This would be:
        # self._apply_change_stacked_widget.setCurrentIndex(1)
        self._apply_change_stacked_widget.setVisible(False)

    def _handle_change_model(self):
        self._combo_box_existing_models.setCurrentIndex(-1)
        self._experiment_name_input.clear()
        self._experiments_model.select_experiment_name(None)
        self._experiments_model.apply_experiment_name(None)
        self._apply_change_stacked_widget.setCurrentIndex(0)

    def _model_combo_handler(self, experiment_name: str) -> None:
        """
        Triggered when the user selects a model from the _combo_box_existing_models.
        Sets the model path in the model.
        """
        if experiment_name == "":
            self._experiments_model.select_experiment_name(None)
        else:
            self._experiments_model.select_experiment_name(experiment_name)

    def _experiment_name_input_handler(self, text: str) -> None:
        """
        Triggered when the user types in the _experiment_name_input.
        Sets the model name in the model.
        """
        self._experiments_model.select_experiment_name(text)

    def _model_radio_handler(self) -> None:
        self._main_model.set_new_model(self._radio_new_model.isChecked())
        if self._radio_new_model.isChecked():
            """
            Triggered when the user selects the "start a new model" radio button.
            Enables and disables relevent controls.
            """
            self._combo_box_existing_models.setCurrentIndex(-1)
            self._combo_box_existing_models.setEnabled(False)
            self._experiment_name_input.setEnabled(True)

            self._experiments_model.select_experiment_name(None)

        if self._radio_existing_model.isChecked():
            """
            Triggered when the user selects the "existing model" radio button.
            Enables and disables relevent controls.
            """
            self._experiments_model.select_experiment_name(None)
            self._combo_box_existing_models.setEnabled(True)
            self._experiment_name_input.setEnabled(False)
            self._experiment_name_input.clear()

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
