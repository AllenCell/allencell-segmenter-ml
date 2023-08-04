from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QSizePolicy,
    QFrame,
    QLabel,
    QGridLayout,
    QComboBox,
    QHBoxLayout,
    QRadioButton,
    QLineEdit,
    QCheckBox,
)

from allencell_ml_segmenter.training.training_model import TrainingModel
from allencell_ml_segmenter.widgets.label_with_hint_widget import LabelWithHint


class ModelSelectionWidget(QWidget):
    """
    A widget for segmentation model selection.
    """

    TITLE_TEXT: str = "Segmentation model"

    def __init__(self, model: TrainingModel):
        super().__init__()

        self._model: TrainingModel = model

        # widget skeleton
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        title: LabelWithHint = LabelWithHint(ModelSelectionWidget.TITLE_TEXT)
        title.setObjectName("title")
        self.layout().addWidget(title)

        frame: QFrame = QFrame()
        frame.setLayout(QVBoxLayout())
        frame.setObjectName("frame")
        self.layout().addWidget(frame)

        # model selection components
        frame.layout().addWidget(LabelWithHint("Select a model:"))

        grid_layout: QGridLayout = QGridLayout()

        self._radio_new_model: QRadioButton = QRadioButton()
        self._radio_new_model.toggled.connect(self._radio_new_model_slot)
        grid_layout.addWidget(self._radio_new_model, 0, 0)

        label_new: LabelWithHint = LabelWithHint("Start a new model")
        grid_layout.addWidget(label_new, 0, 1)

        self._radio_existing_model: QRadioButton = QRadioButton()
        self._radio_existing_model.toggled.connect(
            self._radio_existing_model_slot
        )
        grid_layout.addWidget(self._radio_existing_model, 1, 0)

        label_existing: LabelWithHint = LabelWithHint("Existing model")
        grid_layout.addWidget(label_existing, 1, 1)

        self._combo_box_existing_models: QComboBox = QComboBox()
        self._combo_box_existing_models.setCurrentIndex(-1)
        self._combo_box_existing_models.setPlaceholderText("Select an option")
        self._combo_box_existing_models.setEnabled(False)
        self._combo_box_existing_models.setMinimumWidth(306)
        self._combo_box_existing_models.currentTextChanged.connect(
            lambda path_text: self._model.set_model_path(Path(path_text))
        )
        grid_layout.addWidget(self._combo_box_existing_models, 1, 2)

        frame.layout().addLayout(grid_layout)

        # bottom half
        grid_layout = QGridLayout()

        patch_size_label: LabelWithHint = LabelWithHint("Structure size")
        grid_layout.addWidget(patch_size_label, 0, 0)

        self._patch_size_combo_box: QComboBox = QComboBox()
        self._patch_size_combo_box.setObjectName("structureSizeComboBox")
        self._patch_size_combo_box.setCurrentIndex(-1)
        self._patch_size_combo_box.setPlaceholderText("Select an option")
        self._patch_size_combo_box.addItems(["small", "medium", "large"])
        self._patch_size_combo_box.currentTextChanged.connect(
            lambda size: self._model.set_patch_size(size)
        )
        grid_layout.addWidget(self._patch_size_combo_box, 0, 1)

        image_dimensions_label: LabelWithHint = LabelWithHint(
            "Image dimension"
        )
        grid_layout.addWidget(image_dimensions_label, 1, 0)

        dimension_choice_layout: QHBoxLayout = QHBoxLayout()
        dimension_choice_layout.setSpacing(0)

        self._radio_3d: QRadioButton = QRadioButton()
        self._radio_3d.setObjectName("3DRadio")
        self._radio_3d.toggled.connect(lambda: self._model.set_image_dims(3))
        label_3d: LabelWithHint = LabelWithHint("3D")

        self._radio_2d: QRadioButton = QRadioButton()
        self._radio_2d.toggled.connect(lambda: self._model.set_image_dims(2))
        label_2d: LabelWithHint = LabelWithHint("2D")

        dimension_choice_layout.addWidget(self._radio_3d)
        dimension_choice_layout.addWidget(label_3d)
        dimension_choice_layout.addWidget(
            self._radio_2d, alignment=Qt.AlignLeft
        )
        dimension_choice_layout.addWidget(label_2d, alignment=Qt.AlignLeft)
        dimension_choice_layout.addStretch(10)
        dimension_choice_layout.setContentsMargins(0, 0, 0, 0)

        dimension_choice_dummy: QWidget = (
            QWidget()
        )  # stops interference with other radio buttons
        dimension_choice_dummy.setLayout(dimension_choice_layout)

        grid_layout.addWidget(dimension_choice_dummy, 1, 1)

        max_epoch_label: LabelWithHint = LabelWithHint("Training steps")
        grid_layout.addWidget(max_epoch_label, 2, 0)

        self._max_epoch_input: QLineEdit = QLineEdit()
        self._max_epoch_input.setPlaceholderText("1000")
        self._max_epoch_input.setObjectName("trainingStepInput")
        self._max_epoch_input.textChanged.connect(
            lambda text: self._model.set_max_epoch(int(text))
        )
        grid_layout.addWidget(self._max_epoch_input, 2, 1)

        max_time_layout: QHBoxLayout = QHBoxLayout()
        max_time_layout.setSpacing(0)

        self._max_time_checkbox: QCheckBox = QCheckBox()
        self._max_time_checkbox.setObjectName("timeoutCheckbox")
        self._max_time_checkbox.stateChanged.connect(
            self._timeout_checkbox_slot
        )
        max_time_layout.addWidget(self._max_time_checkbox)

        max_time_left_text: QLabel = QLabel("Time out after")
        max_time_layout.addWidget(max_time_left_text)

        self._max_time_in_hours_input: QLineEdit = QLineEdit()
        self._max_time_in_hours_input.setObjectName("timeoutHourInput")
        self._max_time_in_hours_input.setEnabled(False)
        self._max_time_in_hours_input.setMaximumWidth(30)
        self._max_time_in_hours_input.setPlaceholderText("0")
        # TODO: decide between converting as int(text) or float(text) -> will users want to use decimals? is there a better way to convert from hours to seconds?
        # TODO: how to handle invalid (not convertible to a number) input?
        self._max_time_in_hours_input.textChanged.connect(
            lambda text: self._model.set_max_time(round(float(text) * 3600))
        )
        max_time_layout.addWidget(self._max_time_in_hours_input)

        max_time_right_text: LabelWithHint = LabelWithHint("hours")
        max_time_layout.addWidget(max_time_right_text, alignment=Qt.AlignLeft)
        max_time_layout.addStretch()

        grid_layout.addLayout(max_time_layout, 3, 1)
        grid_layout.setColumnStretch(1, 8)
        grid_layout.setColumnStretch(0, 3)

        frame.layout().addLayout(grid_layout)

    def _radio_new_model_slot(self) -> None:
        """
        Triggered when the user selects the "start a new model" radio button.
        Disables interaction with the combo box below.
        """
        self._model.set_model_path(None)
        self._combo_box_existing_models.setEnabled(False)

    def _radio_existing_model_slot(self) -> None:
        """
        Triggered when the user selects the "existing model" radio button.
        Enables interaction with the neighboring combo box.
        """
        self._combo_box_existing_models.setEnabled(True)

    def _timeout_checkbox_slot(self, checked: Qt.CheckState) -> None:
        """
        Triggered when the user selects the "time out after" _timeout_checkbox.
        Enables/disables interaction with the neighboring hour input based on checkstate.
        """
        if checked == Qt.Checked:
            self._max_time_in_hours_input.setEnabled(True)
        else:
            self._max_time_in_hours_input.setEnabled(False)
