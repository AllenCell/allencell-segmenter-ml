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

from allencell_ml_segmenter.prediction.model import PredictionModel
from allencell_ml_segmenter.widgets.input_button_widget import InputButton
from allencell_ml_segmenter.widgets.label_with_hint_widget import LabelWithHint


class ModelSelectionWidget(QWidget):
    """
    A widget for segmentation model selection.
    """

    TITLE_TEXT: str = "Segmentation model"

    def __init__(self):  # TODO: take in training model as a parameter
        super().__init__()

        # self._model = model

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
        # TODO: convert certain variables to instance variables once model is implemented if their state will be tracked
        frame.layout().addWidget(LabelWithHint("Select a model:"))

        grid_layout: QGridLayout = QGridLayout()

        radio_new: QRadioButton = QRadioButton()
        radio_new.toggled.connect(self._radio_new_slot)
        grid_layout.addWidget(radio_new, 0, 0)

        label_new: LabelWithHint = LabelWithHint("Start a new model")
        grid_layout.addWidget(label_new, 0, 1)

        radio_existing: QRadioButton = QRadioButton()
        radio_existing.toggled.connect(self._radio_existing_slot)
        grid_layout.addWidget(radio_existing, 1, 0)

        label_existing: LabelWithHint = LabelWithHint("Existing model")
        grid_layout.addWidget(label_existing, 1, 1)

        self.combo_box_existing: QComboBox = QComboBox()
        self.combo_box_existing.setCurrentIndex(-1)
        self.combo_box_existing.setPlaceholderText("Select an option")
        grid_layout.addWidget(self.combo_box_existing, 1, 2)

        frame.layout().addLayout(grid_layout)

        # bottom half
        grid_layout: QGridLayout = QGridLayout()

        structure_size_label: LabelWithHint = LabelWithHint("Structure size")
        grid_layout.addWidget(structure_size_label, 0, 0)

        structure_size_combo_box: QComboBox = QComboBox()
        structure_size_combo_box.setCurrentIndex(-1)
        structure_size_combo_box.setPlaceholderText("Select an option")
        structure_size_combo_box.addItems(["small", "med", "large"])
        grid_layout.addWidget(structure_size_combo_box, 0, 1)

        image_dimensions_label: LabelWithHint = LabelWithHint(
            "Image dimension"
        )
        grid_layout.addWidget(image_dimensions_label, 1, 0)

        dimension_choice_layout: QHBoxLayout = QHBoxLayout()

        radio_3d: QRadioButton = QRadioButton()
        label_3d: LabelWithHint = LabelWithHint("3D")

        radio_2d: QRadioButton = QRadioButton()
        label_2d: LabelWithHint = LabelWithHint("2D")

        dimension_choice_layout.addWidget(radio_3d)
        dimension_choice_layout.addWidget(label_3d)
        dimension_choice_layout.addWidget(radio_2d)
        dimension_choice_layout.addWidget(label_2d)

        grid_layout.addLayout(dimension_choice_layout, 1, 1)

        training_step_label: LabelWithHint = LabelWithHint("Training step")
        grid_layout.addWidget(training_step_label, 2, 0)

        training_step_input: QLineEdit = QLineEdit()  # TODO: placeholder text?
        grid_layout.addWidget(training_step_input, 2, 1)

        timeout_layout: QHBoxLayout = QHBoxLayout()

        checkbox: QCheckBox = QCheckBox()
        timeout_layout.addWidget(checkbox)

        left_text: QLabel = QLabel("Time out after")
        timeout_layout.addWidget(left_text)

        hour_input: QLineEdit = QLineEdit()
        timeout_layout.addWidget(hour_input)

        right_text: LabelWithHint = LabelWithHint("hours")
        timeout_layout.addWidget(right_text)

        grid_layout.addLayout(timeout_layout, 3, 1)

        frame.layout().addLayout(grid_layout)

        # apply styling
        self.setStyleSheet("prediction_view.qss")  # TODO: revisit styling

    def _radio_new_slot(self) -> None:
        """
        Triggered when the user selects the "start a new model" radio button.
        Disables interaction with the combo box below.
        """
        # TODO: call corresponding setter once model is implemented
        self.combo_box_existing.setEnabled(False)

    def _radio_existing_slot(self) -> None:
        """
        Triggered when the user selects the "existing model" radio button.
        Enables interaction with the neighboring combo box.
        """
        # TODO: call corresponding setter (likely same as above) once model is implemented
        self.combo_box_existing.setEnabled(True)
