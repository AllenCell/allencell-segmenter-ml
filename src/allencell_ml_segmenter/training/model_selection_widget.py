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
        frame.layout().addWidget(QLabel("Select a model:"))

        grid_layout: QGridLayout = QGridLayout()

        radio_new: QRadioButton = QRadioButton()
        grid_layout.addWidget(radio_new, 0, 0)

        label_new: LabelWithHint = LabelWithHint("Start a new model")
        grid_layout.addWidget(label_new, 0, 1)

        radio_existing: QRadioButton = QRadioButton()
        grid_layout.addWidget(radio_existing, 1, 0)

        label_existing: LabelWithHint = LabelWithHint("Existing model")
        grid_layout.addWidget(label_existing, 1, 1)

        combo_box_existing: QComboBox = QComboBox()
        combo_box_existing.setCurrentIndex(-1)
        combo_box_existing.setPlaceholderText("Select an option")
        grid_layout.addWidget(combo_box_existing, 1, 2)

        frame.layout().addLayout(grid_layout)

        # apply styling
        self.setStyleSheet("prediction_view.qss")  # TODO: revisit styling
