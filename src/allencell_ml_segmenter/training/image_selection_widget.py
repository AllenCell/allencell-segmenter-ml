from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QSizePolicy,
    QFrame,
    QLabel,
    QGridLayout,
    QComboBox,
)

from allencell_ml_segmenter.prediction.model import PredictionModel
from allencell_ml_segmenter.widgets.input_button_widget import InputButton
from allencell_ml_segmenter.widgets.label_with_hint_widget import LabelWithHint


class ImageSelectionWidget(QWidget):
    """
    A widget for training image selection.
    """

    TITLE_TEXT: str = "Training images"

    def __init__(self):  # TODO: take in training model as a parameter
        super().__init__()

        # self._model = model

        # widget skeleton
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        title: LabelWithHint = LabelWithHint(ImageSelectionWidget.TITLE_TEXT)
        title.setObjectName("title")
        self.layout().addWidget(title)

        frame: QFrame = QFrame()
        frame.setLayout(QGridLayout())
        frame.setObjectName("frame")
        self.layout().addWidget(frame)

        # grid contents
        directory_label: LabelWithHint = LabelWithHint("Image directory")
        temp_fake_model: PredictionModel = (
            PredictionModel()
        )  # TODO: get rid of this
        directory_input_button: InputButton = InputButton(
            temp_fake_model, lambda: None, "Select directory..."
        )  # TODO: pass in actual training model
        frame.layout().addWidget(directory_label, 0, 0)

        guide_text: QLabel = QLabel()
        guide_text.setText(
            "Accepts CSV files, <a href='https://www.allencell.org/segmenter.html'>see instructions</a>"
        )
        guide_text.setObjectName("guideText")
        guide_text.setTextFormat(Qt.RichText)
        guide_text.setOpenExternalLinks(True)

        directory_layout: QVBoxLayout = QVBoxLayout()
        directory_layout.addWidget(directory_input_button)
        directory_layout.addWidget(guide_text)
        frame.layout().addLayout(directory_layout, 0, 1)

        channel_label: LabelWithHint = LabelWithHint("Image channel")
        channel_combo_box: QComboBox = QComboBox()
        channel_combo_box.setCurrentIndex(-1) # need to set this to see placeholdertext
        channel_combo_box.setPlaceholderText("Select an option")

        frame.layout().addWidget(channel_label, 1, 0)
        frame.layout().addWidget(channel_combo_box, 1, 1)

        # apply styling
        self.setStyleSheet("prediction_view.qss")  # TODO: revisit styling
