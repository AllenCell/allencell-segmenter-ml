from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QSizePolicy,
    QFrame,
    QLabel,
    QGridLayout,
    QComboBox,
)

from allencell_ml_segmenter.training.training_model import TrainingModel
from allencell_ml_segmenter.widgets.input_button_widget import InputButton
from allencell_ml_segmenter.widgets.label_with_hint_widget import LabelWithHint


class ImageSelectionWidget(QWidget):
    """
    A widget for training image selection.
    """

    TITLE_TEXT: str = "Training images"

    def __init__(self, model: TrainingModel):
        super().__init__()

        self._model: TrainingModel = model

        # widget skeleton
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        title: LabelWithHint = LabelWithHint(ImageSelectionWidget.TITLE_TEXT)
        # TODO: hints for widget titles?
        title.setObjectName("title")
        self.layout().addWidget(title)

        frame: QFrame = QFrame()
        frame.setLayout(QGridLayout())
        frame.layout().setSpacing(0)
        frame.setObjectName("frame")
        self.layout().addWidget(frame)

        # grid contents
        directory_label: LabelWithHint = LabelWithHint("Image directory")
        self._images_directory_input_button: InputButton = InputButton(
            self._model,
            lambda dir: self._model.set_images_directory(dir),
            "Select directory...",
        )
        self._images_directory_input_button.elongate(248)

        frame.layout().addWidget(directory_label, 0, 0, Qt.AlignVCenter)

        guide_text: QLabel = QLabel()
        guide_text.setText(
            "Accepts CSV files, <a href='https://www.allencell.org/segmenter.html'>see instructions</a>"
        )
        guide_text.setObjectName("guideText")
        guide_text.setTextFormat(Qt.RichText)
        guide_text.setOpenExternalLinks(True)

        frame.layout().addWidget(
            self._images_directory_input_button, 0, 1, Qt.AlignVCenter
        )
        frame.layout().addWidget(guide_text, 1, 1, Qt.AlignTop)

        channel_label: LabelWithHint = LabelWithHint("Image channel")

        self._channel_combo_box: QComboBox = QComboBox()
        self._channel_combo_box.setCurrentIndex(
            -1
        )  # need to set this to see placeholdertext
        self._channel_combo_box.setMinimumWidth(306)
        self._channel_combo_box.setPlaceholderText("Select an option")

        self._channel_combo_box.currentTextChanged.connect(
            lambda idx: self._model.set_channel_index(int(idx))
        )

        frame.layout().addWidget(channel_label, 2, 0)
        frame.layout().addWidget(self._channel_combo_box, 2, 1)
