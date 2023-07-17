from PyQt5.QtWidgets import QGridLayout, QLabel
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QRadioButton,
    QComboBox,
)

from allencell_ml_segmenter.widgets.input_button_widget import InputButton
from allencell_ml_segmenter.widgets.label_with_hint_widget import LabelWithHint
from allencell_ml_segmenter.prediction.model import PredictionModel
from allencell_ml_segmenter.widgets.check_box_list_widget import (
    CheckBoxListWidget,
)


class PredictionFileInput(QWidget):
    """
    A widget containing file inputs for the input to a model prediction.
    """

    TOP_TEXT: str = "Select on-screen image(s)"
    BOTTOM_TEXT: str = "Select image(s) from a directory"

    def __init__(self, model: PredictionModel):
        super().__init__()

        self._model = model

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        title: QLabel = QLabel("Input image(s)")
        title.setStyleSheet("background-color: #D9D9D9")
        title.setMaximumHeight(40)
        self.layout().addWidget(title)

        # radiobox for images from napari
        horiz_layout = QHBoxLayout()
        horiz_layout.setSpacing(0)

        self._radio_on_screen = QRadioButton()

        horiz_layout.addWidget(self._radio_on_screen)

        question_label = LabelWithHint(PredictionFileInput.TOP_TEXT)
        horiz_layout.addWidget(question_label)

        horiz_layout.addStretch(60)

        self.layout().addLayout(horiz_layout)

        # list of available images on napari
        self.image_list = CheckBoxListWidget()
        self.layout().addWidget(self.image_list)

        # radiobox for images from directory
        horiz_layout = QHBoxLayout()
        horiz_layout.setSpacing(0)

        self.radio_directory = QRadioButton()

        horiz_layout.addWidget(self.radio_directory)

        question_label = LabelWithHint(PredictionFileInput.BOTTOM_TEXT)
        horiz_layout.addWidget(question_label)

        horiz_layout.addStretch(5)

        self.browse_dir_edit = InputButton(self._model)
        horiz_layout.addWidget(self.browse_dir_edit)
        self.layout().addLayout(horiz_layout)

        grid_layout = QGridLayout()

        image_input_label = LabelWithHint("Image input channel")

        self.channel_select_dropdown = QComboBox()

        output_dir_label = LabelWithHint("Output directory")

        self.browse_output_edit = InputButton(self._model)

        grid_layout.addWidget(image_input_label, 0, 0)
        grid_layout.addWidget(self.channel_select_dropdown, 0, 1)

        grid_layout.addWidget(output_dir_label, 1, 0)
        grid_layout.addWidget(self.browse_output_edit, 1, 1)

        grid_layout.setColumnStretch(0, 1)
        grid_layout.setColumnStretch(1, 0)

        self.layout().addLayout(grid_layout)
