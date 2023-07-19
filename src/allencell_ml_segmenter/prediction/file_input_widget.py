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

    TOP_TEXT: str = "On-screen image(s)"
    BOTTOM_TEXT: str = "Image(s) from a directory"

    def __init__(self, model: PredictionModel):
        super().__init__()

        self._model = model

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        # TODO: have title be in-line with the border if possible (QGroupBox)
        title: QLabel = QLabel("Input image(s)")

        # TODO: remove background color once title has been positioned
        title.setStyleSheet("background-color: #D9D9D9")
        title.setMaximumHeight(20)
        self.layout().addWidget(title)

        # TODO: insert prompt ("Select input image(s):")

        # radiobox for images from napari
        # TODO: indent appropriate elements
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

        # TODO: Gray out input button if associated radio button not selected
        self.radio_directory = QRadioButton()

        horiz_layout.addWidget(self.radio_directory)

        question_label = LabelWithHint(PredictionFileInput.BOTTOM_TEXT)
        horiz_layout.addWidget(question_label)

        horiz_layout.addStretch(5)

        self.browse_dir_edit = InputButton(self._model, "Select directory...")
        horiz_layout.addWidget(self.browse_dir_edit)
        self.layout().addLayout(horiz_layout)

        # TODO: add in disclaimer: “Accept csv file, see instruction” (link leads to tutorial, use dummy link for now)

        grid_layout = QGridLayout()

        image_input_label = LabelWithHint("Image input channel")

        self.channel_select_dropdown = QComboBox()

        # set up disappearing placeholder text
        self.channel_select_dropdown.setCurrentIndex(-1)
        self.channel_select_dropdown.setPlaceholderText(
            "select a channel index"
        )

        output_dir_label = LabelWithHint("Output directory")

        self.browse_output_edit = InputButton(
            self._model, "Select directory..."
        )

        grid_layout.addWidget(image_input_label, 0, 0)
        grid_layout.addWidget(self.channel_select_dropdown, 0, 1)

        grid_layout.addWidget(output_dir_label, 1, 0)
        grid_layout.addWidget(self.browse_output_edit, 1, 1)

        grid_layout.setColumnStretch(0, 1)
        grid_layout.setColumnStretch(1, 0)

        self.layout().addLayout(grid_layout)
