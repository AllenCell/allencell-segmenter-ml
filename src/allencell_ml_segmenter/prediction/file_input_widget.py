from PyQt5.QtWidgets import QGridLayout, QLabel, QFrame, QSizePolicy
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
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        title_frame = QFrame()
        title_frame.setLayout(QVBoxLayout())

        title_frame.setObjectName("tf")
        title_frame.setStyleSheet(
            "#tf {border: 1px solid #AAAAAA; border-radius: 5px}"
        )

        title = QLabel("Input image(s)", self)
        title.setStyleSheet("font-weight: bold")

        self.layout().addWidget(title)
        self.layout().addWidget(title_frame)

        title_frame.layout().addWidget(QLabel("Select input image(s):"))

        # radiobox for images from napari
        horiz_layout = QHBoxLayout()
        horiz_layout.setSpacing(0)

        self._radio_on_screen = QRadioButton()
        self._radio_on_screen.toggled.connect(self._on_screen_slot)
        self._radio_on_screen.setStyleSheet("margin-left: 40px")

        horiz_layout.addWidget(self._radio_on_screen)

        question_label: LabelWithHint = LabelWithHint(
            PredictionFileInput.TOP_TEXT
        )
        question_label.set_hint(
            "Select images to segment from the napari viewer. All images should have the same number and ordering of channels."
        )
        horiz_layout.addWidget(question_label)

        title_frame.layout().addLayout(horiz_layout)

        # list of available images on napari
        self._image_list = CheckBoxListWidget()
        self._image_list.setStyleSheet("margin-left: 40px")
        title_frame.layout().addWidget(self._image_list)

        # radiobox for images from directory
        horiz_layout = QHBoxLayout()
        horiz_layout.setSpacing(0)

        self._radio_directory = QRadioButton()
        self._radio_directory.toggled.connect(self._from_directory_slot)
        self._radio_directory.setStyleSheet("margin-left: 40px")

        horiz_layout.addWidget(self._radio_directory)

        image_dir_layout: QVBoxLayout = QVBoxLayout()

        question_label = LabelWithHint(PredictionFileInput.BOTTOM_TEXT)
        question_label.set_hint(
            "Select images to segment from a folder. All images should have the same number and ordering of channels."
        )
        question_label.add_right_space(10)

        guide_text: QLabel = QLabel()
        guide_text.setText(
            "<font color=#868E93>Accepts CSV files,</font> <a href='https://www.allencell.org/segmenter.html'>see instructions</a>"
        )
        guide_text.setOpenExternalLinks(True)
        guide_text.setStyleSheet("font-size: 8px")

        image_dir_layout.addWidget(question_label)
        image_dir_layout.addWidget(guide_text)

        horiz_layout.addLayout(image_dir_layout)

        horiz_layout.addStretch(5)

        self._browse_dir_edit = InputButton(self._model, "Select directory...")
        horiz_layout.addWidget(self._browse_dir_edit)
        title_frame.layout().addLayout(horiz_layout)

        grid_layout = QGridLayout()

        image_input_label = LabelWithHint("Image input channel")
        image_input_label.set_hint("0-indexed channel in image to segment.")

        self._channel_select_dropdown = QComboBox()

        # set up disappearing placeholder text
        self._channel_select_dropdown.setCurrentIndex(-1)
        self._channel_select_dropdown.setPlaceholderText(
            "select a channel index"
        )

        output_dir_label = LabelWithHint("Output directory")
        output_dir_label.set_hint("Location to save segmentations.")

        self._browse_output_edit = InputButton(
            self._model, "Select directory..."
        )

        grid_layout.addWidget(image_input_label, 0, 0)
        grid_layout.addWidget(self._channel_select_dropdown, 0, 1)

        grid_layout.addWidget(output_dir_label, 1, 0)
        grid_layout.addWidget(self._browse_output_edit, 1, 1)

        grid_layout.setColumnStretch(0, 1)
        grid_layout.setColumnStretch(1, 0)

        title_frame.layout().addLayout(grid_layout)

    def _on_screen_slot(self) -> None:
        """Prohibits usage of non-related input fields if top button is checked."""
        self._image_list.setEnabled(True)
        self._browse_dir_edit.setEnabled(False)

    def _from_directory_slot(self) -> None:
        """Prohibits usage of non-related input fields if bottom button is checked."""
        self._image_list.setEnabled(False)
        self._browse_dir_edit.setEnabled(True)
