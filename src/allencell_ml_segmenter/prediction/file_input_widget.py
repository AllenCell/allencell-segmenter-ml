from pathlib import Path
from typing import List

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QRadioButton,
    QComboBox,
    QGridLayout,
    QLabel,
    QFrame,
    QSizePolicy,
)

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.widgets.input_button_widget import (
    InputButton,
    FileInputMode,
)
from allencell_ml_segmenter.widgets.label_with_hint_widget import LabelWithHint
from allencell_ml_segmenter.prediction.model import (
    PredictionModel,
    PredictionInputMode,
)
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

        self._model: PredictionModel = model

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        frame: QFrame = QFrame()
        frame.setLayout(QVBoxLayout())

        frame.setObjectName("frame")

        title: LabelWithHint = LabelWithHint("Input image(s)")
        # TODO: hints for widget titles?
        title.setObjectName("title")

        self.layout().addWidget(title)
        self.layout().addWidget(frame)

        frame.layout().addWidget(QLabel("Select input image(s):"))

        # radiobox for images from napari
        horiz_layout: QHBoxLayout = QHBoxLayout()
        horiz_layout.setSpacing(0)

        self._radio_on_screen: QRadioButton = QRadioButton()
        self._radio_on_screen.toggled.connect(self._on_screen_slot)
        self._radio_on_screen.setObjectName("onScreen")

        horiz_layout.addWidget(self._radio_on_screen)

        question_label: LabelWithHint = LabelWithHint(
            PredictionFileInput.TOP_TEXT
        )
        question_label.set_hint(
            "Select images to segment from the napari viewer. All images should have the same number and ordering of channels."
        )
        horiz_layout.addWidget(question_label)

        frame.layout().addLayout(horiz_layout)

        # list of available images on napari
        self._image_list: CheckBoxListWidget = CheckBoxListWidget()
        self._image_list.setEnabled(False)
        self._image_list.setObjectName("imageList")
        frame.layout().addWidget(self._image_list)

        # radiobox for images from directory
        horiz_layout = QHBoxLayout()
        horiz_layout.setSpacing(0)

        self._radio_directory: QRadioButton = QRadioButton()
        self._radio_directory.toggled.connect(self._from_directory_slot)
        self._radio_directory.setObjectName("radioDirectory")

        horiz_layout.addWidget(self._radio_directory)

        image_dir_layout: QVBoxLayout = QVBoxLayout()

        question_label = LabelWithHint(PredictionFileInput.BOTTOM_TEXT)
        question_label.set_hint(
            "Select images to segment from a folder. All images should have the same number and ordering of channels."
        )
        question_label.add_right_space(10)

        guide_text: QLabel = QLabel()
        guide_text.setText(
            "Accepts CSV files, <a href='https://www.allencell.org/segmenter.html'>see instructions</a>"
        )
        guide_text.setObjectName("guideText")
        guide_text.setTextFormat(Qt.RichText)
        guide_text.setOpenExternalLinks(True)

        image_dir_layout.addWidget(question_label)
        image_dir_layout.addWidget(guide_text)

        horiz_layout.addLayout(image_dir_layout)

        horiz_layout.addStretch(5)

        self._browse_dir_edit: InputButton = InputButton(
            self._model,
            lambda dir: self._model.set_input_image_path(
                Path(dir), extract_channels=True
            ),
            "Select directory...",
            FileInputMode.DIRECTORY_OR_CSV,
        )
        self._browse_dir_edit.setEnabled(False)
        horiz_layout.addWidget(self._browse_dir_edit)
        frame.layout().addLayout(horiz_layout)

        grid_layout: QGridLayout = QGridLayout()

        image_input_label: LabelWithHint = LabelWithHint("Image input channel")
        image_input_label.set_hint("0-indexed channel in image to segment.")

        self._channel_select_dropdown: QComboBox = QComboBox()

        # set up disappearing placeholder text
        self._channel_select_dropdown.setCurrentIndex(-1)
        self._channel_select_dropdown.setPlaceholderText(
            "select a channel index"
        )
        self._channel_select_dropdown.currentIndexChanged.connect(
            self._model.set_image_input_channel_index
        )
        self._channel_select_dropdown.setEnabled(False)
        # Event to trigger combobox populate on input image directory selection
        self._model.subscribe(
            Event.ACTION_PREDICTION_INPUT_PATH_SELECTED,
            self,
            self._populate_input_channel_combobox,
        )

        output_dir_label: LabelWithHint = LabelWithHint("Output directory")
        output_dir_label.set_hint("Location to save segmentations.")

        self._browse_output_edit: InputButton = InputButton(
            self._model,
            lambda dir: self._model.set_output_directory(dir),
            "Select directory...",
            FileInputMode.DIRECTORY,
        )

        grid_layout.addWidget(image_input_label, 0, 0)
        grid_layout.addWidget(self._channel_select_dropdown, 0, 1)

        grid_layout.addWidget(output_dir_label, 1, 0)
        grid_layout.addWidget(self._browse_output_edit, 1, 1)

        grid_layout.setColumnStretch(0, 1)
        grid_layout.setColumnStretch(1, 0)

        frame.layout().addLayout(grid_layout)

    def _on_screen_slot(self) -> None:
        """Prohibits usage of non-related input fields if top button is checked."""
        self._image_list.setEnabled(True)
        self._browse_dir_edit.setEnabled(False)
        self._model.set_prediction_input_mode(
            PredictionInputMode.FROM_NAPARI_LAYERS
        )

    def _from_directory_slot(self) -> None:
        """Prohibits usage of non-related input fields if bottom button is checked."""
        self._image_list.setEnabled(False)
        self._browse_dir_edit.setEnabled(True)
        self._model.set_prediction_input_mode(PredictionInputMode.FROM_PATH)

    def _populate_input_channel_combobox(self, event: Event = None) -> None:
        values_range: List[str] = [
            str(i) for i in range(self._model.get_max_channels())
        ]
        self._channel_select_dropdown.addItems(values_range)
        self._channel_select_dropdown.setEnabled(True)
