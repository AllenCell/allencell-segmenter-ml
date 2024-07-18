from pathlib import Path
from typing import Optional
import json

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

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.curation.stacked_spinner import StackedSpinner
from allencell_ml_segmenter.main.i_experiments_model import IExperimentsModel
from allencell_ml_segmenter.training.training_model import (
    TrainingModel,
    ImageType,
)
from allencell_ml_segmenter.widgets.input_button_widget import (
    InputButton,
    FileInputMode,
)
from allencell_ml_segmenter.widgets.label_with_hint_widget import LabelWithHint


class ImageSelectionWidget(QWidget):
    """
    A widget for training image selection.
    """

    TITLE_TEXT: str = "Training images"

    def __init__(
        self, model: TrainingModel, experiments_model: IExperimentsModel
    ):
        super().__init__()

        self._model: TrainingModel = model
        self._experiments_model: IExperimentsModel = experiments_model

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
        directory_label: LabelWithHint = LabelWithHint(
            "Curated image data source"
        )
        directory_label.set_hint(
            "The image dataset output by Curation. Directory should contain train.csv, val.csv, and test.csv files."
        )
        self._images_directory_input_button: InputButton = InputButton(
            self._model,
            lambda dir: self._on_input_images_select(dir),
            "Select directory...",
            FileInputMode.DIRECTORY_OR_CSV,
        )
        self._images_directory_input_button.elongate(248)
        self._training_data_stacked_spinner = StackedSpinner(
            self._images_directory_input_button
        )

        frame.layout().setSpacing(10)
        frame.layout().addWidget(directory_label, 0, 0, Qt.AlignVCenter)

        # remove until we provide tutorial for manually creating csv
        # guide_text: QLabel = QLabel()
        # guide_text.setText(
        #    "<a href='https://www.allencell.org/segmenter.html'>See instructions</a> for CSV file"
        # )
        # guide_text.setObjectName("guideText")
        # guide_text.setTextFormat(Qt.RichText)
        # guide_text.setOpenExternalLinks(True)

        frame.layout().addWidget(
            self._training_data_stacked_spinner, 0, 1, Qt.AlignVCenter
        )
        # frame.layout().addWidget(guide_text, 1, 1, Qt.AlignTop)

        self._raw_channel_combo_box: QComboBox = QComboBox()
        self._raw_channel_combo_box.setMinimumWidth(306)
        self._reset_combo_box(self._raw_channel_combo_box, None)
        self._raw_channel_combo_box.currentIndexChanged.connect(
            lambda idx: self._handle_idx_change(idx, ImageType.RAW)
        )
        frame.layout().addWidget(LabelWithHint("Raw image channel"), 2, 0)
        frame.layout().addWidget(self._raw_channel_combo_box, 2, 1)

        self._seg1_channel_combo_box: QComboBox = QComboBox()
        self._seg1_channel_combo_box.setMinimumWidth(306)
        self._reset_combo_box(self._seg1_channel_combo_box, None)
        self._seg1_channel_combo_box.currentIndexChanged.connect(
            lambda idx: self._handle_idx_change(idx, ImageType.SEG1)
        )
        frame.layout().addWidget(LabelWithHint("Seg1 image channel"), 3, 0)
        frame.layout().addWidget(self._seg1_channel_combo_box, 3, 1)

        self._seg2_channel_combo_box: QComboBox = QComboBox()
        self._seg2_channel_combo_box.setMinimumWidth(306)
        self._reset_combo_box(self._seg2_channel_combo_box, None)
        self._seg2_channel_combo_box.currentIndexChanged.connect(
            lambda idx: self._handle_idx_change(idx, ImageType.SEG2)
        )
        frame.layout().addWidget(LabelWithHint("Seg2 image channel"), 4, 0)
        frame.layout().addWidget(self._seg2_channel_combo_box, 4, 1)

        self._experiments_model.subscribe(
            Event.ACTION_EXPERIMENT_APPLIED, self, self.set_inputs_csv
        )

        self._model.signals.num_channels_set.connect(self._update_channels)

    def set_inputs_csv(self, event: Event = None):
        if self._experiments_model.get_csv_path() is not None:
            csv_path = self._experiments_model.get_csv_path() / "train.csv"
            if csv_path.is_file():
                # if the csv exists
                self._images_directory_input_button._text_display.setText(
                    str(self._experiments_model.get_csv_path())
                )
                # This also dispatches channel extraction
                self._model.set_images_directory(
                    self._experiments_model.get_csv_path()
                )
            else:
                self._images_directory_input_button._text_display.setText("")
                self._model.set_images_directory(None)

    def _on_input_images_select(self, dir: Path) -> None:
        self._set_to_loading()
        # This also dispatches channel extraction
        self._model.set_images_directory(dir)

    def _set_to_loading(self) -> None:
        self._training_data_stacked_spinner.start()
        self._reset_combo_box(self._raw_channel_combo_box, None)
        self._reset_combo_box(self._seg1_channel_combo_box, None)
        self._reset_combo_box(self._seg2_channel_combo_box, None)

    def _handle_idx_change(self, idx: int, img_type: ImageType) -> None:
        self._model.set_selected_channel(img_type, idx if idx >= 0 else None)

    def _reset_combo_box(
        self,
        combo_box: QComboBox,
        num_channels: Optional[int],
        default_channel: Optional[int] = None,
    ) -> None:
        if default_channel is None:
            default_channel = 0

        combo_box.clear()
        if num_channels is not None:
            combo_box.addItems([str(x) for x in range(num_channels)])
            combo_box.setCurrentIndex(
                default_channel if default_channel < num_channels else 0
            )
            combo_box.setEnabled(True)
        else:
            combo_box.setPlaceholderText("")
            combo_box.setCurrentIndex(-1)
            combo_box.setEnabled(False)

    def _update_channels(self) -> None:
        self._training_data_stacked_spinner.stop()
        default_channels: dict[ImageType, Optional[int]] = (
            self._model.get_selected_channels()
        )

        self._reset_combo_box(
            self._raw_channel_combo_box,
            self._model.get_num_channels(ImageType.RAW),
            default_channel=default_channels[ImageType.RAW],
        )
        self._reset_combo_box(
            self._seg1_channel_combo_box,
            self._model.get_num_channels(ImageType.SEG1),
            default_channel=default_channels[ImageType.SEG1],
        )
        self._reset_combo_box(
            self._seg2_channel_combo_box,
            self._model.get_num_channels(ImageType.SEG2),
            default_channel=default_channels[ImageType.SEG2],
        )
