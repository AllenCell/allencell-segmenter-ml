from allencell_ml_segmenter.curation.curation_service import CurationService
from allencell_ml_segmenter.core.view import View
from allencell_ml_segmenter.curation.stacked_spinner import StackedSpinner
from allencell_ml_segmenter.widgets.input_button_widget import (
    InputButton,
    FileInputMode,
)
from allencell_ml_segmenter.widgets.label_with_hint_widget import LabelWithHint
from allencell_ml_segmenter.curation.curation_model import (
    CurationModel,
    CurationView,
)
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QSizePolicy,
    QLabel,
    QFrame,
    QGridLayout,
    QVBoxLayout,
    QComboBox,
    QPushButton,
    QWidget,
)
from pathlib import Path
from napari.utils.notifications import show_info


class CurationInputView(QWidget):
    """
    View for Curation UI
    """

    def __init__(self, curation_model: CurationModel) -> None:
        super().__init__()
        self._curation_model: CurationModel = curation_model

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)
        self.layout().setAlignment(Qt.AlignTop)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        self._title: QLabel = QLabel("CURATION UI", self)
        self._title.setObjectName("title")
        self.layout().addWidget(
            self._title, alignment=Qt.AlignHCenter | Qt.AlignTop
        )

        frame: QFrame = QFrame()
        frame.setLayout(QVBoxLayout())
        frame.setObjectName("frame")
        self.layout().addWidget(frame)

        input_images_label: QLabel = QLabel("Input images")
        frame.layout().addWidget(input_images_label)

        raw_grid_layout: QGridLayout = QGridLayout()

        # First Row in Gridlayout
        raw_image_label: LabelWithHint = LabelWithHint("Raw")
        # TODO set hint
        raw_grid_layout.addWidget(
            raw_image_label, 0, 0, alignment=Qt.AlignLeft
        )
        raw_grid_layout.addWidget(
            QLabel("Directory"), 0, 1, alignment=Qt.AlignRight
        )
        self.raw_directory_select: InputButton = InputButton(
            self._curation_model,
            self._on_raw_dir_select,
            "Select directory...",
            FileInputMode.DIRECTORY,
        )
        self.raw_dir_stacked_spinner = StackedSpinner(
            self.raw_directory_select
        )
        raw_grid_layout.addWidget(
            self.raw_dir_stacked_spinner, 0, 2, alignment=Qt.AlignRight
        )

        # Second Row in Gridlayout
        raw_grid_layout.addWidget(
            QLabel("Image channel"), 1, 1, alignment=Qt.AlignRight
        )
        self.raw_image_channel_combo: QComboBox = QComboBox()
        self.raw_image_channel_combo.currentIndexChanged.connect(
            self.raw_channel_selected
        )
        raw_grid_layout.addWidget(
            self.raw_image_channel_combo, 1, 2, alignment=Qt.AlignLeft
        )

        # add grid to frame
        frame.layout().addLayout(raw_grid_layout)

        seg1_grid_layout: QGridLayout = QGridLayout()
        # First Row in Gridlayout
        seg1_image_label: LabelWithHint = LabelWithHint("Seg 1")
        # TODO set hint
        seg1_grid_layout.addWidget(
            seg1_image_label, 0, 0, alignment=Qt.AlignLeft
        )
        seg1_grid_layout.addWidget(
            QLabel("Directory"), 0, 1, alignment=Qt.AlignRight
        )
        # TODO update model accordingly
        self.seg1_directory_select: InputButton = InputButton(
            self._curation_model,
            self._on_seg1_dir_select,
            "Select directory...",
            FileInputMode.DIRECTORY,
        )
        self.seg1_dir_stacked_spinner = StackedSpinner(
            self.seg1_directory_select
        )
        seg1_grid_layout.addWidget(
            self.seg1_dir_stacked_spinner, 0, 2, alignment=Qt.AlignRight
        )

        # Second Row in Gridlayout
        seg1_grid_layout.addWidget(
            QLabel("Image channel"), 1, 1, alignment=Qt.AlignRight
        )
        self.seg1_image_channel_combo: QComboBox = QComboBox()
        self.seg1_image_channel_combo.currentIndexChanged.connect(
            self.seg1_channel_selected
        )
        seg1_grid_layout.addWidget(
            self.seg1_image_channel_combo, 1, 2, alignment=Qt.AlignLeft
        )

        # add grid to frame
        frame.layout().addLayout(seg1_grid_layout)

        seg2_grid_layout: QGridLayout = QGridLayout()
        # First Row in Gridlayout
        seg2_image_label: LabelWithHint = LabelWithHint("Seg 2")
        # TODO set hint
        seg2_grid_layout.addWidget(
            seg2_image_label, 0, 0, alignment=Qt.AlignLeft
        )
        seg2_grid_layout.addWidget(
            QLabel("Directory"), 0, 1, alignment=Qt.AlignRight
        )
        # TODO update model accordingly
        self.seg2_directory_select: InputButton = InputButton(
            self._curation_model,
            self._on_seg2_dir_select,
            "Select directory...",
            FileInputMode.DIRECTORY,
        )
        self.seg2_dir_stacked_spinner = StackedSpinner(
            self.seg2_directory_select
        )
        seg2_grid_layout.addWidget(
            self.seg2_dir_stacked_spinner, 0, 2, alignment=Qt.AlignRight
        )

        # Second Row in Gridlayout
        seg2_grid_layout.addWidget(
            QLabel("Image channel"), 1, 1, alignment=Qt.AlignRight
        )
        self.seg2_image_channel_combo: QComboBox = QComboBox()
        self.seg2_image_channel_combo.currentIndexChanged.connect(
            self.seg2_channel_selected
        )
        seg2_grid_layout.addWidget(
            self.seg2_image_channel_combo, 1, 2, alignment=Qt.AlignLeft
        )

        # add grid to frame
        frame.layout().addLayout(seg2_grid_layout)

        self.start_btn: QPushButton = QPushButton("Start")
        self.start_btn.clicked.connect(self._on_start)
        frame.layout().addWidget(self.start_btn)

        # subscribers
        self._curation_model.raw_image_channel_count_set.connect(
            self.update_raw_channels
        )
        self._curation_model.seg1_image_channel_count_set.connect(
            self.update_seg1_channels
        )
        self._curation_model.seg2_image_channel_count_set.connect(
            self.update_seg2_channels
        )

    def _on_start(self) -> None:
        if any(
            [
                value is None
                for value in [
                    self._curation_model.get_raw_directory(),
                    self._curation_model.get_raw_channel(),
                    self._curation_model.get_seg1_directory(),
                    self._curation_model.get_seg1_channel(),
                ]
            ]
        ):
            show_info(
                "Please select a directory and channel for at least raw and seg1."
            )
            return

        if (
            self._curation_model.get_seg2_directory() is not None
            and self._curation_model.get_seg2_channel() is None
        ):
            show_info("Please select a channel for seg2.")
            return

        self._curation_model.set_current_view(CurationView.MAIN_VIEW)

    def _set_to_loading(
        self, combobox: QComboBox, stacked_spinner: StackedSpinner
    ) -> None:
        stacked_spinner.start()
        combobox.clear()
        combobox.setPlaceholderText("loading channels...")
        combobox.setCurrentIndex(-1)
        combobox.setEnabled(False)

    def _set_to_stopped(
        self,
        combobox: QComboBox,
        stacked_spinner: StackedSpinner,
        input_button: InputButton,
    ) -> None:
        # reset everything to default
        stacked_spinner.stop()
        combobox.clear()
        combobox.setPlaceholderText("")
        combobox.setEnabled(True)
        input_button.clear_selection()

    def _on_raw_dir_select(self, dir: Path) -> None:
        self._set_to_loading(
            self.raw_image_channel_combo, self.raw_dir_stacked_spinner
        )
        self._curation_model.set_raw_directory(dir)

    def _on_seg1_dir_select(self, dir: Path) -> None:
        self._set_to_loading(
            self.seg1_image_channel_combo, self.seg1_dir_stacked_spinner
        )
        self._curation_model.set_seg1_directory(dir)

    def _on_seg2_dir_select(self, dir: Path) -> None:
        self._set_to_loading(
            self.seg2_image_channel_combo, self.seg2_dir_stacked_spinner
        )
        self._curation_model.set_seg2_directory(dir)

    def _populate_channel_combo(
        self, channel_combo: QComboBox, num_channels: int
    ):
        channel_combo.clear()
        if num_channels > 0:
            channel_combo.addItems([str(x) for x in range(num_channels)])
            channel_combo.setCurrentIndex(0)
            channel_combo.setEnabled(True)
        else:
            channel_combo.setPlaceholderText("")
            channel_combo.setEnabled(False)

    def update_raw_channels(self) -> None:
        """
        Event handler when raw image directory is selected. Updates combobox to the correct number of channels in the
        images from the raw directory.
        """
        self.raw_dir_stacked_spinner.stop()
        self._populate_channel_combo(
            self.raw_image_channel_combo,
            self._curation_model.get_raw_image_channel_count(),
        )
        self._curation_model.set_raw_channel(0)

    def update_seg1_channels(self) -> None:
        """
        Event handler when seg1 image directory is selected. Updates combobox to the correct number of channels in the
        images from the seg1 directory.
        """
        self.seg1_dir_stacked_spinner.stop()
        self._populate_channel_combo(
            self.seg1_image_channel_combo,
            self._curation_model.get_seg1_image_channel_count(),
        )
        self._curation_model.set_seg1_channel(0)

    def update_seg2_channels(self) -> None:
        """
        Event handler when seg2 image directory is selected. Updates combobox to the correct number of channels in the
        images from the seg2 directory.
        """
        self.seg2_dir_stacked_spinner.stop()
        self._populate_channel_combo(
            self.seg2_image_channel_combo,
            self._curation_model.get_seg2_image_channel_count(),
        )
        self._curation_model.set_seg2_channel(0)

    def raw_channel_selected(self, index) -> None:
        """
        Event handler when combobox channel selection is made. Sets the raw channel index in the model.
        """
        self._curation_model.set_raw_channel(index)

    def seg1_channel_selected(self, index) -> None:
        """
        Event handler when combobox channel selection is made. Sets the seg1 channel index in the model.
        """
        self._curation_model.set_seg1_channel(index)

    def seg2_channel_selected(self, index) -> None:
        """
        Event handler when combobox channel selection is made. Sets the seg2 channel index in the model.
        """
        self._curation_model.set_seg2_channel(index)
