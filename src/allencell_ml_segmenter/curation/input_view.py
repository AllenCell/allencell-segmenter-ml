from allencell_ml_segmenter.core.view import View
from allencell_ml_segmenter.curation.curation_service import CurationService
from allencell_ml_segmenter.curation.stacked_spinner import StackedSpinner
from allencell_ml_segmenter.widgets.input_button_widget import (
    InputButton,
    FileInputMode,
)
from allencell_ml_segmenter.widgets.label_with_hint_widget import LabelWithHint
from allencell_ml_segmenter.curation.curation_model import CurationModel
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QSizePolicy,
    QLabel,
    QFrame,
    QGridLayout,
    QVBoxLayout,
    QComboBox,
    QPushButton,
)
from allencell_ml_segmenter.core.event import Event
from pathlib import Path


class CurationInputView(View):
    """
    View for Curation UI
    """

    def __init__(
        self, curation_model: CurationModel, curation_service: CurationService
    ) -> None:
        super().__init__()
        self._curation_model: CurationModel = curation_model
        self._curation_service: CurationService = curation_service

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
        self._raw_directory_select: InputButton = InputButton(
            self._curation_model,
            self._on_raw_dir_select,
            "Select directory...",
            FileInputMode.DIRECTORY,
        )
        self._raw_dir_stacked_spinner = StackedSpinner(
            self._raw_directory_select
        )
        raw_grid_layout.addWidget(
            self._raw_dir_stacked_spinner, 0, 2, alignment=Qt.AlignRight
        )

        # Second Row in Gridlayout
        raw_grid_layout.addWidget(
            QLabel("Image channel"), 1, 1, alignment=Qt.AlignRight
        )
        self._raw_image_channel_combo: QComboBox = QComboBox()
        self._raw_image_channel_combo.activated.connect(
            self.raw_channel_selected
        )
        raw_grid_layout.addWidget(
            self._raw_image_channel_combo, 1, 2, alignment=Qt.AlignLeft
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
        self._seg1_directory_select: InputButton = InputButton(
            self._curation_model,
            self._on_seg1_dir_select,
            "Select directory...",
            FileInputMode.DIRECTORY,
        )
        self._seg1_dir_stacked_spinner = StackedSpinner(
            self._seg1_directory_select
        )
        seg1_grid_layout.addWidget(
            self._seg1_dir_stacked_spinner, 0, 2, alignment=Qt.AlignRight
        )

        # Second Row in Gridlayout
        seg1_grid_layout.addWidget(
            QLabel("Image channel"), 1, 1, alignment=Qt.AlignRight
        )
        self._seg1_image_channel_combo: QComboBox = QComboBox()
        self._seg1_image_channel_combo.activated.connect(
            self.seg1_channel_selected
        )
        seg1_grid_layout.addWidget(
            self._seg1_image_channel_combo, 1, 2, alignment=Qt.AlignLeft
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
        self._seg2_directory_select: InputButton = InputButton(
            self._curation_model,
            self._on_seg2_dir_select,
            "Select directory...",
            FileInputMode.DIRECTORY,
        )
        self._seg2_dir_stacked_spinner = StackedSpinner(
            self._seg2_directory_select
        )
        seg2_grid_layout.addWidget(
            self._seg2_dir_stacked_spinner, 0, 2, alignment=Qt.AlignRight
        )

        # Second Row in Gridlayout
        seg2_grid_layout.addWidget(
            QLabel("Image channel"), 1, 1, alignment=Qt.AlignRight
        )
        self._seg2_image_channel_combo: QComboBox = QComboBox()
        self._seg2_image_channel_combo.activated.connect(
            self.seg2_channel_selected
        )
        seg2_grid_layout.addWidget(
            self._seg2_image_channel_combo, 1, 2, alignment=Qt.AlignLeft
        )

        # add grid to frame
        frame.layout().addLayout(seg2_grid_layout)

        self._start_btn: QPushButton = QPushButton("Start")
        self._start_btn.clicked.connect(self._curation_model.set_view)
        frame.layout().addWidget(self._start_btn)

        # subscribers
        self._curation_model.subscribe(
            Event.ACTION_CURATION_RAW_CHANNELS_SET,
            self,
            self.update_raw_channels,
        )
        self._curation_model.subscribe(
            Event.ACTION_CURATION_SEG1_CHANNELS_SET,
            self,
            self.update_seg1_channels,
        )
        self._curation_model.subscribe(
            Event.ACTION_CURATION_SEG2_CHANNELS_SET,
            self,
            self.update_seg2_channels,
        )

        # error handling events
        self._curation_model.subscribe(
            Event.ACTION_CURATION_RAW_THREAD_ERROR,
            self,
            lambda x: self._set_to_stopped(self._raw_image_channel_combo, self._raw_dir_stacked_spinner)
        )
        self._curation_model.subscribe(
            Event.ACTION_CURATION_SEG1_THREAD_ERROR,
            self,
            lambda x: self._set_to_stopped(self._seg1_image_channel_combo, self._seg1_dir_stacked_spinner)
        )
        self._curation_model.subscribe(
            Event.ACTION_CURATION_SEG2_THREAD_ERROR,
            self,
            lambda x: self._set_to_stopped(self._seg2_image_channel_combo, self._seg2_dir_stacked_spinner)
        )

    def _set_to_loading(
        self, combobox: QComboBox, stacked_spinner: StackedSpinner
    ) -> None:
        stacked_spinner.start()
        combobox.clear()
        combobox.setPlaceholderText("loading channels...")
        combobox.setCurrentIndex(-1)
        combobox.setEnabled(False)

    def _set_to_stopped(self, combobox: QComboBox, stacked_spinner: StackedSpinner) -> None:
        # reset everything to default
        stacked_spinner.stop()
        combobox.clear()
        combobox.setPlaceholderText("")
        combobox.setEnabled(True)

    def _on_raw_dir_select(self, dir: Path) -> None:
        self._set_to_loading(
            self._raw_image_channel_combo, self._raw_dir_stacked_spinner
        )
        self._curation_service.select_directory_raw(dir)

    def _on_seg1_dir_select(self, dir: Path) -> None:
        self._set_to_loading(
            self._seg1_image_channel_combo, self._seg1_dir_stacked_spinner
        )
        self._curation_service.select_directory_seg1(dir)

    def _on_seg2_dir_select(self, dir: Path) -> None:
        self._set_to_loading(
            self._seg2_image_channel_combo, self._seg2_dir_stacked_spinner
        )
        self._curation_service.select_directory_seg2(dir)

    def doWork(self) -> None:
        print("work")

    def getTypeOfWork(self) -> None:
        print("getwork")

    def showResults(self) -> None:
        print("show result")

    def update_raw_channels(self, event) -> None:
        """
        Event handler when raw image directory is selected. Updates combobox to the correct number of channels in the
        images from the raw directory.
        """
        self._raw_dir_stacked_spinner.stop()
        self._raw_image_channel_combo.clear()
        self._raw_image_channel_combo.addItems(
            [
                str(x)
                for x in range(
                    self._curation_model.get_total_num_channels_raw()
                )
            ]
        )
        # default first index
        self._raw_image_channel_combo.setCurrentIndex(0)
        self._raw_image_channel_combo.setEnabled(True)
        self._curation_model.set_raw_channel(0)

    def update_seg1_channels(self, event) -> None:
        """
        Event handler when seg1 image directory is selected. Updates combobox to the correct number of channels in the
        images from the seg1 directory.
        """
        self._seg1_dir_stacked_spinner.stop()
        self._seg1_image_channel_combo.clear()
        self._seg1_image_channel_combo.addItems(
            [
                str(x)
                for x in range(
                    self._curation_model.get_total_num_channels_seg1()
                )
            ]
        )
        self._seg1_image_channel_combo.setCurrentIndex(0)
        self._seg1_image_channel_combo.setEnabled(True)
        self._curation_model.set_seg1_channel(0)

    def update_seg2_channels(self, event) -> None:
        """
        Event handler when seg2 image directory is selected. Updates combobox to the correct number of channels in the
        images from the seg2 directory.
        """
        self._seg2_dir_stacked_spinner.stop()
        self._seg2_image_channel_combo.clear()
        self._seg2_image_channel_combo.addItems(
            [
                str(x)
                for x in range(
                    self._curation_model.get_total_num_channels_seg2()
                )
            ]
        )
        self._seg2_image_channel_combo.setCurrentIndex(0)
        self._seg2_image_channel_combo.setEnabled(True)
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
