from allencell_ml_segmenter.core.view import View
from allencell_ml_segmenter.main.main_model import MainModel
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


class CurationInputView(View):
    """
    View for Curation UI
    """

    def __init__(self, curation_model: CurationModel):
        super().__init__()
        self._curation_model = curation_model

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

        input_images_label = QLabel("Input images")
        frame.layout().addWidget(input_images_label)

        raw_grid_layout: QGridLayout = QGridLayout()

        # First Row in Gridlayout
        raw_image_label = LabelWithHint("Raw")
        # TODO set hint
        raw_grid_layout.addWidget(
            raw_image_label, 0, 0, alignment=Qt.AlignLeft
        )
        raw_grid_layout.addWidget(
            QLabel("Directory"), 0, 1, alignment=Qt.AlignRight
        )
        self._raw_directory_select: InputButton = InputButton(
            self._curation_model,
            lambda dir: self._curation_model.set_raw_directory(dir),
            "Select directory...",
            FileInputMode.DIRECTORY,
        )
        raw_grid_layout.addWidget(self._raw_directory_select, 0, 2)

        # Second Row in Gridlayout
        raw_grid_layout.addWidget(
            QLabel("Image channel"), 1, 1, alignment=Qt.AlignRight
        )
        self._raw_image_channel_combo: QComboBox = QComboBox()
        raw_grid_layout.addWidget(
            self._raw_image_channel_combo, 1, 2, alignment=Qt.AlignLeft
        )

        # add grid to frame
        frame.layout().addLayout(raw_grid_layout)

        seg1_grid_layout: QGridLayout = QGridLayout()
        # First Row in Gridlayout
        seg1_image_label = LabelWithHint("Seg 1")
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
            lambda dir: self._curation_model.set_raw_directory(dir),
            "Select directory...",
            FileInputMode.DIRECTORY,
        )
        seg1_grid_layout.addWidget(self._seg1_directory_select, 0, 2)

        # Second Row in Gridlayout
        seg1_grid_layout.addWidget(
            QLabel("Image channel"), 1, 1, alignment=Qt.AlignRight
        )
        self._seg1_image_channel_combo: QComboBox = QComboBox()
        seg1_grid_layout.addWidget(
            self._seg1_image_channel_combo, 1, 2, alignment=Qt.AlignLeft
        )

        # add grid to frame
        frame.layout().addLayout(seg1_grid_layout)

        seg2_grid_layout: QGridLayout = QGridLayout()
        # First Row in Gridlayout
        seg2_image_label = LabelWithHint("Seg 2")
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
            lambda dir: self._curation_model.set_raw_directory(dir),
            "Select directory...",
            FileInputMode.DIRECTORY,
        )
        seg2_grid_layout.addWidget(self._seg2_directory_select, 0, 2)

        # Second Row in Gridlayout
        seg2_grid_layout.addWidget(
            QLabel("Image channel"), 1, 1, alignment=Qt.AlignRight
        )
        self._seg2_image_channel_combo: QComboBox = QComboBox()
        seg2_grid_layout.addWidget(
            self._seg2_image_channel_combo, 1, 2, alignment=Qt.AlignLeft
        )

        # add grid to frame
        frame.layout().addLayout(seg2_grid_layout)

        self._start_btn = QPushButton("Start")
        self._start_btn.clicked.connect(self._curation_model.set_view)
        frame.layout().addWidget(self._start_btn)

        # subscribers
        self._curation_model.subscribe(
            Event.ACTION_CURATION_RAW_SELECTED, self, self.update_raw_channels
        )
        self._curation_model.subscribe(
            Event.ACTION_CURATION_SEG1_SELECTED,
            self,
            self.update_seg1_channels,
        )
        self._curation_model.subscribe(
            Event.ACTION_CURATION_SEG2_SELECTED,
            self,
            self.update_seg2_channels,
        )

    def doWork(self):
        print("work")

    def getTypeOfWork(self):
        print("getwork")

    def showResults(self):
        print("show result")

    def update_raw_channels(self, event):
        self._raw_image_channel_combo.clear()
        self._raw_image_channel_combo.addItems(
            [
                str(x)
                for x in range(
                    self._curation_model.get_total_num_channels_raw()
                )
            ]
        )
        self._raw_image_channel_combo.setCurrentIndex(0)

    def update_seg1_channels(self, event):
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

    def update_seg2_channels(self, event):
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
