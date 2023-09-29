from PyQt5.QtWidgets import QRadioButton, QGridLayout
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QVBoxLayout,
    QSizePolicy,
    QLabel,
    QFrame,
    QHBoxLayout,
    QPushButton,
    QProgressBar,
)

from allencell_ml_segmenter._style import Style
from allencell_ml_segmenter.core.view import View
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.widgets.label_with_hint_widget import LabelWithHint


class CurationMainView(View):
    """
    View for Curation UI
    """

    def __init__(self):
        super().__init__()

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 10, 0, 10)
        self.layout().setSpacing(0)
        self.layout().setAlignment(Qt.AlignTop)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.setStyleSheet(Style.get_stylesheet("curation_main.qss"))

        self._title: QLabel = QLabel("CURATION UI MAIN", self)
        self._title.setObjectName("title")
        self.layout().addWidget(
            self._title, alignment=Qt.AlignHCenter | Qt.AlignTop
        )

        frame: QFrame = QFrame()
        frame.setLayout(QVBoxLayout())
        self.layout().addWidget(frame)

        input_images_label = QLabel("Progress")
        frame.layout().addWidget(input_images_label, alignment=Qt.AlignHCenter)

        progress_bar_layout: QHBoxLayout = QHBoxLayout()
        # Button and progress bar on top row
        self.back_button: QPushButton = QPushButton("◄ Back")
        self.back_button.setObjectName("big_blue_btn")
        progress_bar_layout.addWidget(self.back_button, alignment=Qt.AlignLeft)
        # inner progress bar frame and layout
        inner_progress_frame: QFrame = QFrame()
        inner_progress_frame.setLayout(QHBoxLayout())
        inner_progress_frame.setObjectName("frame")
        progress_bar_label: QLabel = QLabel("Image")
        inner_progress_frame.layout().addWidget(
            progress_bar_label, alignment=Qt.AlignLeft
        )
        self.progress_bar: QProgressBar = QProgressBar()
        inner_progress_frame.layout().addWidget(self.progress_bar)
        self.progress_bar_image_count: QLabel = QLabel("0/0")
        inner_progress_frame.layout().addWidget(
            self.progress_bar_image_count, alignment=Qt.AlignRight
        )
        progress_bar_layout.addWidget(inner_progress_frame)
        self.next_button: QPushButton = QPushButton("Next ►")
        self.next_button.setObjectName("big_blue_btn")
        progress_bar_layout.addWidget(
            self.next_button, alignment=Qt.AlignRight
        )
        self.layout().addLayout(progress_bar_layout)

        self.file_name: QLabel = QLabel("Example_file_1")
        self.layout().addWidget(self.file_name, alignment=Qt.AlignHCenter)

        use_image_frame: QFrame = QFrame()
        use_image_frame.setObjectName("frame")
        use_image_frame.setLayout(QHBoxLayout())
        use_image_frame.layout().addWidget(
            QLabel("Use this image for training")
        )
        self.yes_radio: QRadioButton = QRadioButton("Yes")
        use_image_frame.layout().addWidget(self.yes_radio)
        self.no_radio: QRadioButton = QRadioButton("No")
        use_image_frame.layout().addWidget(self.no_radio)
        self.layout().addWidget(use_image_frame, alignment=Qt.AlignHCenter)

        # Labels for excluding mask
        excluding_mask_labels = QHBoxLayout()
        excluding_mask_label = LabelWithHint("Excluding mask")
        excluding_mask_labels.addWidget(excluding_mask_label)
        excluding_mask_labels.addWidget(
            QLabel("File name..."), alignment=Qt.AlignLeft
        )
        self.layout().addLayout(excluding_mask_labels)
        # buttons for excluding mask
        excluding_mask_buttons = QHBoxLayout()
        excluding_create_button = QPushButton("+ Create")
        excluding_create_button.setObjectName("small_blue_btn")
        excluding_propagate_button = QPushButton("Propagate in 3D")
        excluding_delete_button = QPushButton("Delete")
        excluding_save_button = QPushButton("Save")
        excluding_save_button.setObjectName("small_blue_btn")
        excluding_mask_buttons.addWidget(excluding_create_button)
        excluding_mask_buttons.addWidget(excluding_propagate_button)
        excluding_mask_buttons.addWidget(excluding_delete_button)
        excluding_mask_buttons.addWidget(excluding_save_button)
        self.layout().addLayout(excluding_mask_buttons)

        # Label for Merging mask
        merging_mask_label = LabelWithHint("Merging mask")
        self.layout().addWidget(merging_mask_label)
        # buttons for merging mask
        merging_mask_buttons = QHBoxLayout()
        merging_create_button = QPushButton("+ Create")
        merging_create_button.setObjectName("small_blue_btn")
        merging_propagate_button = QPushButton("Propagate in 3D")
        merging_delete_button = QPushButton("Delete")
        merging_save_button = QPushButton("Save")
        merging_save_button.setObjectName("small_blue_btn")
        merging_mask_buttons.addWidget(merging_create_button)
        merging_mask_buttons.addWidget(merging_propagate_button)
        merging_mask_buttons.addWidget(merging_delete_button)
        merging_mask_buttons.addWidget(merging_save_button)

        self.layout().addLayout(merging_mask_buttons)

    def doWork(self):
        print("work")

    def getTypeOfWork(self):
        print("getwork")

    def showResults(self):
        print("show result")
