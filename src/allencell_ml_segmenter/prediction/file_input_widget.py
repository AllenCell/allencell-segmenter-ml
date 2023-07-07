from PyQt5.QtWidgets import QFrame, QListWidget
from qtpy.QtGui import QPixmap
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSizePolicy,
    QRadioButton,
    QLineEdit,
    QPushButton,
    QComboBox,
)

from allencell_ml_segmenter.prediction.label_with_hint_widget import (
    LabelWithHint,
)
from allencell_ml_segmenter.views.view import View
from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.core.directories import Directories
from allencell_ml_segmenter.widgets.check_box_list_widget import (
    CheckBoxListWidget,
)


class PredictionFileInput(QWidget):
    """
    A widget containing file inputs for the input to a model prediction.
    """

    TOP_TEXT: str = "Select on-screen image(s)"
    BOTTOM_TEXT: str = "Select image(s) from a directory"

    def __init__(self):
        super().__init__()
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setAlignment(Qt.AlignTop)

        # radiobox for images from napari
        horiz_layout = QHBoxLayout()
        horiz_layout.setSpacing(0)  # NEW

        # OLD: self.radio_on_screen = QRadioButton("Select on-screen image:")
        self._radio_on_screen = QRadioButton()  # NEW

        horiz_layout.addWidget(self._radio_on_screen)

        # OLD: question_label = QLabel()
        # OLD: question_label.setPixmap(QPixmap(f"{Directories.get_assets_dir()}/icons/question-circle.svg"))
        question_label = LabelWithHint(PredictionFileInput.TOP_TEXT)  # NEW

        horiz_layout.addWidget(question_label)

        horiz_layout.addStretch(60)  # NEW

        self.layout().addLayout(horiz_layout)

        # seperator_line = QFrame()
        # seperator_line.setFrameShape(QFrame.HLine)
        # seperator_line.setFrameShadow(QFrame.Sunken)
        # horiz_layout.addWidget(seperator_line)

        # list of available images on napari
        self.image_list = CheckBoxListWidget()
        self.layout().addWidget(self.image_list)

        # radiobox for images from directory
        horiz_layout = QHBoxLayout()
        horiz_layout.setSpacing(0)  # NEW

        # OLD: self.radio_directory = QRadioButton("Select image(s) from directory:")
        self.radio_directory = QRadioButton()  # NEW

        horiz_layout.addWidget(self.radio_directory)

        # OLD: question_label = QLabel()
        # OLD: question_label.setPixmap(QPixmap(f"{Directories.get_assets_dir()}/icons/question-circle.svg"))
        question_label = LabelWithHint(PredictionFileInput.BOTTOM_TEXT)  # NEW

        horiz_layout.addWidget(question_label)
        self.browse_dir_edit = QLineEdit()

        horiz_layout.addStretch(5)  # NEW

        horiz_layout.addWidget(self.browse_dir_edit)
        self.browse_dir_button = QPushButton("Browse")
        horiz_layout.addWidget(self.browse_dir_button)
        self.layout().addLayout(horiz_layout)

        horiz_layout = QHBoxLayout()
        image_input_label = LabelWithHint("Image input channel: ")
        self.channel_select_dropdown = QComboBox()
        horiz_layout.addWidget(image_input_label)
        horiz_layout.addWidget(self.channel_select_dropdown)
        self.layout().addLayout(horiz_layout)

        horiz_layout = QHBoxLayout()
        output_dir_label = LabelWithHint("Output directory: ")
        horiz_layout.addWidget(output_dir_label)
        self.out_dir_edit = QLineEdit()
        horiz_layout.addWidget(self.out_dir_edit)
        self.browse_output_dir_button = QPushButton("Browse")
        horiz_layout.addWidget(self.browse_output_dir_button)
        self.layout().addLayout(horiz_layout)

        # seperator_line = QFrame()
        self.layout().addStretch()  # add stretch to bottom to push to top
