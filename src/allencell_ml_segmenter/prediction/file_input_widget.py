from qtpy.QtGui import QPixmap
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSizePolicy, QRadioButton, QLineEdit, QPushButton
from allencell_ml_segmenter.views.view import View
from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.core.directories import Directories

class PredictionFileInput(QWidget):
    """
    A widget containing file inputs for the input to a model prediction.

    """
    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        # radiobox with hint
        horiz_layout = QHBoxLayout()
        horiz_layout.setSpacing(0)
        horiz_layout.addWidget(self.radio_on_screen)
        question_label = QLabel()
        question_label.setStyleSheet("margin-left: 0px; margin-right: 0px;")
        question_label.setPixmap(QPixmap(Directories.get_assets_dir() / "icons" / "question.png"))
        horiz_layout.addWidget(question_label)
        self.layout().addLayout(horiz_layout)

        horiz_layout = QHBoxLayout()
        self.radio_directory = QRadioButton("Select image(s) from directory:")
        horiz_layout.addWidget(self.radio_directory)
        self.directory_input = QLineEdit()
        self.submit_directory_input = QPushButton("Submit")
        horiz_layout.addWidget(self.directory_input)
        horiz_layout.addWidget(self.submit_directory_input)
        self.layout().addLayout(horiz_layout)


