from qtpy.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
    QLabel,
)
from typing import Callable


class SelectionWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        # Controller
        self.training_button = QPushButton("Training View")
        self.training_button.clicked.connect(self.start_training)
        self.prediction_button = QPushButton("Prediction View")

        # add buttons
        self.layout().addWidget(self.training_button)
        self.layout().addWidget(self.prediction_button)



