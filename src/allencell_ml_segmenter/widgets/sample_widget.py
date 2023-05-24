from qtpy.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
    QLabel,
)
from typing import Callable


class SampleWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        self.btn: QPushButton = QPushButton("Start Training")
        self.layout().addWidget(self.btn)

        self.label: QLabel = QLabel("")
        self.layout().addWidget(self.label)

    def setLabelText(self, text: str) -> None:
        self.label.setText(text)

    def connectSlots(self, function: Callable):
        self.btn.clicked.connect(function)
