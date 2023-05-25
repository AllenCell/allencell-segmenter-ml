from qtpy.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
    QLabel,
)
from typing import Callable, List


class SampleWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        self.btns: List[QPushButton] = []

        self.btn: QPushButton = QPushButton("Start Training")
        self.layout().addWidget(self.btn)
        self.btns.append(self.btn)

        self.label: QLabel = QLabel("")
        self.layout().addWidget(self.label)

        self.return_btn: QPushButton = QPushButton("Return")
        self.layout().addWidget(self.return_btn)
        self.btns.append(self.return_btn)

    def setLabelText(self, text: str) -> None:
        self.label.setText(text)

    def connectSlots(self, functions: List[Callable]):
        print("buttons connected")
        for idx, function in enumerate(functions):
            self.btns[idx].clicked.connect(function)

