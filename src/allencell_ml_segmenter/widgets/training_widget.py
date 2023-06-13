from typing import Callable, List

from qtpy.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)


class TrainingWidget(QWidget):
    """
    A sample widget for training a models.

    """

    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        self.btns: List[QPushButton] = []

        self.btn: QPushButton = QPushButton("Start Training")
        self.layout().addWidget(self.btn)
        self.btns.append(self.btn)

        self.return_btn: QPushButton = QPushButton("Return")
        self.layout().addWidget(self.return_btn)
        self.btns.append(self.return_btn)

    def connectSlots(self, functions: List[Callable]) -> None:
        for idx, function in enumerate(functions):
            self.btns[idx].clicked.connect(function)
