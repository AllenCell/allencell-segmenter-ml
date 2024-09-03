from qtpy.QtWidgets import (
    QDialog,
    QWidget,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QPushButton,
)
from typing import Optional


class DialogBox(QDialog):
    """
    Warning message Widget with a yellow warning sign icon on the left.
    """

    def __init__(self, message: str, parent: Optional[QWidget] = None):
        super().__init__(parent=parent)
        self.selection: Optional[bool] = None
        layout: QHBoxLayout = QHBoxLayout()
        self.setLayout(layout)

        self._text = QLabel(message)

        layout.addStretch()
        layout.addWidget(self._text)
        layout.addStretch()
        btns = QVBoxLayout()
        self.yes_btn = QPushButton("Yes")
        self.no_btn = QPushButton("No")
        self.yes_btn.clicked.connect(self.yes_selected)
        self.no_btn.clicked.connect(self.no_selected)
        btns.addWidget(self.yes_btn)
        btns.addWidget(self.no_btn)
        layout.addLayout(btns)

    @property
    def message(self) -> str:
        return self.getMessage()

    def yes_selected(self) -> None:
        self.selection = True
        self.accept()

    def no_selected(self) -> None:
        self.selection = False
        self.reject()

    def setMessage(self, message: str) -> None:
        self._text.setText(message)

    def getMessage(self) -> str:
        return self._text.text()

    def get_selection(self) -> Optional[bool]:
        return self.selection
