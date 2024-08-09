from typing import Optional
from qtpy.QtWidgets import (
    QDialog,
    QWidget,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QPushButton,
)


class InfoDialogBox(QDialog):
    """
    Warning message Widget with a yellow warning sign icon on the left.
    """

    def __init__(self, message: str, parent: Optional[QWidget] = None):
        super().__init__(parent=parent)
        layout: QHBoxLayout = QHBoxLayout()
        self.setLayout(layout)
        layout.addStretch()
        layout.addWidget(QLabel(message))
        layout.addStretch()
        btns = QVBoxLayout()
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self._on_close_clicked)
        btns.addWidget(self.close_btn)
        layout.addLayout(btns)

    def _on_close_clicked(self) -> None:
        self.close()
