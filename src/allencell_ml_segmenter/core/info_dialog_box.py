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

    def __init__(self, message: str, parent: QWidget = None):
        super().__init__(parent=parent)
        self.setLayout(QHBoxLayout())

        self.layout().addStretch()
        self.layout().addWidget(QLabel(message))
        self.layout().addStretch()
        btns = QVBoxLayout()
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        btns.addWidget(self.close_btn)
        self.layout().addLayout(btns)
