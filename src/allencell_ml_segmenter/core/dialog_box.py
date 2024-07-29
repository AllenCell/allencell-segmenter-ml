from qtpy.QtWidgets import (
    QDialog,
    QWidget,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QPushButton,
)


class DialogBox(QDialog):
    """
    Warning message Widget with a yellow warning sign icon on the left.
    """

    def __init__(self, message: str, parent: QWidget = None):
        super().__init__(parent=parent)
        self.selection = None

        self.setLayout(QHBoxLayout())

        self._text = QLabel(message)

        self.layout().addStretch()
        self.layout().addWidget(self._text)
        self.layout().addStretch()
        btns = QVBoxLayout()
        self.yes_btn = QPushButton("Yes")
        self.no_btn = QPushButton("No")
        self.yes_btn.clicked.connect(self.yes_selected)
        self.no_btn.clicked.connect(self.no_selected)
        btns.addWidget(self.yes_btn)
        btns.addWidget(self.no_btn)
        self.layout().addLayout(btns)

    @property
    def message(self):
        return self.getMessage()

    def yes_selected(self):
        self.selection = True
        self.accept()

    def no_selected(self):
        self.selection = False
        self.reject()

    def setMessage(self, message: str):
        self._text.setText(message)

    def getMessage(self):
        return self._text.text()

    def get_selection(self):
        return self.selection
