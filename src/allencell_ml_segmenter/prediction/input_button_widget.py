from qtpy.QtWidgets import (
    QWidget,
    QSizePolicy,
    QHBoxLayout,
    QPushButton,
    QLineEdit,
)
from qtpy.QtCore import Qt


class InputButton(QWidget):
    """
    Compound widget consisting of QLineEdit and QPushButton side-by-side.
    Useful for selecting files and displaying the chosen file path.
    """

    def __init__(self):
        super().__init__()

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.setLayout(QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

        # text box that will eventually display the chosen file path
        self.text_display: QLineEdit = QLineEdit()
        self.text_display.setPlaceholderText("Choose a file...")
        self.text_display.setStyleSheet(
            "border-left: 2px solid gray; "
            + "border-top: 2px solid gray; "
            + "border-bottom: 2px solid gray; "
            + "padding-top: 4px; "
            + "padding-bottom: 4px"
        )
        self.text_display.setReadOnly(True)

        # button to open file explorer
        self.button: QPushButton = QPushButton("Browse")
        self.button.setStyleSheet("padding: 5px; border: 2px solid gray; background-color: #e8ecfc")

        # add widgets to layout
        self.layout().addWidget(self.text_display, alignment=Qt.AlignLeft)
        self.layout().addWidget(self.button, alignment=Qt.AlignLeft)
