from qtpy.QtWidgets import (
    QWidget,
    QSizePolicy,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
)
from qtpy.QtCore import Qt


class InputButton(QWidget):
    def __init__(self):
        super().__init__()

        # TODO: decide on size policy
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.setLayout(QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

        # TODO: make the text that is displayed responsive to whatever file is selected

        # self.text_display: QLabel = QLabel("Choose a model")
        # self.text_display.setStyleSheet("border: 2px solid gray; margin-right: 2px")
        # self.layout().addWidget(self.text_display, alignment=Qt.AlignLeft)

        self.text_display: QLineEdit = QLineEdit()
        self.text_display.setPlaceholderText("Choose a file...")
        self.text_display.setStyleSheet(
            "border-left: 2px solid gray; "
            + "border-top: 2px solid gray; "
            + "border-bottom: 2px solid gray; "
            + "padding-top: 4px; "
            + "padding-bottom: 4px"
        )
        self.text_display.setReadOnly(
            True
        )  # TODO: potentially problematic for event handlers!!!

        self.layout().addWidget(self.text_display, alignment=Qt.AlignLeft)

        # TODO: does the border on the button render it unusable?
        self.button: QPushButton = QPushButton("Browse")
        # self.button.setStyleSheet("padding-left: 12px; padding-right: 12px; padding-top: 4px; padding-bottom: 2px;")
        self.button.setStyleSheet(
            "padding: 5px; border: 2px solid gray;"
        )  # background-color?
        self.layout().addWidget(self.button, alignment=Qt.AlignLeft)
