from qtpy.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QSizePolicy,
    QRadioButton,
    QLabel,
)


class RadioButtonEntry(QWidget):
    def __init__(self, text):
        super().__init__()

        # TODO: decide on size policy
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.setLayout(QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

        self.button: QRadioButton = QRadioButton()
        self.button.setStyleSheet("margin-left: 25px; margin-right: 6 px")
        # self.button.setStyleSheet("margin: 5px 6px 5px 25px")
        self.layout().addWidget(self.button)

        description: QLabel = QLabel(text)
        self.layout().addWidget(description)
        self.layout().addStretch(6)
