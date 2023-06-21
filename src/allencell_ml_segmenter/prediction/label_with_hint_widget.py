from qtpy.QtWidgets import QWidget, QSizePolicy, QHBoxLayout, QLabel
from qtpy.QtGui import QPixmap
from qtpy.QtCore import Qt


class LabelWithHint(QWidget):
    """
    Compound widget with text label and question mark icon for clear access to tool tips.
    """
    def __init__(self):
        super().__init__()

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.setLayout(QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

        self.label: QLabel = QLabel("")
        self.label.setStyleSheet("margin-left: 8px")
        self.layout().addWidget(self.label, alignment=Qt.AlignLeft)

        self.question_mark: QLabel = QLabel()
        self.question_mark.setPixmap(QPixmap("../assets/icons/question-circle.svg"))
        self.question_mark.setStyleSheet("margin-right: 10px")

        self.layout().addWidget(self.question_mark, alignment=Qt.AlignLeft)
        self.layout().addStretch(6)

    def set_label_text(self, text: str) -> None:
        self.label.setText(text)

    def set_hint(self, hint: str):
        self.question_mark.setToolTip(hint)
