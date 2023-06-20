from qtpy.QtWidgets import QWidget, QSizePolicy, QHBoxLayout, QLabel
from qtpy.QtGui import QPixmap
from qtpy.QtCore import Qt


class LabelWithHint(QWidget):
    # eventually pass in (a) text for the QLabel and (b) pop-up text for the question button
    def __init__(self, text: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: decide on size policy
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.setLayout(QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

        title: QLabel = QLabel(text)
        title.setStyleSheet("margin-left: 8px")
        self.layout().addWidget(title, alignment=Qt.AlignLeft)

        # TODO: hook this up to an event handler responsive to mouse hovers
        self.pixmap_container: QLabel = QLabel()
        im = QPixmap("../assets/icons/question-circle.svg")
        self.pixmap_container.setPixmap(im)
        self.pixmap_container.setStyleSheet("margin-right: 10px")
        self.layout().addWidget(self.pixmap_container, alignment=Qt.AlignLeft)
        self.layout().addStretch(6)
