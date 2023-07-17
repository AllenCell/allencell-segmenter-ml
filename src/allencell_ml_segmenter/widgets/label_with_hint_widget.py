from PyQt5.QtGui import QPainter, QPalette
from qtpy.QtWidgets import QWidget, QSizePolicy, QHBoxLayout, QLabel
from qtpy.QtGui import QPixmap

from allencell_ml_segmenter.core.directories import Directories


class LabelWithHint(QWidget):
    """
    Compound widget with text label and question mark icon for clear access to tool tips.
    """

    def __init__(self, label_text: str = ""):
        super().__init__()

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.setLayout(QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

        self._label: QLabel = QLabel("")
        self._label.setText(label_text)
        self.layout().addWidget(self._label)

        self._question_mark: QLabel = QLabel()
        self._question_mark.setPixmap(
            QPixmap(
                f"{Directories.get_assets_dir()}/icons/question-circle.svg"
            )
        )
        self._question_mark.setStyleSheet("margin-right: 10px")

        self.layout().addWidget(self._question_mark)
        self.layout().addStretch(6)

    def set_label_text(self, text: str) -> None:
        """
        Sets the text of the label.
        """
        self._label.setText(text)

    def set_hint(self, hint: str) -> None:
        """
        Sets the tooltip to be displayed when the question icon is hovered over.
        """
        self._question_mark.setToolTip(hint)

        # guard against errant highlighting
        self._question_mark.setStyleSheet(
            "QToolTip {background-color: #282c34}"
        )

    def paintEvent(self, event):
        """
        Overrides the default paint event to set the background color.
        """
        painter = QPainter(self)
        painter.fillRect(
            self.rect(), self.palette().color(QPalette.Background)
        )
