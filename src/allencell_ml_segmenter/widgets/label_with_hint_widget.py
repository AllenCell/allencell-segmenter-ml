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
        self._question_mark.setObjectName("qm")

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

    def add_right_space(self, marg: int) -> None:
        """
        Sets margin-right such that the question mark icon is not cramped.
        """
        self._question_mark.setStyleSheet(
            "#qm {margin-right: " + str(marg) + "}"
        )
