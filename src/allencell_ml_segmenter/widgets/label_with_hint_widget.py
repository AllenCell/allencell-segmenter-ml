from qtpy.QtWidgets import QWidget, QSizePolicy, QHBoxLayout, QLabel
from qtpy.QtGui import QPixmap

from allencell_ml_segmenter.core.directories import Directories


class LabelWithHint(QWidget):
    """
    Compound widget with text label and question mark icon for clear access to tool tips.
    """

    def __init__(
        self, label_text: str = "", value_text: str = "", hint: str = ""
    ):
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
                f"{str(Directories.get_assets_dir())}/icons/question-circle.svg"
            )
        )
        self._question_mark.setObjectName("questionMark")
        self._question_mark.setToolTip(hint)

        self.layout().addWidget(self._question_mark)
        self._question_mark.setVisible(bool(hint))

        self._label: QLabel = QLabel(value_text)
        self.layout().addWidget(self._label)
        self.layout().addStretch(6)

    def set_label_text(self, text: str) -> None:
        """
        Sets the text of the label.
        """
        self._label.setText(text)

    def set_value_text(self, value_text: str) -> None:
        """
        Sets the text of the value.
        """
        self._label.setText(value_text)

    def set_hint(self, hint: str) -> None:
        """
        Sets the tooltip to be displayed when the question icon is hovered over.
        """
        self._question_mark.setToolTip(hint)
        self._question_mark.setVisible(bool(hint))

    def add_right_space(self, marg: int) -> None:
        """
        Sets margin-right such that the question mark icon is not cramped.
        """
        self._question_mark.setStyleSheet(
            "#questionMark {margin-right: " + str(marg) + "}"
        )
