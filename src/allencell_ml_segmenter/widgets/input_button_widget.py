from typing import Callable

from PyQt5.QtWidgets import QFileDialog
from qtpy.QtWidgets import (
    QWidget,
    QSizePolicy,
    QHBoxLayout,
    QPushButton,
    QLineEdit,
)
from qtpy.QtCore import Qt

from allencell_ml_segmenter._style import Style
from allencell_ml_segmenter.prediction.model import PredictionModel


class InputButton(QWidget):
    """
    Compound widget consisting of QLineEdit and QPushButton side-by-side.
    Useful for selecting files and displaying the chosen file path.
    """

    def __init__(
        self,
        model: PredictionModel,
        model_set_file_path_function: Callable,
        placeholder: str = "Select file...",
    ):
        super().__init__()

        self._model: PredictionModel = model

        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

        self.setLayout(QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

        self._set_path_function = model_set_file_path_function

        # text box that will eventually display the chosen file path
        self._text_display: QLineEdit = QLineEdit()
        self._text_display.setPlaceholderText(placeholder)
        self._text_display.setObjectName("textDisplay")
        self._text_display.setReadOnly(True)

        # button to open file explorer
        self._button: QPushButton = QPushButton("Browse")
        self._button.setObjectName("button")

        # add widgets to layout
        self.layout().addWidget(self._text_display, alignment=Qt.AlignLeft)
        self.layout().addWidget(self._button, alignment=Qt.AlignLeft)

        # connect to slot
        self._button.clicked.connect(
            lambda: self._update_path_text(
                QFileDialog.getOpenFileName(self, "Open file")[0]
            )
        )
        # connect to stylesheet
        self.setStyleSheet(Style.get_stylesheet("input_button_widget.qss"))

    def _update_path_text(self, path: str) -> None:
        """
        Displays path on label portion of input button.
        """
        self._text_display.setReadOnly(False)
        self._text_display.setText(path)
        self._text_display.setReadOnly(True)
        self._set_path_function(path)
