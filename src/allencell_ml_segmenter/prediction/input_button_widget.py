from PyQt5.QtWidgets import QFileDialog
from qtpy.QtWidgets import (
    QWidget,
    QSizePolicy,
    QHBoxLayout,
    QPushButton,
    QLineEdit,
)
from qtpy.QtCore import Qt

from allencell_ml_segmenter.prediction.model import PredictionModel


class InputButton(QWidget):
    """
    Compound widget consisting of QLineEdit and QPushButton side-by-side.
    Useful for selecting files and displaying the chosen file path.
    """

    def __init__(self, model: PredictionModel):
        super().__init__()

        self._model: PredictionModel = model

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.setLayout(QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

        # text box that will eventually display the chosen file path
        self._text_display: QLineEdit = QLineEdit()
        self._text_display.setPlaceholderText("Choose a file...")
        self._text_display.setStyleSheet(
            "border-left: 2px solid gray; "
            + "border-top: 2px solid gray; "
            + "border-bottom: 2px solid gray; "
            + "padding-top: 3px; "
            + "padding-bottom: 3px"
        )
        self._text_display.setReadOnly(True)

        # button to open file explorer
        self._button: QPushButton = QPushButton("Browse")
        self._button.setStyleSheet(
            "padding: 5px; border: 2px solid gray; background-color: darkorchid"
        )

        # add widgets to layout
        self.layout().addWidget(self._text_display, alignment=Qt.AlignLeft)
        self.layout().addWidget(self._button, alignment=Qt.AlignLeft)

        # connect to slot
        self._button.clicked.connect(self._update_file_text)

    def _update_file_text(self) -> None:
        """
        Gets and displays file path on label portion of input button.
        Caution - currently operates under the assumption that only
        one input button is hooked up to the model.
        """
        file_path: str = QFileDialog.getOpenFileName(self, "Open file")[0]
        self._text_display.setReadOnly(False)
        self._text_display.setText(file_path)
        self._text_display.setReadOnly(True)
        self._model.set_file_path(file_path)
