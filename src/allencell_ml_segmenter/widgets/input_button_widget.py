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
        model_set_file_path_function=None,
        placeholder: str = "Select file...",
    ):
        super().__init__()

        self._model: PredictionModel = model

        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

        self.setLayout(QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

        self._model_set_file_path_function = model_set_file_path_function

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
        self._button.clicked.connect(self._update_file_text)

        # connect to stylesheet
        self.setStyleSheet(Style.get_stylesheet("input_button_widget.qss"))

    def _update_file_text(self) -> None:
        """
        Gets and displays file path on label portion of input button.
        Caution - currently operates under the assumption that only
        one input button is hooked up to the model.
        """
        # TODO: shouldn't always be a file; sometimes should be a directory
        file_path: str = QFileDialog.getOpenFileName(self, "Open file")[0]
        self._text_display.setReadOnly(False)
        self._text_display.setText(file_path)
        self._text_display.setReadOnly(True)
        if self._model_set_file_path_function:
            self._model_set_file_path_function(file_path)
