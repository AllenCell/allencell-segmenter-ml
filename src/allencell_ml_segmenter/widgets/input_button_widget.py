from enum import Enum
from pathlib import Path
from typing import Callable

from qtpy.QtWidgets import (
    QWidget,
    QSizePolicy,
    QHBoxLayout,
    QPushButton,
    QLineEdit,
    QFileDialog,
)
from qtpy.QtCore import Qt

from allencell_ml_segmenter._style import Style
from allencell_ml_segmenter.core.publisher import Publisher
from allencell_ml_segmenter.widgets.directory_or_csv_file_dialog import (
    DirectoryOrCSVFileDialog,
)


class FileInputMode(Enum):
    DIRECTORY = "dir"  # allows CSV as well
    FILE = "file"


class InputButton(QWidget):
    """
    Compound widget consisting of QLineEdit and QPushButton side-by-side.
    Useful for selecting files and displaying the chosen file path.
    """

    def __init__(
        self,
        model: Publisher,
        model_set_file_path_function: Callable,
        placeholder: str = "Select file...",
        mode: FileInputMode = FileInputMode.FILE,
        accept_csv: bool = False,
    ):
        super().__init__()

        self._model: Publisher = model

        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

        self.setLayout(QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

        self._set_path_function: Callable = model_set_file_path_function
        self._mode: FileInputMode = mode
        self._accept_csv: bool = accept_csv

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
        self._button.clicked.connect(self._select_path)
        # connect to stylesheet
        self.setStyleSheet(Style.get_stylesheet("input_button_widget.qss"))

    def _update_path_text(self, path_text: str) -> None:
        """
        Displays path on label portion of input button.
        """
        self._text_display.setReadOnly(False)
        self._text_display.setText(path_text)
        self._text_display.setReadOnly(True)

        # convert path_text to the appropriate path
        path: Path = Path(path_text)

        self._set_path_function(path)

    def elongate(self, min_width: int) -> None:
        """
        Increases the width of the input button _text_display.
        """
        self._text_display.setMinimumWidth(min_width)

    def _select_path(self) -> None:
        """
        Called whenever an input button is clicked. Takes into account the type of files that can be selected,
        as well as the custom setter that gets passed upon instantiation.
        """
        if self._mode == FileInputMode.FILE:
            file_path: str = QFileDialog.getOpenFileName(
                self,
                "Select a file",
                options=QFileDialog.Option.DontUseNativeDialog
                | QFileDialog.Option.DontUseCustomDirectoryIcons,
            )[0]
        elif self._accept_csv:
            custom_dialog: DirectoryOrCSVFileDialog = (
                DirectoryOrCSVFileDialog()
            )
            if custom_dialog.exec_() == QFileDialog.Accepted:
                file_path = custom_dialog.selectedFiles()[0]
            else:
                file_path = None
        else:
            file_path: str = QFileDialog.getExistingDirectory(
                self,
                "Select a directory",
                options=QFileDialog.Option.DontUseNativeDialog
                | QFileDialog.Option.DontUseCustomDirectoryIcons
                | QFileDialog.ShowDirsOnly,
            )
        # TODO: boolean field for csv_file_selected (for some service to read it and extract info)?

        if file_path:
            self._update_path_text(file_path)
