from enum import Enum
from pathlib import Path
from typing import Callable, Any, Optional

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
from allencell_ml_segmenter.widgets.directory_or_csv_file_dialog import (
    DirectoryOrCSVFileDialog,
)


class FileInputMode(Enum):
    DIRECTORY = "dir"
    FILE = "file"
    DIRECTORY_OR_CSV = "dir_or_csv"


class InputButton(QWidget):
    """
    Compound widget consisting of QLineEdit and QPushButton side-by-side.
    Useful for selecting files and displaying the chosen file path.
    """

    def __init__(
        self,
        model: Any,
        model_set_file_path_function: Callable,
        placeholder: str = "Select file...",
        mode: FileInputMode = FileInputMode.FILE,
    ):
        super().__init__()
        self._default_placeholder = placeholder
        self._model: Any = model

        self.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed
        )

        layout: QHBoxLayout = QHBoxLayout()
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._set_path_function: Callable = model_set_file_path_function
        self._mode: FileInputMode = mode

        # text box that will eventually display the chosen file path
        self._text_display: QLineEdit = QLineEdit()
        self._text_display.setPlaceholderText(self._default_placeholder)
        self._text_display.setObjectName("textDisplay")
        self._text_display.setReadOnly(True)

        # button to open file explorer
        self.button: QPushButton = QPushButton("Browse")
        self.button.setObjectName("button")

        # add widgets to layout
        layout.addWidget(
            self._text_display, alignment=Qt.AlignmentFlag.AlignLeft
        )
        layout.addWidget(self.button, alignment=Qt.AlignmentFlag.AlignLeft)

        # connect to slot
        self.button.clicked.connect(self._select_path)
        # connect to stylesheet
        self.setStyleSheet(Style.get_stylesheet("input_button_widget.qss"))

    def _update_path_text(self, path_text: Optional[str]) -> None:
        """
        Displays chosen file or directory path on label portion of input button.
        """
        if path_text is not None:
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
        Called whenever an input button is clicked. Takes into account the type of files that can be selected
        and makes a call to _update_path_text if the user selects a compatible file or directory.
        """
        file_path: Optional[str]
        if self._mode == FileInputMode.FILE:
            file_path = QFileDialog.getOpenFileName(
                self,
                "Select a file",
                options=QFileDialog.Option.DontUseNativeDialog
                | QFileDialog.Option.DontUseCustomDirectoryIcons,  # type: ignore
            )[0]
        elif self._mode == FileInputMode.DIRECTORY_OR_CSV:
            custom_dialog: DirectoryOrCSVFileDialog = (
                DirectoryOrCSVFileDialog()
            )
            if custom_dialog.exec_() == QFileDialog.Accepted:
                file_path = custom_dialog.selectedFiles()[0]
            else:
                file_path = None
        else:  # FileInputMode.DIRECTORY
            file_path = QFileDialog.getExistingDirectory(
                self,
                "Select a directory",
                options=QFileDialog.Option.DontUseNativeDialog
                | QFileDialog.Option.DontUseCustomDirectoryIcons
                | QFileDialog.ShowDirsOnly,
            )
        # TODO: boolean field for csv_file_selected (for some service to read it and extract info)?

        if file_path:
            self._update_path_text(file_path)

    def clear_selection(self) -> None:
        self._text_display.clear()
        self._text_display.setPlaceholderText(self._default_placeholder)
