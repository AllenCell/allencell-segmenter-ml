from allencell_ml_segmenter.core.directories import Directories
from allencell_ml_segmenter.widgets.input_button_widget import (
    InputButton,
)
from pathlib import Path
from qtpy.QtWidgets import QStackedWidget, QLabel, QSizePolicy
from qtpy.QtGui import QMovie
from qtpy.QtCore import QSize, Qt


class StackedSpinner(QStackedWidget):
    def __init__(self, input_button: InputButton = None):
        super().__init__()

        self.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum
        )
        if input_button is not None:
            self.resize(input_button.width(), input_button.height())
        else:
            self.resize(260, 50)

        self.spinner = QLabel(self)
        self.spinner.setAlignment(Qt.AlignCenter)
        gif_path: Path = Directories.get_assets_dir() / "loading.gif"
        self.movie = QMovie(str(gif_path.resolve()))
        self.spinner.setMovie(self.movie)

        self.input_button = input_button
        self.addWidget(self.spinner)
        self.addWidget(self.input_button)
        self.setCurrentWidget(self.input_button)
        self._is_spinning = False

    def start(self):
        self.setCurrentWidget(self.spinner)
        min_dim: int = min(self.spinner.width(), self.spinner.height())
        padding: int = 2  # necessary to make sure the gif isn't cut off
        size: QSize = QSize(min_dim - padding, min_dim - padding)
        # resizing on start in case window size has changed
        self.movie.setScaledSize(size)
        self.movie.start()
        self._is_spinning = True

    def stop(self):
        self.setCurrentWidget(self.input_button)
        self.movie.stop()
        self._is_spinning = False

    def is_spinning(self) -> bool:
        return self._is_spinning
