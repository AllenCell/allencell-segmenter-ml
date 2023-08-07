import os

from qtpy.QtWidgets import QFileDialog


class CustomFileDialog(QFileDialog):
    def __init__(self):
        super().__init__()
        self.setOption(QFileDialog.DontUseNativeDialog)
        self.setFileMode(QFileDialog.Directory)
        self.currentChanged.connect(self._selected)
        self.setNameFilter("Directories and CSV files (*.csv)")

    def _selected(self, name):
        if os.path.isdir(name):
            self.setFileMode(QFileDialog.Directory)
            self.setNameFilter(
                "Directories and CSV files (*.csv)"
            )  # without this, the filter is changed to just "Directories"
        else:
            self.setFileMode(QFileDialog.ExistingFile)
