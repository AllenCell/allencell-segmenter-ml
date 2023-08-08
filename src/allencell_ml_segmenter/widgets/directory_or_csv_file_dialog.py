import os

from qtpy.QtWidgets import QFileDialog


class DirectoryOrCSVFileDialog(QFileDialog):
    """
    A custom QFileDialog that allows the user to select either a directory or a CSV file.
    Used in relation to InputButton.
    """

    def __init__(self):
        super().__init__()
        self.setOption(QFileDialog.DontUseNativeDialog)
        self.setFileMode(QFileDialog.Directory)
        self.currentChanged.connect(self._selected)
        self.setNameFilter("Directories and CSV files (*.csv)")

    def _selected(self, name: str) -> None:
        """
        Called whenever the user selects a new option in the File Dialog menu.
        """
        if os.path.isdir(name):
            self.setFileMode(QFileDialog.Directory)
            self.setNameFilter(
                "Directories and CSV files (*.csv)"
            )  # without this, the filter is changed to just "Directories"
        else:
            self.setFileMode(QFileDialog.ExistingFile)
