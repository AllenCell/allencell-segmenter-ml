from pathlib import Path
from qtpy.QtCore import QSettings
from qtpy.QtWidgets import QFileDialog
from qtpy.QtWidgets import QMessageBox
from qtpy.QtWidgets import QWidget

from allencell_ml_segmenter.config.i_user_settings import IUserSettings

CYTO_DL_HOME_PATH = "/Users/chrishu/dev/code/test2/cyto-dl"
EXPERIMENTS_HOME_KEY = "experimentshome"


class UserSettings(IUserSettings):
    def __init__(
        self, settings: QSettings = QSettings("AICS", "Segmenter ML")
    ):
        self.settings = settings

        # still hardcoding this for now, hoping that cytodl api will make it unecessary
        self._cyto_dl_home_path: Path = Path(CYTO_DL_HOME_PATH)

    def get_cyto_dl_home_path(self) -> Path:
        return self._cyto_dl_home_path

    def get_user_experiments_path(self) -> Path:
        if self.settings.value(EXPERIMENTS_HOME_KEY) is None:
            return None
        else:
            return Path(self.settings.value(EXPERIMENTS_HOME_KEY))

    def set_user_experiments_path(self, path: Path):
        self.settings.setValue(EXPERIMENTS_HOME_KEY, path)

    def prompt_for_user_experiments_home(self, parent: QWidget):
        message_dialog = QMessageBox(
            parent=parent,
            text="Please select a location to store your Segmenter ML data.",
        )
        message_dialog.exec()
        path: Path = self._prompt_for_directory(parent)
        self.set_user_experiments_path(path)

    def display_change_user_experiments_home(self, parent: QWidget):
        buttonReply = QMessageBox.question(
            parent,
            "Experiments Home",
            "Experiments Home: "
            + str(self.get_user_experiments_path())
            + "\n\n Would you like to change the Experiments Home?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if buttonReply == QMessageBox.Yes:
            path: Path = self._prompt_for_directory(parent)
            self.set_user_experiments_path(path)

    def _prompt_for_directory(self, parent: QWidget) -> Path:
        directory_dialog = QFileDialog(parent=parent)
        directory_dialog.setFileMode(QFileDialog.Directory)
        file_path: str = QFileDialog.getExistingDirectory(
            parent,
            "Select a directory",
            options=QFileDialog.Option.DontUseNativeDialog
            | QFileDialog.Option.DontUseCustomDirectoryIcons
            | QFileDialog.ShowDirsOnly,
        )
        return Path(file_path)
