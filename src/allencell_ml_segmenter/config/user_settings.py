from pathlib import Path
from qtpy.QtCore import QSettings
from qtpy.QtWidgets import QFileDialog
from qtpy.QtWidgets import QMessageBox
from qtpy.QtWidgets import QWidget

from allencell_ml_segmenter.config.i_user_settings import IUserSettings

CYTO_DL_HOME_PATH = "/Users/chrishu/dev/code/test2/cyto-dl"
EXPERIMENTS_HOME_KEY = "experimentshome"


class UserSettings(IUserSettings):
    def __init__(self):

        self.settings = QSettings("AIiCS", "Segmenter ML")

        # still hardcoding this for now, hoping that cytodl api will make it unecessary
        self._cyto_dl_home_path: Path = Path(CYTO_DL_HOME_PATH)

    def get_cyto_dl_home_path(self) -> Path:
        return self._cyto_dl_home_path

    def get_user_experiments_path(self) -> Path:
        if(self.settings.value(EXPERIMENTS_HOME_KEY) is None):
            return None
        else:
            return Path(self.settings.value(EXPERIMENTS_HOME_KEY))

    def set_user_experiments_path(self, path: Path):
        self.settings.setValue(EXPERIMENTS_HOME_KEY, path)

    def prompt_for_user_experiments_home(self, parent: QWidget) -> Path:
        message_dialog = QMessageBox(
            parent=parent,
            text="Please select a location to store your Segmenter ML data.",
        )
        message_dialog.exec()
        directory_dialog = QFileDialog(parent=parent)
        directory_dialog.setFileMode(QFileDialog.Directory)
        return Path(directory_dialog.getExistingDirectory())
