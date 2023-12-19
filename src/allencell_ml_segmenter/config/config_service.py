from allencell_ml_segmenter.widgets.experimentsHomeDialog import (
    ExperimentsHomeDialog,
)
from allencell_ml_segmenter.config.cyto_dl_config import UserConfig

from qtpy.QtWidgets import QFileDialog
from qtpy.QtCore import QSettings

CYTO_DL_HOME_PATH = "/Users/chrishu/dev/code/test2/cyto-dl"
EXPERIMENTS_HOME = "experimentshome"

class ConfigService:

    def __init__(self) -> None:
        pass

    def get_user_config(self) -> UserConfig:
        settings = QSettings("AICS", "Segmenter ML")
        experiments_home_path = settings.value(EXPERIMENTS_HOME)
        if experiments_home_path is None:
            custom_dialog: ExperimentsHomeDialog = ExperimentsHomeDialog()
            if custom_dialog.exec_() == QFileDialog.Accepted:
                experiments_home_path = custom_dialog.selectedFiles()[0]
                settings.setValue(EXPERIMENTS_HOME, experiments_home_path)
        return UserConfig(CYTO_DL_HOME_PATH, experiments_home_path)
