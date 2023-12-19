from allencell_ml_segmenter.widgets.experimentsHomeDialog import (
    ExperimentsHomeDialog,
)
from allencell_ml_segmenter.config.cyto_dl_config import UserConfig

from qtpy.QtWidgets import QFileDialog
from qtpy.QtCore import QSettings
from allencell_ml_segmenter.constants import (
    CYTO_DL_HOME_PATH,
)


class ConfigService:
    def __init__(self) -> None:
        pass

    def get_user_config(self) -> UserConfig:
        settings = QSettings("AICS", "Segmenter ML")
        experiments_home_path = settings.value("experimentshome")
        if experiments_home_path is None:
            custom_dialog: ExperimentsHomeDialog = ExperimentsHomeDialog()
            if custom_dialog.exec_() == QFileDialog.Accepted:
                experiments_home_path = custom_dialog.selectedFiles()[0]
                settings.setValue("experimentshome", experiments_home_path)
        return UserConfig(CYTO_DL_HOME_PATH, experiments_home_path)
