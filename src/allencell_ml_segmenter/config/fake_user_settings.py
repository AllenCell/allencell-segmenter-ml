from pathlib import Path
from allencell_ml_segmenter.config.i_user_settings import IUserSettings
from qtpy.QtWidgets import QWidget


class FakeUserSettings(IUserSettings):
    def __init__(self, prompt_response: Path = None):
        self.prompt_response: Path = prompt_response

    def get_cyto_dl_home_path(self) -> Path:
        return self.cyto_dl_home_path

    def set_cyto_dl_home_path(self, path: Path) -> Path:
        self.cyto_dl_home_path = path

    def get_user_experiments_path(self) -> Path:
        return self.user_experiments_path

    def set_user_experiments_path(self, path: str):
        self.user_experiments_path = path

    def prompt_for_user_experiments_home(self, parent: QWidget) -> Path:
        return Path(self.prompt_response)
