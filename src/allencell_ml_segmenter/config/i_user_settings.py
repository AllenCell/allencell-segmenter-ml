from abc import ABC, abstractmethod
from pathlib import Path
from qtpy.QtWidgets import QWidget


class IUserSettings(ABC):
    def __init__(self):
        super.__init__(self)

    @abstractmethod
    def get_cyto_dl_home_path(self) -> Path:
        pass

    @abstractmethod
    def get_user_experiments_path(self) -> Path:
        pass

    @abstractmethod
    def set_user_experiments_path(self, path: Path):
        pass

    @abstractmethod
    def prompt_for_user_experiments_home(self, parent: QWidget):
        pass

    @abstractmethod
    def display_change_user_experiments_home(self, parent: QWidget):
        pass
