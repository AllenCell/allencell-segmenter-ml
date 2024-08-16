from abc import ABC, abstractmethod
from pathlib import Path
from qtpy.QtWidgets import QWidget
from typing import Optional


class IUserSettings(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_cyto_dl_home_path(self) -> Path:
        pass

    @abstractmethod
    def get_user_experiments_path(self) -> Optional[Path]:
        pass

    @abstractmethod
    def set_user_experiments_path(self, path: Path) -> None:
        pass

    @abstractmethod
    def prompt_for_user_experiments_home(self, parent: QWidget) -> None:
        pass

    @abstractmethod
    def display_change_user_experiments_home(self, parent: QWidget) -> None:
        pass
