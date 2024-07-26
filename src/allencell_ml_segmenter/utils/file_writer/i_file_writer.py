from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
from typing import List, Union


class IFileWriter(ABC):
    def __init__(self):
        raise RuntimeError(
            "Cannot initialize new singleton, please use .global_instance() instead"
        )

    @abstractmethod
    def np_save(self, path: Path, arr: np.ndarray) -> None:
        """
        Saves :param arr: to :param path:. Creates directories as necessary.
        """
        pass

    @abstractmethod
    def csv_open_write_mode(self, path: Path) -> None:
        """
        Opens a CSV at :param path: in write mode. Can only call csv_write_row
        on the path after running this method. Must call csv_close with the same
        path at some point after calling this method. Creates directories as necessary.
        """
        pass

    @abstractmethod
    def csv_write_row(self, path: Path, row: List[str]) -> None:
        """
        Appends :param row: to the open CSV at :param path:
        """
        pass

    @abstractmethod
    def csv_close(self, path: Path) -> None:
        """
        Closes the open CSV at :param path:
        """
        pass

    @abstractmethod
    def write_json(self, json_like_obj: Union[list, dict], path: Path) -> None:
        pass
