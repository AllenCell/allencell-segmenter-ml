from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
import csv
from typing import List

class IFileWriter(ABC):
    def __init__(self):
        raise RuntimeError(
            "Cannot initialize new singleton, please use .global_instance() instead"
        )
    
    @abstractmethod
    def np_save(self, path: Path, arr: np.ndarray) -> None:
        """
        Saves :param arr: to :param path:
        """
        pass

    @abstractmethod
    def csv_write_row(self, writer: csv.writer, row: List[str]) -> None:
        """
        Appends :param row: to the provided :param writer:
        """
        pass
    
    @classmethod
    @abstractmethod
    def global_instance(cls):
        pass