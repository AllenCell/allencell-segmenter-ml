from .i_file_writer import IFileWriter
import numpy as np
from pathlib import Path
import csv
from typing import List


class FileWriter(IFileWriter):
    _instance = None

    def np_save(self, path: Path, arr: np.ndarray) -> None:
        """
        Saves :param arr: to :param path:
        """
        np.save(path, arr)

    def csv_write_row(self, writer: csv.writer, row: List[str]) -> None:
        """
        Appends :param row: to the provided :param writer:
        """
        writer.writerow(row)
    
    @classmethod
    def global_instance(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance 