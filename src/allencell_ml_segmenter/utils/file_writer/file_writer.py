from .i_file_writer import IFileWriter
import numpy as np
from pathlib import Path
import csv
import json
from io import TextIOBase
from typing import List, Dict, Tuple, Union


class FileWriter(IFileWriter):
    _instance = None

    # maps path to an open csv file and the corresponding csv writer
    _open_files: Dict[Path, Tuple[TextIOBase, csv.writer]] = {}

    def np_save(self, path: Path, arr: np.ndarray) -> None:
        """
        Saves :param arr: to :param path:. Creates directories as necessary.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)

    def csv_open_write_mode(self, path: Path) -> None:
        """
        Opens a CSV at :param path: in write mode. Can only call csv_write_row
        on the path after running this method. Must call csv_close with the same
        path at some point after calling this method. Creates directories as necessary.
        """
        path = path.resolve()
        if path in self._open_files:
            raise RuntimeError(f"{path} already open")
        path.parent.mkdir(parents=True, exist_ok=True)
        open_file: TextIOBase = open(path, "w", newline="")
        writer: csv.writer = csv.writer(open_file, delimiter=",")
        self._open_files[path] = (open_file, writer)

    def csv_write_row(self, path: Path, row: List[str]) -> None:
        """
        Appends :param row: to the open CSV at :param path:
        """
        path = path.resolve()
        if path not in self._open_files:
            raise RuntimeError(
                f"{path} must be opened with csv_open_write_mode before writing"
            )
        self._open_files[path][1].writerow(row)

    def csv_close(self, path: Path) -> None:
        """
        Closes the open CSV at :param path:
        """
        path = path.resolve()
        if path in self._open_files:
            self._open_files[path][0].close()
            del self._open_files[path]

    def write_json(self, json_like_obj: Union[list, dict], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fw:
            json.dump(json_like_obj, fw)

    @classmethod
    def global_instance(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
