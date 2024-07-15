from .i_file_writer import IFileWriter
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Union


class FakeFileWriter(IFileWriter):
    _instance = None

    def __init__(self):
        # {path: saved_array}
        self.np_save_state: Dict[Path, np.ndarray] = {}

        # {path: {"open": T/F, "rows": [[header1, header2...], [col1, col2...]]}}
        self.csv_state: Dict[Path, Dict[str, Any]] = {}

        # {path: json-like-obj}
        self.json_state: dict[Path, Union[list, dict]] = {}

    def np_save(self, path: Path, arr: np.ndarray) -> None:
        """
        Saves :param arr: to :param path:
        """
        self.np_save_state[path.resolve()] = arr

    def csv_open_write_mode(self, path: Path) -> None:
        """
        Opens a CSV at :param path: in write mode. Can only call csv_write_row
        on the path after running this method. Must call csv_close with the same
        path at some point after calling this method.
        """
        path = path.resolve()
        if path in self.csv_state and self.csv_state[path]["open"]:
            raise RuntimeError(f"{path} already open")
        self.csv_state[path] = {"open": True, "rows": []}

    def csv_write_row(self, path: Path, row: List[str]) -> None:
        """
        Appends :param row: to the open CSV at :param path:
        """
        path = path.resolve()
        if path not in self.csv_state or not self.csv_state[path]["open"]:
            raise RuntimeError(
                f"{path} must be opened with csv_open_write_mode before writing"
            )
        self.csv_state[path]["rows"].append(row)

    def csv_close(self, path: Path) -> None:
        """
        Closes the open CSV at :param path:
        """
        path = path.resolve()
        if path in self.csv_state:
            self.csv_state[path]["open"] = False

    def write_json(self, json_like_obj: Union[list, dict], path: Path) -> None:
        self.json_state[path] = json_like_obj
