from pathlib import Path
from typing import List

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.publisher import Publisher


class CurationModel(Publisher):
    """
    Stores state relevant to prediction processes.
    """

    def __init__(self):
        super().__init__()
        self._raw_image_directory: Path = None
        self._seg1_directory: Path = None
        self._seg2_directory: Path = None

    def set_raw_directory(self, dir: Path):
        self._raw_image_directory = dir

    def get_raw_directory(self) -> Path:
        return self._raw_image_directory

    def set_seg1_directory(self, dir: Path):
        self._seg1_directory = dir

    def get_seg1_directory(self) -> Path:
        return self._seg1_directory

    def set_seg2_directory(self, dir: Path):
        self._seg2_directory = dir

    def get_seg2_directory(self) -> Path:
        return self._seg2_directory


