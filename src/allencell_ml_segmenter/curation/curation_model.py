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
        self._raw_image_directory = None

    def set_raw_directory(self, dir):
        self._raw_image_directory = dir