from pathlib import Path
from typing import Dict

from allencell_ml_segmenter.core.directories import Directories


class Style:
    """
    Helper class that enables the retrieval of custom Qt stylesheets. Implementation adapted from Segmenter Classic.
    """

    cache: Dict[str, str] = dict()

    @classmethod
    def get_stylesheet(cls, name: str) -> str:
        """
        Retrieve a stylesheet from the style directory. Stylesheets are cached in memory for efficiency.
        """
        if name is None:
            raise ValueError("Stylesheet name can't be None")
        if not name.endswith(".qss"):
            raise ValueError("Stylesheet must be a qss file (.qss)")

        if name not in cls.cache:
            cls.cache[name] = cls._load_from_file(name)

        return cls.cache[name]

    @classmethod
    def _load_from_file(cls, name: str) -> str:
        """
        Helper method that loads a stylesheet that has not yet been cached.
        """
        path: Path = Directories.get_style_dir() / name
        with open(path, "r") as handle:
            return handle.read()
