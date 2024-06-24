from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from allencell_ml_segmenter.core.image_data_extractor import ImageData


class IImageDataExtractor(ABC):
    def __init__(self):
        raise RuntimeError(
            "Cannot initialize new singleton, please use .global_instance() instead"
        )

    @abstractmethod
    def extract_image_data(
        self,
        img_path: Path,
        channel: int = 0,
        dims: bool = True,
        np_data: bool = True,
        seg: Optional[int] = None,
    ) -> ImageData:
        pass

    @classmethod
    @abstractmethod
    def global_instance(cls):
        pass
