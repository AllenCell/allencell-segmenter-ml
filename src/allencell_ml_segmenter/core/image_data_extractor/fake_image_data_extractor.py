from pathlib import Path
from typing import Optional

import numpy as np
from allencell_ml_segmenter.core.image_data_extractor import (
    IImageDataExtractor,
    ImageData,
)


class FakeImageDataExtractor(IImageDataExtractor):
    """
    Returns dummy image data for testing.
    """

    _instance = None

    def extract_image_data(
        self,
        img_path: Path,
        channel: int = 0,
        dims: bool = True,
        np_data: bool = True,
        seg: Optional[int] = None,
    ) -> ImageData:
        return ImageData(
            1 if dims else None,
            2 if dims else None,
            3 if dims else None,
            4 if dims else None,
            np.zeros((5, 5)) if np_data else None,
            img_path,
        )

    @classmethod
    def global_instance(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
