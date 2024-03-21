from pathlib import Path
from allencell_ml_segmenter.core.image_data_extractor import (
    IImageDataExtractor,
    ImageData,
)
from aicsimageio.aics_image import AICSImage


class AICSImageDataExtractor(IImageDataExtractor):
    """
    Extracts image data using aicsimageio.
    """

    _instance = None

    def extract_image_data(
        self, img_path: Path, dims: bool = True, np_data: bool = True
    ) -> ImageData:
        aics_img: AICSImage = AICSImage(img_path)
        return ImageData(
            aics_img.dims.X if dims else None,
            aics_img.dims.Y if dims else None,
            aics_img.dims.Z if dims else None,
            aics_img.dims.C if dims else None,
            aics_img.data if np_data else None,
            img_path,
        )

    @classmethod
    def global_instance(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
