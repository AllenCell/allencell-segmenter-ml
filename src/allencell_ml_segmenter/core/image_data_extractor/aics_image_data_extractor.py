from pathlib import Path
from typing import Optional

from allencell_ml_segmenter.core.image_data_extractor import (
    IImageDataExtractor,
    ImageData,
)
from bioio.bio_image import BioImage

from allencell_ml_segmenter.utils.image_processing import (
    set_all_nonzero_values_to,
)


class AICSImageDataExtractor(IImageDataExtractor):
    """
    Extracts image data using aicsimageio.
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

        aics_img: BioImage = BioImage(img_path)
        if aics_img.dims.T > 1:
            raise RuntimeError("Cannot load timeseries images")

        img_data = None
        if np_data:
            img_data = aics_img.get_image_dask_data("ZYX", C=channel).compute()
            if seg:
                # if this image is a segmentation, replace all values in image with 1 or 2,
                # so it renders correctly as a napari labels layer.
                img_data = set_all_nonzero_values_to(img_data, seg)

        return ImageData(
            aics_img.dims.X if dims else None,
            aics_img.dims.Y if dims else None,
            aics_img.dims.Z if dims else None,
            aics_img.dims.C if dims else None,
            img_data,
            img_path,
        )

    @classmethod
    def global_instance(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
