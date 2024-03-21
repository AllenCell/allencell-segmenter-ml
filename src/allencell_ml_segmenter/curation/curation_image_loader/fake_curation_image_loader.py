from allencell_ml_segmenter.curation.curation_image_loader import (
    ICurationImageLoader,
)
from typing import List, Optional
from pathlib import Path
from allencell_ml_segmenter.core.q_runnable_manager import (
    IQRunnableManager,
    GlobalQRunnableManager,
)
from allencell_ml_segmenter.core.image_data_extractor import (
    IImageDataExtractor,
    AICSImageDataExtractor,
    FakeImageDataExtractor,
    ImageData,
)


class FakeCurationImageLoader(ICurationImageLoader):
    def __init__(
        self,
        raw_images: List[Path],
        seg1_images: List[Path],
        seg2_images: Optional[List[Path]] = None,
        qr_manager: IQRunnableManager = GlobalQRunnableManager.global_instance(),
        img_data_extractor: IImageDataExtractor = AICSImageDataExtractor.global_instance(),
    ):
        self._raw_images = raw_images
        self._seg1_images = seg1_images
        self._seg2_images = seg2_images
        self._cursor = 0
        self._img_data_extractor = FakeImageDataExtractor.global_instance()

    def get_num_images(self) -> int:
        """
        Returns number of image sets (one set includes raw + its segmentations) in
        this image loader.
        """
        return len(self._raw_images)

    def get_current_index(self) -> int:
        """
        Returns the current index of our 'cursor' within the image sets (always <
        num images)
        """
        return self._cursor

    def get_raw_image_data(self) -> ImageData:
        """
        Returns the image data for the raw image in the set that the 'cursor' is
        currently pointing at.
        """
        return self._img_data_extractor.extract_image_data(
            self._raw_images[self._cursor]
        )

    def get_seg1_image_data(self) -> ImageData:
        """
        Returns the image data for the seg1 image in the set that the 'cursor' is
        currently pointing at.
        """
        return self._img_data_extractor.extract_image_data(
            self._seg1_images[self._cursor]
        )

    def get_seg2_image_data(self) -> Optional[ImageData]:
        """
        Returns the image data for the seg2 image in the set that the 'cursor' is
        currently pointing at. If this loader does not have two segmentations, returns None.
        """
        return self._img_data_extractor.extract_image_data(
            self._seg2_images[self._cursor]
        )

    def has_next(self) -> bool:
        """
        Returns true iff next() can be safely called.
        """
        return self._cursor + 1 < len(self._raw_images)

    def has_prev(self) -> bool:
        """
        Returns true iff prev() can be safely called.
        """
        return self._cursor > 0

    def next(self) -> None:
        """
        Advance to the next set of images in this image loader.
        """
        if not self.has_next():
            raise RuntimeError()
        self._cursor += 1

    def prev(self) -> None:
        """
        Move to the previous set of images in this image loader.
        """
        if not self.has_prev():
            raise RuntimeError()
        self._cursor -= 1
