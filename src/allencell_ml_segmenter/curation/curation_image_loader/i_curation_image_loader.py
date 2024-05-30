from abc import ABC, abstractmethod
from qtpy.QtCore import QObject, Signal
from typing import List, Optional
from pathlib import Path
from allencell_ml_segmenter.core.task_executor import ITaskExecutor

from allencell_ml_segmenter.core.image_data_extractor import (
    IImageDataExtractor,
    ImageData,
)


class CurationImageLoaderSignals(QObject):
    # emitted when safe to call prev/next (all currently loading images are completely loaded)
    is_idle: Signal = Signal()


class ICurationImageLoader(ABC):
    def __init__(
        self,
        raw_images: List[Path],
        seg1_images: List[Path],
        seg2_images: Optional[List[Path]],
        img_data_extractor: IImageDataExtractor,
        task_executor: ITaskExecutor,
    ):
        super().__init__()
        self.signals = CurationImageLoaderSignals()
        self._raw_images: List[Path] = list(raw_images)
        self._seg1_images: List[Path] = list(seg1_images)
        self._seg2_images: Optional[List[Path]] = (
            list(seg2_images) if seg2_images else None
        )

        if len(self._raw_images) != len(self._seg1_images) or (
            self._seg2_images
            and len(self._seg1_images) != len(self._seg2_images)
        ):
            raise ValueError("provided image lists must be of same length")
        elif len(self._raw_images) < 1:
            raise ValueError("cannot load images from empty image list")

        self._num_images = len(self._raw_images)

        self._img_data_extractor: IImageDataExtractor = img_data_extractor
        self._task_executor: ITaskExecutor = task_executor
        self._cursor: int = 0

    def get_num_images(self) -> int:
        """
        Returns number of image sets (one set includes raw + its segmentations) in
        this image loader.
        """
        return self._num_images

    def get_current_index(self) -> int:
        """
        Returns the current index of our 'cursor' within the image sets (always <
        num images)
        """
        return self._cursor

    @abstractmethod
    def start(self) -> None:
        """
        Starts the image loader on at least the first set of images.
        """
        pass

    @abstractmethod
    def is_busy(self) -> bool:
        pass

    @abstractmethod
    def get_raw_image_data(self) -> ImageData:
        """
        Returns the image data for the raw image in the set that the 'cursor' is
        currently pointing at.
        """
        pass

    @abstractmethod
    def get_seg1_image_data(self) -> ImageData:
        """
        Returns the image data for the seg1 image in the set that the 'cursor' is
        currently pointing at.
        """
        pass

    @abstractmethod
    def get_seg2_image_data(self) -> Optional[ImageData]:
        """
        Returns the image data for the seg2 image in the set that the 'cursor' is
        currently pointing at. If this loader does not have two segmentations, returns None.
        """
        pass

    def has_next(self) -> bool:
        """
        Returns true iff next() can be safely called.
        """
        return self._cursor + 1 < self._num_images

    def has_prev(self) -> bool:
        """
        Returns true iff prev() can be safely called.
        """
        return self._cursor > 0

    @abstractmethod
    def next(self) -> None:
        """
        Advance to the next set of images in this image loader.
        """
        pass

    @abstractmethod
    def prev(self) -> None:
        """
        Move to the previous set of images in this image loader.
        """
        pass
