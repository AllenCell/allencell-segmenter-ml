from typing import List, Dict, Optional, Callable
from pathlib import Path
import time
from allencell_ml_segmenter.core.image_data_extractor import (
    ImageData,
    IImageDataExtractor,
    AICSImageDataExtractor,
)
from allencell_ml_segmenter.core.task_executor import (
    ITaskExecutor,
    NapariThreadTaskExecutor,
)
from allencell_ml_segmenter.curation.curation_image_loader import (
    ICurationImageLoader,
)


class CurationImageLoader(ICurationImageLoader):
    """
    CurationImageLoader manages image data for curation with the invariant
    that the getter functions will never be blocking.
    """

    def __init__(
        self,
        raw_images: List[Path],
        seg1_images: List[Path],
        seg2_images: Optional[List[Path]] = None,
        img_data_extractor: IImageDataExtractor = AICSImageDataExtractor.global_instance(),
        task_executor: ITaskExecutor = NapariThreadTaskExecutor.global_instance(),
    ):
        """
        Raises ValueError if provided lists are unequal in size or empty.
        :param raw_images: paths to raw images
        :param seg1_images: paths to segmentations
        :param seg2_images: (optional) paths to second segmentations
        """
        super().__init__(
            raw_images,
            seg1_images,
            seg2_images,
            img_data_extractor,
            task_executor,
        )

        # private invariant: _next_img_data will only have < _num_data_dict_keys keys if
        # a thread is currently updating _next_img_data. Same goes for prev and curr
        self._num_data_dict_keys: int = 3 if self._seg2_images else 2
        self._curr_img_data: Dict[str, Optional[ImageData]] = self._get_placeholder_dict()
        self._next_img_data: Dict[str, Optional[ImageData]] = self._get_placeholder_dict()
        self._prev_img_data: Dict[str, Optional[ImageData]] = self._get_placeholder_dict()

        # threads are currently running for extraction
        self._is_busy: bool = False

    def start(self) -> None:
        # start threads for first and next images
        if self.has_next():
            self._extract_image_data(curr=True, next=True)
        else:
            self._extract_image_data(curr=True)

    def is_busy(self) -> bool:
        return self._is_busy

    def _get_placeholder_dict(self) -> Dict[str, Optional[ImageData]]:
        return {"raw": None, "seg1": None} if self._num_data_dict_keys == 2 else {"raw": None, "seg1": None, "seg2": None}
    
    def _wait_on_data_dicts(self) -> None:
        """
        This should never be called in the main/UI thread. It is used exclusively by the
        monitor thread in _extract_image_data.
        """
        while len(self._prev_img_data) < self._num_data_dict_keys:
            time.sleep(0.1)
        while len(self._curr_img_data) < self._num_data_dict_keys:
            time.sleep(0.1)
        while len(self._next_img_data) < self._num_data_dict_keys:
            time.sleep(0.1)

    def _on_extraction_finished(self) -> None:
        self._is_busy = False
        self.signals.images_ready.emit()

    def _extract_image_data(self, prev: bool=False, curr: bool=False, next: bool=False) -> None:
        """
        Begins image data extraction for previous images (based on current _cursor location) iff :param prev:.
        Same pattern applies for current and next images. Emits images_ready signal when all extractions are
        completed.
        """
        if prev:
            self._start_extraction_threads(self._cursor - 1, self._prev_img_data)
        if curr:
            self._start_extraction_threads(self._cursor, self._curr_img_data)
        if next:
            self._start_extraction_threads(self._cursor + 1, self._next_img_data)
        
        if any([prev, curr, next]):
            self._is_busy = True
            # this is thread safe due to GIL: https://docs.python.org/3/glossary.html#term-global-interpreter-lock
            self._task_executor.exec(self._wait_on_data_dicts, on_finish=self._on_extraction_finished)
        
    def _start_extraction_threads(
        self,
        img_index: int,
        data_dict: Dict[str, ImageData],
    ) -> None:
        """
        Clears :param data_dict:, begins extraction of image data from raw, seg1, and seg2 (if applicable) at
        :param img_index:. Extracted data will be added to :param data_dict: as it becomes available.
        """
        data_dict.clear()
        self._task_executor.exec(
            lambda: self._img_data_extractor.extract_image_data(
                self._raw_images[img_index]
            ),
            on_return=lambda img_data: data_dict.update({"raw": img_data}),
        )
        self._task_executor.exec(
            lambda: self._img_data_extractor.extract_image_data(
                self._seg1_images[img_index]
            ),
            on_return=lambda img_data: data_dict.update({"seg1": img_data}),
        )

        if self._seg2_images:
            self._task_executor.exec(
                lambda: self._img_data_extractor.extract_image_data(
                    self._seg2_images[img_index]
                ),
                on_return=lambda img_data: data_dict.update(
                    {"seg2": img_data}
                ),
            )

    def get_raw_image_data(self) -> ImageData:
        """
        Returns the image data for the raw image in the set that the 'cursor' is
        currently pointing at.
        """
        return self._curr_img_data["raw"]

    def get_seg1_image_data(self) -> ImageData:
        """
        Returns the image data for the seg1 image in the set that the 'cursor' is
        currently pointing at.
        """
        return self._curr_img_data["seg1"]

    def get_seg2_image_data(self) -> Optional[ImageData]:
        """
        Returns the image data for the seg2 image in the set that the 'cursor' is
        currently pointing at. If this loader does not have two segmentations, returns None.
        """
        return (
            self._curr_img_data["seg2"]
            if "seg2" in self._curr_img_data
            else None
        )

    def next(self) -> None:
        """
        Advance to the next set of images in this image loader.
        """
        if self.is_busy():
            raise RuntimeError("cannot call next when image loader is busy")

        if not self.has_next():
            raise RuntimeError("cannot move cursor past end of image lists")

        self._prev_img_data = self._curr_img_data
        self._curr_img_data = self._next_img_data
        self._next_img_data = self._get_placeholder_dict()
        self._cursor += 1
        if self.has_next():
            self._extract_image_data(next=True)

    def prev(self) -> None:
        """
        Move to the previous set of images in this image loader.
        """
        if self.is_busy():
            raise RuntimeError("cannot call prev when image loader is busy")

        if not self.has_prev():
            raise RuntimeError(
                "cannot move cursor before beginning of image lists"
            )

        self._next_img_data = self._curr_img_data
        self._curr_img_data = self._prev_img_data
        self._prev_img_data = self._get_placeholder_dict()
        self._cursor -= 1
        if self.has_prev():
            self._extract_image_data(prev=True)
