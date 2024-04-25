from typing import List, Dict, Optional, Callable
from pathlib import Path
import time
from allencell_ml_segmenter.core.image_data_extractor import (
    ImageData,
    IImageDataExtractor,
    AICSImageDataExtractor,
)
from allencell_ml_segmenter.core.q_runnable_manager import (
    IQRunnableManager,
    GlobalQRunnableManager,
)
from allencell_ml_segmenter.curation.curation_image_loader import (
    ICurationImageLoader,
)
from napari.qt.threading import thread_worker, FunctionWorker
from qtpy.QtCore import Signal

# TODO: worker creator/ worker manager interface to make testing code that uses thread worker easier
class CurationImageLoader(ICurationImageLoader):
    """
    CurationImageLoader manages image data for curation with the invariant
    that the getter functions will never be blocking.
    """

    is_idle: Signal = Signal()

    def __init__(
        self,
        raw_images: List[Path],
        seg1_images: List[Path],
        seg2_images: Optional[List[Path]] = None,
        img_data_extractor: IImageDataExtractor = AICSImageDataExtractor.global_instance(),
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
        )

        # private invariant: _next_img_data will only have < _num_data_dict_keys keys if there is
        # no next image or a thread is currently updating _next_img_data. Same goes for _prev_img_data
        self._num_data_dict_keys: int = 3 if self._seg2_images else 2
        self._curr_img_data: Dict[str, ImageData] = {}
        self._next_img_data: Dict[str, ImageData] = {}
        self._prev_img_data: Dict[str, ImageData] = {}

        # if threads are currently running for prev, curr, next img data
        self._is_busy = [False, False, False]

        # start threads for first and possibly next images
        self._set_curr_is_busy(True)
        curr_worker: FunctionWorker = self._start_extraction_threads(0, self._curr_img_data)
        curr_worker.finished.connect(lambda: self._set_curr_is_busy(False))

        if self.has_next():
            self._set_next_is_busy(True)
            next_worker: FunctionWorker = self._start_extraction_threads(1, self._next_img_data)
            next_worker.finished.connect(lambda: self._set_next_is_busy(False))

    def is_busy(self):
        return any(self._is_busy)

    def _set_curr_is_busy(self, busy: bool):
        self._is_busy[1] = busy
        if not self.is_busy():
            self.is_idle.emit()

    def _set_prev_is_busy(self, busy: bool):
        self._is_busy[0] = busy
        if not self.is_busy():
            self.is_idle.emit()
    
    def _set_next_is_busy(self, busy: bool):
        self._is_busy[2] = busy
        if not self.is_busy():
            self.is_idle.emit()

    @thread_worker
    def _start_img_data_extraction(self, img_path: Path) -> ImageData:
        return self._img_data_extractor.extract_image_data(img_path)

    @thread_worker
    def _start_extraction_threads(
        self, img_index: int, data_dict: Dict[str, ImageData]
    ) -> FunctionWorker:
        data_dict.clear()
        raw_worker: FunctionWorker = self._start_img_data_extraction(self._raw_images[img_index])
        raw_worker.returned.connect(lambda img_data: data_dict.update({"raw": img_data}))
        
        seg1_worker: FunctionWorker = self._start_img_data_extraction(self._seg1_images[img_index])
        seg1_worker.returned.connect(lambda img_data: data_dict.update({"seg1": img_data}))

        workers: List[FunctionWorker] = [raw_worker, seg1_worker]
        if self._seg2_images:
            seg2_worker: FunctionWorker = self._start_img_data_extraction(self._seg2_images[img_index])
            seg2_worker.returned.connect(lambda img_data: data_dict.update({"seg2": img_data}))
            workers.append(seg2_worker)
        
        while any([worker.is_running() for worker in workers]):
            time.sleep(0.1)


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
        self._next_img_data = {}
        self._cursor += 1
        if self.has_next():
            self._set_next_is_busy(True)
            next_worker: FunctionWorker = self._start_extraction_threads(
                self._cursor + 1, self._next_img_data
            )
            next_worker.finished.connect(lambda: self._set_next_is_busy(False))

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
        self._prev_img_data = {}
        self._cursor -= 1
        if self.has_prev():
            self._set_prev_is_busy(True)
            prev_worker: FunctionWorker = self._start_extraction_threads(
                self._cursor - 1, self._prev_img_data
            )
            prev_worker.finished.connect(lambda: self._set_next_is_busy(False))
