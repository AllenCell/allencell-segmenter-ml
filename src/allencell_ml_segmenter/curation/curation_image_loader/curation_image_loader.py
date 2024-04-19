from typing import List, Dict, Optional, Callable
from pathlib import Path
import time
from qtpy.QtCore import QRunnable
from allencell_ml_segmenter.core.image_data_extractor import (
    ImageData,
    IImageDataExtractor,
    AICSImageDataExtractor,
)
from allencell_ml_segmenter.curation.curation_image_loader import (
    ICurationImageLoader,
)
from qtpy.QtCore import QThread, Signal, QObject
from PyQt5.QtCore import pyqtSlot


class LoaderThread(QThread):
    task_failed: Signal = Signal(Exception)

    def __init__(self, runnable: QRunnable, parent: QObject = None):
        super().__init__(parent)
        self._runnable = runnable

    def run(self):
        try:
            self._runnable.run()
        except Exception as e:
            self.task_failed.emit(e)

class CountdownLatchThread(QThread):
    
    def __init__(self, count: int, parent: QObject = None):
        super().__init__(parent)
        self.count = count

    @pyqtSlot()
    def count_down(self):
        self.count = self.count - 1
        print(f'countdown {self.count}')

    def run(self):
        while(self.count > 0):
            time.sleep(.5)



class Worker(QRunnable):
    """
    Generic implementation of QRunnable that simply runs the
    provided do_work function.
    """

    def __init__(self, do_work: Callable):
        """
        :param do_work: function that this worker will call in a thread
        """
        super().__init__()
        self._do_work = do_work

    def run(self):
        self._do_work()


class CurationImageLoader(ICurationImageLoader):
    """
    CurationImageLoader manages image data for curation with the invariant
    that the getter functions will never be blocking.
    """

    def __init__(
        self,
        raw_channel: int,
        seg1_channel: int,
        seg2_channel: int,
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

        self._raw_channel: int = raw_channel
        self._seg1_channel: int = seg1_channel
        self._seg2_channel: int = seg2_channel
        # private invariant: _next_img_data will only have < _num_data_dict_keys keys if there is
        # no next image or a thread is currently updating _next_img_data. Same goes for _prev_img_data
        self._num_data_dict_keys: int = 3 if self._seg2_images else 2
        self._curr_img_data: Dict[str, ImageData] = {}
        self._next_img_data: Dict[str, ImageData] = {}
        self._prev_img_data: Dict[str, ImageData] = {}
        self._running_thread = []


    def initialize(self, callback: Callable = None) -> None:
        """
        Initialize the image loader.
        """
        self._start_extraction_threads(0, self._curr_img_data, callback)
        if self.has_next():
            self._start_extraction_threads(1, self._next_img_data)
    

    def _update_data_dict(
        self, data_dict: Dict[str, ImageData], key: str, img_path: Path, channel: int
    ) -> None:
        img_data: ImageData = self._img_data_extractor.extract_image_data(
            img_path=img_path, channel=channel
        )
        data_dict[key] = img_data

    def _start_extraction_threads(
        self, img_index: int, data_dict: Dict[str, ImageData], callback: Callable = lambda: None
    ) -> None:
        print(f"start extraction threads {img_index}\n")
        data_dict.clear()
        raw_worker: Worker = Worker(
            lambda: self._update_data_dict(
                data_dict, "raw", self._raw_images[img_index], self._raw_channel
            )
        )
        seg1_worker: Worker = Worker(
            lambda: self._update_data_dict(
                data_dict, "seg1", self._seg1_images[img_index], self._seg1_channel
            )
        )
        workers = [raw_worker, seg1_worker]
        if self._seg2_images:
            seg2_worker: Worker = Worker(
                lambda: self._update_data_dict(
                    data_dict, "seg2", self._seg2_images[img_index], self._seg2_channel
                )
            )
            workers.append(seg2_worker)
        latch_thread = CountdownLatchThread(len(workers))
        latch_thread.finished.connect(callback)
        self._running_thread.append(latch_thread)
        latch_thread.start()
        for worker in workers:
            self._run_with_callback(worker, latch_thread, lambda: None)

    def _run_with_callback(self, runnable: QRunnable, latch_thread: CountdownLatchThread, fail_callback: Callable) -> None:
        thread = LoaderThread(runnable)
        thread.task_failed.connect(fail_callback)
        self._running_thread.append(thread)
        thread.finished.connect(latch_thread.count_down)
        # thread.finished.connect(lambda: self._pop_thread(thread))
        thread.start()

    def _pop_thread(self, thread):
        self._running_thread.remove(thread)

    def _wait_on_data_dicts(self) -> None:
        """
        Wait for any ongoing updates to prev and next data dicts to finish.
        """
        expected_length: int = self._num_data_dict_keys
        if self.has_prev():
            while len(self._prev_img_data) < expected_length:
                time.sleep(0.1)
        if self.has_next():
            while len(self._next_img_data) < expected_length:
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
        if not self.has_next():
            raise RuntimeError("cannot move cursor past end of image lists")
        self._wait_on_data_dicts()
        self._prev_img_data = self._curr_img_data
        self._curr_img_data = self._next_img_data
        self._next_img_data = {}
        self._cursor += 1
        if self.has_next():
            self._start_extraction_threads(
                self._cursor + 1, self._next_img_data
            )

    def prev(self) -> None:
        """
        Move to the previous set of images in this image loader.
        """
        if not self.has_prev():
            raise RuntimeError(
                "cannot move cursor before beginning of image lists"
            )
        self._wait_on_data_dicts()
        self._next_img_data = self._curr_img_data
        self._curr_img_data = self._prev_img_data
        self._prev_img_data = {}
        self._cursor -= 1
        if self.has_prev():
            self._start_extraction_threads(
                self._cursor - 1, self._prev_img_data
            )
