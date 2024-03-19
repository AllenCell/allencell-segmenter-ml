from typing import List, Dict, Optional, Callable
from pathlib import Path
from dataclasses import dataclass
import time
import numpy as np
from aicsimageio.aics_image import AICSImage
from PyQt5.QtCore import QThreadPool, QRunnable


@dataclass
class ImageData:
    dim_x: int
    dim_y: int
    dim_z: int
    np_data: np.ndarray
    path: Path


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


class CurationImageLoader:
    """
    CurationImageLoader manages image data for curation with the invariant
    that the getter functions will never be blocking.
    """

    def __init__(
        self,
        raw_images: List[Path],
        seg1_images: List[Path],
        seg2_images: Optional[List[Path]] = None,
    ):
        """
        Raises ValueError if provided lists are unequal in size or empty.
        :param raw_images: paths to raw images
        :param seg1_images: paths to segmentations
        :param seg2_images: (optional) paths to second segmentations
        """
        if len(raw_images) != len(seg1_images) or (
            seg2_images and len(seg1_images) != len(seg2_images)
        ):
            raise ValueError("provided image lists must be of same length")
        elif len(raw_images) < 1:
            raise ValueError("cannot load images from empty image list")

        self._num_images = len(raw_images)

        self._raw_images: List[Path] = list(raw_images)
        self._seg1_images: List[Path] = list(seg1_images)
        self._seg2_images: Optional[List[Path]] = (
            list(seg2_images) if seg2_images else None
        )

        # private invariant: _next_img_data will only have < _num_data_dict_keys keys if there is
        # no next image or a thread is currently updating _next_img_data. Same goes for _prev_img_data
        self._num_data_dict_keys: int = 3 if self._seg2_images else 2
        self._curr_img_data: Dict[str, ImageData] = {}
        self._next_img_data: Dict[str, ImageData] = {}
        self._prev_img_data: Dict[str, ImageData] = {}

        self._cursor: int = 0
        # grab data for first images synchronously, start thread for next images
        self._curr_img_data["raw"] = self._get_image_data(self._raw_images[0])
        self._curr_img_data["seg1"] = self._get_image_data(
            self._seg1_images[0]
        )
        if seg2_images:
            self._curr_img_data["seg2"] = self._get_image_data(
                self._seg2_images[0]
            )

        if self.has_next():
            self._start_extraction_threads(1, self._next_img_data)

    def _get_image_data(self, img_path: Path) -> ImageData:
        aics_img: AICSImage = AICSImage(img_path)
        return ImageData(
            aics_img.dims.X,
            aics_img.dims.Y,
            aics_img.dims.Z,
            aics_img.data,
            img_path,
        )

    def _update_data_dict(
        self, data_dict: Dict[str, ImageData], key: str, img_path: Path
    ) -> None:
        img_data: ImageData = self._get_image_data(img_path)
        data_dict[key] = img_data

    def _start_extraction_threads(
        self, img_index: int, data_dict: Dict[str, ImageData]
    ) -> None:
        data_dict.clear()
        raw_worker: Worker = Worker(
            lambda: self._update_data_dict(
                data_dict, "raw", self._raw_images[img_index]
            )
        )
        seg1_worker: Worker = Worker(
            lambda: self._update_data_dict(
                data_dict, "seg1", self._seg1_images[img_index]
            )
        )
        if self._seg2_images:
            seg2_worker: Worker = Worker(
                lambda: self._update_data_dict(
                    data_dict, "seg2", self._seg2_images[img_index]
                )
            )
            QThreadPool.globalInstance().start(seg2_worker)
        QThreadPool.globalInstance().start(raw_worker)
        QThreadPool.globalInstance().start(seg1_worker)

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

    def get_num_images(self) -> int:
        return self._num_images

    def get_current_index(self) -> int:
        return self._cursor

    def get_raw_image_data(self) -> ImageData:
        return self._curr_img_data["raw"]

    def get_seg1_image_data(self) -> ImageData:
        return self._curr_img_data["seg1"]

    def get_seg2_image_data(self) -> Optional[ImageData]:
        return (
            self._curr_img_data["seg2"]
            if "seg2" in self._curr_img_data
            else None
        )

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
