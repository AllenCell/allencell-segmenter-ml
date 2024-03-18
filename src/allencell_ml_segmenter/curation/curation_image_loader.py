from typing import List, Dict, Optional, Callable
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import time
import numpy as np
from aicsimageio.aics_image import AICSImage
from PyQt5.QtCore import QObject, QThreadPool, QRunnable, pyqtSignal


@dataclass
class ImageData:
    dim_x: int
    dim_y: int
    dim_z: int
    np_data: np.ndarray


class Worker(QRunnable):
    def __init__(self, do_work: Callable):
        super().__init__()
        self._do_work = do_work

    def run(self):
        self._do_work()


class CurationImageLoader:
    def __init__(
        self,
        raw_images: List[Path],
        seg1_images: List[Path],
        seg2_images: Optional[List[Path]] = None,
    ):
        if len(raw_images) != len(seg1_images) or (
            seg2_images and len(seg1_images) != len(seg2_images)
        ):
            raise ValueError("provided image lists must be of same length")
        elif len(raw_images) < 1:
            raise ValueError("cannot load images from empty image list")

        self._num_images = len(raw_images)
        self._raw_images: List[Path] = list(raw_images)
        self._seg1_images: List[Path] = list(seg1_images)
        self._seg2_images: Optional[List[Path]] = list(seg2_images)
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

        if self._num_images > 1:
            self._start_extraction_threads(1, self._next_img_data)

    def _get_image_data(self, img_path: Path) -> ImageData:
        aics_img: AICSImage = AICSImage(img_path)
        return ImageData(
            aics_img.dims.X, aics_img.dims.Y, aics_img.dims.Z, aics_img.data
        )

    def _update_data_dict(
        self, data_dict: Dict[str, ImageData], key: str, img_path: Path
    ) -> None:
        img_data: ImageData = self._get_image_data(img_path)
        data_dict[key] = img_data

    def _start_extraction_threads(
        self, img_index: int, data_dict: Dict[str, ImageData]
    ):
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

    def _wait_on_data_dicts(self):
        expected_length: int = 3 if self._seg2_images else 2
        if self._cursor > 0:
            while len(self._prev_img_data) < expected_length:
                time.sleep(0.1)
                print("waiting prev")
        if self._cursor < self._num_images - 1:
            while len(self._next_img_data) < expected_length:
                time.sleep(0.1)
                print("waiting next")

    def get_raw_image_data(self):
        return self._curr_img_data["raw"]

    def get_seg1_image_data(self):
        return self._curr_img_data["seg1"]

    def get_seg2_image_data(self):
        return self._curr_img_data["seg2"]

    def next(self):
        if self._cursor == self._num_images - 1:
            raise RuntimeError("cannot move cursor past end of image lists")
        self._wait_on_data_dicts()
        self._prev_img_data = self._curr_img_data
        self._curr_img_data = self._next_img_data
        self._next_img_data = {}
        self._cursor += 1
        if self._cursor + 1 < self._num_images:
            self._start_extraction_threads(
                self._cursor + 1, self._next_img_data
            )

    def prev(self):
        if self._cursor == 0:
            raise RuntimeError(
                "cannot move cursor before beginning of image lists"
            )
        self._wait_on_data_dicts()
        self._next_img_data = self._curr_img_data
        self._curr_img_data = self._prev_img_data
        self._prev_img_data = {}
        self._cursor -= 1
        if self._cursor > 0:
            self._start_extraction_threads(
                self._cursor - 1, self._prev_img_data
            )
