from typing import List, Dict, Optional
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from aicsimageio.aics_image import AICSImage
from PyQt5.QtCore import QObject, QThreadPool, QRunnable, pyqtSignal

@dataclass
class ImageData:
    dim_x: int
    dim_y: int
    dim_z: int
    np_data: np.ndarray


def _get_image_data(img_path: Path) -> ImageData:
    aics_img: AICSImage = AICSImage(img_path)
    return ImageData(aics_img.dims.X, aics_img.dims.Y, aics_img.dims.Z, aics_img.data)


# signals cannot live directly in QRunnable since signals only work in QObject subclasses
class ImageDataExtractionSignals(QObject):
    data_ready: pyqtSignal = pyqtSignal(ImageData)


class ImageDataExtractionThread(QRunnable):
    def __init__(self, img_path: Path):
        super().__init__()
        self.signals = ImageDataExtractionSignals()
        self._img_path = img_path

    def run(self):
        img_data: ImageData = _get_image_data(self._img_path)
        self.signals.data_ready.emit(img_data)


class CurationImageLoader():
    def __init__(self, raw_images: List[Path], seg1_images: List[Path], seg2_images: Optional[List[Path]]=None):
        self._raw_images: List[Path] = raw_images
        self._seg1_images: List[Path] = seg1_images
        self._seg2_images: Optional[List[Path]] = seg2_images
        self._curr_img_data: Optional[Dict[str, ImageData]] = None
        self._next_img_data: Optional[Dict[str, ImageData]] = None
        self._prev_img_data: Optional[Dict[str, ImageData]] = None
        self._cursor: int = 0
        # grab data for first images synchronously, start thread for next images

    def get_raw_image_data(self):
        return self._curr_img_data["raw"]
    
    def get_seg1_image_data(self):
        return self._curr_img_data["seg1"]
    
    def get_seg2_image_data(self):
        return self._curr_img_data["seg2"]
    
    def next(self):
        pass # wait until next data is filled by thread, then move cursor and reset thread

    def prev(self):
        pass
