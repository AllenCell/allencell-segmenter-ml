from pathlib import Path
from typing import Generator, List
import csv

from aicsimageio.exceptions import UnsupportedFileFormatError
from qtpy.QtCore import QObject, QThread, Signal
from aicsimageio import AICSImage

from allencell_ml_segmenter.core.image_data_extractor import (
    AICSImageDataExtractor,
    ImageData,
)


def get_img_path_from_csv(csv_path: Path) -> Path:
    """
    Returns path of an image in the 'raw' column of the csv.
    :param csv_path: path to a csv with a 'raw' column
    """
    with open(csv_path) as csv_file:
        reader: csv.reader = csv.DictReader(csv_file)
        img_path: str = next(reader)["raw"]
    return Path(img_path).resolve()

class ChannelExtractionThreadSignals(QObject):
    channels_ready: Signal = Signal(int)  # num_channels
    image_data_ready: Signal = Signal(ImageData)
    task_failed: Signal = Signal(Exception)



class ChannelExtractionThread(QThread):
    """
    A ChannelExtractionThread will extract the number of channels from
    the provided image. If the parent thread has not requested an interruption
    during the thread execution, the number of channels will be emitted through
    the channels_ready signal. If the parent thread has requested an interruption,
    the thread will have no side effects.

    """
    def __init__(
        self,
        img_path: Path,
        emit_image_data: bool = False,
        parent: QObject = None,
    ):
        """
        :param img_path: path to image (must exist, otherwise ValueError)
        :param emit_image_data: True to return image_data (dimensions and channel) through image_data_ready singal
                               False to return only num_channels through channels_ready signal
        :param parent: (optional) parent QObject for this thread, if any.
        """
        super().__init__(parent)
        self.signals: ChannelExtractionThreadSignals = ChannelExtractionThreadSignals()
        self._img_path: Path = img_path
        self._emit_image_data: bool = emit_image_data
        self._image_extractor: AICSImageDataExtractor = (
            AICSImageDataExtractor.global_instance()
        )

    # override
    def run(self):
        # will show up as a pop-up in the UI, does not force napari to quit
        if not self._img_path.exists():
            raise ValueError(f"{self._img_path} does not exist")

        try:
            image_data: ImageData = self._image_extractor.extract_image_data(
                self._img_path, dims=True, np_data=False
            )
        except UnsupportedFileFormatError as ex:
            self.signals.task_failed.emit(ex)
            return  # return instead of reraise to surprss error message in napari console
        except FileNotFoundError as ex:
            self.signals.task_failed.emit(ex)
            return

        if not QThread.currentThread().isInterruptionRequested():
            if self._emit_image_data:
                self.signals.image_data_ready.emit(image_data)
            else:
                self.signals.channels_ready.emit(image_data.channels)
