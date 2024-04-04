from pathlib import Path
from typing import Generator
import csv
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from aicsimageio import AICSImage


def extract_channels_from_image(img_path: Path) -> int:
    """
    Returns number of channels in the given img_path.
    :param img_path: image to extract channels from
    """
    return AICSImage(str(img_path)).dims.C

def get_img_path_from_csv(csv_path: Path) -> Path:
    """
    Returns path of an image in the 'raw' column of the csv.
    :param csv_path: path to a csv with a 'raw' column
    """
    with open(csv_path) as csv_file:
        reader: csv.reader = csv.DictReader(csv_file)
        img_path: str = next(reader)["raw"]
    return Path(img_path).resolve()


class ChannelExtractionThread(QThread):
    """
    A ChannelExtractionThread will extract the number of channels from
    the provided image. If the parent thread has not requested an interruption
    during the thread execution, the number of channels will be emitted through
    the channels_ready signal. If the parent thread has requested an interruption,
    the thread will have no side effects.
    """

    channels_ready: pyqtSignal = pyqtSignal(int)  # num_channels

    def __init__(self, img_path: Path, parent: QObject = None):
        """
        :param img_path: path to image (must exist, otherwise ValueError)
        :param id: id for this thread instance, provided by parent thread
        """
        super().__init__(parent)
        self._img_path: Path = img_path

    # override
    def run(self):
        # will show up as a pop-up in the UI, does not force napari to quit
        if not self._img_path.exists():
            raise ValueError(f"{self._img_path} does not exist")

        channels: int = extract_channels_from_image(self._img_path)

        if not QThread.currentThread().isInterruptionRequested():
            self.channels_ready.emit(channels)
