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


def get_img_path_from_folder(folder: Path) -> Path:
    """
    Returns path of an image in the folder.
    :param folder: path to a folder containing images
    """
    # we expect user to have the same number of channels for all images in their folders
    # and that only images are stored in those folders

    path_generator: Generator[Path] = folder.glob("*")
    image: Path = next(path_generator)
    # ignore hidden files
    while str(image.name).startswith("."):
        image: Path = next(path_generator)
    return image.resolve()


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
    the channels_ready signal. If the parent thread has requested an
    interruption, the id of the interrupted thread will be emitted through
    the interrupted_thread_finished signal when it is safe to call wait, then drop references
    to the thread.
    """

    channels_ready: pyqtSignal = pyqtSignal(int, int)  # id, num_channels
    interrupted_thread_finished: pyqtSignal = pyqtSignal(int)  # id

    def __init__(self, img_path: Path, id: int, parent: QObject = None):
        """
        :param img_path: path to image (must exist, otherwise ValueError)
        :param id: id for this thread instance, provided by parent thread
        """
        super().__init__(parent)
        if not img_path.exists():
            raise ValueError(f"{img_path} does not exist")

        self._img_path: Path = img_path
        self._id: int = id

    def get_id(self) -> int:
        return self._id

    # override
    def run(self):
        channels: int = extract_channels_from_image(self._img_path)

        if QThread.currentThread().isInterruptionRequested():
            self.interrupted_thread_finished.emit(self._id)
        else:
            self.channels_ready.emit(self._id, channels)
