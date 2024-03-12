from pathlib import Path
from typing import Callable, Generator, List, Union, Optional, Dict
import csv
from dataclasses import dataclass

from aicsimageio import AICSImage

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.prediction.model import PredictionModel

from PyQt5.QtCore import QThread, pyqtSignal
import threading

class ChannelExtractionThread(QThread):
    channels_ready: pyqtSignal = pyqtSignal(int)
    dep_thread_done: pyqtSignal = pyqtSignal(int)

    def __init__(self, extract_channels: Callable[[], int], id, parent=None):
        super().__init__(parent)
        self._extract_channels = extract_channels
        self._id = id

    # override
    def run(self):
        channels: int = self._extract_channels()
        if QThread.currentThread().isInterruptionRequested():
            self.dep_thread_done.emit(self._id)
        else:
            self.channels_ready.emit(channels)

@dataclass
class ThreadTracker():
    id: int
    thread: ChannelExtractionThread

class ModelFileService(Subscriber):
    """
    Parses the chosen model file to extract the preprocessing method.
    """

    def __init__(self, model: PredictionModel):
        super().__init__()
        self._model: PredictionModel = model
        self._current_thread: Optional[ThreadTracker] = (
            None
        )
        self._deprecated_threads: Dict[int, ChannelExtractionThread] = {}
        self._threads_created = 0

        self._model.subscribe(
            Event.ACTION_PREDICTION_MODEL_FILE,
            self,
            lambda e: self.extract_preprocessing_method(),
        )

        self._model.subscribe(
            Event.ACTION_PREDICTION_EXTRACT_CHANNELS,
            self,
            lambda e: self._initiate_channel_extraction(),
        )

    def handle_event(self, event: Event) -> None:
        pass

    def extract_preprocessing_method(self) -> None:
        """
        Calls the prediction model's setter for the preprocessing method. Currently set up with a dummy value.
        """
        # TODO: replace dummy implementation
        self._model.set_preprocessing_method("foo")

    def extract_num_channels(self) -> int:
        """
        Extracts the number of channels from the prediction model. If model's
        input_image_path is set, will attempt to infer channels from that field;
        otherwise will attempt to infer channels from model's selected_paths field
        (the images currently checked in the prediction widget). Throws ValueError
        if both fields uninitialized.
        """
        path: Path = self._model.get_input_image_path()
        img_path: Union[str, Path] = None
        if not path:  # using viewer input method
            paths: List[Path] = self._model.get_selected_paths()
            if not paths or len(paths) <= 0:
                raise ValueError(
                    "expected input_image_path or selected_paths to be initialized and non-empty"
                )
            img_path = paths[0]
        elif path.is_dir():
            img_path = self._get_img_path_from_folder(path)
        elif path.suffix == ".csv":
            img_path = self._get_img_path_from_csv(path)
        else:
            raise ValueError(f"unrecognized input image path in model: {path}")

        return self._extract_num_channels_from_image(img_path)

    def _get_img_path_from_folder(self, path: Path) -> str:
        """
        Determine total number of channels for image in a set folder
        """
        # we expect user to have the same number of channels for all images in their folders
        # and that only images are stored in those folders
        # Get first image path
        path_generator: Generator[Path] = path.glob("*")
        image: Path = next(path_generator)
        # ignore hidden files
        while str(image.name).startswith("."):
            image = next(path_generator)
        return str(image.resolve())

    def _get_img_path_from_csv(self, path: Path) -> str:
        with open(path) as file:
            reader: csv.reader = csv.DictReader(file)
            line_data_path: str = next(reader)["raw"]
        return line_data_path

    def _extract_num_channels_from_image(self, path: Union[str, Path]) -> int:
        img: AICSImage = AICSImage(str(path))
        return img.dims.C

    def stop_channel_extraction(self) -> None:
        if (
            self._current_thread
            and self._current_thread.thread.isRunning()
        ):
            self._deprecated_threads[self._current_thread.id] = self._current_thread.thread
            self._current_thread.thread.requestInterruption()

    def _initiate_channel_extraction(self) -> None:
        # must wait for the thread to finish naturally; otherwise
        # we would be forcing an exit within AICSImage code, which
        # could have unforeseen consequences
        self.stop_channel_extraction()

        self._current_thread = ThreadTracker(self._threads_created, ChannelExtractionThread(
            self.extract_num_channels, self._threads_created))
        self._threads_created += 1
        
        self._current_thread.thread.channels_ready.connect(
            self._model.set_max_channels
        )
        self._current_thread.thread.dep_thread_done.connect(
            lambda id: self._deprecated_threads.pop(id)
        )
        self._current_thread.thread.start()
