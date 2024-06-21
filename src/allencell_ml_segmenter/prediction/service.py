from pathlib import Path
from typing import List, Union, Optional, Dict

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.prediction.model import PredictionModel
from allencell_ml_segmenter.core.channel_extraction import (
    ChannelExtractionThread,
    get_img_path_from_csv,
)
from allencell_ml_segmenter.utils.file_utils import FileUtils
from napari.utils.notifications import show_error


class ModelFileService(Subscriber):
    """
    Parses the chosen model file to extract the preprocessing method.
    """

    def __init__(self, model: PredictionModel):
        super().__init__()
        self._model: PredictionModel = model
        self._current_thread: Optional[ChannelExtractionThread] = None
        self._running_threads: Dict[int, ChannelExtractionThread] = {}
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

    def stop_channel_extraction(self) -> None:
        if self._current_thread and self._current_thread.isRunning():
            self._current_thread.requestInterruption()

    def _get_img_path_from_model(self) -> Path:
        """
        Returns path of an image to be predicted from the prediction model. If model's
        input_image_path is set, will attempt to infer image from that field;
        otherwise will attempt to infer image from model's selected_paths field
        (the images currently checked in the prediction widget). Throws ValueError
        if both fields uninitialized.
        """
        path: Path = self._model.get_input_image_path()
        img_path: Path = None
        if not path:  # using viewer input method
            paths: List[Path] = self._model.get_selected_paths()
            if not paths or len(paths) <= 0:
                raise ValueError(
                    "expected input_image_path or selected_paths to be initialized and non-empty"
                )
            img_path = Path(paths[0])
        elif path.is_dir():
            img_path = FileUtils.get_img_path_from_folder(path)
        elif path.suffix == ".csv":
            img_path = get_img_path_from_csv(path)
        else:
            raise ValueError(f"unrecognized input image path in model: {path}")

        return img_path

    def _initiate_channel_extraction(self) -> None:
        img_path: Path = self._get_img_path_from_model()

        self.stop_channel_extraction()

        self._current_thread = ChannelExtractionThread(img_path)
        thread_id: int = self._threads_created
        self._threads_created += 1
        self._running_threads[thread_id] = self._current_thread

        self._current_thread.channels_ready.connect(
            self._model.set_max_channels
        )
        self._current_thread.task_failed.connect(
            self._on_channel_extraction_failed
        )
        self._current_thread.finished.connect(
            lambda: self._running_threads.pop(thread_id)
        )
        self._current_thread.start()

    def _on_channel_extraction_failed(self, err: Exception) -> None:
        # setting max channels to none will result in the spinner stopping
        # but no channel selection being available
        self._model.set_max_channels(None)
        show_error(f"Channel extraction failed: {err}")
