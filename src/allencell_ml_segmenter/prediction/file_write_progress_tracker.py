from pathlib import Path
from watchdog.observers.api import BaseObserver
from watchdog.observers import Observer
from allencell_ml_segmenter.core.progress_tracker import ProgressTracker
from allencell_ml_segmenter.prediction.file_write_event_handler import (
    FileWriteEventHandler,
)
from typing import Optional


class FileWriteProgressTracker(ProgressTracker):
    """
    A FileWriteProgressTracker measures progress by observing a folder in
    which cyto-dl will create images and incrementing progress when
    a new prediction image is created and placed in this folder.
    """

    def __init__(self, write_folder_path: Path, num_preds: int):
        """
        :param progress_folder_path: path to the output directory for predictions
        :param num_preds: total number of new pred files expected to be written to the folder
        """
        super().__init__(
            progress_minimum=0,
            progress_maximum=num_preds,
            label_text="Prediction progress",
        )

        if not write_folder_path.exists():
            write_folder_path.mkdir(parents=True)
        self._write_folder_path: Path = write_folder_path

        self._observer: Optional[BaseObserver] = None

    # override
    def start_tracker(self) -> None:
        self.stop_tracker()
        self._observer = Observer()
        event_handler: FileWriteEventHandler = FileWriteEventHandler(
            self.set_progress
        )
        self._observer.schedule(
            event_handler,
            path=str(self._write_folder_path.resolve()),
            recursive=False,
        )
        self._observer.start()

    # override
    def stop_tracker(self) -> None:
        if self._observer:
            self._observer.stop()
