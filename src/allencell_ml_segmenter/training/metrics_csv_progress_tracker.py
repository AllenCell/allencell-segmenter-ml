from pathlib import Path
from watchdog.observers.api import BaseObserver
from watchdog.observers import Observer
from allencell_ml_segmenter.core.progress_tracker import ProgressTracker
from allencell_ml_segmenter.training.metrics_csv_event_handler import (
    MetricsCSVEventHandler,
)
from typing import Optional


class MetricsCSVProgressTracker(ProgressTracker):
    """
    A MetricsCSVProgressTracker measures progress by observing a metrics CSV file
    produced by cyto-dl and taking the greatest epoch listed inside of it as its
    measure of progress. Relies heavily on current cyto-dl file logging procedure.
    """

    def __init__(self, csv_path: Path, current_epoch: int, num_epochs: int, version_number: int):
        """
        :param csv_path: path to cyto-dl csv directory for an experiment
        :param num_epochs: maximum number of epochs that will be recorded in the csv
        :param version_number: experiment version to track
        """
        super().__init__(progress_minimum=current_epoch, progress_maximum=num_epochs)

        self._csv_path: Path = csv_path
        if not csv_path.exists():
            csv_path.mkdir(parents=True)

        self._target_path: Path = (
            csv_path / f"version_{version_number}" / "metrics.csv"
        )
        self._observer: Optional[BaseObserver] = None

    # override
    def start_tracker(self) -> None:
        self.stop_tracker()
        self._observer = Observer()
        event_handler: MetricsCSVEventHandler = MetricsCSVEventHandler(
            self._target_path, self.set_progress, self.get_progress_minimum()
        )
        self._observer.schedule(
            event_handler, path=self._csv_path, recursive=True
        )
        self._observer.start()

    # override
    def stop_tracker(self) -> None:
        if self._observer:
            self._observer.stop()
