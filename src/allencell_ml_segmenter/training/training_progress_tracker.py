from pathlib import Path
from watchdog.observers.api import BaseObserver
from watchdog.observers import Observer
from allencell_ml_segmenter.core.progress_tracker import ProgressTracker
from allencell_ml_segmenter.training.metrics_csv_event_handler import (
    MetricsCSVEventHandler,
)
from allencell_ml_segmenter.training.cache_dir_event_handler import (
    CacheDirEventHandler,
)
from typing import Optional


class TrainingProgressTracker(ProgressTracker):
    """
    A MetricsCSVProgressTracker measures progress by observing a metrics CSV file
    produced by cyto-dl and using the number of epochs written to it as a
    measure of progress. Relies heavily on current cyto-dl file logging procedure.
    """

    def __init__(
        self,
        csv_path: Path,
        cache_path: Path,
        num_epochs: int,
        total_files: int,
        version_number: int,
    ):
        """
        :param csv_path: path to cyto-dl csv directory for an experiment
        :param cache_path: path to cache directory for an experiment
        :param num_epochs: number of epochs that will be recorded in the csv
        :param total_files: total number of unique files used for training
        :param version_number: experiment version to track
        """
        # set to 0, 0 to initially have a spinning progress bar
        super().__init__(
            progress_minimum=0,
            progress_maximum=0,
            label_text=f"Files cached: 0 / {total_files}",
        )
        self._num_epochs = num_epochs

        self._csv_path: Path = csv_path
        # these paths will be created by cyto-dl anyways, but we need to make
        # sure they exist before we try to set an observer on them
        if not csv_path.exists():
            csv_path.mkdir(parents=True)

        self._cache_path: Path = cache_path
        if not cache_path.exists():
            cache_path.mkdir(parents=True)

        self._target_path: Path = (
            csv_path / f"version_{version_number}" / "metrics.csv"
        )
        self._observer: Optional[BaseObserver] = None
        self._total_files: int = total_files

    def _set_progress(self, progress: int) -> None:
        # on the first call to _set_progress, we expect the progress bar to
        # be in a spinning state (max = 0), we want to change that to the real max
        # to show deterministic progress here
        if self.get_progress_maximum() != self._num_epochs:
            self.set_progress_maximum(self._num_epochs)
        self.set_progress(progress)

    # override
    def start_tracker(self) -> None:
        self.stop_tracker()
        self._observer = Observer()
        csv_handler: MetricsCSVEventHandler = MetricsCSVEventHandler(
            self._target_path, self._set_progress, self.set_label_text
        )
        self._observer.schedule(
            csv_handler, path=self._csv_path, recursive=True
        )
        cache_handler: CacheDirEventHandler = CacheDirEventHandler(
            self._set_cache_progress_text
        )
        self._observer.schedule(
            cache_handler, path=self._cache_path, recursive=True
        )
        self._observer.start()

    # override
    def stop_tracker(self) -> None:
        if self._observer:
            self._observer.stop()

    def _set_cache_progress_text(self, num_cached: int) -> None:
        self.set_label_text(
            f"Files cached: {num_cached} / {self._total_files}"
        )
