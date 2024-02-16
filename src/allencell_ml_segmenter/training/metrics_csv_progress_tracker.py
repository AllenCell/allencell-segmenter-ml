from pathlib import Path
from watchdog.observers.api import BaseObserver
from watchdog.observers import Observer
from allencell_ml_segmenter.core.progress_tracker import ProgressTracker
from allencell_ml_segmenter.training.metrics_csv_event_handler import MetricsCSVEventHandler

class MetricsCSVProgressTracker(ProgressTracker):
    """
    A MetricsCSVProgressTracker measures progress by observing a metrics CSV file
    produced by cyto-dl and taking the greatest epoch listed inside of it as its
    measure of progress. Relies heavily on current cyto-dl file logging procedure.
    """

    def __init__(self, csv_path: Path, progress_minimum: int=0, progress_maximum: int=0):
        super().__init__(progress_minimum, progress_maximum)

        self._csv_path: Path = csv_path
        if not csv_path.exists():
            csv_path.mkdir(parents=True)

        self._target_path: Path = (
            csv_path / f"version_{self._get_last_csv_version() + 1}" / "metrics.csv"
        )
        self._observer: BaseObserver = None

    def start_tracker(self) -> None:
        self.stop_tracker()
        self._observer = Observer()
        event_handler: MetricsCSVEventHandler = MetricsCSVEventHandler(self._target_path, self.set_progress)
        self._observer.schedule(event_handler,  path=self._csv_path,  recursive=True)
        self._observer.start()
    
    def stop_tracker(self) -> None:
        if self._observer:
            self._observer.stop()

    def _get_last_csv_version(self) -> int:
        """
        Returns version number of the most recent version directory within
        the cyto-dl CSV folder (self._csv_path) or -1 if no version directories
        exist
        """
        last_version: int = -1
        if self._csv_path.exists():
            for child in self._csv_path.glob("version_*"):
                if child.is_dir():
                    version_str: str = child.name.split("_")[-1]
                    try:
                        last_version = int(version_str) if int(version_str) > last_version else last_version
                    except ValueError:
                        continue
        return last_version