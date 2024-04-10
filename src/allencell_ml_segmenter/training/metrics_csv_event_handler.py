from watchdog.events import FileSystemEvent, FileSystemEventHandler
from pathlib import Path
from csv import DictReader
from typing import Callable


class MetricsCSVEventHandler(FileSystemEventHandler):
    """
    A MetricsCSVEventHandler calls progress_callback upon any changes to
    the provided target_path CSV file, passing the latest epoch in the modified
    CSV to the callback.
    """

    def __init__(
        self, target_path: Path, progress_callback: Callable, min_epoch: int
    ):
        super().__init__()
        self._target_path: Path = target_path
        self._progress_callback: Callable = progress_callback
        self._min_epoch: int = min_epoch

    def _get_latest_epoch(self) -> int:
        if not self._target_path.exists():
            return self._min_epoch

        latest: int = self._min_epoch
        with self._target_path.open("r", newline="") as fr:
            dict_reader: DictReader = DictReader(fr)
            for row in dict_reader:
                latest = (
                    int(row["epoch"]) if int(row["epoch"]) > latest else latest
                )

        return latest

    # override
    def on_any_event(self, event: FileSystemEvent) -> None:
        if self._target_path.exists() and self._target_path.samefile(
            event.src_path
        ):
            self._progress_callback(self._get_latest_epoch())
