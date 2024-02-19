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

    def __init__(self, target_path: Path, progress_callback: Callable):
        super().__init__()
        self._target_path: Path = target_path
        self._progress_callback: Callable = progress_callback

    def _get_latest_epoch(self) -> int:
        if not self._target_path.exists():
            return 0

        latest: int = 0
        with self._target_path.open("r", newline="") as fr:
            dict_reader: DictReader = DictReader(fr)
            for row in dict_reader:
                latest = (
                    int(row["epoch"]) if int(row["epoch"]) > latest else latest
                )

        return latest

    def on_any_event(self, event: FileSystemEvent) -> None:
        if self._target_path.exists() and self._target_path.samefile(
            event.src_path
        ):
            self._progress_callback(self._get_latest_epoch())
