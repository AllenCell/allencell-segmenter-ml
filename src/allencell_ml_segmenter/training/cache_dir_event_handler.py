from watchdog.events import FileSystemEvent, FileSystemEventHandler
from typing import Callable


class CacheDirEventHandler(FileSystemEventHandler):
    """
    A CacheDirEventHandler calls :param progress_callback: when a .pt file
    is created in the watched directory with the number of .pt files that have
    been created.
    """

    def __init__(
        self,
        progress_callback: Callable,
    ):
        super().__init__()
        self._progress_callback: Callable = progress_callback
        self._num_files = 0

    # override
    def on_created(self, event: FileSystemEvent) -> None:
        if event.src_path.endswith(".pt"):
            self._num_files += 1
            self._progress_callback(self._num_files)
