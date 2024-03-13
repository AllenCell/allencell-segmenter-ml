from watchdog.events import FileSystemEvent, FileSystemEventHandler
from typing import Callable, Set


class PredictionFolderEventHandler(FileSystemEventHandler):
    """
    A PredictionFolderEventHandler calls progress_callback upon creation of a file
    with an extension contained in PRED_FILE_EXTS. The argument to progress_callback
    will be how many files that fit these criteria have been created.
    """

    PRED_FILE_EXTS: Set[str] = {".tif", ".tiff"}

    def __init__(self, progress_callback: Callable):
        super().__init__()
        self._progress_callback: Callable = progress_callback
        self._num_pred_files_created = 0

    # override
    def on_created(self, event: FileSystemEvent) -> None:
        if any([event.src_path.endswith(ext) for ext in self.PRED_FILE_EXTS]):
            self._num_pred_files_created += 1
            self._progress_callback(self._num_pred_files_created)
