from watchdog.events import FileSystemEvent, FileSystemEventHandler
from pathlib import Path
from csv import DictReader
from typing import Callable, Tuple, Set, Optional


class MetricsCSVEventHandler(FileSystemEventHandler):
    """
    A MetricsCSVEventHandler calls progress_callback upon any changes to
    the provided target_path CSV file, passing the latest epoch in the modified
    CSV to the callback.
    """

    def __init__(
        self,
        target_path: Path,
        progress_callback: Callable,
        label_text_callback: Callable,
    ):
        super().__init__()
        self._target_path: Path = target_path
        self._progress_callback: Callable = progress_callback
        self._label_text_callback: Callable = label_text_callback

    def _get_csv_data(self) -> Tuple[int, Optional[float]]:
        epochs: Set[int] = set()
        latest_loss: Optional[float] = None
        with self._target_path.open("r", newline="") as fr:
            dict_reader: DictReader = DictReader(fr)
            latest_epoch: int = -1
            for row in dict_reader:
                epoch: int = int(row["epoch"])
                epochs.add(epoch)
                if epoch > latest_epoch:
                    try:
                        # this data is not available in all rows for a given epoch
                        latest_loss = float(row["val/loss_epoch"])
                        latest_epoch = epoch
                    except Exception:
                        pass

        return len(epochs), latest_loss

    # override
    def on_any_event(self, event: FileSystemEvent) -> None:
        if self._target_path.exists() and self._target_path.samefile(
            event.src_path
        ):
            epochs, loss = self._get_csv_data()
            self._progress_callback(epochs)
            self._label_text_callback(
                f"Current loss: {'  N/A' if loss is None else '{:.3f}'.format(loss)}"
            )
