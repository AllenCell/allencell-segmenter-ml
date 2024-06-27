from abc import abstractmethod
from qtpy.QtCore import QObject, Signal


class ProgressTrackerSignals(QObject):
    progress_changed: Signal = Signal(int)
    label_text_changed: Signal = Signal(str)
    progress_max_changed: Signal = Signal(int)


class ProgressTracker:
    """
    Base class for all ProgressTrackers to inherit from. A ProgressTracker
    maintains an integer measure of progress between progress_minimum and
    progress_maximum. The progress value can be used by PyQt progress bars
    for example. It also maintains a string which can be used as a label for
    the progress.
    """

    def __init__(
        self,
        progress_minimum: int = 0,
        progress_maximum: int = 0,
        label_text: str = "Progress",
    ):
        self._progress_minimum: int = progress_minimum
        self._progress_maximum: int = progress_maximum
        self._progress: int = progress_minimum
        self._label_text: str = label_text
        self.signals: ProgressTrackerSignals = ProgressTrackerSignals()

    def get_progress_minimum(self) -> int:
        return self._progress_minimum

    def get_progress_maximum(self) -> int:
        return self._progress_maximum

    def set_progress_maximum(self, maximum: int) -> None:
        self._progress_maximum = maximum
        self.signals.progress_max_changed.emit(self._progress_maximum)

    def get_progress(self) -> int:
        return self._progress

    def set_progress(self, progress: int) -> None:
        """
        If param progress > progress_maximum, throws ValueError.
        If param progress < progress minimum, throws ValueError.
        Otherwise sets this trackers progress to param progress.
        """
        if progress > self._progress_maximum:
            raise ValueError(
                "cannot set progress to value greater than progress_maximum"
            )
        if progress < self._progress_minimum:
            raise ValueError(
                "cannot set progress to value less than progress_minimum"
            )

        self._progress = progress
        self.signals.progress_changed.emit(self._progress)

    def get_label_text(self) -> str:
        return self._label_text

    def set_label_text(self, label_text: str) -> None:
        self._label_text = label_text
        self.signals.label_text_changed.emit(self._label_text)

    @abstractmethod
    def start_tracker(self) -> None:
        """
        Enables updates to the progress measure from another thread.
        """
        pass

    @abstractmethod
    def stop_tracker(self) -> None:
        """
        Stops any threads that may be active for progress updates.
        Must be called before losing reference to the instance of the ProgressTracker.
        """
        pass
