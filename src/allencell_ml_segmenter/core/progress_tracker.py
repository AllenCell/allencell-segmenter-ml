from abc import abstractmethod


class ProgressTracker:
    """
    Base class for all ProgressTrackers to inherit from. A ProgressTracker
    maintains an integer measure of progress between progress_minimum and
    progress_maximum. The progress value can be used by PyQt progress bars
    for example.
    """

    def __init__(self, progress_minimum: int = 0, progress_maximum: int = 0):
        self._progress_minimum: int = progress_minimum
        self._progress_maximum: int = progress_maximum
        self._progress: int = progress_minimum

    def get_progress_minimum(self) -> int:
        return self._progress_minimum

    def get_progress_maximum(self) -> int:
        return self._progress_maximum

    def get_progress(self) -> int:
        return self._progress

    def set_progress_minimum(self, progress_minimum: int) -> None:
        self._progress_minimum = progress_minimum

    def set_progress_maximum(self, progress_maximum: int) -> None:
        self._progress_maximum = progress_maximum

    def set_progress(self, progress: int) -> None:
        self._progress = (
            progress
            if progress >= self._progress_minimum
            and progress <= self._progress_maximum
            else self._progress
        )

    @abstractmethod
    def start_tracker(self) -> None:
        pass

    @abstractmethod
    def stop_tracker(self) -> None:
        pass
