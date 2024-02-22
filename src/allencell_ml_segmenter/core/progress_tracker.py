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

    def set_progress(self, progress: int) -> None:
        """
        If param progress > progress_maximum, throws ValueError.
        If param progress < progress minimum, throws ValueError.
        Otherwise sets this trackers progress to param progress.
        """
        if progress > self._progress_maximum:
            raise ValueError("cannot set progress to value greater than progress_maximum")
        if progress < self._progress_minimum:
            raise ValueError("cannot set progress to value less than progress_minimum")
        
        self._progress = progress

    @abstractmethod
    def start_tracker(self) -> None:
        pass

    @abstractmethod
    def stop_tracker(self) -> None:
        pass
