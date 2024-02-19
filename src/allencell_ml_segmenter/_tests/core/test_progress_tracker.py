from allencell_ml_segmenter.core.progress_tracker import ProgressTracker


def test_set_progress_within_bounds():
    tracker: ProgressTracker = ProgressTracker(
        progress_minimum=0, progress_maximum=10
    )
    tracker.set_progress(2)
    assert tracker.get_progress() == 2
    tracker.set_progress(9)
    assert tracker.get_progress() == 9
    tracker.set_progress(0)
    assert tracker.get_progress() == 0
    tracker.set_progress(10)
    assert tracker.get_progress() == 10


def test_set_progress_greater_than_max():
    tracker: ProgressTracker = ProgressTracker(
        progress_minimum=0, progress_maximum=10
    )
    tracker.set_progress(11)
    assert tracker.get_progress() == 10
    tracker.set_progress(10394)
    assert tracker.get_progress() == 10


def test_set_progress_less_than_min():
    tracker: ProgressTracker = ProgressTracker(
        progress_minimum=0, progress_maximum=10
    )
    tracker.set_progress(-1)
    assert tracker.get_progress() == 0
    tracker.set_progress(-1948)
    assert tracker.get_progress() == 0
