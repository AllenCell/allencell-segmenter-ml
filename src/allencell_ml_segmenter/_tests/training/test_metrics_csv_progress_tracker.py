from pathlib import Path
import allencell_ml_segmenter
from allencell_ml_segmenter.training.metrics_csv_progress_tracker import (
    MetricsCSVProgressTracker,
)


def test_get_last_csv_version_versions():
    test_csv_path: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "metrics_csv"
        / "version_subdirs"
    )
    tracker: MetricsCSVProgressTracker = MetricsCSVProgressTracker(
        test_csv_path, 10
    )
    assert tracker.get_last_csv_version() == 1


def test_get_last_csv_version_no_versions():
    test_csv_path: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "metrics_csv"
        / "no_version_subdirs"
    )
    tracker: MetricsCSVProgressTracker = MetricsCSVProgressTracker(
        test_csv_path, 10
    )
    assert tracker.get_last_csv_version() == -1
