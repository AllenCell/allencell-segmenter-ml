from pathlib import Path
import allencell_ml_segmenter
from allencell_ml_segmenter.training.metrics_csv_event_handler import (
    MetricsCSVEventHandler,
)
from unittest.mock import Mock


def test_csv_3_epochs():
    callback_mock: Mock = Mock()
    label_mock: Mock = Mock()
    test_csv_path: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "main"
        / "experiments_home"
        / "1_exp"
        / "csv"
        / "version_1"
        / "test_metrics_csv_3_epochs.csv"
    )
    handler: MetricsCSVEventHandler = MetricsCSVEventHandler(
        test_csv_path, callback_mock, label_mock
    )
    fs_event_mock: Mock = Mock(src_path=test_csv_path)
    handler.on_any_event(fs_event_mock)
    callback_mock.assert_called_with(3)


def test_empty_csv():
    callback_mock: Mock = Mock()
    label_mock: Mock = Mock()
    test_csv_path: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "main"
        / "experiments_home"
        / "1_exp"
        / "csv"
        / "version_0"
        / "test_metrics_csv_empty.csv"
    )
    handler: MetricsCSVEventHandler = MetricsCSVEventHandler(
        test_csv_path, callback_mock, label_mock
    )
    fs_event_mock: Mock = Mock(src_path=test_csv_path)
    handler.on_any_event(fs_event_mock)
    # since test_metrics_csv_empty.csv is empty, we expect 0
    callback_mock.assert_called_with(0)


def test_nonexistent_csv():
    callback_mock: Mock = Mock()
    label_mock: Mock = Mock()
    test_csv_path: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "main"
        / "experiments_home"
        / "0_exp"
        / "csv"
        / "version_0"
        / "test_metrics_does_not_exist.csv"
    )
    handler: MetricsCSVEventHandler = MetricsCSVEventHandler(
        test_csv_path, callback_mock, label_mock
    )
    fs_event_mock: Mock = Mock(src_path=test_csv_path)
    handler.on_any_event(fs_event_mock)
    callback_mock.assert_not_called()
