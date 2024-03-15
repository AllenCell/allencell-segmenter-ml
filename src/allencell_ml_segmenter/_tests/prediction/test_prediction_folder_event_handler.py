from allencell_ml_segmenter.prediction.prediction_folder_event_handler import (
    PredictionFolderEventHandler,
)
from unittest.mock import Mock


def test_file_creation_omesegpredtif():
    # ARRANGE
    fake_file_path: str = "/path/to/some.ome_seg_pred.tif"
    progress_callback_mock: Mock = Mock()
    fs_file_created_event_mock: Mock = Mock(src_path=fake_file_path)
    handler: PredictionFolderEventHandler = PredictionFolderEventHandler(
        progress_callback_mock
    )

    # ACT/ASSERT
    handler.on_created(fs_file_created_event_mock)
    progress_callback_mock.assert_called_with(1)

    handler.on_created(fs_file_created_event_mock)
    progress_callback_mock.assert_called_with(2)

    handler.on_created(fs_file_created_event_mock)
    progress_callback_mock.assert_called_with(3)


def test_file_creation_omesegpredtiff():
    # ARRANGE
    fake_file_path: str = "/path/to/some.ome_seg_pred.tiff"
    progress_callback_mock: Mock = Mock()
    fs_file_created_event_mock: Mock = Mock(src_path=fake_file_path)
    handler: PredictionFolderEventHandler = PredictionFolderEventHandler(
        progress_callback_mock
    )

    # ACT/ASSERT
    handler.on_created(fs_file_created_event_mock)
    progress_callback_mock.assert_called_with(1)

    handler.on_created(fs_file_created_event_mock)
    progress_callback_mock.assert_called_with(2)

    handler.on_created(fs_file_created_event_mock)
    progress_callback_mock.assert_called_with(3)


def test_file_creation_bad_ext():
    # ARRANGE
    progress_callback_mock: Mock = Mock()
    handler: PredictionFolderEventHandler = PredictionFolderEventHandler(
        progress_callback_mock
    )

    # ACT/ASSERT
    handler.on_created(Mock(src_path="/bad/file/path.png"))
    progress_callback_mock.assert_not_called()

    handler.on_created(Mock(src_path="/bad/file/.DS_Store"))
    progress_callback_mock.assert_not_called()

    handler.on_created(Mock(src_path="/bad/file/path.zip"))
    progress_callback_mock.assert_not_called()
