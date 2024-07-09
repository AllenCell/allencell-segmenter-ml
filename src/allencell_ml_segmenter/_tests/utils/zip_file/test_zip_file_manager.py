from pathlib import Path
from unittest.mock import Mock, mock_open, patch, create_autospec
from zipfile import ZipFile

import allencell_ml_segmenter
from allencell_ml_segmenter.utils.file_utils import FileUtils
from allencell_ml_segmenter.utils.zip_file import ZipFileManager


def test_write_to_file() -> None:
    # Arrange
    to_write: bytes = bytes("abcde", "utf-8")
    test_path: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "zip_files"
        / "test_zip.zip"
    )
    open_file_mock: Mock = mock_open()
    file_manager = ZipFileManager.global_instance()

    # Act
    with patch(
        "allencell_ml_segmenter.utils.zip_file.zip_file_manager.open",
        open_file_mock,
    ):
        file_manager.write_zip_file(test_path, to_write)

    # Assert
    open_file_mock.assert_called_with(test_path, "wb")
    open_file_mock.return_value.write.assert_called_once_with(to_write)


def test_unzip_zipped_file_and_delete_zip():
    # Arrange
    file_manager = ZipFileManager.global_instance()
    mock_extract_all = Mock()
    mock_path = create_autospec(Path)
    mock_path.parent = "parent"
    with patch(
        "allencell_ml_segmenter.utils.zip_file.zip_file_manager.ZipFile"
    ) as zipfile_mock:
        zipfile_mock.return_value.__enter__.return_value.extractall = (
            mock_extract_all
        )

        # Act
        file_manager.unzip_zipped_file_and_delete_zip(mock_path)

    # Assert
    mock_extract_all.assert_called_once_with(mock_path.parent)
    mock_path.unlink.assert_called_once()
