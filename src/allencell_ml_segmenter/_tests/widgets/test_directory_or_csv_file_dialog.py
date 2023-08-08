from unittest.mock import patch

import pytest
from qtpy.QtWidgets import QFileDialog
from pytestqt.qtbot import QtBot

from allencell_ml_segmenter.widgets.directory_or_csv_file_dialog import (
    DirectoryOrCSVFileDialog,
)


@pytest.fixture
def directory_or_csv_file_dialog(qtbot: QtBot) -> DirectoryOrCSVFileDialog:
    """
    Fixture that creates an instance of CustomFileDialog for testing.
    """
    return DirectoryOrCSVFileDialog()


def test_selected(
    directory_or_csv_file_dialog: DirectoryOrCSVFileDialog,
) -> None:
    """
    Test the _selected method of CustomFileDialog.
    """
    # ARRANGE
    mock_path: str = "/path/to/file"

    # ACT
    with patch("os.path.isdir", return_value=True):
        directory_or_csv_file_dialog._selected(mock_path)

    # ASSERT
    assert directory_or_csv_file_dialog.fileMode() == QFileDialog.Directory

    # ACT
    with patch("os.path.isdir", return_value=False):
        directory_or_csv_file_dialog._selected(mock_path)

    # ASSERT
    assert directory_or_csv_file_dialog.fileMode() == QFileDialog.ExistingFile
