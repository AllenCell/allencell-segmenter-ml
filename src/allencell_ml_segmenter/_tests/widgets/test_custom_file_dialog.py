from unittest.mock import patch

import pytest
from qtpy.QtWidgets import QFileDialog
from pytestqt.qtbot import QtBot

from allencell_ml_segmenter.widgets.custom_file_dialog import CustomFileDialog


@pytest.fixture
def custom_file_dialog(qtbot: QtBot) -> CustomFileDialog:
    """
    Fixture that creates an instance of CustomFileDialog for testing.
    """
    return CustomFileDialog()


def test_selected(custom_file_dialog: CustomFileDialog) -> None:
    """
    Test the _selected method of CustomFileDialog.
    """
    # ARRANGE
    mock_path: str = "/path/to/file"

    # ACT
    with patch("os.path.isdir", return_value=True):
        custom_file_dialog._selected(mock_path)

    # ASSERT
    assert custom_file_dialog.fileMode() == QFileDialog.Directory

    # ACT
    with patch("os.path.isdir", return_value=False):
        custom_file_dialog._selected(mock_path)

    # ASSERT
    assert custom_file_dialog.fileMode() == QFileDialog.ExistingFile
