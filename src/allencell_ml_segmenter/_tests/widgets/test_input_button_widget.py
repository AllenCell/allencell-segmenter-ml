from unittest.mock import patch, Mock

import pytest
from PyQt5.QtWidgets import QFileDialog
from pytestqt.qtbot import QtBot
from qtpy.QtCore import Qt

from allencell_ml_segmenter.widgets.input_button_widget import InputButton
from allencell_ml_segmenter.prediction.model import PredictionModel


@pytest.fixture
def input_button(qtbot: QtBot) -> InputButton:
    """
    Fixture that creates an instance of InputButton for testing.
    """
    return InputButton(
        Mock(spec=PredictionModel), model_set_file_path_function=Mock()
    )


def test_set_file_text(
    qtbot: QtBot, input_button: InputButton, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Test the _update_path_text method of InputButton for file use cases.
    """
    # Arrange
    with patch.object(
        QFileDialog, "getOpenFileName", return_value=("/path/to/file", "")
    ):
        # Act
        qtbot.mouseClick(input_button._button, Qt.LeftButton)
        assert input_button._text_display.text() == "/path/to/file"


def test_elongate(input_button: InputButton) -> None:
    """
    Test the elongate method of InputButton.
    """
    # ARRANGE
    initial_width: int = input_button._text_display.width()
    expected_width: int = initial_width + 100

    # ACT
    input_button.elongate(expected_width)

    # ASSERT
    assert input_button._text_display.width() == expected_width
