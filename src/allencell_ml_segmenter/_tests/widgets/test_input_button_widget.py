from unittest.mock import patch, Mock

import pytest
from qtpy.QtWidgets import QFileDialog
from pytestqt.qtbot import QtBot
from qtpy.QtCore import Qt

from allencell_ml_segmenter.widgets.input_button_widget import InputButton
from allencell_ml_segmenter.prediction.model import PredictionModel


@pytest.fixture
def model_set_file_path_function() -> Mock:
    """
    Fixture that creates a mock function for setting a file path.
    """
    return Mock()


@pytest.fixture
def input_button(
    qtbot: QtBot, model_set_file_path_function: Mock
) -> InputButton:
    """
    Fixture that creates an instance of InputButton for testing. This InputButton is meant to mock selecting files.
    """
    return InputButton(
        Mock(spec=PredictionModel), model_set_file_path_function
    )


def test_set_text(
    qtbot: QtBot, input_button: InputButton, model_set_file_path_function: Mock
) -> None:
    """
    Tests the _update_path_text method of InputButton for file use cases.
    """
    # ARRANGE
    mock_path: str = "/path/to/file"

    with patch.object(
        QFileDialog, "getOpenFileName", return_value=(mock_path, "")
    ):
        # ACT
        qtbot.mouseClick(input_button.button, Qt.LeftButton)

        # ASSERT
        assert input_button._text_display.text() == mock_path
        model_set_file_path_function.assert_called_once()


def test_elongate(input_button: InputButton) -> None:
    """
    Tests the elongate method of InputButton.
    """
    # ARRANGE
    initial_width: int = input_button._text_display.width()
    expected_width: int = initial_width + 100

    # ACT
    input_button.elongate(expected_width)

    # ASSERT
    assert input_button._text_display.width() == expected_width
