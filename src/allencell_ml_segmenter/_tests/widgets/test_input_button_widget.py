from unittest.mock import patch, Mock

import pytest
from PyQt5.QtWidgets import QFileDialog
from qtpy.QtCore import Qt

from allencell_ml_segmenter.widgets.input_button_widget import InputButton
from allencell_ml_segmenter.prediction.model import PredictionModel


@pytest.fixture
def input_button(qtbot):
    """
    Fixture that creates an instance of InputButton for testing.
    """
    return InputButton(Mock(spec=PredictionModel))

def test_set_file_text(qtbot, input_button, monkeypatch):
    """
    Test the set_file_text method of InputButton.
    """
    # Arrange
    with patch.object(QFileDialog, "getOpenFileName", return_value=("/path/to/file", "")):
        # Act
        qtbot.mouseClick(input_button._button, Qt.LeftButton)
        assert input_button._text_display.text() == "/path/to/file"