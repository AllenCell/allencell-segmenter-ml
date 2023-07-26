from unittest.mock import patch

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog

from allencell_ml_segmenter.prediction.file_input_widget import (
    PredictionFileInput,
)
from allencell_ml_segmenter.prediction.model import PredictionModel


@pytest.fixture
def file_input_widget(qtbot):
    """
    Fixture that creates an instance of ModelInputWidget for testing.
    """
    return PredictionFileInput(PredictionModel())


def test_top_radio_button_slot(qtbot, file_input_widget):
    """
    Test the _top_radio_button_slot method of PredictionFileInput.
    """
    # Disable the bottom input button and enable the top checkbox list widget
    file_input_widget._on_screen_slot()

    assert file_input_widget._image_list.isEnabled()
    assert not file_input_widget._browse_dir_edit.isEnabled()


def test_bottom_radio_button_slot(qtbot, file_input_widget):
    """
    Test the _bottom_radio_button_slot method of PredictionFileInput.
    """
    # Enable the bottom input button and disable the top checkbox list widget
    file_input_widget._from_directory_slot()

    assert not file_input_widget._image_list.isEnabled()
    assert file_input_widget._browse_dir_edit.isEnabled()


def test_preprocessing_method(qtbot, file_input_widget, monkeypatch):
    """
    Test that the input buttons in file_input_widget do not affect the state related
    to the model_input_widget. This test was introduced because any input button instance
    used to manipulate the model_path state in the prediction model.
    """
    # ARRANGE
    with patch.object(
        QFileDialog, "getOpenFileName", return_value=("/path/to/file", "")
    ):
        # ACT
        qtbot.mouseClick(
            file_input_widget._browse_dir_edit._button, Qt.LeftButton
        )
        assert file_input_widget._model.get_preprocessing_method() is None

        qtbot.mouseClick(
            file_input_widget._browse_output_edit._button, Qt.LeftButton
        )
        assert file_input_widget._model.get_preprocessing_method() is None
