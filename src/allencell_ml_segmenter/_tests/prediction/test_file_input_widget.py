import pytest
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
