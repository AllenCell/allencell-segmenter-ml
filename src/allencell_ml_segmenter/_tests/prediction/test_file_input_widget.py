from unittest.mock import patch

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog
from pytestqt.qtbot import QtBot

from allencell_ml_segmenter.prediction.file_input_widget import (
    PredictionFileInput,
)
from allencell_ml_segmenter.prediction.model import PredictionModel


@pytest.fixture
def file_input_widget(qtbot: QtBot):
    """
    Fixture that creates an instance of ModelInputWidget for testing.
    """
    return PredictionFileInput(PredictionModel())


def test_top_radio_button_slot(
    qtbot: QtBot, file_input_widget: PredictionFileInput
) -> None:
    """
    Test the _top_radio_button_slot method of PredictionFileInput.
    """
    # ARRANGE - explicitly disable file_input_widget._image_list and enable file_input_widget._browse_dir_edit
    file_input_widget._image_list.setEnabled(False)
    file_input_widget._browse_dir_edit.setEnabled(True)

    # ACT
    with qtbot.waitSignals([file_input_widget._radio_on_screen.toggled]):
        file_input_widget._radio_on_screen.click()

    # ASSERT - states should have flipped
    assert file_input_widget._image_list.isEnabled()
    assert not file_input_widget._browse_dir_edit.isEnabled()


def test_bottom_radio_button_slot(
    qtbot: QtBot, file_input_widget: PredictionFileInput
) -> None:
    """
    Test the _bottom_radio_button_slot method of PredictionFileInput.
    """
    # ARRANGE - explicitly enable file_input_widget._image_list and disable file_input_widget._browse_dir_edit
    file_input_widget._image_list.setEnabled(True)
    file_input_widget._browse_dir_edit.setEnabled(False)

    # ACT
    with qtbot.waitSignals([file_input_widget._radio_directory.toggled]):
        file_input_widget._radio_directory.click()

    # ASSERT - states should have flipped
    assert not file_input_widget._image_list.isEnabled()
    assert file_input_widget._browse_dir_edit.isEnabled()


def test_preprocessing_method(
    qtbot: QtBot,
    file_input_widget: PredictionFileInput,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

        # ASSERT
        assert file_input_widget._model.get_preprocessing_method() is None

        # ACT
        qtbot.mouseClick(
            file_input_widget._browse_output_edit._button, Qt.LeftButton
        )

        # ASSERT
        assert file_input_widget._model.get_preprocessing_method() is None
