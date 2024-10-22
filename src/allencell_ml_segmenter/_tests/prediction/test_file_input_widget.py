from unittest.mock import patch, Mock

import pytest
from qtpy.QtWidgets import QFileDialog
from qtpy.QtCore import Qt
from pytestqt.qtbot import QtBot

from allencell_ml_segmenter._tests.fakes.fake_viewer import FakeViewer
from allencell_ml_segmenter.prediction.file_input_widget import (
    FileInputWidget,
)
from allencell_ml_segmenter.prediction.model import (
    PredictionModel,
)
from allencell_ml_segmenter.core.file_input_model import InputMode

MOCK_PATH: str = "/path/to/file"


@pytest.fixture
def prediction_model(qtbot: QtBot) -> PredictionModel:
    return PredictionModel()


@pytest.fixture
def file_input_widget(
    qtbot: QtBot, prediction_model: PredictionModel
) -> FileInputWidget:
    """
    Fixture that creates an instance of ModelInputWidget for testing.
    """
    return FileInputWidget(
        prediction_model, viewer=FakeViewer(), service=None
    )


def test_top_radio_button_slot(
    qtbot: QtBot,
    file_input_widget: FileInputWidget,
    prediction_model: PredictionModel,
) -> None:
    """
    Test the _top_radio_button_slot method of PredictionFileInput.
    """
    # ARRANGE - explicitly disable file_input_widget._image_list and enable file_input_widget._browse_dir_edit
    file_input_widget._image_list.setEnabled(False)
    file_input_widget._browse_dir_edit.setEnabled(True)

    # ACT
    with qtbot.waitSignal(file_input_widget._radio_on_screen.toggled):
        file_input_widget._radio_on_screen.click()

    # ASSERT - states should have flipped
    assert file_input_widget._image_list.isEnabled()
    assert not file_input_widget._browse_dir_edit.isEnabled()
    assert prediction_model.get_input_mode() == InputMode.FROM_NAPARI_LAYERS


def test_bottom_radio_button_slot(
    qtbot: QtBot,
    file_input_widget: FileInputWidget,
    prediction_model: PredictionModel,
) -> None:
    """
    Test the _bottom_radio_button_slot method of PredictionFileInput.
    """
    # ARRANGE - explicitly enable file_input_widget._image_list and disable file_input_widget._browse_dir_edit
    file_input_widget._image_list.setEnabled(True)
    file_input_widget._browse_dir_edit.setEnabled(False)

    # ACT
    with qtbot.waitSignal(file_input_widget._radio_directory.toggled):
        file_input_widget._radio_directory.click()

    # ASSERT - states should have flipped
    assert not file_input_widget._image_list.isEnabled()
    assert file_input_widget._browse_dir_edit.isEnabled()
    assert prediction_model.get_input_mode() == InputMode.FROM_PATH


# decorator used to stub QFileDialog and avoid nested context managers
@patch.multiple(
    QFileDialog,
    exec_=Mock(return_value=QFileDialog.Accepted),
    selectedFiles=Mock(return_value=[MOCK_PATH]),
    getExistingDirectory=Mock(return_value=MOCK_PATH),
)
def test_populate_input_channel_combobox(qtbot: QtBot) -> None:
    # Arrange
    prediction_model: PredictionModel = PredictionModel()
    prediction_file_input: FileInputWidget = FileInputWidget(
        prediction_model, viewer=FakeViewer(), service=None
    )
    prediction_model.set_max_channels(6)

    # Act
    prediction_file_input._populate_input_channel_combobox()

    # Assert
    assert prediction_file_input._channel_select_dropdown.isEnabled()
