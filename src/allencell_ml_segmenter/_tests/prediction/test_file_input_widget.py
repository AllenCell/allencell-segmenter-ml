from unittest.mock import patch, Mock
from pathlib import Path

import pytest
from qtpy.QtWidgets import QFileDialog
from pytestqt.qtbot import QtBot

from allencell_ml_segmenter._tests.fakes.fake_viewer import FakeViewer
from allencell_ml_segmenter.core.file_input_widget import (
    FileInputWidget,
)
from allencell_ml_segmenter.prediction.model import (
    PredictionModel,
)

from allencell_ml_segmenter.core.file_input_model import (
    InputMode,
    FileInputModel,
)

MOCK_PATH: str = "/path/to/file"


@pytest.fixture
def file_input_model(qtbot: QtBot) -> FileInputModel:
    return FileInputModel()


@pytest.fixture
def file_input_widget(
    qtbot: QtBot, file_input_model: FileInputModel
) -> FileInputWidget:
    """
    Fixture that creates an instance of ModelInputWidget for testing.
    """
    return FileInputWidget(file_input_model, viewer=FakeViewer(), service=None)


def test_top_radio_button_slot(
    qtbot: QtBot,
    file_input_widget: FileInputWidget,
    file_input_model: FileInputModel,
) -> None:
    """
    Test the _top_radio_button_slot method of FileInputWidget.
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
    assert file_input_model.get_input_mode() == InputMode.FROM_NAPARI_LAYERS


def test_bottom_radio_button_slot(
    qtbot: QtBot,
    file_input_widget: FileInputWidget,
    file_input_model: FileInputModel,
) -> None:
    """
    Test the _bottom_radio_button_slot method of FileInputWidget.
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
    assert file_input_model.get_input_mode() == InputMode.FROM_PATH


# decorator used to stub QFileDialog and avoid nested context managers
@patch.multiple(
    QFileDialog,
    exec_=Mock(return_value=QFileDialog.Accepted),
    selectedFiles=Mock(return_value=[MOCK_PATH]),
    getExistingDirectory=Mock(return_value=MOCK_PATH),
)
def test_populate_input_channel_combobox(qtbot: QtBot) -> None:
    # Arrange
    file_input_model: FileInputModel = FileInputModel()
    prediction_file_input: FileInputWidget = FileInputWidget(
        file_input_model, viewer=FakeViewer(), service=None
    )
    file_input_model.set_max_channels(6)

    # Act
    prediction_file_input._populate_input_channel_combobox()

    # Assert
    assert prediction_file_input._channel_select_dropdown.isEnabled()


def test_input_image_paths(file_input_model: FileInputModel) -> None:
    """
    Tests that the input image paths are set and retrieved properly.
    """
    # ARRANGE
    dummy_paths: List[Path] = [
        Path("example path " + str(i)) for i in range(10)
    ]

    # ACT
    file_input_model.set_input_image_path(dummy_paths)

    # ASSERT
    assert file_input_model.get_input_image_path() == dummy_paths


def test_image_input_channel_index(file_input_model: FileInputModel) -> None:
    """
    Tests that the channel index is set and retrieved properly.
    """
    for i in range(10):
        # ACT
        file_input_model.set_image_input_channel_index(i)

        # ASSERT
        assert file_input_model.get_image_input_channel_index() == i


def test_output_directory(file_input_model: FileInputModel) -> None:
    """
    Tests that the output directory is set and retrieved properly.
    """
    # ARRANGE
    dummy_path: Path = Path("example path")

    # ACT
    file_input_model.set_output_directory(dummy_path)

    # ASSERT
    assert file_input_model.get_output_directory() == dummy_path
