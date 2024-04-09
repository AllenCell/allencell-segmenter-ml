from pathlib import Path
from typing import List
from unittest.mock import patch, Mock

import pytest
from qtpy.QtWidgets import QFileDialog
from qtpy.QtCore import Qt
from pytestqt.qtbot import QtBot

import allencell_ml_segmenter
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter._tests.fakes.fake_experiments_model import (
    FakeExperimentsModel,
)
from allencell_ml_segmenter._tests.fakes.fake_image_selection_widget import FakeImageSelectionWidget

from allencell_ml_segmenter.training.image_selection_widget import (
    ImageSelectionWidget,
)
from allencell_ml_segmenter.training.training_model import TrainingModel


MOCK_PATH: str = "/path/to/file"


@pytest.fixture
def experiments_model() -> FakeExperimentsModel:
    """
    Fixture that creates an instance of FakeExperimentsModel for testing.
    """
    return FakeExperimentsModel()


@pytest.fixture
def training_model(experiments_model) -> TrainingModel:
    """
    Fixture that creates an instance of TrainingModel for testing.
    """
    return TrainingModel(MainModel(), experiments_model=experiments_model)


@pytest.fixture
def image_selection_widget(
    training_model: TrainingModel, experiments_model: FakeExperimentsModel
) -> ImageSelectionWidget:
    """
    Fixture that creates an instance of ImageSelectionWidget for testing.
    """
    return ImageSelectionWidget(training_model, FakeExperimentsModel())


# decorator used to stub QFileDialog and avoid nested context managers
@patch.multiple(
    QFileDialog,
    exec_=Mock(return_value=QFileDialog.Accepted),
    selectedFiles=Mock(return_value=[MOCK_PATH]),
)
def test_set_images_directory(
    qtbot: QtBot,
    image_selection_widget: ImageSelectionWidget,
    training_model: TrainingModel,
) -> None:
    """
    Tests that the slot connected to _directory_input_button properly sets the images directory field.
    """
    # ACT
    qtbot.mouseClick(
        image_selection_widget._images_directory_input_button._button,
        Qt.LeftButton,
    )

    # ASSERT
    assert training_model.get_images_directory() == Path(MOCK_PATH)


def test_set_channel_index(
    image_selection_widget: ImageSelectionWidget,
    training_model: TrainingModel,
) -> None:
    """
    Tests that the slot connected to the image channel QComboBox properly sets the image channel index field.
    """
    # ARRANGE - add arbitrary channel index options to the QComboBox, since it does not come with default choices
    mock_choices: List[str] = [str(i) for i in range(10)]
    image_selection_widget._channel_combo_box.addItems(mock_choices)

    for i in range(10):
        # ACT
        image_selection_widget._channel_combo_box.setCurrentIndex(i)

        # ASSERT
        assert training_model.get_channel_index() == i

def test_update_channels_subscription(
    experiments_model: FakeExperimentsModel
):

    # Arrange
    training_model: TrainingModel = TrainingModel(MainModel(), experiments_model=experiments_model)
    training_model.set_images_directory((
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "images"))
    fake_image_selection_widget: FakeImageSelectionWidget = FakeImageSelectionWidget(training_model, experiments_model)

    # Act
    training_model.dispatch_channel_extraction()

    # Assert
    assert fake_image_selection_widget.channels_updated_with_max == training_model.get_max_channel()

