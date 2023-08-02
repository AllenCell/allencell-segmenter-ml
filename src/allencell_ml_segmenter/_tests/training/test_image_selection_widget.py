from pathlib import Path
from unittest.mock import patch

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog
from pytestqt.qtbot import QtBot

from allencell_ml_segmenter.training.image_selection_widget import (
    ImageSelectionWidget,
)
from allencell_ml_segmenter.training.training_model import TrainingModel


@pytest.fixture
def training_model() -> TrainingModel:
    """
    Fixture that creates an instance of TrainingModel for testing.
    """
    return TrainingModel()


@pytest.fixture
def image_selection_widget(
    qtbot: QtBot, training_model: TrainingModel
) -> ImageSelectionWidget:
    """
    Fixture that creates an instance of ImageSelectionWidget for testing.
    """
    return ImageSelectionWidget(training_model)


def test_set_images_directory(
    qtbot: QtBot,
    image_selection_widget: ImageSelectionWidget,
    training_model: TrainingModel,
) -> None:
    """
    Tests that the slot connected to the InputButton properly sets the images directory field.
    """
    # TODO: replace QFileDialog details after Brian enables acceptance of either a directory or a file
    # ARRANGE
    with patch.object(
        QFileDialog, "getOpenFileName", return_value=("/path/to/file", "")
    ):
        # ACT
        qtbot.mouseClick(
            image_selection_widget._directory_input_button._button,
            Qt.LeftButton,
        )

    # ASSERT
    assert training_model.get_images_directory() == Path("/path/to/file")


def test_set_channel_index(
    qtbot: QtBot,
    image_selection_widget: ImageSelectionWidget,
    training_model: TrainingModel,
) -> None:
    """
    Tests that the slot connected to the image channel QComboBox properly sets the image channel index field.
    """
    # ARRANGE - add arbitrary channel index options to the QComboBox, since it does not come with default choices
    image_selection_widget._channel_combo_box.addItems(
        [str(i) for i in range(10)]
    )

    # ACT
    image_selection_widget._channel_combo_box.setCurrentIndex(5)

    # ASSERT
    assert training_model.get_channel_index() == 5

    # ACT
    image_selection_widget._channel_combo_box.setCurrentIndex(9)

    # ASSERT
    assert training_model.get_channel_index() == 9
