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
    # ACT
    # TODO: replace QFileDialog details after Brian enables acceptance of either a directory or a file
    with patch.object(
        QFileDialog, "getOpenFileName", return_value=("/path/to/file", "")
    ):
        qtbot.mouseClick(
            image_selection_widget._directory_input_button._button,
            Qt.LeftButton,
        )

    # ASSERT
    assert training_model.get_images_directory() == Path("/path/to/file")
