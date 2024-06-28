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

from allencell_ml_segmenter.training.image_selection_widget import (
    ImageSelectionWidget,
)
from allencell_ml_segmenter.training.training_model import TrainingModel, TrainingImageType


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
    return ImageSelectionWidget(training_model, experiments_model)


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
        image_selection_widget._images_directory_input_button.button,
        Qt.LeftButton,
    )

    # ASSERT
    assert training_model.get_images_directory() == Path(MOCK_PATH)

def test_combos_update_when_num_channels_set(
        qtbot: QtBot,
        image_selection_widget: ImageSelectionWidget,
        training_model: TrainingModel,
    ) -> None:
    # ASSERT (sanity check)
    assert image_selection_widget._raw_channel_combo_box.count() == 0
    assert image_selection_widget._seg1_channel_combo_box.count() == 0
    assert image_selection_widget._seg2_channel_combo_box.count() == 0

    assert image_selection_widget._raw_channel_combo_box.currentIndex() == -1
    assert image_selection_widget._seg1_channel_combo_box.currentIndex() == -1
    assert image_selection_widget._seg2_channel_combo_box.currentIndex() == -1

    assert training_model.get_selected_channel(TrainingImageType.RAW) == None
    assert training_model.get_selected_channel(TrainingImageType.SEG1) == None
    assert training_model.get_selected_channel(TrainingImageType.SEG2) == None

    # ACT (image selection widget should react to this event)
    training_model.set_all_num_channels({
        TrainingImageType.RAW: 1,
        TrainingImageType.SEG1: 2,
        TrainingImageType.SEG2: 3,
    })

    # ASSERT
    assert image_selection_widget._raw_channel_combo_box.count() == 1
    assert image_selection_widget._seg1_channel_combo_box.count() == 2
    assert image_selection_widget._seg2_channel_combo_box.count() == 3

    assert image_selection_widget._raw_channel_combo_box.currentIndex() == 0
    assert image_selection_widget._seg1_channel_combo_box.currentIndex() == 0
    assert image_selection_widget._seg2_channel_combo_box.currentIndex() == 0

    assert training_model.get_selected_channel(TrainingImageType.RAW) == 0
    assert training_model.get_selected_channel(TrainingImageType.SEG1) == 0
    assert training_model.get_selected_channel(TrainingImageType.SEG2) == 0

def test_combos_index_controls_model_state(
        qtbot: QtBot,
        image_selection_widget: ImageSelectionWidget,
        training_model: TrainingModel,
    ) -> None:
    # ARRANGE
    training_model.set_all_num_channels({
        TrainingImageType.RAW: 4,
        TrainingImageType.SEG1: 4,
        TrainingImageType.SEG2: 4,
    })

    # ACT
    image_selection_widget._raw_channel_combo_box.setCurrentIndex(1)
    image_selection_widget._seg1_channel_combo_box.setCurrentIndex(2)
    image_selection_widget._seg2_channel_combo_box.setCurrentIndex(3)

    # ASSERT
    assert training_model.get_selected_channel(TrainingImageType.RAW) == 1
    assert training_model.get_selected_channel(TrainingImageType.SEG1) == 2
    assert training_model.get_selected_channel(TrainingImageType.SEG2) == 3

def test_reset_num_channels_to_none(
        qtbot: QtBot,
        image_selection_widget: ImageSelectionWidget,
        training_model: TrainingModel,
    ) -> None:
    # ARRANGE
    training_model.set_all_num_channels({
        TrainingImageType.RAW: 4,
        TrainingImageType.SEG1: 4,
        TrainingImageType.SEG2: 4,
    })

    # ACT
    training_model.set_all_num_channels({
        TrainingImageType.RAW: None,
        TrainingImageType.SEG1: None,
        TrainingImageType.SEG2: None,
    })

    # ASSERT
    assert training_model.get_selected_channel(TrainingImageType.RAW) == None
    assert training_model.get_selected_channel(TrainingImageType.SEG1) == None
    assert training_model.get_selected_channel(TrainingImageType.SEG2) == None

