from pathlib import Path

import pytest
from PyQt5.QtCore import Qt
from pytestqt.qtbot import QtBot

from allencell_ml_segmenter.training.model_selection_widget import (
    ModelSelectionWidget,
)
from allencell_ml_segmenter.training.training_model import (
    TrainingModel,
    PatchSize,
)


@pytest.fixture
def training_model() -> TrainingModel:
    """
    Fixture that creates an instance of TrainingModel for testing.
    """
    return TrainingModel()


@pytest.fixture
def model_selection_widget(
    training_model: TrainingModel,
) -> ModelSelectionWidget:
    """
    Fixture that creates an instance of ModelSelectionWidget for testing.
    """
    return ModelSelectionWidget(training_model)


def test_radio_new_slot(
    qtbot: QtBot, model_selection_widget: ModelSelectionWidget
) -> None:
    """
    Test the slot connected to the "new model" radio button.
    """
    # ACT (disable combo box)
    model_selection_widget._combo_box_existing_models.setEnabled(
        True
    )  # explicitly enable the combobox to see if it gets disabled
    model_selection_widget._radio_new_model_slot()

    # ASSERT
    assert not model_selection_widget._combo_box_existing_models.isEnabled()


def test_radio_existing_slot(
    qtbot: QtBot, model_selection_widget: ModelSelectionWidget
) -> None:
    """
    Test the slot connected to the "existing model" radio button.
    """
    # ACT (enable combo box)
    model_selection_widget._combo_box_existing_models.setEnabled(
        False
    )  # explicitly disable the combobox to see if it gets enabled
    model_selection_widget._radio_existing_model_slot()

    # ASSERT
    assert model_selection_widget._combo_box_existing_models.isEnabled()


def test_checkbox_slot(
    qtbot: QtBot, model_selection_widget: ModelSelectionWidget
) -> None:
    """
    Test the slot connected to the timeout checkbox.
    """
    # ASSERT (QLineEdit related to timeout limit is disabled by default)
    assert not model_selection_widget._max_time_in_hours_input.isEnabled()

    # ACT (enable QLineEdit related to timeout limit)
    model_selection_widget._timeout_checkbox_slot(Qt.Checked)

    # ASSERT
    assert model_selection_widget._max_time_in_hours_input.isEnabled()

    # ACT (disable QLineEdit related to timeout limit)
    model_selection_widget._timeout_checkbox_slot(Qt.Unchecked)

    # ASSERT
    assert not model_selection_widget._max_time_in_hours_input.isEnabled()


def test_set_model_path(
    qtbot: QtBot,
    model_selection_widget: ModelSelectionWidget,
    training_model: TrainingModel,
) -> None:
    """
    Tests that the slots connected to the "start a new model" radio button and the existing model QCombBox properly set the model path field.
    """
    # ARRANGE - add arbitrary model path options to the QComboBox, since it does not come with default choices
    model_selection_widget._combo_box_existing_models.addItems(
        [f"dummy path {i}" for i in range(10)]
    )

    # ACT
    with qtbot.waitSignal(
        model_selection_widget._radio_existing_model.toggled
    ):
        model_selection_widget._radio_existing_model.click()  # enables the combo box

    model_selection_widget._combo_box_existing_models.setCurrentIndex(8)

    # ASSERT
    assert training_model.get_model_path() == Path("dummy path 8")

    # ACT
    model_selection_widget._combo_box_existing_models.setCurrentIndex(3)

    # ASSERT
    assert training_model.get_model_path() == Path("dummy path 3")

    # ACT - press "start a new model" radio button, which should set model_path to None
    with qtbot.waitSignal(model_selection_widget._radio_new_model.toggled):
        model_selection_widget._radio_new_model.click()

    # ASSERT
    assert training_model.get_model_path() is None


def test_set_patch_size(
    model_selection_widget: ModelSelectionWidget,
    training_model: TrainingModel,
) -> None:
    """
    Tests that using the associated combo box properly sets the patch size field.
    """
    # ACT
    model_selection_widget._structure_size_combo_box.setCurrentIndex(
        0
    )  # small

    # ASSERT
    assert training_model.get_patch_size() == PatchSize.SMALL

    # ACT
    model_selection_widget._structure_size_combo_box.setCurrentIndex(
        2
    )  # large

    # ASSERT
    assert training_model.get_patch_size() == PatchSize.LARGE


def test_set_image_dimensions(
    qtbot: QtBot,
    model_selection_widget: ModelSelectionWidget,
    training_model: TrainingModel,
) -> None:
    """
    Tests that checking the associated radio buttons properly sets the image dimensions.
    """
    # ACT
    with qtbot.waitSignal(model_selection_widget._radio_2d.toggled):
        model_selection_widget._radio_2d.click()

    # ASSERT
    assert training_model.get_image_dims() == 2

    # ACT
    with qtbot.waitSignal(model_selection_widget._radio_3d.toggled):
        model_selection_widget._radio_3d.click()

    # ASSERT
    assert training_model.get_image_dims() == 3


def test_set_max_epoch(
    qtbot: QtBot,
    model_selection_widget: ModelSelectionWidget,
    training_model: TrainingModel,
) -> None:
    """
    Tests that the max epoch field is properly set by the associated QLineEdit.
    """
    # ACT
    qtbot.keyClicks(model_selection_widget._max_epoch_input, "100")

    # ASSERT
    assert training_model.get_max_epoch() == 100


def test_set_max_time(
    qtbot: QtBot,
    model_selection_widget: ModelSelectionWidget,
    training_model: TrainingModel,
) -> None:
    """
    Tests that the max time field is properly set by the associated QLineEdit.
    """
    # ACT
    with qtbot.waitSignal(model_selection_widget._timeout_checkbox.toggled):
        model_selection_widget._timeout_checkbox.click()  # enables the QLineEdit

    qtbot.keyClicks(model_selection_widget._max_time_in_hours_input, "1")

    # ASSERT
    assert training_model.get_max_time() == 3600
