from pathlib import Path

import pytest
from PyQt5.QtCore import Qt
from pytestqt.qtbot import QtBot

from allencell_ml_segmenter.training.model_selection_widget import (
    ModelSelectionWidget,
)
from allencell_ml_segmenter.training.training_model import TrainingModel


@pytest.fixture
def training_model() -> TrainingModel:
    """
    Fixture that creates an instance of TrainingModel for testing.
    """
    return TrainingModel()


@pytest.fixture
def model_selection_widget(
    qtbot: QtBot, training_model: TrainingModel
) -> ModelSelectionWidget:
    """
    Fixture that creates an instance of ModelSelectionWidget for testing.
    """
    return ModelSelectionWidget(training_model)


def test_radio_new_slot(
    qtbot: QtBot, model_selection_widget: ModelSelectionWidget
) -> None:
    """
    Test the slot connected to the top radio button.
    """
    # ARRANGE - explicitly enable model_selection_widget._combo_box_existing
    model_selection_widget._combo_box_existing.setEnabled(True)

    # ACT (disable combo box)
    with qtbot.waitSignals([model_selection_widget._radio_new_model.toggled]):
        model_selection_widget._radio_new_model.click()

    # ASSERT
    assert not model_selection_widget._combo_box_existing.isEnabled()


def test_radio_existing_slot(
    qtbot: QtBot, model_selection_widget: ModelSelectionWidget
) -> None:
    """
    Test the slot connected to the bottom radio button.
    """
    # ARRANGE - explicitly disable model_selection_widget._combo_box_existing
    model_selection_widget._combo_box_existing.setEnabled(False)

    # ACT (enable combo box)
    with qtbot.waitSignals(
        [model_selection_widget._radio_existing_model.toggled]
    ):
        model_selection_widget._radio_existing_model.click()

    # ASSERT
    assert model_selection_widget._combo_box_existing.isEnabled()


def test_checkbox_slot(
    qtbot: QtBot, model_selection_widget: ModelSelectionWidget
) -> None:
    """
    Test the slot connected to the checkbox.
    """
    # ASSERT (QLineEdit related to timeout limit is disabled by default)
    assert not model_selection_widget._timeout_hour_input.isEnabled()

    # ACT (enable QLineEdit related to timeout limit)
    with qtbot.waitSignals(
        [model_selection_widget._timeout_checkbox.stateChanged]
    ):
        model_selection_widget._timeout_checkbox.click()

    # ASSERT
    assert model_selection_widget._timeout_hour_input.isEnabled()

    # ACT (disabled QLineEdit related to timeout limit)
    with qtbot.waitSignals(
        [model_selection_widget._timeout_checkbox.stateChanged]
    ):
        model_selection_widget._timeout_checkbox.click()

    # ASSERT
    assert not model_selection_widget._timeout_hour_input.isEnabled()


def test_set_model_path(
    qtbot: QtBot,
    model_selection_widget: ModelSelectionWidget,
    training_model: TrainingModel,
) -> None:
    """
    Tests that the slots connected to the "start a new model" radio button and the existing model QCombBox properly set the model path field.
    """
    # ARRANGE - add arbitrary model path options to the QComboBox, since it does not come with default choices
    model_selection_widget._combo_box_existing.addItems(
        [f"dummy path {i}" for i in range(10)]
    )
    model_selection_widget._combo_box_existing.setEnabled(True)
    # TODO: enable the combo box instead by clicking the "existing model" radio button

    # ACT
    model_selection_widget._combo_box_existing.setCurrentIndex(8)

    # ASSERT
    assert training_model.get_model_path() == Path("dummy path 8")

    # ACT
    model_selection_widget._combo_box_existing.setCurrentIndex(3)

    # ASSERT
    assert training_model.get_model_path() == Path("dummy path 3")

    # ACT - press "start a new model" radio button, which should set model_path to None

    # TODO: find out why calling .click() doesn't send a clicked or toggled signal, and why timeout does not work
    # qtbot.mouseClick(model_selection_widget._radio_new_model, Qt.LeftButton)
    # while qtbot.waitSignals(
    #     [model_selection_widget._radio_new_model.toggled], timeout=1000
    # ):
    #     model_selection_widget._radio_new_model.click()

    # ASSERT
    # TODO: uncomment after above is solved
    # assert training_model.get_model_path() is None
