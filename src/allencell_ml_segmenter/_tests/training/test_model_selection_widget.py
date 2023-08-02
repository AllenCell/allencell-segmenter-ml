import pytest
from PyQt5.QtCore import Qt
from pytestqt.qtbot import QtBot

from allencell_ml_segmenter.training.model_selection_widget import (
    ModelSelectionWidget,
)
from allencell_ml_segmenter.training.training_model import TrainingModel


@pytest.fixture
def model_selection_widget(qtbot: QtBot) -> ModelSelectionWidget:
    """
    Fixture that creates an instance of ModelSelectionWidget for testing.
    """
    return ModelSelectionWidget(TrainingModel())


def test_radio_new_slot(
    qtbot: QtBot, model_selection_widget: ModelSelectionWidget
) -> None:
    """
    Test the slot connected to the top radio button.
    """
    # ACT (disable combo box)
    model_selection_widget._combo_box_existing.setEnabled(
        True
    )  # explicitly enable the combobox to see if it gets disabled
    model_selection_widget._radio_new_model_slot()

    # ASSERT
    assert not model_selection_widget._combo_box_existing.isEnabled()


def test_radio_existing_slot(
    qtbot: QtBot, model_selection_widget: ModelSelectionWidget
) -> None:
    """
    Test the slot connected to the bottom radio button.
    """
    # ACT (enable combo box)
    model_selection_widget._combo_box_existing.setEnabled(
        False
    )  # explicitly disable the combobox to see if it gets enabled
    model_selection_widget._radio_existing_model_slot()

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
    model_selection_widget._timeout_checkbox_slot(Qt.Checked)

    # ASSERT
    assert model_selection_widget._timeout_hour_input.isEnabled()

    # ACT (disable QLineEdit related to timeout limit)
    model_selection_widget._timeout_checkbox_slot(Qt.Unchecked)

    # ASSERT
    assert not model_selection_widget._timeout_hour_input.isEnabled()
