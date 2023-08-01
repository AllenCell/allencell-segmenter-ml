import pytest
from PyQt5.QtCore import Qt
from pytestqt.qtbot import QtBot

from allencell_ml_segmenter.training.model_selection_widget import (
    ModelSelectionWidget,
)


@pytest.fixture
def model_selection_widget() -> ModelSelectionWidget:
    """
    Fixture that creates an instance of ModelSelectionWidget for testing.
    """
    return ModelSelectionWidget()


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
