import pytest
from PyQt5.QtCore import Qt

from allencell_ml_segmenter.training.model_selection_widget import (
    ModelSelectionWidget,
)


@pytest.fixture
def model_selection_widget(qtbot):
    """
    Fixture that creates an instance of ModelSelectionWidget for testing.
    """
    return ModelSelectionWidget()


def test_radio_new_slot(qtbot, model_selection_widget):
    """
    Test the slot connected to the top radio button.
    """
    # ACT (disable combo box)
    model_selection_widget._radio_new_slot()

    # ASSERT
    assert not model_selection_widget.combo_box_existing.isEnabled()


def test_radio_existing_slot(qtbot, model_selection_widget):
    """
    Test the slot connected to the bottom radio button.
    """
    # ACT (enable combo box)
    model_selection_widget._radio_existing_slot()

    # ASSERT
    assert model_selection_widget.combo_box_existing.isEnabled()


def test_checkbox_slot(qtbot, model_selection_widget):
    """
    Test the slot connected to the checkbox.
    """
    # ASSERT (QLineEdit related to timeout limit is disabled by default)
    assert not model_selection_widget.hour_input.isEnabled()

    # ACT (enable QLineEdit related to timeout limit)
    model_selection_widget._checkbox_slot(Qt.Checked)

    # ASSERT
    assert model_selection_widget.hour_input.isEnabled()

    # ACT (disable QLineEdit related to timeout limit)
    model_selection_widget._checkbox_slot(Qt.Unchecked)

    # ASSERT
    assert not model_selection_widget.hour_input.isEnabled()
