import pytest
from qtpy.QtWidgets import QPushButton
from unittest.mock import Mock
from allencell_ml_segmenter.widgets.training_widget import TrainingWidget

@pytest.fixture
def training_widget(qtbot):
    return TrainingWidget()


def test_init(training_widget):
    assert isinstance(training_widget.btn, QPushButton)
    assert training_widget.btn.text() == "Start Training"
    assert isinstance(training_widget.return_btn, QPushButton)
    assert training_widget.return_btn.text() == "Return"

def test_connect_slots(training_widget):
    functions = [Mock(), Mock()]

    training_widget.connectSlots(functions)

    training_widget.btn.click()
    training_widget.return_btn.click()

    functions[0].assert_called_once()
    functions[1].assert_called_once()
