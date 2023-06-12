import pytest
from qtpy.QtWidgets import QPushButton
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.models.main_model import MainModel
from allencell_ml_segmenter.widgets.selection_widget import SelectionWidget


@pytest.fixture
def selection_widget(qtbot):
    model = FakeModel()
    return SelectionWidget(model)


class FakeModel(MainModel):
    def __init__(self):
        super().__init__()
        self.dispatch_called = False
        self.dispatch_event = None

    def dispatch(self, event):
        self.dispatch_called = True
        self.dispatch_event = event


def test_init(selection_widget):
    assert isinstance(selection_widget.training_button, QPushButton)
    assert selection_widget.training_button.text() == "Training View"
    assert isinstance(selection_widget.prediction_button, QPushButton)
    assert selection_widget.prediction_button.text() == "Example View"


def test_training_button(selection_widget):
    selection_widget.training_button.click()
    assert selection_widget.model.dispatch_called
    assert selection_widget.model.dispatch_event == Event.TRAINING_SELECTED


def test_prediction_button_click(selection_widget):
    selection_widget.prediction_button.click()
    assert selection_widget.model.dispatch_called
    assert selection_widget.model.dispatch_event == Event.EXAMPLE_SELECTED
