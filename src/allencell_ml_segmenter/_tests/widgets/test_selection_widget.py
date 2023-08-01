import pytest
from pytestqt.qtbot import QtBot
from qtpy.QtWidgets import QPushButton
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.widgets.selection_widget import SelectionWidget


@pytest.fixture
def selection_widget(qtbot: QtBot) -> SelectionWidget:
    model: FakeModel = FakeModel()
    return SelectionWidget(model)


class FakeModel(MainModel):
    def __init__(self):
        super().__init__()
        self.dispatch_called: bool = False
        self.dispatch_event: Event = None

    def dispatch(self, event):
        self.dispatch_called = True
        self.dispatch_event = event


def test_init(selection_widget: SelectionWidget) -> None:
    assert isinstance(selection_widget.training_button, QPushButton)
    assert selection_widget.training_button.text() == "Training View"
    assert isinstance(selection_widget.prediction_button, QPushButton)
    assert selection_widget.prediction_button.text() == "Prediction View"


def test_training_button(selection_widget: SelectionWidget) -> None:
    selection_widget.training_button.click()
    assert selection_widget.model.dispatch_called
    assert (
        selection_widget.model.dispatch_event == Event.VIEW_SELECTION_TRAINING
    )


def test_prediction_button_click(selection_widget: SelectionWidget) -> None:
    selection_widget.prediction_button.click()
    assert selection_widget.model.dispatch_called
    assert (
        selection_widget.model.dispatch_event
        == Event.VIEW_SELECTION_PREDICTION
    )
