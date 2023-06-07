import pytest
from unittest.mock import Mock
from allencell_ml_segmenter.models.main_model import MainModel
from allencell_ml_segmenter.core.publisher import Event
from allencell_ml_segmenter.views.training_view import TrainingView
from allencell_ml_segmenter.widgets.training_widget import TrainingWidget


@pytest.fixture
def main_model():
    return Mock(spec=MainModel)


@pytest.fixture
def training_view(main_model, qtbot):
    return TrainingView(main_model)


def test_init(training_view, main_model, qtbot):
    assert training_view._main_model == main_model
    assert training_view.layout().count() == 1
    assert isinstance(
        training_view.layout().itemAt(0).widget(), TrainingWidget
    )


def test_model_property(training_view, main_model):
    assert training_view._main_model == main_model


def test_handle_event_training_selected(training_view, main_model, qtbot):
    training_view.handle_event(Event.TRAINING_SELECTED)

    main_model.set_current_view.assert_called_once_with(training_view)


def test_back_to_main(training_view, main_model, qtbot):
    training_view.back_to_main()

    main_model.dispatch.assert_called_once_with(Event.MAIN_SELECTED)
