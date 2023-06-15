import pytest
from unittest.mock import Mock
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.core.publisher import Event
from allencell_ml_segmenter.sample.sample_view import SampleView


@pytest.fixture
def main_model():
    return Mock(spec=MainModel)


@pytest.fixture
def sample_view(main_model, qtbot):
    return SampleView(main_model)


def test_model_property(sample_view, main_model):
    assert sample_view._main_model == main_model


def integration_test_handle_event_training_selected(training_view):
    # ARRANGE
    model = MainModel()
    SampleView(model)

    # ACT
    model.dispatch(Event.VIEW_SELECTION_TRAINING)

    # ASSERT
    model.set_current_view.assert_called_once_with(training_view)
