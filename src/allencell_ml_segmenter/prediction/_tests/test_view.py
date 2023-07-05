import pytest

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.prediction.view import PredictionView


@pytest.fixture
def main_model():
    return MainModel()


@pytest.fixture
def prediction_view(main_model, qtbot):
    return PredictionView(main_model)


def test_prediction_view(prediction_view, main_model):
    # ACT
    main_model.dispatch(Event.VIEW_SELECTION_PREDICTION)

    # ASSERT
    assert main_model.get_current_view() == prediction_view
