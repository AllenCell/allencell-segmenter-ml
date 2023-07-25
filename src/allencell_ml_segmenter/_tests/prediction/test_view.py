import pytest

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.prediction.view import PredictionView


@pytest.fixture
def prediction_view(qtbot):
    return PredictionView()


# def test_prediction_view(prediction_view):
#     # ACT
#     main_model.dispatch(Event.VIEW_SELECTION_PREDICTION)
#
#     # ASSERT
#     assert main_model.get_current_view() == prediction_view
