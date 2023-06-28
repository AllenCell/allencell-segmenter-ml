import pytest
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.prediction.view import PredictionView


@pytest.fixture
def main_model():
    return MainModel()


@pytest.fixture
def prediction_view(main_model, qtbot):
    return PredictionView(main_model)


def test_model_property(prediction_view, main_model):
    assert prediction_view._main_model == main_model


def test_prediction_view(prediction_view, main_model):
    prediction_view._main_model.set_current_view(prediction_view)
    assert prediction_view._main_model.get_current_view() == prediction_view
