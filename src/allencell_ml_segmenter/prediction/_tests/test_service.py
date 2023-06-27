import pytest
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.prediction.model import PredictionModel
from allencell_ml_segmenter.prediction.service import ModelFileService


@pytest.fixture
def prediction_model():
    return PredictionModel()


def test_service(prediction_model):
    model_file_service = ModelFileService(prediction_model)
    prediction_model.dispatch(Event.ACTION_PREDICTION_MODEL_FILE_SELECTED)

    assert prediction_model.get_preprocessing_method() == "foo"
