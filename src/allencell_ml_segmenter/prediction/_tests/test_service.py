import pytest
from allencell_ml_segmenter.prediction.model import PredictionModel
from allencell_ml_segmenter.prediction.service import ModelFileService


@pytest.fixture
def prediction_model():
    return PredictionModel()


def test_service(prediction_model):
    model_file_service = ModelFileService(prediction_model)
    model_file_service._model.set_file_path("random string")
    assert prediction_model.get_preprocessing_method() == "foo"
