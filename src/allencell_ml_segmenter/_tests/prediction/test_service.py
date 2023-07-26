import pytest
from allencell_ml_segmenter.prediction.model import PredictionModel
from allencell_ml_segmenter.prediction.service import ModelFileService


@pytest.fixture
def prediction_model():
    return PredictionModel()


def test_service(prediction_model):
    # ARRANGE
    ModelFileService(prediction_model)

    # ACT
    prediction_model.set_model_path("random string")

    # ASSERT STATE
    assert prediction_model.get_preprocessing_method() == "foo"
