from pathlib import Path

import pytest
from allencell_ml_segmenter.prediction.model import PredictionModel
from allencell_ml_segmenter.prediction.service import ModelFileService


@pytest.fixture
def prediction_model() -> PredictionModel:
    return PredictionModel()


def test_service(prediction_model: PredictionModel) -> None:
    # ARRANGE
    ModelFileService(prediction_model)

    # ACT
    prediction_model.set_model_path(Path("example path"))

    # ASSERT STATE
    assert prediction_model.get_preprocessing_method() == "foo"
