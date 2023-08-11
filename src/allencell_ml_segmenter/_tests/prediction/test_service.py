from pathlib import Path

import pytest
from allencell_ml_segmenter.prediction.model import PredictionModel
from allencell_ml_segmenter.prediction.service import ModelFileService


@pytest.fixture
def prediction_model() -> PredictionModel:
    """
    Returns a PredictionModel instance for testing.
    """
    return PredictionModel()


def test_service(prediction_model: PredictionModel) -> None:
    """
    Tests that the ModelFileService correctly sets the preprocessing
    method (currently a dummy string) in response to model path being set.
    """
    # ARRANGE
    ModelFileService(prediction_model)

    # ACT
    prediction_model.set_model_path(Path("example path"))

    # ASSERT STATE
    # TODO: replace assertion after implementing real functionality
    assert prediction_model.get_preprocessing_method() == "foo"
