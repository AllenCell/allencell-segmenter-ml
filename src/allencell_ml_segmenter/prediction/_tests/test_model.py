import pytest

from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.prediction.model_input_widget import (
    ModelInputWidget,
)
from allencell_ml_segmenter.prediction.view import PredictionView


@pytest.fixture
def main_model():
    return MainModel()
