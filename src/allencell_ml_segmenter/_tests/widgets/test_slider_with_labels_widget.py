import pytest
from unittest.mock import Mock
from allencell_ml_segmenter.prediction.model import PredictionModel
from allencell_ml_segmenter.widgets.slider_with_labels_widget import (
    SliderWithLabels,
)
from PyQt5.QtTest import QTest

#TODO redo tests once replaced with magicgui floatslider
@pytest.fixture
def prediction_model():
    return PredictionModel()

@pytest.fixture
def slider_with_labels(prediction_model, qtbot):
    return SliderWithLabels(0, 100, prediction_model)

def test_label_update_model(slider_with_labels):
    # ACT
    QTest.keyClicks(slider_with_labels._label, "3")

    # Assert
    assert slider_with_labels._model.get_postprocessing_simple_threshold() == 3.0

    # ACT
    QTest.keyClicks(slider_with_labels._label, "30")

    # Assert
    assert slider_with_labels._model.get_postprocessing_simple_threshold() == 30

