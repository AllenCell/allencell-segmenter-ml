import pytest
from qtpy.QtCore import Qt

from allencell_ml_segmenter.prediction.model import PredictionModel
from allencell_ml_segmenter.prediction.model_input_widget import (
    ModelInputWidget,
)


@pytest.fixture
def prediction_model():
    return PredictionModel()


@pytest.fixture
def model_input_widget(prediction_model, qtbot):
    return ModelInputWidget(prediction_model)


def test_postprocessing_method(model_input_widget, prediction_model, qtbot):
    # ACT
    qtbot.mouseClick(model_input_widget._top_button, Qt.LeftButton)

    # ASSERT
    assert (
        prediction_model.get_postprocessing_method()
        == "simple threshold cutoff"
    )

    # ACT
    qtbot.mouseClick(model_input_widget._bottom_button, Qt.LeftButton)

    # ASSERT
    assert prediction_model.get_postprocessing_method() == "auto threshold"


def test_postprocessing_auto_threshold(model_input_widget, prediction_model):
    # ACT
    model_input_widget._bottom_input_box.setCurrentIndex(4)

    # ASSERT
    assert prediction_model.get_postprocessing_auto_threshold() == "minimum"

    # ACT
    model_input_widget._bottom_input_box.setCurrentIndex(6)

    # ASSERT
    assert prediction_model.get_postprocessing_auto_threshold() == "niblack"
