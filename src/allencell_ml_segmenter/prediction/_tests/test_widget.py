import pytest

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


# def test_preprocessing_method(model_input_widget, prediction_model):
#     # TODO: call the setter directly
#     # ACT
#     prediction_model.set_file_path("random string")
#
#     # ASSERT
#     # TODO: rename method
#     assert model_input_widget.method.text() == "foo"


# TODO: potentially move to sliderwithlabel widget tests
def test_postprocessing_simple_threshold_ui(model_input_widget):
    # ACT
    model_input_widget.top_input_box.slider.setValue(29)

    # ASSERT
    assert model_input_widget.top_input_box.label.text() == "0.29"

    # ACT
    model_input_widget.top_input_box.label.setText("0.77")

    # ASSERT
    assert model_input_widget.top_input_box.slider.value() == 77

    # ACT
    model_input_widget.top_input_box.slider.setValue(41)

    # ASSERT
    assert model_input_widget.top_input_box.label.text() == "0.41"
