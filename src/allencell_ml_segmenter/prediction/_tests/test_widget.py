import pytest

from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.prediction.view import PredictionView


@pytest.fixture
def main_model():
    return MainModel()


@pytest.fixture
def prediction_view(main_model, qtbot):
    return PredictionView(main_model)


def test_preprocessing_method(prediction_view):
    # ACT
    prediction_view._prediction_model.set_file_path("random string")

    # ASSERT
    assert prediction_view.model_input_widget.method.text() == "foo"


def test_postprocessing_simple_threshold_ui(prediction_view):
    # ACT
    prediction_view.model_input_widget.top_input_box.slider.setValue(29)

    # ASSERT
    assert (
        prediction_view.model_input_widget.top_input_box.label.text() == "0.29"
    )

    # ACT
    prediction_view.model_input_widget.top_input_box.label.setText("0.77")

    # ASSERT
    assert (
        prediction_view.model_input_widget.top_input_box.slider.value() == 77
    )

    # ACT
    prediction_view.model_input_widget.top_input_box.slider.setValue(41)

    # ASSERT
    assert (
        prediction_view.model_input_widget.top_input_box.label.text() == "0.41"
    )
