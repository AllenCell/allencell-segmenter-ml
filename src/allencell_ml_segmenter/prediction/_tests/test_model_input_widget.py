import pytest

from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.prediction.model_input_widget import (
    ModelInputWidget,
)
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


def test_postprocessing_method(prediction_view):
    # ARRANGE
    model_input_widget: ModelInputWidget = ModelInputWidget(
        prediction_view._prediction_model
    )

    # ACT
    model_input_widget.top_radio_button_slot()

    # ASSERT
    assert (
        prediction_view._prediction_model.get_postprocessing_method()
        == "simple threshold cutoff"
    )

    # ACT
    model_input_widget.mid_radio_button_slot()

    # ASSERT
    assert (
        prediction_view._prediction_model.get_postprocessing_method()
        == "auto threshold"
    )
