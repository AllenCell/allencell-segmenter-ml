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


def test_postprocessing_auto_threshold(prediction_view):
    # ACT
    prediction_view.model_input_widget.mid_input_box.setCurrentIndex(4)

    # ASSERT
    assert (
        prediction_view._prediction_model.get_postprocessing_auto_threshold()
        == "minimum"
    )

    # ACT
    prediction_view.model_input_widget.mid_input_box.setCurrentIndex(6)

    # ASSERT
    assert (
        prediction_view._prediction_model.get_postprocessing_auto_threshold()
        == "niblack"
    )


def test_postprocessing_simple_threshold(prediction_view):
    # ACT
    prediction_view.model_input_widget.top_input_box.slider.setValue(29)

    # ASSERT
    assert (
        prediction_view._prediction_model.get_postprocessing_simple_threshold()
        == 0.29
    )

    # ACT
    prediction_view.model_input_widget.top_input_box.label.setText("0.77")

    # ASSERT
    assert (
        prediction_view._prediction_model.get_postprocessing_simple_threshold()
        == 0.77
    )

    # ACT
    prediction_view.model_input_widget.top_input_box.slider.setValue(41)

    # ASSERT
    assert (
        prediction_view._prediction_model.get_postprocessing_simple_threshold()
        == 0.41
    )
