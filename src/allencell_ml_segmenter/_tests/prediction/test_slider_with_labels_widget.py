import pytest

from allencell_ml_segmenter.prediction.model import PredictionModel
from allencell_ml_segmenter.widgets.slider_with_labels_widget import (
    SliderWithLabels,
)


@pytest.fixture
def prediction_model():
    return PredictionModel()


@pytest.fixture
def slider_with_labels_widget(prediction_model, qtbot):
    return SliderWithLabels(0, 1, prediction_model)


def test_postprocessing_simple_threshold(
    slider_with_labels_widget, prediction_model
):
    # ACT
    slider_with_labels_widget.set_slider_value(0.29)

    # ASSERT
    assert prediction_model.get_postprocessing_simple_threshold() == 0.29
    assert slider_with_labels_widget._label.text() == "0.29"

    # ACT
    slider_with_labels_widget.set_label_value(0.77)

    # ASSERT
    assert prediction_model.get_postprocessing_simple_threshold() == 0.77
    assert slider_with_labels_widget._slider.value() == 77

    # ACT
    slider_with_labels_widget.set_slider_value(0.41)

    # ASSERT
    assert prediction_model.get_postprocessing_simple_threshold() == 0.41
    assert slider_with_labels_widget._label.text() == "0.41"
