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


def test_postprocessing_method(model_input_widget):
    # ACT
    # TODO: use qtbot actions instead
    model_input_widget._top_radio_button_slot()

    # ASSERT
    assert (
        model_input_widget._model.get_postprocessing_method()
        == "simple threshold cutoff"
    )

    # ACT
    model_input_widget._bottom_radio_button_slot()

    # ASSERT
    assert (
        model_input_widget._model.get_postprocessing_method()
        == "auto threshold"
    )


def test_postprocessing_auto_threshold(model_input_widget):
    # ACT
    model_input_widget._bottom_input_box.setCurrentIndex(4)

    # ASSERT
    assert (
        model_input_widget._model.get_postprocessing_auto_threshold()
        == "minimum"
    )

    # ACT
    model_input_widget._bottom_input_box.setCurrentIndex(6)

    # ASSERT
    assert (
        model_input_widget._model.get_postprocessing_auto_threshold()
        == "niblack"
    )
