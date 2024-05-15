from pathlib import Path

import pytest
from aicsimageio import AICSImage
from pytestqt.qtbot import QtBot
from numpy import array_equal

import allencell_ml_segmenter
from allencell_ml_segmenter._tests.fakes.fake_viewer import FakeViewer
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.prediction.model import (
    PredictionModel,
    PredictionInputMode,
)
from allencell_ml_segmenter.prediction.view import PredictionView


@pytest.fixture
def main_model() -> MainModel:
    """
    Returns a MainModel instance for testing.
    """
    return MainModel()


@pytest.fixture
def prediction_view(main_model: MainModel, qtbot: QtBot) -> PredictionView:
    """
    Returns a PredictionView instance for testing.
    """
    prediction_model: PredictionModel = PredictionModel()
    return PredictionView(main_model, prediction_model, FakeViewer())


def test_prediction_view(
    prediction_view: PredictionView, main_model: MainModel
) -> None:
    """
    Tests that the PredictionView correctly sets the current view to itself
    """
    # ACT
    main_model.dispatch(Event.PROCESS_TRAINING_COMPLETE)

    # ASSERT
    assert main_model.get_current_view() == prediction_view


def test_show_results(main_model: MainModel) -> None:
    """
    Testing the showresults that runs after a prediction run
    """
    # ARRANGE
    prediction_model: PredictionModel = PredictionModel()
    prediction_model.set_output_directory(
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "output_test_folder"
    )
    prediction_model.set_prediction_input_mode(
        PredictionInputMode.FROM_NAPARI_LAYERS
    )
    fake_viewer: FakeViewer = FakeViewer()

    prediction_view: PredictionView = PredictionView(
        main_model, prediction_model, fake_viewer
    )

    # ACT
    prediction_view.showResults()

    # ASSERT
    assert (
        len(fake_viewer.get_all_images()) == 2
    )  # num img files in output_test_folder
