from pathlib import Path

import pytest
from aicsimageio import AICSImage
from pytestqt.qtbot import QtBot
from numpy import array_equal

import allencell_ml_segmenter
from allencell_ml_segmenter._tests.fakes.fake_viewer import FakeViewer
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.prediction.model import PredictionModel
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

def test_show_results(
main_model: MainModel) -> None:
    """
    Testing the showresults that runs after a prediction run
    """
    # ARRANGE
    prediction_model: PredictionModel = PredictionModel()
    prediction_model.set_output_directory(Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "output_test_folder")
    fake_viewer: FakeViewer = FakeViewer()

    prediction_view: PredictionView = PredictionView(main_model, prediction_model, fake_viewer)

    # ACT
    prediction_view.showResults()

    # ASSERT
    assert len(fake_viewer.images_added) == 2 # correct number
    image: Path = Path(allencell_ml_segmenter.__file__).parent / "_tests" / "test_files" / "output_test_folder" / "seg" / "output_1.tiff"
    assert array_equal(fake_viewer.images_added[image.name], AICSImage(image).data, equal_nan=True)





