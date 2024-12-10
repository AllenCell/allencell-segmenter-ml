from pathlib import Path

import pytest
from pytestqt.qtbot import QtBot

import allencell_ml_segmenter
from allencell_ml_segmenter._tests.fakes.fake_viewer import FakeViewer
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.prediction.model import (
    PredictionModel,
)
from allencell_ml_segmenter.core.image_data_extractor import (
    FakeImageDataExtractor,
)
from allencell_ml_segmenter.core.file_input_model import (
    InputMode,
    FileInputModel,
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
    return PredictionView(
        main_model, prediction_model, FileInputModel(), FakeViewer()
    )


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
    file_input_model: FileInputModel = FileInputModel()
    prediction_model: PredictionModel = PredictionModel()
    file_input_model.set_output_directory(
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "output_test_folder"
    )
    file_input_model.set_input_mode(InputMode.FROM_NAPARI_LAYERS)
    file_input_model.set_selected_paths(
        [Path("output_1.tiff"), Path("output_2.tiff")]
    )
    file_input_model.set_image_input_channel_index(0)
    fake_viewer: FakeViewer = FakeViewer()

    prediction_view: PredictionView = PredictionView(
        main_model,
        prediction_model,
        file_input_model,
        fake_viewer,
        img_data_extractor=FakeImageDataExtractor.global_instance(),
    )

    # ACT
    prediction_view.showResults()

    # ASSERT
    assert len(fake_viewer.get_all_labels()) == 2
    assert len(fake_viewer.get_all_images()) == 2
    assert fake_viewer.contains_layer("[raw] output_1.tiff")
    assert fake_viewer.contains_layer("[seg] output_1.tiff")
    assert fake_viewer.contains_layer("[raw] output_2.tiff")
    assert fake_viewer.contains_layer("[seg] output_2.tiff")


def test_show_results_non_empty_folder(main_model: MainModel) -> None:
    """
    Testing that only the new images in a folder will be shown after prediction.
    """
    # ARRANGE
    file_input_model: FileInputModel = FileInputModel()
    prediction_model: PredictionModel = PredictionModel()
    file_input_model.set_output_directory(
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "output_test_folder_extra"
    )
    file_input_model.set_input_mode(InputMode.FROM_NAPARI_LAYERS)
    file_input_model.set_selected_paths(
        [Path("output_3.tiff"), Path("output_4.tiff")]
    )
    file_input_model.set_image_input_channel_index(0)
    fake_viewer: FakeViewer = FakeViewer()

    prediction_view: PredictionView = PredictionView(
        main_model,
        prediction_model,
        file_input_model,
        fake_viewer,
        img_data_extractor=FakeImageDataExtractor.global_instance(),
    )

    # ACT
    prediction_view.showResults()

    # ASSERT
    assert len(fake_viewer.get_all_labels()) == 2
    assert len(fake_viewer.get_all_images()) == 2
    assert fake_viewer.contains_layer("[raw] output_3.tiff")
    assert fake_viewer.contains_layer("[seg] output_3.tiff")
    assert fake_viewer.contains_layer("[raw] output_4.tiff")
    assert fake_viewer.contains_layer("[seg] output_4.tiff")
