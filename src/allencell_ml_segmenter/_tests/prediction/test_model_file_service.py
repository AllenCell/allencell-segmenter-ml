from pathlib import Path

import allencell_ml_segmenter
from allencell_ml_segmenter.prediction.service import ModelFileService
from allencell_ml_segmenter.prediction.model import PredictionModel

import pytest


def test_extract_num_channels_from_folder() -> None:
    # Arrange
    prediction_model: PredictionModel = PredictionModel()
    prediction_model.set_input_image_path(
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "images"
    )
    model_file_service: ModelFileService = ModelFileService(prediction_model)

    # Act / Assert
    assert model_file_service.extract_num_channels() == 3


def test_extract_num_channels_from_csv() -> None:
    # Arrange
    prediction_model: PredictionModel = PredictionModel()
    prediction_model.set_input_image_path(
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "csv"
        / "test_csv.csv"
    )
    model_file_service: ModelFileService = ModelFileService(prediction_model)

    # Act / Assert
    assert model_file_service.extract_num_channels() == 3


def test_extract_num_channels_from_viewer() -> None:
    # Arrange
    prediction_model: PredictionModel = PredictionModel()
    prediction_model.set_selected_paths(
        [
            Path(allencell_ml_segmenter.__file__).parent
            / "_tests"
            / "test_files"
            / "images"
            / "test_2_channels.tiff"
        ]
    )
    model_file_service: ModelFileService = ModelFileService(prediction_model)

    # Act / Assert
    assert model_file_service.extract_num_channels() == 3


def test_extract_num_channels_empty_selected_paths() -> None:
    # Arrange
    prediction_model: PredictionModel = PredictionModel()
    prediction_model.set_selected_paths([])
    model_file_service: ModelFileService = ModelFileService(prediction_model)

    with pytest.raises(ValueError):
        model_file_service.extract_num_channels()


def test_extract_num_channels_fields_unset() -> None:
    # Arrange
    prediction_model: PredictionModel = PredictionModel()
    model_file_service: ModelFileService = ModelFileService(prediction_model)

    with pytest.raises(ValueError):
        model_file_service.extract_num_channels()


def test_extract_num_channels_bad_input_path() -> None:
    # Arrange
    prediction_model: PredictionModel = PredictionModel()
    prediction_model.set_input_image_path(Path("./x/y/z/bad.png"))
    model_file_service: ModelFileService = ModelFileService(prediction_model)

    with pytest.raises(ValueError):
        model_file_service.extract_num_channels()
