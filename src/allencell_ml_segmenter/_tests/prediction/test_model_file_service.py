from pathlib import Path

import allencell_ml_segmenter
from allencell_ml_segmenter.prediction.service import ModelFileService
from allencell_ml_segmenter.prediction.model import PredictionModel


class PredicitonModel:
    pass


def test_extract_num_channels_in_a_folder() -> None:
    # Arrange
    prediction_model: PredictionModel = PredictionModel()
    model_file_service: ModelFileService = ModelFileService(prediction_model)

    # Act / Assert
    assert model_file_service.extract_num_channels_in_folder(Path(allencell_ml_segmenter.__file__).parent / "_tests" / "test_files") == 3
