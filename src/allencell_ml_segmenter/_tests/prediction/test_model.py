from pathlib import Path
from typing import List

import pytest

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.prediction.model import PredictionModel
from allencell_ml_segmenter._tests.fakes.fake_subscriber import FakeSubscriber


@pytest.fixture
def prediction_model() -> PredictionModel:
    """
    Fixture that creates an instance of PredictionModel for testing.
    """
    return PredictionModel()


def test_input_image_paths(prediction_model: PredictionModel) -> None:
    """
    Tests that the input image paths are set and retrieved properly.
    """
    # ARRANGE
    dummy_paths: List[Path] = [
        Path("example path " + str(i)) for i in range(10)
    ]

    # ACT
    prediction_model.set_input_image_path(dummy_paths)

    # ASSERT
    assert prediction_model.get_input_image_path() == dummy_paths


def test_image_input_channel_index(prediction_model: PredictionModel) -> None:
    """
    Tests that the channel index is set and retrieved properly.
    """
    for i in range(10):
        # ACT
        prediction_model.set_image_input_channel_index(i)

        # ASSERT
        assert prediction_model.get_image_input_channel_index() == i


def test_output_directory(prediction_model: PredictionModel) -> None:
    """
    Tests that the output directory is set and retrieved properly.
    """
    # ARRANGE
    dummy_path: Path = Path("example path")

    # ACT
    prediction_model.set_output_directory(dummy_path)

    # ASSERT
    assert prediction_model.get_output_directory() == dummy_path
