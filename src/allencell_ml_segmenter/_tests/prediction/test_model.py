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


def test_model_path(prediction_model: PredictionModel) -> None:
    """
    Tests that the model path is set and retrieved properly, and that the correct event is fired off.
    """
    # ARRANGE
    event_under_test: Event = Event.ACTION_PREDICTION_MODEL_FILE
    subscriber: FakeSubscriber = FakeSubscriber()
    prediction_model.subscribe(event_under_test, subscriber, subscriber.handle)
    dummy_path: Path = Path("example path")

    # ACT
    prediction_model.set_model_path(dummy_path)

    # ASSERT
    assert prediction_model.get_model_path() == dummy_path
    assert subscriber.was_handled(event_under_test)


def test_preprocessing_method(prediction_model: PredictionModel) -> None:
    """
    Tests that the preprocessing method is set and retrieved properly, and that the correct event is fired off.
    """
    # ARRANGE
    event_under_test: Event = Event.ACTION_PREDICTION_PREPROCESSING_METHOD
    subscriber: FakeSubscriber = FakeSubscriber()
    prediction_model.subscribe(event_under_test, subscriber, subscriber.handle)

    # ACT
    prediction_model.set_preprocessing_method("example method")

    # ASSERT
    assert prediction_model.get_preprocessing_method() == "example method"
    assert subscriber.was_handled(event_under_test)


def test_postprocessing_method(prediction_model: PredictionModel) -> None:
    """
    Tests that the postprocessing method is set and retrieved properly, and that the correct event is fired off.
    """
    # ARRANGE
    event_under_test: Event = Event.ACTION_PREDICTION_POSTPROCESSING_METHOD
    subscriber: FakeSubscriber = FakeSubscriber()
    prediction_model.subscribe(event_under_test, subscriber, subscriber.handle)

    # ACT
    prediction_model.set_postprocessing_method("example method")

    # ASSERT
    assert prediction_model.get_postprocessing_method() == "example method"
    assert subscriber.was_handled(event_under_test)


def test_postprocessing_simple_threshold(
    prediction_model: PredictionModel,
) -> None:
    """
    Tests that the postprocessing simple threshold is set and retrieved properly, and that the correct event is fired off.
    """
    # ARRANGE
    event_under_test: Event = (
        Event.ACTION_PREDICTION_POSTPROCESSING_SIMPLE_THRESHOLD
    )
    subscriber: FakeSubscriber = FakeSubscriber()
    prediction_model.subscribe(event_under_test, subscriber, subscriber.handle)

    # ACT
    prediction_model.set_postprocessing_simple_threshold(0.01)

    # ASSERT
    assert prediction_model.get_postprocessing_simple_threshold() == 0.01
    assert subscriber.was_handled(event_under_test)


def test_postprocessing_auto_threshold(
    prediction_model: PredictionModel,
) -> None:
    """
    Tests that the postprocessing auto threshold is set and retrieved properly, and that the correct event is fired off.
    """
    # ARRANGE
    event_under_test: Event = (
        Event.ACTION_PREDICTION_POSTPROCESSING_AUTO_THRESHOLD
    )
    subscriber: FakeSubscriber = FakeSubscriber()
    prediction_model.subscribe(event_under_test, subscriber, subscriber.handle)

    # ACT
    prediction_model.set_postprocessing_auto_threshold("example threshold")
    assert subscriber.was_handled(event_under_test)

    # ASSERT
    assert (
        prediction_model.get_postprocessing_auto_threshold()
        == "example threshold"
    )
