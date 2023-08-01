import pytest

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.prediction.model import PredictionModel
from allencell_ml_segmenter._tests.fakes.fake_subscriber import FakeSubscriber


@pytest.fixture
def prediction_model() -> PredictionModel:
    return PredictionModel()


def test_input_image_paths(prediction_model: PredictionModel) -> None:
    # no event currently dispatched
    # ACT
    prediction_model.set_input_image_paths(
        ["example path " + str(i) for i in range(10)]
    )

    # ASSERT
    assert prediction_model.get_input_image_paths() == [
        "example path " + str(i) for i in range(10)
    ]


def test_image_input_channel_index(prediction_model: PredictionModel) -> None:
    # ACT
    prediction_model.set_image_input_channel_index(17)

    # ASSERT
    assert prediction_model.get_image_input_channel_index() == 17


def test_output_directory(prediction_model: PredictionModel) -> None:
    # ACT
    prediction_model.set_output_directory("example directory")

    # ASSERT
    assert prediction_model.get_output_directory() == "example directory"


def test_model_path(prediction_model: PredictionModel) -> None:
    # ARRANGE
    event_under_test: Event = Event.ACTION_PREDICTION_MODEL_FILE
    subscriber: FakeSubscriber = FakeSubscriber()
    prediction_model.subscribe(event_under_test, subscriber, subscriber.handle)

    # ACT
    prediction_model.set_model_path("example path")

    # ASSERT
    assert prediction_model.get_model_path() == "example path"
    assert subscriber.was_handled(event_under_test)


def test_preprocessing_method(prediction_model: PredictionModel) -> None:
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
