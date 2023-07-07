import pytest

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.prediction.model import PredictionModel
from allencell_ml_segmenter._tests.fakes.fake_subscriber import FakeSubscriber


@pytest.fixture
def prediction_model():
    return PredictionModel()


def test_file_path(prediction_model):
    # ARRANGE
    event_under_test: Event = Event.ACTION_PREDICTION_MODEL_FILE
    subscriber: FakeSubscriber = FakeSubscriber()
    prediction_model.subscribe(event_under_test, subscriber)

    # ACT
    prediction_model.set_file_path("example path")

    # ASSERT
    assert prediction_model.get_file_path() == "example path"
    assert subscriber.handled_event == event_under_test


def test_preprocessing_method(prediction_model):
    # ARRANGE
    event_under_test: Event = Event.ACTION_PREDICTION_PREPROCESSING_METHOD
    subscriber: FakeSubscriber = FakeSubscriber()
    prediction_model.subscribe(event_under_test, subscriber)

    # ACT
    prediction_model.set_preprocessing_method("example method")

    # ASSERT
    assert prediction_model.get_preprocessing_method() == "example method"
    assert subscriber.handled_event == event_under_test


def test_postprocessing_method(prediction_model):
    # ARRANGE
    event_under_test: Event = Event.ACTION_PREDICTION_POSTPROCESSING_METHOD
    subscriber: FakeSubscriber = FakeSubscriber()
    prediction_model.subscribe(event_under_test, subscriber)

    # ACT
    prediction_model.set_postprocessing_method("example method")

    # ASSERT
    assert prediction_model.get_postprocessing_method() == "example method"
    assert subscriber.handled_event == event_under_test


def test_postprocessing_simple_threshold_typed(prediction_model):
    # ARRANGE
    event_under_test: Event = (
        Event.ACTION_PREDICTION_POSTPROCESSING_SIMPLE_THRESHOLD_TYPED
    )
    subscriber: FakeSubscriber = FakeSubscriber()
    prediction_model.subscribe(event_under_test, subscriber)

    # ACT
    prediction_model.set_postprocessing_simple_threshold(0.01)
    prediction_model.dispatch(event_under_test)  # dispatch separately

    # ASSERT
    assert prediction_model.get_postprocessing_simple_threshold() == 0.01
    assert subscriber.handled_event == event_under_test


def test_postprocessing_simple_threshold_moved(prediction_model):
    # ARRANGE
    event_under_test: Event = (
        Event.ACTION_PREDICTION_POSTPROCESSING_SIMPLE_THRESHOLD_MOVED
    )
    subscriber: FakeSubscriber = FakeSubscriber()
    prediction_model.subscribe(event_under_test, subscriber)

    # ACT
    prediction_model.set_postprocessing_simple_threshold(0.01)
    prediction_model.dispatch(event_under_test)  # dispatch separately

    # ASSERT
    assert prediction_model.get_postprocessing_simple_threshold() == 0.01
    assert subscriber.handled_event == event_under_test


def test_postprocessing_auto_threshold(prediction_model):
    # ARRANGE
    event_under_test: Event = (
        Event.ACTION_PREDICTION_POSTPROCESSING_AUTO_THRESHOLD
    )
    subscriber: FakeSubscriber = FakeSubscriber()
    prediction_model.subscribe(event_under_test, subscriber)

    # ACT
    prediction_model.set_postprocessing_auto_threshold("example threshold")

    # ASSERT
    assert (
        prediction_model.get_postprocessing_auto_threshold()
        == "example threshold"
    )
    assert subscriber.handled_event == event_under_test
