import pytest

from allencell_ml_segmenter._tests.fakes.fake_subscriber import FakeSubscriber
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.thresholding.thresholding_model import (
    ThresholdingModel,
)


@pytest.fixture
def thresholding_model() -> ThresholdingModel:
    model = ThresholdingModel()
    return model


def test_set_thresholding_value_dispatches_event(thresholding_model):
    fake_subscriber: FakeSubscriber = FakeSubscriber()
    thresholding_model.subscribe(
        Event.ACTION_EXECUTE_THRESHOLDING,
        fake_subscriber,
        fake_subscriber.handle,
    )

    thresholding_model.set_thresholding_value(2)

    assert fake_subscriber.was_handled(Event.ACTION_EXECUTE_THRESHOLDING)


def test_set_autothresholding_enabled_dispatches_event(thresholding_model):
    fake_subscriber: FakeSubscriber = FakeSubscriber()
    thresholding_model.subscribe(
        Event.ACTION_EXECUTE_THRESHOLDING,
        fake_subscriber,
        fake_subscriber.handle,
    )

    thresholding_model.set_autothresholding_enabled(True)

    assert fake_subscriber.was_handled(Event.ACTION_EXECUTE_THRESHOLDING)


def test_dispatch_save_thresholded_images(thresholding_model):
    fake_subscriber: FakeSubscriber = FakeSubscriber()
    thresholding_model.subscribe(
        Event.ACTION_SAVE_THRESHOLDING_IMAGES,
        fake_subscriber,
        fake_subscriber.handle,
    )

    thresholding_model.dispatch_save_thresholded_images()

    assert fake_subscriber.was_handled(Event.ACTION_SAVE_THRESHOLDING_IMAGES)
